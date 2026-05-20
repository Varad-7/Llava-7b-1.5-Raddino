"""
LLaVA-RADDino: LLaVA with RAD-DINO Vision Encoder.

Architecture:
  Image (224×224×3)
    → RAD-DINO encoder          → [B, 256, 768]
    → MLP Projector (768→4096)  → [B, 256, 4096]
    → Concat with text embeds   → [B, 256+T, 4096]
    → Vicuna-7B LLM             → text generation

The projector is a 2-layer MLP that bridges the RAD-DINO feature space
to the Vicuna-7B embedding space. It is the ONLY randomly initialized
component and must be trained first (Stage 1) before fine-tuning the
LLM with LoRA (Stage 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

from raddino_encoder import RadDinoEncoder
import config as cfg


class MultimodalProjector(nn.Module):
    """
    2-layer MLP projector: Linear(768→4096) → GELU → Linear(4096→4096).

    Bridges RAD-DINO's 768-dim features to Vicuna-7B's 4096-dim embedding space.
    This is initialized from scratch and must be trained.
    """

    def __init__(
        self,
        vision_hidden_size: int = cfg.RADDINO_HIDDEN_SIZE,
        llm_hidden_size: int = cfg.LLM_HIDDEN_SIZE,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size)

        # Xavier initialization for stable training start
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 768] vision features from RAD-DINO

        Returns:
            [B, N, 4096] projected features for LLM
        """
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class LLaVARaddino(nn.Module):
    """
    Full LLaVA-RADDino model.

    Components:
      1. RadDinoEncoder  — frozen vision encoder (768-dim output)
      2. MultimodalProjector — trainable bridge (768 → 4096)
      3. Vicuna-7B LLM   — language model (frozen in Stage 1, LoRA in Stage 2)

    The forward pass:
      1. Encode image → project → [B, 256, 4096]
      2. Embed text tokens → [B, T, 4096]
      3. Concatenate: [prefix_embeds | image_embeds | suffix_embeds | report_embeds]
      4. Forward through LLM with causal LM loss
    """

    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()

        # ---- Vision encoder (frozen) ----
        self.vision_encoder = RadDinoEncoder(cfg.RADDINO_MODEL_NAME)

        # ---- Multimodal projector (trainable, randomly initialized) ----
        self.projector = MultimodalProjector(
            vision_hidden_size=cfg.RADDINO_HIDDEN_SIZE,
            llm_hidden_size=cfg.LLM_HIDDEN_SIZE,
        ).to(dtype=torch.bfloat16)  # Match LLM dtype to prevent float vs bf16 mismatch

        # ---- Language model ----
        print(f"[LLaVARaddino] Loading LLM: {cfg.LLM_MODEL_NAME} ...")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            cfg.LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        # Resize embeddings if tokenizer was modified
        self.language_model.resize_token_embeddings(len(tokenizer))
        print(f"[LLaVARaddino] LLM loaded. Parameters: {sum(p.numel() for p in self.language_model.parameters()) / 1e6:.1f}M")

        # ---- Pre-tokenize fixed prompt parts ----
        # These are stored as buffers so they move with the model to the right device
        prefix_tokens = tokenizer(
            cfg.PROMPT_USER,
            add_special_tokens=True,  # adds BOS <s>
            return_tensors="pt",
        )
        suffix_tokens = tokenizer(
            cfg.PROMPT_INSTRUCTION,
            add_special_tokens=False,
            return_tensors="pt",
        )
        self.register_buffer("prefix_ids", prefix_tokens.input_ids)  # [1, T_prefix]
        self.register_buffer("suffix_ids", suffix_tokens.input_ids)  # [1, T_suffix]

        print(f"[LLaVARaddino] Prompt: prefix={self.prefix_ids.shape[1]} tokens, "
              f"suffix={self.suffix_ids.shape[1]} tokens, "
              f"image={cfg.NUM_IMAGE_TOKENS} tokens")

    @property
    def llm_dtype(self) -> torch.dtype:
        """Get the dtype of the LLM weights (bf16)."""
        return next(self.language_model.parameters()).dtype

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images through RAD-DINO + projector.

        Args:
            pixel_values: [B, 3, 224, 224]

        Returns:
            image_embeds: [B, 256, 4096] in LLM dtype (bf16)
        """
        # RAD-DINO features (frozen, no grad)
        patch_features = self.vision_encoder(pixel_values)  # [B, 256, 768]

        # Cast to projector dtype (bf16) and project to LLM dim
        patch_features = patch_features.to(dtype=self.projector.linear_1.weight.dtype)
        image_embeds = self.projector(patch_features)  # [B, 256, 4096]

        # Ensure output matches LLM dtype
        image_embeds = image_embeds.to(dtype=self.llm_dtype)

        # Dimension check
        B = pixel_values.shape[0]
        assert image_embeds.shape == (B, cfg.NUM_IMAGE_TOKENS, cfg.LLM_HIDDEN_SIZE), (
            f"Projector output mismatch: expected ({B}, {cfg.NUM_IMAGE_TOKENS}, "
            f"{cfg.LLM_HIDDEN_SIZE}), got {image_embeds.shape}"
        )

        return image_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        report_ids: torch.Tensor,
        report_attention_mask: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> dict:
        """
        Full forward pass for training.

        The input sequence is constructed as:
          [BOS USER: ] [image_tokens × 256] [\\nGenerate...\\nASSISTANT: ] [report EOS]

        Labels are -100 for everything BEFORE the report, and real token IDs
        for the report tokens. This means the model only learns to generate
        the report conditioned on the image + prompt.

        Args:
            pixel_values:          [B, 3, 224, 224] preprocessed images
            report_ids:            [B, T_report]     tokenized report + EOS (padded)
            report_attention_mask: [B, T_report]     1 for real tokens, 0 for padding
            label_smoothing:       float, label smoothing factor (0.0 = none)

        Returns:
            dict with 'loss' and 'logits'
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # ---- 1. Encode image ----
        image_embeds = self.encode_image(pixel_values)  # [B, 256, 4096]

        # ---- 2. Get text embeddings ----
        embed_fn = self.language_model.get_input_embeddings()

        prefix_embeds = embed_fn(self.prefix_ids.expand(B, -1))   # [B, T1, 4096]
        suffix_embeds = embed_fn(self.suffix_ids.expand(B, -1))   # [B, T2, 4096]
        report_embeds = embed_fn(report_ids)                       # [B, T3, 4096]

        T1 = prefix_embeds.shape[1]
        T_img = image_embeds.shape[1]  # 256
        T2 = suffix_embeds.shape[1]
        T3 = report_ids.shape[1]
        T_total = T1 + T_img + T2 + T3

        # ---- 3. Concatenate all embeddings ----
        inputs_embeds = torch.cat(
            [prefix_embeds, image_embeds, suffix_embeds, report_embeds], dim=1
        ).to(dtype=self.llm_dtype)  # [B, T_total, 4096] — ensure all bf16

        # ---- 4. Build attention mask ----
        prompt_mask = torch.ones(B, T1 + T_img + T2, dtype=torch.long, device=device)
        attention_mask = torch.cat([prompt_mask, report_attention_mask], dim=1)
        # [B, T_total]

        # ---- 5. Build labels ----
        # -100 for prefix + image + suffix (model should NOT learn to predict these)
        prompt_labels = torch.full(
            (B, T1 + T_img + T2), fill_value=-100, dtype=torch.long, device=device
        )
        # Real token IDs for report, -100 for padding
        report_labels = report_ids.clone()
        report_labels[report_attention_mask == 0] = -100
        labels = torch.cat([prompt_labels, report_labels], dim=1)  # [B, T_total]

        # ---- 6. Forward through LLM ----
        if label_smoothing == 0.0:
            # Use built-in loss computation
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits
        else:
            # Compute loss with label smoothing manually
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = self._label_smoothed_loss(logits, labels, smoothing=label_smoothing)

        return {"loss": loss, "logits": logits}

    def _label_smoothed_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.1
    ) -> torch.Tensor:
        """
        Cross-entropy loss with label smoothing.

        This prevents the model from becoming overconfident (a key
        contributor to model collapse in fine-tuning).
        """
        # Shift for next-token prediction (same as HuggingFace internally)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        vocab_size = shift_logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Mask: only compute loss on non-ignored positions
        mask = shift_labels != -100
        active_logits = shift_logits[mask]
        active_labels = shift_labels[mask]

        if active_labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Standard NLL loss
        log_probs = F.log_softmax(active_logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, active_labels, reduction="mean")

        # Smooth loss: uniform distribution over vocabulary
        smooth_loss = -log_probs.mean(dim=-1).mean()

        return (1.0 - smoothing) * nll_loss + smoothing * smooth_loss

    @torch.no_grad()
    def generate_report(
        self,
        pixel_values: torch.Tensor,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = cfg.GEN_MAX_NEW_TOKENS,
        num_beams: int = cfg.GEN_NUM_BEAMS,
        repetition_penalty: float = cfg.GEN_REPETITION_PENALTY,
        length_penalty: float = cfg.GEN_LENGTH_PENALTY,
    ) -> list:
        """
        Generate radiology reports for given images.

        Args:
            pixel_values:       [B, 3, 224, 224] preprocessed images
            tokenizer:          Tokenizer for decoding
            max_new_tokens:     Maximum tokens to generate
            num_beams:          Beam search width
            repetition_penalty: Penalty for repeated tokens
            length_penalty:     Length penalty for beam search

        Returns:
            List of B generated report strings.
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Encode image
        image_embeds = self.encode_image(pixel_values)  # [B, 256, 4096]

        # Build prompt embeddings (no report — model generates it)
        embed_fn = self.language_model.get_input_embeddings()
        prefix_embeds = embed_fn(self.prefix_ids.expand(B, -1))
        suffix_embeds = embed_fn(self.suffix_ids.expand(B, -1))

        inputs_embeds = torch.cat(
            [prefix_embeds, image_embeds, suffix_embeds], dim=1
        ).to(dtype=self.llm_dtype)  # ensure bf16
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )

        prompt_length = inputs_embeds.shape[1]

        # Generate
        output_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=True,
        )

        # Extract generated tokens only (skip prompt positions)
        if output_ids.shape[1] > max_new_tokens:
            generated_ids = output_ids[:, prompt_length:]
        else:
            generated_ids = output_ids

        # Decode
        reports = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Clean up any residual prompt artifacts
        cleaned = []
        for text in reports:
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1]
            cleaned.append(text.strip())

        return cleaned

    # ================================================================
    # Training stage control
    # ================================================================

    def freeze_for_stage1(self):
        """
        Stage 1: Train ONLY the projector.
        Freeze: vision encoder + LLM.
        """
        # Vision encoder is already frozen in __init__
        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.projector.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Stage 1] Trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.4f}%)")

    def prepare_for_stage2(self):
        """
        Stage 2: Train projector + LoRA adapters on LLM.
        Freeze: vision encoder.
        """
        # Projector stays trainable
        for param in self.projector.parameters():
            param.requires_grad = True

        # Apply LoRA to LLM
        lora_config = LoraConfig(
            r=cfg.LORA_RANK,
            lora_alpha=cfg.LORA_ALPHA,
            target_modules=cfg.LORA_TARGET_MODULES,
            lora_dropout=cfg.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.print_trainable_parameters()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Stage 2] Total trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.4f}%)")

    def save_checkpoint(self, path: str, stage: int):
        """Save projector weights and (if stage 2) LoRA adapters."""
        import os
        os.makedirs(path, exist_ok=True)

        # Always save projector
        torch.save(
            self.projector.state_dict(),
            os.path.join(path, "projector.pth"),
        )

        # Save LoRA adapters in stage 2
        if stage == 2:
            lora_path = os.path.join(path, "lora_adapters")
            self.language_model.save_pretrained(lora_path)

        # Save config info
        import json
        with open(os.path.join(path, "model_info.json"), "w") as f:
            json.dump({
                "stage": stage,
                "raddino_model": cfg.RADDINO_MODEL_NAME,
                "llm_model": cfg.LLM_MODEL_NAME,
                "raddino_hidden_size": cfg.RADDINO_HIDDEN_SIZE,
                "llm_hidden_size": cfg.LLM_HIDDEN_SIZE,
                "num_image_tokens": cfg.NUM_IMAGE_TOKENS,
            }, f, indent=2)

        print(f"[Checkpoint] Saved stage {stage} checkpoint to {path}")

    def load_checkpoint(self, path: str, tokenizer: AutoTokenizer):
        """Load projector weights and LoRA adapters from a checkpoint."""
        import os

        # Load projector
        projector_path = os.path.join(path, "projector.pth")
        if os.path.exists(projector_path):
            self.projector.load_state_dict(
                torch.load(projector_path, map_location="cpu", weights_only=True)
            )
            print(f"[Checkpoint] Loaded projector from {projector_path}")

        # Load LoRA adapters if present
        lora_path = os.path.join(path, "lora_adapters")
        if os.path.isdir(lora_path):
            from peft import PeftModel
            self.language_model = PeftModel.from_pretrained(
                self.language_model, lora_path
            )
            print(f"[Checkpoint] Loaded LoRA adapters from {lora_path}")
