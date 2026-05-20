"""
Two-Stage Training Script for LLaVA-RADDino.

Stage 1 — Projector Alignment:
  - Trains ONLY the projector MLP (768→4096→4096)
  - Freezes: RAD-DINO encoder + Vicuna-7B LLM
  - Higher LR (1e-3) since projector is randomly initialized
  - Purpose: Align RAD-DINO vision features to Vicuna-7B's embedding space

Stage 2 — LoRA Fine-Tuning:
  - Trains: Projector + LoRA adapters on Vicuna-7B (q_proj, v_proj)
  - Freezes: RAD-DINO encoder
  - Lower LR (2e-5) to prevent catastrophic forgetting
  - Label smoothing (0.1) to prevent model collapse
  - Early stopping with patience=3

Anti-collapse measures:
  - Label smoothing in Stage 2
  - Low LoRA rank (16) to limit parameter count
  - Gradient clipping at 1.0
  - Cosine LR schedule with warmup
  - Early stopping on validation loss
  - Diversity monitoring (sample generation every epoch)

Usage:
  python train.py --stage 1
  python train.py --stage 2 --stage1_ckpt /path/to/stage1/checkpoint
"""

import os
import sys
import json
import argparse
import random
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

import config as cfg
from model import LLaVARaddino
from dataset import create_dataloaders
from raddino_encoder import RadDinoEncoder
from metrics import compute_all_metrics


def set_seed(seed: int = cfg.SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> tuple:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def validate(model, val_loader, device, label_smoothing=0.0):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            report_ids = batch["report_ids"].to(device)
            report_mask = batch["report_attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=pixel_values,
                    report_ids=report_ids,
                    report_attention_mask=report_mask,
                    label_smoothing=label_smoothing,
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    model.train()
    return avg_loss


def generate_samples(model, val_loader, tokenizer, device, num_samples=3):
    """
    Generate sample reports for diversity monitoring.
    Prints side-by-side comparison of ground truth vs generated.
    """
    model.eval()
    batch = next(iter(val_loader))

    pixel_values = batch["pixel_values"][:num_samples].to(device)
    ground_truths = batch["ground_truth"][:num_samples]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate_report(pixel_values, tokenizer)

    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS (diversity check)")
    print("=" * 70)
    for i in range(min(num_samples, len(generated))):
        print(f"\n--- Sample {i+1} ---")
        print(f"  GT:  {ground_truths[i][:200]}...")
        print(f"  GEN: {generated[i][:200]}...")
    print("=" * 70 + "\n")

    # Check diversity: are all outputs identical?
    unique_outputs = len(set(generated))
    if unique_outputs == 1 and num_samples > 1:
        print("⚠️  WARNING: All generated outputs are identical! Possible model collapse.")
    else:
        print(f"✓ Output diversity: {unique_outputs}/{num_samples} unique outputs")

    model.train()
    return generated


def train_stage(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    stage: int,
    output_dir: str,
):
    """
    Run one stage of training.

    Args:
        model:        LLaVARaddino model
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        tokenizer:    Tokenizer for sample generation
        device:       CUDA device
        stage:        1 or 2
        output_dir:   Directory for checkpoints
    """
    # ---- Hyperparameters ----
    if stage == 1:
        lr = cfg.STAGE1_LR
        epochs = cfg.STAGE1_EPOCHS
        warmup_ratio = cfg.STAGE1_WARMUP_RATIO
        weight_decay = cfg.STAGE1_WEIGHT_DECAY
        label_smoothing = 0.0
        max_grad_norm = None
        patience = None  # No early stopping in Stage 1
        grad_accum_steps = 1
    else:
        lr = cfg.STAGE2_LR
        epochs = cfg.STAGE2_EPOCHS
        warmup_ratio = cfg.STAGE2_WARMUP_RATIO
        weight_decay = cfg.STAGE2_WEIGHT_DECAY
        label_smoothing = cfg.STAGE2_LABEL_SMOOTHING
        max_grad_norm = cfg.STAGE2_MAX_GRAD_NORM
        patience = cfg.STAGE2_PATIENCE
        grad_accum_steps = cfg.STAGE2_GRAD_ACCUM_STEPS

    # ---- Optimizer (only trainable params) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ---- LR Scheduler ----
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Training state ----
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    log_interval = 50  # Log every N steps

    print(f"\n{'='*70}")
    print(f"STAGE {stage} TRAINING")
    print(f"{'='*70}")
    print(f"  Epochs:          {epochs}")
    print(f"  Learning rate:   {lr}")
    print(f"  Batch size:      {train_loader.batch_size} × {grad_accum_steps} accum = {train_loader.batch_size * grad_accum_steps}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Warmup steps:    {warmup_steps}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Grad clip norm:  {max_grad_norm}")
    total_p, train_p = count_parameters(model)
    print(f"  Parameters:      {train_p:,} trainable / {total_p:,} total")
    print(f"{'='*70}\n")

    model.train()
    # Keep encoder in eval mode always (BatchNorm etc.)
    model.vision_encoder.model.eval()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Stage {stage} | Epoch {epoch}/{epochs}",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            report_ids = batch["report_ids"].to(device)
            report_mask = batch["report_attention_mask"].to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=pixel_values,
                    report_ids=report_ids,
                    report_attention_mask=report_mask,
                    label_smoothing=label_smoothing,
                )
                loss = outputs["loss"] / grad_accum_steps

            # Backward
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += outputs["loss"].item()
            epoch_steps += 1

            # Progress bar update
            pbar.set_postfix({
                "loss": f"{outputs['loss'].item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Periodic logging
            if global_step > 0 and global_step % log_interval == 0:
                avg_recent = epoch_loss / epoch_steps
                print(
                    f"  [Step {global_step}] "
                    f"loss={outputs['loss'].item():.4f} "
                    f"avg_loss={avg_recent:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # ---- End of epoch ----
        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        # Validation
        val_loss = validate(model, val_loader, device, label_smoothing)

        print(f"\n📊 Epoch {epoch}/{epochs} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   LR:         {scheduler.get_last_lr()[0]:.2e}")

        # Generate samples for diversity monitoring
        generate_samples(model, val_loader, tokenizer, device, num_samples=3)

        # ---- Checkpoint best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, f"stage{stage}_best")
            model.save_checkpoint(ckpt_path, stage=stage)
            print(f"   ✅ New best! Val loss improved to {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience if patience else '∞'}")

        # Save latest checkpoint every epoch
        latest_path = os.path.join(output_dir, f"stage{stage}_latest")
        model.save_checkpoint(latest_path, stage=stage)

        # ---- Early stopping ----
        if patience is not None and patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {patience} epochs)")
            break

        print()

    print(f"\n{'='*70}")
    print(f"STAGE {stage} COMPLETE — Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(output_dir, f'stage{stage}_best')}")
    print(f"{'='*70}\n")

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train LLaVA-RADDino for medical report generation"
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2],
        help="Training stage: 1 (projector only) or 2 (projector + LoRA)"
    )
    parser.add_argument(
        "--stage1_ckpt", type=str, default=None,
        help="Path to Stage 1 checkpoint (required for Stage 2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=cfg.OUTPUT_DIR,
        help="Output directory for checkpoints"
    )
    args = parser.parse_args()

    # ---- Validate arguments ----
    if args.stage == 2 and args.stage1_ckpt is None:
        print("ERROR: --stage1_ckpt is required for Stage 2 training.")
        print("  Run Stage 1 first, then pass the checkpoint path.")
        sys.exit(1)

    # ---- Setup ----
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer: {cfg.LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Image transform ----
    transform = RadDinoEncoder.get_image_transform()

    # ---- Data ----
    batch_size = (
        cfg.STAGE1_BATCH_SIZE if args.stage == 1 else cfg.STAGE2_BATCH_SIZE
    )
    print(f"\nLoading datasets...")
    train_loader, val_loader, _ = create_dataloaders(
        tokenizer=tokenizer,
        transform=transform,
        batch_size_train=batch_size,
        batch_size_eval=batch_size,
        num_workers=cfg.NUM_WORKERS,
    )
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # ---- Model ----
    print(f"\nBuilding model...")
    model = LLaVARaddino(tokenizer=tokenizer)

    if args.stage == 1:
        model.freeze_for_stage1()
    elif args.stage == 2:
        # Load Stage 1 projector weights first
        print(f"\nLoading Stage 1 checkpoint: {args.stage1_ckpt}")
        model.load_checkpoint(args.stage1_ckpt, tokenizer)
        model.prepare_for_stage2()

    model = model.to(device)

    # ---- Train ----
    train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        stage=args.stage,
        output_dir=args.output_dir,
    )

    # ---- Save final training info ----
    info = {
        "stage": args.stage,
        "completed_at": datetime.now().isoformat(),
        "output_dir": args.output_dir,
        "stage1_ckpt": args.stage1_ckpt,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
    }
    with open(os.path.join(args.output_dir, f"stage{args.stage}_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
