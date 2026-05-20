"""
RAD-DINO Vision Encoder Wrapper.

Wraps Microsoft's RAD-DINO (DINOv2 ViT-B/14 fine-tuned on chest X-rays)
to produce patch token features compatible with a LLaVA-style pipeline.

Input:  [B, 3, 224, 224] images
Output: [B, 256, 768]     patch token features (CLS token stripped)
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model
from torchvision import transforms

from config import (
    RADDINO_MODEL_NAME,
    RADDINO_HIDDEN_SIZE,
    IMAGE_SIZE,
    NUM_IMAGE_TOKENS,
    IMAGE_MEAN,
    IMAGE_STD,
)


class RadDinoEncoder(nn.Module):
    """
    Frozen RAD-DINO encoder that extracts patch-level features from chest X-rays.

    Architecture: DINOv2 ViT-B/14
    - hidden_size: 768
    - patch_size:  14×14
    - For 224×224 input: 16×16 = 256 patch tokens + 1 CLS token = 257 total
    - We return only the 256 patch tokens (CLS is stripped).
    """

    def __init__(self, model_name: str = RADDINO_MODEL_NAME):
        super().__init__()
        print(f"[RadDinoEncoder] Loading {model_name} ...")
        self.model = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = RADDINO_HIDDEN_SIZE

        # Freeze all parameters — the radiology features are already strong
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[RadDinoEncoder] Loaded. hidden_size={self.hidden_size}, frozen=True")

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract patch token features from images.

        Args:
            pixel_values: [B, 3, 224, 224] preprocessed images

        Returns:
            patch_features: [B, NUM_IMAGE_TOKENS, 768] patch token features
        """
        # Forward through DINOv2
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state shape: [B, 1 + num_patches, 768]
        # Position 0 is CLS token, positions 1..N are patch tokens
        all_tokens = outputs.last_hidden_state

        # Strip the CLS token (position 0), keep only patch tokens
        patch_features = all_tokens[:, 1:, :]

        # ---- Dimension assertions ----
        B = pixel_values.shape[0]
        assert patch_features.shape == (B, NUM_IMAGE_TOKENS, self.hidden_size), (
            f"Expected patch_features shape ({B}, {NUM_IMAGE_TOKENS}, {self.hidden_size}), "
            f"got {patch_features.shape}"
        )

        return patch_features

    @staticmethod
    def get_image_transform() -> transforms.Compose:
        """
        Returns the image preprocessing pipeline for RAD-DINO.

        Uses ImageNet normalization (same as DINOv2 training).
        Input images are resized to 224×224 to produce 256 patch tokens.
        """
        return transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
