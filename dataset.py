"""
IU-Xray Dataset Loader for LLaVA-RADDino.

Handles multiple common JSON formats for IU-Xray annotations:
  - List format:  [{"id": ..., "image_path": [...], "report": ...}, ...]
  - Dict format:  {"train": [...], "val": [...], "test": [...]}
  - Various key names: "image_path"/"image"/"images", "report"/"findings"/"impression"

Each sample returns:
  - pixel_values:          [3, 224, 224]  preprocessed image
  - report_ids:            [T]            tokenized report + EOS
  - report_attention_mask: [T]            1=real, 0=padding (applied in collator)
  - ground_truth:          str            original report text
  - sample_id:             str            sample identifier
  - image_path:            str            relative image path
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from functools import partial

import config as cfg


class IUXrayDataset(Dataset):
    """
    Dataset for IU-Xray medical report generation.

    Automatically detects the JSON format and extracts:
    - Image path (uses first/frontal image if multiple exist)
    - Report text (combines 'findings' + 'impression' if separate)
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        tokenizer,
        transform,
        max_report_len: int = cfg.MAX_REPORT_LENGTH,
        split_key: str = None,
    ):
        """
        Args:
            json_path:      Path to the annotation JSON file.
            image_dir:      Directory containing the X-ray images.
            tokenizer:      HuggingFace tokenizer (Vicuna).
            transform:      Image preprocessing transform (from RadDinoEncoder).
            max_report_len: Maximum number of tokens for the report.
            split_key:      If JSON is a dict, which key to use (e.g., "train").
        """
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_report_len = max_report_len

        # Load annotations
        with open(json_path, "r") as f:
            data = json.load(f)

        # Handle various JSON structures
        if isinstance(data, list):
            self.samples = data
        elif isinstance(data, dict):
            if split_key and split_key in data:
                self.samples = data[split_key]
            else:
                # Try common keys
                for key in ["train", "val", "test", "data", "annotations"]:
                    if key in data:
                        self.samples = data[key]
                        break
                else:
                    # Assume the dict itself is keyed by IDs
                    self.samples = [
                        {"id": k, **v} if isinstance(v, dict) else {"id": k, "data": v}
                        for k, v in data.items()
                    ]
        else:
            raise ValueError(f"Unexpected JSON format in {json_path}")

        # Filter out samples with missing reports
        valid_samples = []
        for s in self.samples:
            report = self._extract_report(s)
            if report and report.strip():
                valid_samples.append(s)
        
        skipped = len(self.samples) - len(valid_samples)
        if skipped > 0:
            print(f"[Dataset] Skipped {skipped} samples with empty reports")
        self.samples = valid_samples

        print(f"[Dataset] Loaded {len(self.samples)} samples from {json_path}")

    def _extract_image_path(self, sample: dict) -> str:
        """Extract image path from sample, handling various key names."""
        for key in ["image_path", "image", "images", "img_path", "file_name"]:
            if key in sample:
                path = sample[key]
                if isinstance(path, list):
                    path = path[0]  # Use first (frontal) image
                return path
        raise KeyError(f"No image path found in sample keys: {list(sample.keys())}")

    def _extract_report(self, sample: dict) -> str:
        """Extract report text, combining findings + impression if needed."""
        # Try direct "report" key first
        if "report" in sample:
            return sample["report"].strip()

        # Try combining findings + impression
        parts = []
        for key in ["findings", "impression", "conclusion"]:
            if key in sample and sample[key]:
                text = sample[key].strip()
                if text:
                    parts.append(text)

        if parts:
            return " ".join(parts)

        # Try "text" key
        if "text" in sample:
            return sample["text"].strip()

        return ""

    def _extract_id(self, sample: dict, idx: int) -> str:
        """Extract sample ID."""
        for key in ["id", "study_id", "subject_id", "uid", "ID"]:
            if key in sample:
                return str(sample[key])
        return str(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ---- Image ----
        image_rel_path = self._extract_image_path(sample)
        image_full_path = os.path.join(self.image_dir, image_rel_path)

        try:
            image = Image.open(image_full_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image: {image_full_path}\n"
                f"Make sure IMAGE_DIR in config.py points to the correct directory.\n"
                f"Error: {e}"
            )

        pixel_values = self.transform(image)  # [3, 224, 224]

        # ---- Report ----
        report_text = self._extract_report(sample)

        # Tokenize report + EOS
        report_tokens = self.tokenizer(
            report_text,
            add_special_tokens=False,
            max_length=self.max_report_len - 1,  # Reserve space for EOS
            truncation=True,
            return_tensors="pt",
        )

        # Append EOS token
        eos_id = torch.tensor([[self.tokenizer.eos_token_id]], dtype=torch.long)
        eos_mask = torch.ones(1, 1, dtype=torch.long)

        report_ids = torch.cat(
            [report_tokens.input_ids, eos_id], dim=1
        ).squeeze(0)  # [T_report]

        report_attention_mask = torch.cat(
            [report_tokens.attention_mask, eos_mask], dim=1
        ).squeeze(0)  # [T_report]

        return {
            "pixel_values": pixel_values,
            "report_ids": report_ids,
            "report_attention_mask": report_attention_mask,
            "ground_truth": report_text,
            "sample_id": self._extract_id(sample, idx),
            "image_path": image_rel_path,
        }


def collate_fn(batch: list, pad_token_id: int) -> dict:
    """
    Custom collation function that pads report tokens to the same length.

    Args:
        batch:        List of dicts from IUXrayDataset.__getitem__
        pad_token_id: Token ID to use for padding

    Returns:
        Batched dict with padded tensors.
    """
    B = len(batch)

    # Stack images (all same size)
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    # Pad reports to max length in this batch
    max_report_len = max(b["report_ids"].shape[0] for b in batch)

    report_ids = torch.full((B, max_report_len), pad_token_id, dtype=torch.long)
    report_attention_mask = torch.zeros(B, max_report_len, dtype=torch.long)

    for i, b in enumerate(batch):
        length = b["report_ids"].shape[0]
        report_ids[i, :length] = b["report_ids"]
        report_attention_mask[i, :length] = b["report_attention_mask"]

    return {
        "pixel_values": pixel_values,
        "report_ids": report_ids,
        "report_attention_mask": report_attention_mask,
        # Metadata (not tensors — for evaluation)
        "ground_truth": [b["ground_truth"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def create_dataloaders(
    tokenizer,
    transform,
    batch_size_train: int,
    batch_size_eval: int,
    num_workers: int = cfg.NUM_WORKERS,
) -> tuple:
    """
    Create train, validation, and test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = IUXrayDataset(
        json_path=cfg.TRAIN_JSON,
        image_dir=cfg.IMAGE_DIR,
        tokenizer=tokenizer,
        transform=transform,
        max_report_len=cfg.MAX_REPORT_LENGTH,
    )

    val_dataset = IUXrayDataset(
        json_path=cfg.VAL_JSON,
        image_dir=cfg.IMAGE_DIR,
        tokenizer=tokenizer,
        transform=transform,
        max_report_len=cfg.MAX_REPORT_LENGTH,
    )

    test_dataset = IUXrayDataset(
        json_path=cfg.TEST_JSON,
        image_dir=cfg.IMAGE_DIR,
        tokenizer=tokenizer,
        transform=transform,
        max_report_len=cfg.MAX_REPORT_LENGTH,
    )

    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
