"""
Central configuration for LLaVA-RADDino project.
Edit the PATHS section below to match your server setup.
"""

import os

# ============================================================
# PATHS — EDIT THESE TO MATCH YOUR SERVER
# ============================================================
# Root directory containing the IU-Xray data
DATA_ROOT = "/path/to/IU-Xray"

# Directory containing the X-ray images (*.dcm.png files)
# From screenshot: IU-Xray/datasets/raddar/chest-xrays-indiana.../versions/2/images/images_normalized/
IMAGE_DIR = "/path/to/IU-Xray/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized"

# CSV file with reports — columns should include UID and report text
REPORTS_CSV = "/path/to/processed_indiana_reports.csv"

# Directory where train/val/test JSON splits will be generated
SPLITS_DIR = "/path/to/IU-Xray/splits"

# Generated split files (created by prepare_data.py)
TRAIN_JSON = "/path/to/IU-Xray/splits/train.json"
VAL_JSON = "/path/to/IU-Xray/splits/val.json"
TEST_JSON = "/path/to/IU-Xray/splits/test.json"

# Output directory for checkpoints + results
OUTPUT_DIR = "/path/to/outputs/llava_raddino"

# ============================================================
# MODEL NAMES (HuggingFace Hub IDs)
# ============================================================
RADDINO_MODEL_NAME = "microsoft/rad-dino"
LLM_MODEL_NAME = "lmsys/vicuna-7b-v1.5"

# ============================================================
# ARCHITECTURE CONSTANTS
# ============================================================
RADDINO_HIDDEN_SIZE = 768       # RAD-DINO (DINOv2 ViT-B/14) output dim
LLM_HIDDEN_SIZE = 4096          # Vicuna-7B hidden dim
IMAGE_SIZE = 224                # Input image resolution
PATCH_SIZE = 14                 # ViT patch size
NUM_IMAGE_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256

# ============================================================
# TOKENIZATION
# ============================================================
MAX_REPORT_LENGTH = 256         # Max tokens for generated report
PROMPT_USER = "USER: "
PROMPT_INSTRUCTION = "\nGenerate a detailed radiology report for this chest X-ray.\nASSISTANT: "

# ============================================================
# DATA SPLIT RATIOS (used by prepare_data.py)
# ============================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ============================================================
# STAGE 1 — PROJECTOR ALIGNMENT
# ============================================================
STAGE1_LR = 1e-3
STAGE1_EPOCHS = 5
STAGE1_BATCH_SIZE = 16
STAGE1_WARMUP_RATIO = 0.1
STAGE1_WEIGHT_DECAY = 0.0       # No weight decay for alignment

# ============================================================
# STAGE 2 — LoRA FINE-TUNING
# ============================================================
STAGE2_LR = 2e-5
STAGE2_EPOCHS = 10
STAGE2_BATCH_SIZE = 8
STAGE2_GRAD_ACCUM_STEPS = 2     # Effective batch = 16
STAGE2_WARMUP_RATIO = 0.05
STAGE2_WEIGHT_DECAY = 0.01
STAGE2_LABEL_SMOOTHING = 0.1
STAGE2_MAX_GRAD_NORM = 1.0
STAGE2_PATIENCE = 3             # Early stopping patience (epochs)

# LoRA hyperparameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ============================================================
# GENERATION SETTINGS
# ============================================================
GEN_MAX_NEW_TOKENS = 256
GEN_NUM_BEAMS = 4
GEN_REPETITION_PENALTY = 1.2
GEN_LENGTH_PENALTY = 1.0

# ============================================================
# IMAGE NORMALIZATION (DINOv2 / ImageNet stats)
# ============================================================
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# ============================================================
# MISC
# ============================================================
SEED = 42
NUM_WORKERS = 4
DTYPE = "bfloat16"  # A6000 supports bf16 natively
