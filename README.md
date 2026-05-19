# LLaVA-RADDino: Medical Report Generation with RAD-DINO Vision Encoder

## Overview

This project replaces LLaVA-1.5-7B's default CLIP ViT-L/14 vision encoder with **Microsoft's RAD-DINO** — a radiology-specialized DINOv2 ViT-B/14 encoder — and fine-tunes it on dataset for chest X-ray report generation.

### Architecture

```
Chest X-ray Image (224×224×3)
    ↓
RAD-DINO Encoder (frozen)           →  [B, 256, 768]
    ↓
MLP Projector (trainable)            →  [B, 256, 4096]
    ↓                                    768→4096→4096
Concatenate with text embeddings     →  [B, 256+T, 4096]
    ↓
Vicuna-7B LLM (LoRA fine-tuned)     →  Generated Report
```

### Key Design Decisions

| Component | Original LLaVA | Our Replacement |
|---|---|---|
| Vision Encoder | CLIP ViT-L/14 (1024-dim) | RAD-DINO ViT-B/14 (768-dim) |
| Projector | Linear(1024→4096)→GELU→Linear(4096→4096) | Linear(**768**→4096)→GELU→Linear(4096→4096) |
| LLM | Vicuna-7B | Vicuna-7B (same, LoRA fine-tuned) |

---

**Image naming:** `{UID}_IM-{XXXX}-{XXXX}.dcm.png`  
**UID** = first number in filename (e.g., `106`, `107`, `108`)

---

## Step-by-Step Guide

### Step 0: Connect to Server

```bash
# SSH into your remote server with gpu
ssh your_username@your_server_ip

# Start a tmux session (persists if SSH disconnects)
tmux new -s llava_raddino
```

> **tmux cheat sheet:**
> - Detach: `Ctrl+B`, then `D`
> - Re-attach: `tmux attach -t llava_raddino`
> - List sessions: `tmux ls`
> - Kill session: `tmux kill-session -t llava_raddino`

---

### Step 1: Create Environment

```bash
# Create conda environment
conda create -n llava_raddino python=3.10 -y
conda activate llava_raddino

# Navigate to project directory
cd /path/to/your/project
mkdir -p llava_raddino
cd llava_raddino

# Copy all the code files here (config.py, model.py, etc.)
# Then install dependencies:
pip install -r requirements.txt

# Download NLTK data (needed for BLEU)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

### Step 2: Edit Configuration

Open `config.py` and update the **PATHS section**:

```python
# ============ PATHS — EDIT THESE ============
DATA_ROOT = "/home/your_user/data/IU-Xray"

IMAGE_DIR = "/home/your_user/data/IU-Xray/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized"

REPORTS_CSV = "/home/your_user/data/processed_indiana_reports.csv"

SPLITS_DIR = "/home/your_user/data/IU-Xray/splits"
TRAIN_JSON = "/home/your_user/data/IU-Xray/splits/train.json"
VAL_JSON   = "/home/your_user/data/IU-Xray/splits/val.json"
TEST_JSON  = "/home/your_user/data/IU-Xray/splits/test.json"

OUTPUT_DIR = "/home/your_user/outputs/llava_raddino"
```

---

### Step 3: Prepare Data (Create Train/Val/Test Splits)

This reads your CSV + scans your image folder and creates JSON split files:

```bash
conda activate llava_raddino
cd /path/to/llava_raddino

python prepare_data.py
```

**What it does:**
1. Reads `processed_indiana_reports.csv` → extracts UID → report mapping
2. Scans `images_normalized/` → extracts UID → image files mapping
3. Matches UIDs between CSV and images
4. Creates **patient-level** train (70%) / val (10%) / test (20%) splits
5. Saves `train.json`, `val.json`, `test.json`

**Expected output:**
```
📄 Loading reports from: /path/to/processed_indiana_reports.csv
[CSV] Columns found: ['uid', 'report', ...]
[CSV] Loaded 3,955 reports

🖼️  Scanning images in: /path/to/images_normalized
[Images] Found 7,470 images across 3,955 UIDs

[Match] UIDs with both report + images: 3,955
[Split] Train: 2,768 samples (2,768 UIDs)
[Split] Val:   396 samples (396 UIDs)
[Split] Test:  791 samples (791 UIDs)

✅ Saved train: /path/to/splits/train.json
✅ Saved val:   /path/to/splits/val.json
✅ Saved test:  /path/to/splits/test.json
```

> **If UIDs don't match:** The script will show sample UIDs from both CSV and images so you can debug the mismatch.

---

### Step 4: Verify Setup (Quick Sanity Check)

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import json
from config import TRAIN_JSON, VAL_JSON, TEST_JSON
for name, path in [('Train', TRAIN_JSON), ('Val', VAL_JSON), ('Test', TEST_JSON)]:
    with open(path) as f:
        data = json.load(f)
    print(f'{name}: {len(data)} samples')
"
```

---

### Step 5: Stage 1 Training — Projector Alignment

This trains **only the projector MLP** (768→4096→4096) to align RAD-DINO features with Vicuna-7B's embedding space. The vision encoder and LLM are frozen.

**Start in tmux** (so it survives SSH disconnects):

```bash
# Make sure you're in the tmux session
tmux attach -t llava_raddino

# Activate env
conda activate llava_raddino

# Run Stage 1
cd /path/to/llava_raddino
python train.py --stage 1 --output_dir /path/to/outputs/llava_raddino

# This will:
#   1. Download RAD-DINO (~350MB) and Vicuna-7B (~14GB) from HuggingFace
#   2. Train projector for 5 epochs
#   3. Save best checkpoint to: outputs/llava_raddino/stage1_best/
```

**Expected output:**
```
[RadDinoEncoder] Loading microsoft/rad-dino ...
[LLaVARaddino] Loading LLM: lmsys/vicuna-7b-v1.5 ...
[Stage 1] Trainable: 19,926,016 / 6,758,xxx,xxx (0.29%)

Stage 1 | Epoch 1/5: 100%|████████| 173/173 [05:30<00:00]
  Train Loss: 4.2345
  Val Loss:   3.8901
```

**Estimated time:** ~30-60 minutes on A6000 (depends on dataset size).

After Stage 1, you can **detach tmux** if needed: `Ctrl+B, D`

---

### Step 6: Stage 2 Training — LoRA Fine-Tuning

This trains the **projector + LoRA adapters** on Vicuna-7B's attention layers. The vision encoder stays frozen.

```bash
# Re-attach tmux
tmux attach -t llava_raddino
conda activate llava_raddino

# Run Stage 2 (pass Stage 1 checkpoint)
python train.py \
    --stage 2 \
    --stage1_ckpt /path/to/outputs/llava_raddino/stage1_best \
    --output_dir /path/to/outputs/llava_raddino

# This will:
#   1. Load Stage 1 projector weights
#   2. Apply LoRA (rank=16) to q_proj, v_proj of Vicuna-7B
#   3. Train for up to 10 epochs with early stopping (patience=3)
#   4. Save best checkpoint to: outputs/llava_raddino/stage2_best/
```

**Expected output:**
```
[Checkpoint] Loaded projector from .../stage1_best/projector.pth
trainable params: 6,815,744 || all params: 6,764,xxx,xxx || trainable%: 0.1008
[Stage 2] Total trainable: 26,741,760 / 6,764,xxx,xxx (0.3955%)

Stage 2 | Epoch 1/10: 100%|████████| 346/346 [12:00<00:00]
  Train Loss: 2.1456
  Val Loss:   2.0123
  ✅ New best! Val loss improved to 2.0123

SAMPLE GENERATIONS (diversity check)
  Sample 1 GT:  The heart size is normal...
  Sample 1 GEN: Normal heart size. The lungs...
  ✓ Output diversity: 3/3 unique outputs
```

**Estimated time:** ~2-4 hours on A6000.

> **⚠️ Watch for model collapse signs:**
> - All sample outputs become identical
> - Val loss suddenly spikes or flatlines
> - Training loss drops to near 0 extremely fast

---

### Step 7: Run Inference on Test Set

```bash
conda activate llava_raddino

python inference.py \
    --checkpoint /path/to/outputs/llava_raddino/stage2_best \
    --output /path/to/outputs/llava_raddino/results.json \
    --batch_size 4 \
    --num_beams 4

# This will:
#   1. Load the best Stage 2 model
#   2. Generate reports for all test samples
#   3. Compute BLEU (1-4) and ROUGE (1, 2, L) per sample
#   4. Save everything to results.json
```

**Expected output:**
```
INFERENCE COMPLETE
  Samples processed: 791
  Total time:        600.3s (0.76s/sample)

  AGGREGATE METRICS:
  ────────────────────────────────────
      bleu_1: 0.4123
      bleu_2: 0.2845
      bleu_3: 0.2134
      bleu_4: 0.1678
     rouge_1: 0.4567
     rouge_2: 0.2012
     rouge_l: 0.3890
  ────────────────────────────────────

  Results saved to: results.json
```

---

### Step 8: Inspect Results

The output JSON (`results.json`) has this structure:

```json
{
  "model_info": {
    "vision_encoder": "microsoft/rad-dino",
    "language_model": "lmsys/vicuna-7b-v1.5",
    "checkpoint": "/path/to/stage2_best",
    "num_beams": 4,
    "max_new_tokens": 256
  },
  "aggregate_metrics": {
    "bleu_1": 0.4123,
    "bleu_2": 0.2845,
    "bleu_3": 0.2134,
    "bleu_4": 0.1678,
    "rouge_1": 0.4567,
    "rouge_2": 0.2012,
    "rouge_l": 0.3890
  },
  "num_samples": 791,
  "inference_time_seconds": 600.3,
  "results": [
    {
      "id": "106",
      "image_path": "106_IM-0042-1001.dcm.png",
      "ground_truth": "The heart size is normal...",
      "generated_report": "Normal heart size. Clear lungs...",
      "metrics": {
        "bleu_1": 0.45,
        "bleu_2": 0.32,
        "bleu_3": 0.25,
        "bleu_4": 0.20,
        "rouge_1": 0.48,
        "rouge_2": 0.22,
        "rouge_l": 0.40
      }
    }
  ]
}
```

---

## Complete Command Sequence (Copy-Paste Ready)

```bash
# ======== ONE-TIME SETUP ========
ssh your_user@your_server
tmux new -s llava_raddino

conda create -n llava_raddino python=3.10 -y
conda activate llava_raddino

cd /path/to/project/llava_raddino
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Edit config.py paths!
nano config.py

# ======== PREPARE DATA ========
python prepare_data.py

# ======== STAGE 1 (in tmux) ========
python train.py --stage 1 --output_dir /path/to/outputs/llava_raddino
# (Ctrl+B, D to detach if needed)

# ======== STAGE 2 (in tmux) ========
# (tmux attach -t llava_raddino)
python train.py --stage 2 \
    --stage1_ckpt /path/to/outputs/llava_raddino/stage1_best \
    --output_dir /path/to/outputs/llava_raddino

# ======== INFERENCE ========
python inference.py \
    --checkpoint /path/to/outputs/llava_raddino/stage2_best \
    --output /path/to/outputs/llava_raddino/results.json

# ======== VIEW RESULTS ========
python -c "
import json
with open('/path/to/outputs/llava_raddino/results.json') as f:
    data = json.load(f)
print('Aggregate Metrics:')
for k, v in data['aggregate_metrics'].items():
    print(f'  {k}: {v:.4f}')
"
```

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `STAGE2_BATCH_SIZE` in `config.py` from 8 to 4
- Increase `STAGE2_GRAD_ACCUM_STEPS` from 2 to 4 (keeps effective batch=16)

### No UIDs Match (prepare_data.py error)
- Check CSV column names: the script tries `uid`, `UID`, `id`, `ID`, etc.
- Check image naming: expects `{UID}_IM-...` format
- The script prints sample UIDs from both sources for comparison

### Model Collapse (all outputs identical)
- Check if `STAGE2_LR` is too high — try `1e-5`
- Increase `STAGE2_LABEL_SMOOTHING` from 0.1 to 0.2
- Reduce `LORA_RANK` from 16 to 8
- Make sure Stage 1 completed successfully (check val loss decreased)

### HuggingFace Login Required
```bash
# If Vicuna-7B requires auth:
pip install huggingface_hub
huggingface-cli login
# Enter your HuggingFace token
```

### Slow Data Loading
- Increase `NUM_WORKERS` in config.py
- Ensure images are stored on SSD, not network storage

---

## File Descriptions

| File | Purpose |
|---|---|
| `config.py` | All paths, hyperparameters, and model settings |
| `prepare_data.py` | **NEW** — Reads CSV + images, creates train/val/test JSON splits |
| `raddino_encoder.py` | RAD-DINO vision encoder wrapper (frozen, 768-dim output) |
| `model.py` | Full LLaVA-RADDino model with projector and LLM |
| `dataset.py` | Xray data loading with flexible JSON format support |
| `train.py` | Two-stage training (projector alignment → LoRA fine-tuning) |
| `inference.py` | Test-set inference with metric computation |
| `metrics.py` | BLEU (1-4) and ROUGE (1, 2, L) computation |
| `requirements.txt` | Python dependencies |

---

## Memory Estimates (A6000 48GB, bf16)

| Component | Memory |
|---|---|
| RAD-DINO (frozen, no grad) | ~170 MB |
| Projector | ~36 MB |
| Vicuna-7B (bf16) | ~14 GB |
| LoRA adapters | ~14 MB |
| Activations + optimizer (Stage 2) | ~15-20 GB |
| **Total peak** | **~30-35 GB** |

Fits comfortably on A6000 (48GB) with room for larger batch sizes.
