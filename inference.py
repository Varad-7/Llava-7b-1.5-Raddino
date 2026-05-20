"""
Inference Script for LLaVA-RADDino.

Runs the trained model on the test set and produces a JSON output file with:
  - Per-sample: ground truth, generated report, BLEU (1-4), ROUGE (1, 2, L)
  - Aggregate: Mean of all metrics across the test set

Usage:
  python inference.py --checkpoint /path/to/stage2_best --output results.json
"""

import os
import sys
import json
import argparse
import time

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

import config as cfg
from model import LLaVARaddino
from dataset import IUXrayDataset, collate_fn
from raddino_encoder import RadDinoEncoder
from metrics import compute_all_metrics, aggregate_metrics
from functools import partial
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained LLaVA-RADDino model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint directory (e.g., stage2_best)"
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Inference batch size"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=cfg.GEN_MAX_NEW_TOKENS,
        help="Maximum tokens to generate per report"
    )
    parser.add_argument(
        "--num_beams", type=int, default=cfg.GEN_NUM_BEAMS,
        help="Beam search width"
    )
    args = parser.parse_args()

    # ---- Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer: {cfg.LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Model ----
    print(f"\nBuilding model...")
    model = LLaVARaddino(tokenizer=tokenizer)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model.load_checkpoint(args.checkpoint, tokenizer)
    model = model.to(device)
    model.eval()

    # ---- Test Data ----
    transform = RadDinoEncoder.get_image_transform()

    test_dataset = IUXrayDataset(
        json_path=cfg.TEST_JSON,
        image_dir=cfg.IMAGE_DIR,
        tokenizer=tokenizer,
        transform=transform,
        max_report_len=cfg.MAX_REPORT_LENGTH,
    )

    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=_collate,
        pin_memory=True,
    )

    print(f"Test set: {len(test_dataset)} samples, {len(test_loader)} batches")

    # ---- Inference ----
    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE")
    print(f"  Max new tokens:    {args.max_new_tokens}")
    print(f"  Beam width:        {args.num_beams}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"{'='*70}\n")

    all_results = []
    all_metrics_list = []
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating reports"):
            pixel_values = batch["pixel_values"].to(device)
            ground_truths = batch["ground_truth"]
            sample_ids = batch["sample_id"]
            image_paths = batch["image_path"]

            # Generate reports
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_reports = model.generate_report(
                    pixel_values=pixel_values,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )

            # Compute per-sample metrics
            for i in range(len(generated_reports)):
                gt = ground_truths[i]
                gen = generated_reports[i]

                sample_metrics = compute_all_metrics(gt, gen)
                all_metrics_list.append(sample_metrics)

                result = {
                    "id": sample_ids[i],
                    "image_path": image_paths[i],
                    "ground_truth": gt,
                    "generated_report": gen,
                    "metrics": sample_metrics,
                }
                all_results.append(result)

    elapsed = time.time() - start_time

    # ---- Aggregate Metrics ----
    agg_metrics = aggregate_metrics(all_metrics_list)

    # ---- Build Output JSON ----
    output = {
        "model_info": {
            "vision_encoder": cfg.RADDINO_MODEL_NAME,
            "language_model": cfg.LLM_MODEL_NAME,
            "checkpoint": args.checkpoint,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
        },
        "aggregate_metrics": agg_metrics,
        "num_samples": len(all_results),
        "inference_time_seconds": round(elapsed, 2),
        "results": all_results,
    }

    # ---- Save ----
    output_path = args.output
    if not output_path.endswith(".json"):
        output_path += ".json"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ---- Print Summary ----
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"  Samples processed: {len(all_results)}")
    print(f"  Total time:        {elapsed:.1f}s ({elapsed/len(all_results):.2f}s/sample)")
    print(f"\n  AGGREGATE METRICS:")
    print(f"  {'─'*40}")
    for metric, value in agg_metrics.items():
        print(f"    {metric:>10s}: {value:.4f}")
    print(f"  {'─'*40}")
    print(f"\n  Results saved to: {output_path}")
    print(f"{'='*70}\n")

    # ---- Print a few example outputs ----
    print("SAMPLE OUTPUTS:")
    for i, r in enumerate(all_results[:3]):
        print(f"\n--- Sample {i+1} (ID: {r['id']}) ---")
        print(f"  GT:  {r['ground_truth'][:200]}...")
        print(f"  GEN: {r['generated_report'][:200]}...")
        print(f"  BLEU-4: {r['metrics']['bleu_4']:.4f}  ROUGE-L: {r['metrics']['rouge_l']:.4f}")


if __name__ == "__main__":
    main()
