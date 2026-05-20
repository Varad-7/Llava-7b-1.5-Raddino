"""
Data Preparation Script for IU-Xray.

Reads the CSV reports file and the image directory, matches images to UIDs,
and creates train/val/test JSON splits for training.

Your data structure:
  IU-Xray/
  ├── datasets/raddar/chest-xrays-indiana.../versions/2/images/images_normalized/
  │   ├── 106_IM-0042-1001.dcm.png
  │   ├── 106_IM-0042-2001.dcm.png
  │   ├── 107_IM-0049-1001.dcm.png
  │   └── ...
  └── processed_indiana_reports.csv

Image naming convention:
  {UID}_IM-{XXXX}-{XXXX}.dcm.png
  └── UID is the first number, e.g., 106, 107, 108

This script:
  1. Reads the CSV to get UID → report mapping
  2. Scans the image directory to get UID → [image_files] mapping
  3. Matches them together
  4. Splits into train/val/test by UID (patient-level split to prevent leakage)
  5. Saves JSON files

Usage:
  python prepare_data.py
"""

import os
import sys
import json
import csv
import random
import re
from collections import defaultdict

import config as cfg


def extract_uid_from_filename(filename: str) -> str:
    """
    Extract UID from image filename.
    
    Examples:
      '106_IM-0042-1001.dcm.png' → '106'
      '1234_IM-0042-1001.dcm.png' → '1234'
    """
    # UID is everything before the first underscore
    match = re.match(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    return None


def load_reports_csv(csv_path: str) -> dict:
    """
    Load reports from CSV file.
    
    Automatically detects the UID column and report column(s).
    Tries common column names for UID and report text.
    
    Returns:
        dict mapping UID (str) → report (str)
    """
    uid_to_report = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Sniff the delimiter
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
        except csv.Error:
            reader = csv.DictReader(f)
        
        # Get column names
        fieldnames = reader.fieldnames
        print(f"[CSV] Columns found: {fieldnames}")
        
        # --- Detect UID column ---
        uid_col = None
        for candidate in ['uid', 'UID', 'Uid', 'id', 'ID', 'Id', 
                          'study_id', 'subject_id', 'patient_id',
                          'report_id', 'image_id']:
            if candidate in fieldnames:
                uid_col = candidate
                break
        
        if uid_col is None:
            # Try the first column
            uid_col = fieldnames[0]
            print(f"[CSV] WARNING: No standard UID column found. Using first column: '{uid_col}'")
        else:
            print(f"[CSV] UID column: '{uid_col}'")
        
        # --- Detect report column(s) ---
        report_col = None
        findings_col = None
        impression_col = None
        
        for candidate in ['report', 'Report', 'REPORT', 'text', 'Text',
                          'report_text', 'radiology_report']:
            if candidate in fieldnames:
                report_col = candidate
                break
        
        for candidate in ['findings', 'Findings', 'FINDINGS']:
            if candidate in fieldnames:
                findings_col = candidate
                break
        
        for candidate in ['impression', 'Impression', 'IMPRESSION',
                          'conclusion', 'Conclusion']:
            if candidate in fieldnames:
                impression_col = candidate
                break
        
        if report_col:
            print(f"[CSV] Report column: '{report_col}'")
        elif findings_col or impression_col:
            print(f"[CSV] Findings column: '{findings_col}', Impression column: '{impression_col}'")
        else:
            # Try to find any text-like column
            for col in fieldnames:
                if col != uid_col:
                    report_col = col
                    break
            print(f"[CSV] WARNING: No standard report column found. Using: '{report_col}'")
        
        # --- Read rows ---
        for row in reader:
            uid = str(row.get(uid_col, '')).strip()
            if not uid:
                continue
            
            # Build report text
            if report_col and row.get(report_col, '').strip():
                report = row[report_col].strip()
            else:
                parts = []
                if findings_col and row.get(findings_col, '').strip():
                    parts.append(row[findings_col].strip())
                if impression_col and row.get(impression_col, '').strip():
                    parts.append(row[impression_col].strip())
                report = ' '.join(parts)
            
            if report:
                uid_to_report[uid] = report
    
    print(f"[CSV] Loaded {len(uid_to_report)} reports")
    return uid_to_report


def scan_images(image_dir: str) -> dict:
    """
    Scan image directory and group images by UID.
    
    Returns:
        dict mapping UID (str) → list of image filenames
    """
    uid_to_images = defaultdict(list)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.dcm.png'}
    
    for filename in sorted(os.listdir(image_dir)):
        # Check if it's an image
        lower = filename.lower()
        if not any(lower.endswith(ext) for ext in image_extensions):
            continue
        
        uid = extract_uid_from_filename(filename)
        if uid:
            uid_to_images[uid].append(filename)
    
    print(f"[Images] Found {sum(len(v) for v in uid_to_images.values())} images "
          f"across {len(uid_to_images)} UIDs")
    
    return dict(uid_to_images)


def create_splits(
    uid_to_report: dict,
    uid_to_images: dict,
    train_ratio: float = cfg.TRAIN_RATIO,
    val_ratio: float = cfg.VAL_RATIO,
    test_ratio: float = cfg.TEST_RATIO,
    seed: int = cfg.SEED,
) -> tuple:
    """
    Create train/val/test splits at the UID (patient) level.
    
    Patient-level splitting prevents data leakage — all images for a given
    patient stay in the same split.
    
    Returns:
        (train_samples, val_samples, test_samples) — each is a list of dicts
    """
    # Find UIDs that exist in BOTH reports and images
    common_uids = sorted(set(uid_to_report.keys()) & set(uid_to_images.keys()))
    
    report_only = set(uid_to_report.keys()) - set(uid_to_images.keys())
    image_only = set(uid_to_images.keys()) - set(uid_to_report.keys())
    
    print(f"\n[Match] UIDs with both report + images: {len(common_uids)}")
    if report_only:
        print(f"[Match] UIDs with report but no images: {len(report_only)} (skipped)")
    if image_only:
        print(f"[Match] UIDs with images but no report: {len(image_only)} (skipped)")
    
    if len(common_uids) == 0:
        print("\n❌ ERROR: No UIDs match between CSV and images!")
        print(f"   Sample CSV UIDs: {list(uid_to_report.keys())[:5]}")
        print(f"   Sample image UIDs: {list(uid_to_images.keys())[:5]}")
        print("   Check if the UID format matches (e.g., '106' vs 'CXR106')")
        sys.exit(1)
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(common_uids)
    
    n_total = len(common_uids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # n_test is the remainder
    
    train_uids = common_uids[:n_train]
    val_uids = common_uids[n_train:n_train + n_val]
    test_uids = common_uids[n_train + n_val:]
    
    def build_samples(uids):
        samples = []
        for uid in uids:
            images = uid_to_images[uid]
            report = uid_to_report[uid]
            # Use the first (frontal) image as primary
            # Store all images for reference
            samples.append({
                "id": uid,
                "image_path": images[0],       # Primary image (frontal)
                "all_images": images,           # All images for this UID
                "report": report,
            })
        return samples
    
    train_samples = build_samples(train_uids)
    val_samples = build_samples(val_uids)
    test_samples = build_samples(test_uids)
    
    print(f"\n[Split] Train: {len(train_samples)} samples ({len(train_uids)} UIDs)")
    print(f"[Split] Val:   {len(val_samples)} samples ({len(val_uids)} UIDs)")
    print(f"[Split] Test:  {len(test_samples)} samples ({len(test_uids)} UIDs)")
    
    return train_samples, val_samples, test_samples


def main():
    print("=" * 60)
    print("IU-Xray Data Preparation")
    print("=" * 60)
    
    # ---- Validate paths ----
    if not os.path.exists(cfg.REPORTS_CSV):
        print(f"\n❌ CSV not found: {cfg.REPORTS_CSV}")
        print("   Edit REPORTS_CSV in config.py")
        sys.exit(1)
    
    if not os.path.isdir(cfg.IMAGE_DIR):
        print(f"\n❌ Image directory not found: {cfg.IMAGE_DIR}")
        print("   Edit IMAGE_DIR in config.py")
        sys.exit(1)
    
    # ---- Load data ----
    print(f"\n📄 Loading reports from: {cfg.REPORTS_CSV}")
    uid_to_report = load_reports_csv(cfg.REPORTS_CSV)
    
    print(f"\n🖼️  Scanning images in: {cfg.IMAGE_DIR}")
    uid_to_images = scan_images(cfg.IMAGE_DIR)
    
    # ---- Show sample data ----
    print("\n--- Sample reports (first 3) ---")
    for uid in list(uid_to_report.keys())[:3]:
        print(f"  UID {uid}: {uid_to_report[uid][:100]}...")
    
    print("\n--- Sample images (first 3 UIDs) ---")
    for uid in list(uid_to_images.keys())[:3]:
        print(f"  UID {uid}: {uid_to_images[uid]}")
    
    # ---- Create splits ----
    train, val, test = create_splits(uid_to_report, uid_to_images)
    
    # ---- Save JSON files ----
    os.makedirs(cfg.SPLITS_DIR, exist_ok=True)
    
    for split_name, data, path in [
        ("train", train, cfg.TRAIN_JSON),
        ("val", val, cfg.VAL_JSON),
        ("test", test, cfg.TEST_JSON),
    ]:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {split_name}: {path} ({len(data)} samples)")
    
    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Train JSON: {cfg.TRAIN_JSON}")
    print(f"  Val JSON:   {cfg.VAL_JSON}")
    print(f"  Test JSON:  {cfg.TEST_JSON}")
    print(f"\n  Total matched samples: {len(train) + len(val) + len(test)}")
    print(f"  Split ratios: {cfg.TRAIN_RATIO}/{cfg.VAL_RATIO}/{cfg.TEST_RATIO}")
    print(f"\n  Sample entry:")
    if train:
        print(f"  {json.dumps(train[0], indent=4)}")
    print(f"{'='*60}")
    print(f"\n  Next step: python train.py --stage 1")


if __name__ == "__main__":
    main()
