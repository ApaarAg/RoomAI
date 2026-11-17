"""Audit script to inspect all datasets, YAML configs, image counts, and label distributions."""

import os
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

print("=" * 70)
print("RoomAI Dataset Audit: Canonical Datasets and Splits")
print("=" * 70)

yaml_files = list(DATA_DIR.rglob("*.yaml")) + list(DATA_DIR.rglob("*.yml"))
for yf in yaml_files:
    print(f"\n--- Found Dataset YAML: {yf.relative_to(REPO_ROOT)} ---")
    try:
        with open(yf, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Names: {cfg.get('names')}")
        print(f"nc: {cfg.get('nc')}")
        for split in ["train", "val", "test"]:
            split_path = cfg.get(split)
            if split_path:
                resolved = Path(split_path)
                if not resolved.is_absolute():
                    resolved = yf.parent / split_path
                if resolved.exists():
                    img_count = len(list(resolved.glob("*.*")))
                    print(f"  {split}: {resolved.name} -> {img_count} images")
                else:
                    print(f"  {split}: {split_path} (NOT FOUND at {resolved})")
    except Exception as e:
        print(f"  Error loading YAML: {e}")

# Check directories in data
print("\n" + "=" * 70)
print("Scanning Data Folders for Images and Labels:")
print("=" * 70)
for sub in ["indoor_canonical", "exterior_canonical", "parking_canonical", "deepfurniture"]:
    p = DATA_DIR / sub
    if p.exists():
        print(f"\nFolder: data/{sub}")
        for split in ["train", "val", "test"]:
            img_dir = p / "images" / split
            lbl_dir = p / "labels" / split
            n_imgs = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
            n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            print(f"  Split '{split}': {n_imgs} images, {n_lbls} label files")
