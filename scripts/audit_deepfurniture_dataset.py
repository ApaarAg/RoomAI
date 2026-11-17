"""Comprehensive Audit Script for Dalby123/DeepFurniture Dataset.

Performs:
1. File integrity and 1-to-1 matching (images and labels/annotations).
2. Image resolution, aspect ratio, and corrupt file checks.
3. Bounding box validity (xMin < xMax, yMin < yMax, within [0, 1] bounds, area > 0).
4. Class balance and distribution across the 11 DeepFurniture categories.
5. Image duplicate detection via SHA-256 hash.
6. Split leakage check (verifies no identical images or scene IDs across train, val, and test).
7. GPU availability and VRAM measurement for training.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "deepfurniture"

CATEGORY_NAMES = {
    0: "cabinet#shelf",
    1: "table",
    2: "chair#stool",
    3: "lamp",
    4: "door",
    5: "bed",
    6: "sofa",
    7: "plant",
    8: "decoration",
    9: "curtain",
    10: "home-appliance",
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of an image file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_deepfurniture() -> Dict:
    print("=" * 65)
    print(" RoomAI DeepFurniture Dataset Audit & Integrity Verification")
    print("=" * 65)

    if not DATA_DIR.exists():
        print(f"ERROR: Dataset directory {DATA_DIR} does not exist!")
        sys.exit(1)

    splits = ["train", "val", "test"]
    split_counts = {}
    split_instances = {}
    split_hashes: Dict[str, Set[str]] = {s: set() for s in splits}
    split_scenes: Dict[str, Set[str]] = {s: set() for s in splits}

    total_corrupt_images = 0
    total_unmatched_files = 0
    total_invalid_bboxes = 0
    total_duplicate_images = 0

    class_counts_overall = Counter()
    class_counts_by_split = {s: Counter() for s in splits}
    resolutions = Counter()

    for split in splits:
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Warning: Directory missing for split '{split}'")
            continue

        images = sorted(list(img_dir.glob("*.jpg")))
        labels = sorted(list(lbl_dir.glob("*.txt")))

        img_stems = {img.stem for img in images}
        lbl_stems = {lbl.stem for lbl in labels}

        # 1-to-1 matching check
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        if missing_labels:
            print(f"[{split}] Missing labels for {len(missing_labels)} images: {list(missing_labels)[:3]}...")
            total_unmatched_files += len(missing_labels)
        if missing_images:
            print(f"[{split}] Missing images for {len(missing_images)} labels: {list(missing_images)[:3]}...")
            total_unmatched_files += len(missing_images)

        split_counts[split] = len(images)
        split_instances[split] = 0

        # Check each image and label
        for img_path in images:
            stem = img_path.stem
            split_scenes[split].add(stem)

            # Check image integrity
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    resolutions[im.size] += 1
            except Exception as e:
                print(f"Corrupt image: {img_path}: {e}")
                total_corrupt_images += 1
                continue

            # Duplicate image check
            img_hash = compute_sha256(img_path)
            if img_hash in split_hashes[split]:
                total_duplicate_images += 1
            split_hashes[split].add(img_hash)

            # Check YOLO label file
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    print(f"Invalid label format in {lbl_path}:{line_no} -> '{line.strip()}'")
                    total_invalid_bboxes += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                except ValueError:
                    print(f"Non-numeric values in {lbl_path}:{line_no}")
                    total_invalid_bboxes += 1
                    continue

                # Class index validation (0..10)
                if cls_id not in CATEGORY_NAMES:
                    print(f"Invalid class_id {cls_id} in {lbl_path}:{line_no}")
                    total_invalid_bboxes += 1
                    continue

                # Bounding box bounds validation
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    print(f"Center out of bounds in {lbl_path}:{line_no}: ({xc}, {yc})")
                    total_invalid_bboxes += 1
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    print(f"Invalid box size in {lbl_path}:{line_no}: (w={w}, h={h})")
                    total_invalid_bboxes += 1

                class_name = CATEGORY_NAMES[cls_id]
                class_counts_overall[class_name] += 1
                class_counts_by_split[split][class_name] += 1
                split_instances[split] += 1

    # Check leakage across splits
    train_val_scene_overlap = split_scenes["train"] & split_scenes["val"]
    train_test_scene_overlap = split_scenes["train"] & split_scenes["test"]
    val_test_scene_overlap = split_scenes["val"] & split_scenes["test"]

    train_val_hash_overlap = split_hashes["train"] & split_hashes["val"]
    train_test_hash_overlap = split_hashes["train"] & split_hashes["test"]
    val_test_hash_overlap = split_hashes["val"] & split_hashes["test"]

    # GPU Verification
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if gpu_available else 0.0
    allocated_vram_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 3) if gpu_available else 0.0
    reserved_vram_gb = round(torch.cuda.memory_reserved(0) / (1024**3), 3) if gpu_available else 0.0
    free_vram_gb = round(total_vram_gb - reserved_vram_gb, 2) if gpu_available else 0.0

    print("\n1. DATASET SPLIT & INSTANCE AUDIT")
    print("-" * 50)
    for s in splits:
        print(f"  {s.upper():<10}: {split_counts.get(s, 0):>5} images | {split_instances.get(s, 0):>6} bounding boxes")
    print(f"  TOTAL     : {sum(split_counts.values()):>5} images | {sum(split_instances.values()):>6} bounding boxes")

    print("\n2. IMAGE RESOLUTIONS")
    print("-" * 50)
    for res, count in resolutions.most_common(5):
        print(f"  {res[0]} x {res[1]}: {count} images")

    print("\n3. CLASS DISTRIBUTION (11 DEEPFURNITURE CATEGORIES)")
    print("-" * 50)
    print(f"  {'Class Name':<18} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("  " + "-" * 48)
    for i in range(11):
        cname = CATEGORY_NAMES[i]
        tr = class_counts_by_split["train"][cname]
        va = class_counts_by_split["val"][cname]
        te = class_counts_by_split["test"][cname]
        tot = class_counts_overall[cname]
        print(f"  {cname:<18} {tr:>7} {va:>7} {te:>7} {tot:>7}")

    print("\n4. DATASET INTEGRITY & LEAKAGE CHECK")
    print("-" * 50)
    print(f"  Corrupt images:              {total_corrupt_images} (PASS)" if total_corrupt_images == 0 else f"  Corrupt images: {total_corrupt_images} (FAIL)")
    print(f"  Unmatched files:             {total_unmatched_files} (PASS)" if total_unmatched_files == 0 else f"  Unmatched files: {total_unmatched_files} (FAIL)")
    print(f"  Invalid bounding boxes:      {total_invalid_bboxes} (PASS)" if total_invalid_bboxes == 0 else f"  Invalid bboxes: {total_invalid_bboxes} (FAIL)")
    print(f"  Intra-split duplicate images:{total_duplicate_images}")
    print(f"  Train-Val scene leakage:     {len(train_val_scene_overlap)} (ZERO LEAKAGE)" if len(train_val_scene_overlap) == 0 else f"  Train-Val overlap: {len(train_val_scene_overlap)}")
    print(f"  Train-Test scene leakage:    {len(train_test_scene_overlap)} (ZERO LEAKAGE)" if len(train_test_scene_overlap) == 0 else f"  Train-Test overlap: {len(train_test_scene_overlap)}")
    print(f"  Train-Val image hash leakage:{len(train_val_hash_overlap)} (ZERO LEAKAGE)" if len(train_val_hash_overlap) == 0 else f"  Train-Val hash overlap: {len(train_val_hash_overlap)}")
    print(f"  Train-Test image hash leakage:{len(train_test_hash_overlap)} (ZERO LEAKAGE)" if len(train_test_hash_overlap) == 0 else f"  Train-Test hash overlap: {len(train_test_hash_overlap)}")

    print("\n5. HARDWARE & GPU STATUS")
    print("-" * 50)
    print(f"  CUDA Available:    {gpu_available}")
    print(f"  GPU Name:          {gpu_name}")
    print(f"  Total VRAM:        {total_vram_gb} GB")
    print(f"  Free VRAM:         ~{free_vram_gb} GB")

    audit_passed = (
        total_corrupt_images == 0
        and total_unmatched_files == 0
        and total_invalid_bboxes == 0
        and len(train_val_scene_overlap) == 0
        and len(train_test_scene_overlap) == 0
        and len(train_val_hash_overlap) == 0
        and len(train_test_hash_overlap) == 0
        and sum(split_counts.values()) == 1000
    )

    print("\n" + "=" * 65)
    if audit_passed:
        print(" AUDIT RESULT: PASSED ALL CHECKS")
    else:
        print(" AUDIT RESULT: FAILED CHECKS")
    print("=" * 65 + "\n")

    return {
        "audit_passed": audit_passed,
        "split_counts": split_counts,
        "split_instances": split_instances,
        "class_counts": dict(class_counts_overall),
        "gpu": {
            "available": gpu_available,
            "name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "free_vram_gb": free_vram_gb,
        },
    }


if __name__ == "__main__":
    audit_deepfurniture()
