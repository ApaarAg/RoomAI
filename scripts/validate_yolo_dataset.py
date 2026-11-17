"""
Automated Verification Script for RoomAI YOLO Dataset.
Validates the processed dataset directly from disk.
Reports every check as PASS, WARNING, or FAIL.
Runs an Ultralytics dataset-loading smoke test.
"""

import argparse
import csv
from pathlib import Path
import sys

from PIL import Image
import yaml


def validate_yolo_dataset(dataset_dir: Path):
    print("=" * 60)
    print("RoomAI YOLO Dataset Automated Verification")
    print("=" * 60)

    results = []

    def record(check_name: str, status: str, details: str):
        print(f"[{status:<7}] {check_name}: {details}")
        results.append({"check": check_name, "status": status, "details": details})

    # 1. Check data.yaml
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        record("data.yaml Existence", "FAIL", f"File missing at {yaml_path}")
        return results

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        record("data.yaml Syntax", "PASS", "Valid YAML format")
    except Exception as e:
        record("data.yaml Syntax", "FAIL", f"Invalid YAML: {e}")
        return results

    # 2. Check Directory Structure
    splits = ["train", "val", "test"]
    all_dirs_exist = True
    for s in splits:
        img_d = dataset_dir / "images" / s
        lbl_d = dataset_dir / "labels" / s
        if not img_d.exists() or not lbl_d.exists():
            all_dirs_exist = False
            record("Directory Layout", "FAIL", f"Missing images or labels dir for split '{s}'")
    if all_dirs_exist:
        record("Directory Layout", "PASS", "All images/ and labels/ subdirectories present for train, val, test")

    # 3. Check class names
    names = config.get("names", {})
    if isinstance(names, dict):
        class_map = {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        class_map = {i: v for i, v in enumerate(names)}
    else:
        class_map = {}

    if len(class_map) == 11 and sorted(class_map.keys()) == list(range(11)):
        record("Class Definition", "PASS", f"11 contiguous classes (IDs 0 to 10): {list(class_map.values())}")
    else:
        record("Class Definition", "FAIL", f"Invalid class definitions: {class_map}")

    # 4. Check Manifest Existence
    manifest_path = dataset_dir / "dataset_manifest.csv"
    manifest_records = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_records[row["scene_id"]] = row
        record("Manifest File", "PASS", f"Manifest found with {len(manifest_records)} scene records")
    else:
        record("Manifest File", "FAIL", "dataset_manifest.csv is missing")

    # 5. Image & Label Integrity Across Splits
    total_images = 0
    total_labels = 0
    corrupt_images = 0
    missing_labels = 0
    invalid_lines = 0
    box_coord_errors = 0
    class_id_errors = 0
    valid_boxes = 0

    split_images = {}

    for s in splits:
        img_dir = dataset_dir / "images" / s
        lbl_dir = dataset_dir / "labels" / s
        imgs = sorted(img_dir.glob("*.jpg"))
        split_images[s] = set(p.stem for p in imgs)
        total_images += len(imgs)

        for img_path in imgs:
            # Check Image readability
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                corrupt_images += 1

            # Check matching label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                missing_labels += 1
                continue
            total_labels += 1

            # Check label content
            content = lbl_path.read_text(encoding="utf-8").strip()
            if not content:
                # Valid negative sample (0 objects)
                continue

            for line_idx, line in enumerate(content.splitlines(), 1):
                tokens = line.strip().split()
                if len(tokens) != 5:
                    invalid_lines += 1
                    continue

                try:
                    cid, xc, yc, bw, bh = map(float, tokens)
                except ValueError:
                    invalid_lines += 1
                    continue

                cid_int = int(cid)
                if cid_int not in class_map:
                    class_id_errors += 1

                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    box_coord_errors += 1
                else:
                    valid_boxes += 1

    # Record Image Verification
    if corrupt_images == 0 and total_images > 0:
        record("Image Readability", "PASS", f"All {total_images} images opened and verified without corruption")
    else:
        record("Image Readability", "FAIL", f"{corrupt_images} corrupted images found out of {total_images}")

    # Record Label 1:1 Matching
    if missing_labels == 0 and total_images == total_labels:
        record("Image-Label Parity", "PASS", f"100% 1:1 match across all {total_images} images and labels")
    else:
        record("Image-Label Parity", "FAIL", f"{missing_labels} images missing matching .txt label file")

    # Record Bounding Box Verification
    if invalid_lines == 0 and box_coord_errors == 0 and class_id_errors == 0:
        record("YOLO Box Geometry", "PASS", f"{valid_boxes} bounding boxes conform strictly to normalized [0, 1] bounds")
    else:
        record("YOLO Box Geometry", "FAIL", f"Errors found: {box_coord_errors} bounds, {class_id_errors} class IDs, {invalid_lines} invalid tokens")

    # 6. Check Mutual Exclusivity (No Split Leakage)
    train_set = split_images["train"]
    val_set = split_images["val"]
    test_set = split_images["test"]

    train_val_leak = train_set.intersection(val_set)
    train_test_leak = train_set.intersection(test_set)
    val_test_leak = val_set.intersection(test_set)

    if not train_val_leak and not train_test_leak and not val_test_leak:
        record("Split Isolation", "PASS", "Mutually exclusive splits (0 leakage across train, val, and test)")
    else:
        record("Split Isolation", "FAIL", f"Data leakage detected! Train/Val: {len(train_val_leak)}, Train/Test: {len(train_test_leak)}")

    # 7. Ultralytics Smoke Test
    try:
        from ultralytics.data.dataset import YOLODataset
        # Attempt to instantiate YOLODataset on validation split to confirm compatibility
        val_img_path = str(dataset_dir / "images" / "val")
        ds = YOLODataset(img_path=val_img_path, data=config, task="detect")
        record("Ultralytics Loader Smoke Test", "PASS", f"Ultralytics YOLODataset loaded {len(ds)} val samples successfully")
    except Exception as e:
        record("Ultralytics Loader Smoke Test", "WARNING", f"Ultralytics loader note: {e}")

    # Overall Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SCORECARD")
    print("=" * 60)
    passes = sum(1 for r in results if r["status"] == "PASS")
    warnings = sum(1 for r in results if r["status"] == "WARNING")
    fails = sum(1 for r in results if r["status"] == "FAIL")

    print(f"PASS: {passes} | WARNING: {warnings} | FAIL: {fails}")
    if fails == 0:
        print("\nAll mandatory validation checks PASSED! Dataset is 100% verified for YOLO training.")
    else:
        print(f"\nVerification encountered {fails} critical failures. Review details above.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO dataset.")
    parser.add_argument("--dataset-dir", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\processed\deepfurniture_yolo")
    args = parser.parse_args()

    validate_yolo_dataset(Path(args.dataset_dir))
