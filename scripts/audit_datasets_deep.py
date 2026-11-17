"""Deep Dataset Audit Script for RoomAI.

Audits:
  - Split counts and label completeness
  - Class frequency distributions and imbalance ratios
  - Split contamination / exact duplicate image leakage (MD5 hash collision)
  - Scene / property correlation leakage across splits
  - Annotation geometry anomalies (out-of-bounds, degenerate, tiny, huge boxes)
  - Background (empty) images
  - Outputs reports/data_quality_audit.json
"""

import hashlib
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPORT_DIR = REPO_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR = REPO_ROOT / "artifacts" / "audit_galleries"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name": "indoor_canonical",
        "yaml": "data/indoor_canonical/indoor_canonical.yaml",
        "path": "data/indoor_canonical",
        "classes": [
            "cabinet#shelf", "table", "chair#stool", "lamp", "door", "bed",
            "sofa", "plant", "decoration", "curtain", "home-appliance", "window"
        ],
    },
    {
        "name": "exterior_canonical",
        "yaml": "data/exterior_canonical/exterior_canonical.yaml",
        "path": "data/exterior_canonical",
        "classes": [
            "building", "facade", "window", "door", "roof_cornice",
            "balcony", "porch", "pillar_column", "shutter_blind", "molding_deco"
        ],
    },
    {
        "name": "parking_canonical",
        "yaml": "data/parking_canonical/parking_canonical.yaml",
        "path": "data/parking_canonical",
        "classes": [
            "car_vehicle", "motorcycle", "bicycle", "outdoor_seating",
            "outdoor_table", "plant_bush", "tree", "outdoor_fixture"
        ],
    },
    {
        "name": "deepfurniture",
        "yaml": "data/deepfurniture/deepfurniture.yaml",
        "path": "data/deepfurniture",
        "classes": [
            "cabinet#shelf", "table", "chair#stool", "lamp", "door", "bed",
            "sofa", "plant", "decoration", "curtain", "home-appliance"
        ],
    },
]


def compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def audit_single_dataset(ds_info: dict) -> dict:
    ds_path = REPO_ROOT / ds_info["path"]
    classes = ds_info["classes"]
    print(f"\n=======================================================")
    print(f"Auditing Dataset: {ds_info['name']}")
    print(f"=======================================================")

    report = {
        "dataset_name": ds_info["name"],
        "num_classes": len(classes),
        "class_names": classes,
        "splits": {},
        "leakage_analysis": {},
        "class_distribution": {},
        "geometry_anomalies": {},
        "background_images": {},
    }

    # 1. Image and label counts, MD5 hashing for leakage
    hash_to_split_files = defaultdict(list)
    stem_to_split = defaultdict(set)

    for split in ["train", "val", "test"]:
        img_dir = ds_path / "images" / split
        lbl_dir = ds_path / "labels" / split

        img_files = sorted(list(img_dir.glob("*.*"))) if img_dir.exists() else []
        lbl_files = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []

        img_stems = {f.stem for f in img_files}
        lbl_stems = {f.stem for f in lbl_files}

        # Track missing labels vs missing images
        imgs_without_lbl = sorted(list(img_stems - lbl_stems))
        lbls_without_img = sorted(list(lbl_stems - img_stems))

        report["splits"][split] = {
            "image_count": len(img_files),
            "label_count": len(lbl_files),
            "images_without_label_file": len(imgs_without_lbl),
            "labels_without_image_file": len(lbls_without_img),
        }
        print(f"Split {split.upper()}: {len(img_files)} images, {len(lbl_files)} label files")

        # Hash each image
        for img_p in img_files:
            try:
                h = compute_file_hash(img_p)
                hash_to_split_files[h].append((split, img_p.name))
                # Property/scene stem: extract prefix before sequence/instance id if exists
                stem = img_p.stem
                stem_to_split[stem].add(split)
            except Exception as e:
                pass

    # 2. Check for exact duplicate image leakage across splits
    exact_duplicates_across_splits = []
    for h, locations in hash_to_split_files.items():
        splits_present = {loc[0] for loc in locations}
        if len(splits_present) > 1:
            exact_duplicates_across_splits.append({
                "hash": h,
                "splits": list(splits_present),
                "occurrences": [f"{loc[0]}/{loc[1]}" for loc in locations],
            })

    report["leakage_analysis"]["exact_duplicate_leakage_count"] = len(exact_duplicates_across_splits)
    report["leakage_analysis"]["exact_duplicates"] = exact_duplicates_across_splits[:20]  # sample
    print(f"  Exact Duplicate Cross-Split Leakage: {len(exact_duplicates_across_splits)} instances")

    # 3. Check property/scene stem correlation leakage
    # For instance, if scene ID (e.g. df_L3D121...) is shared
    scene_id_leakage = []
    scene_counter = defaultdict(set)
    for stem, splits in stem_to_split.items():
        # Heuristic: split by '_' to get base scene ID
        parts = stem.split("_")
        if len(parts) >= 2:
            scene_key = "_".join(parts[:2])
            for s in splits:
                scene_counter[scene_key].add(s)

    for sc, splits_in_sc in scene_counter.items():
        if len(splits_in_sc) > 1 and "test" in splits_in_sc:
            scene_id_leakage.append({"scene_prefix": sc, "splits": list(splits_in_sc)})

    report["leakage_analysis"]["correlated_scene_leakage_count"] = len(scene_id_leakage)
    report["leakage_analysis"]["sample_correlated_scenes"] = scene_id_leakage[:15]
    print(f"  Correlated Scene Prefix Leakage to Test: {len(scene_id_leakage)} scene groups")

    # 4. Class Distribution & Box Geometry Anomalies
    total_class_counts = Counter()
    split_class_counts = {s: Counter() for s in ["train", "val", "test"]}
    anomalies = {
        "out_of_bounds_coords": 0,
        "degenerate_boxes": 0,
        "tiny_boxes_sub_0_1_pct": 0,
        "giant_boxes_gt_90_pct": 0,
        "invalid_class_id": 0,
        "empty_label_files": 0,
    }

    for split in ["train", "val", "test"]:
        lbl_dir = ds_path / "labels" / split
        if not lbl_dir.exists():
            continue

        for lbl_p in lbl_dir.glob("*.txt"):
            lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                anomalies["empty_label_files"] += 1
                continue

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w, h = [float(v) for v in parts[1:5]]

                    if cid < 0 or cid >= len(classes):
                        anomalies["invalid_class_id"] += 1
                        continue

                    total_class_counts[classes[cid]] += 1
                    split_class_counts[split][classes[cid]] += 1

                    # Box checks
                    if w <= 0 or h <= 0:
                        anomalies["degenerate_boxes"] += 1
                    if x < 0 or x > 1 or y < 0 or y > 1 or w > 1 or h > 1:
                        anomalies["out_of_bounds_coords"] += 1
                    if (x - w / 2 < -0.05) or (x + w / 2 > 1.05) or (y - h / 2 < -0.05) or (y + h / 2 > 1.05):
                        anomalies["out_of_bounds_coords"] += 1

                    area = w * h
                    if area < 0.001:
                        anomalies["tiny_boxes_sub_0_1_pct"] += 1
                    if area > 0.90:
                        anomalies["giant_boxes_gt_90_pct"] += 1

                except ValueError:
                    pass

    # Imbalance Ratio
    counts_list = [total_class_counts[c] for c in classes]
    max_c = max(counts_list) if counts_list else 0
    min_c = min(counts_list) if counts_list else 0
    imbalance_ratio = round(max_c / (min_c + 1e-6), 2)

    report["class_distribution"] = {
        "total_instances": sum(counts_list),
        "per_class_counts": {c: total_class_counts[c] for c in classes},
        "per_split_counts": {s: {c: split_class_counts[s][c] for c in classes} for s in ["train", "val", "test"]},
        "max_class_count": max_c,
        "min_class_count": min_c,
        "imbalance_ratio": imbalance_ratio,
    }
    report["geometry_anomalies"] = anomalies

    print(f"  Total Box Instances: {sum(counts_list):,}")
    print(f"  Imbalance Ratio: {imbalance_ratio}:1 (Max: {max_c}, Min: {min_c})")
    print(f"  Geometry Anomalies: {anomalies}")

    return report


def main():
    print("=" * 80)
    print("RoomAI Comprehensive Data Quality & Split Audit")
    print("=" * 80)

    all_reports = {}
    for ds in DATASETS:
        all_reports[ds["name"]] = audit_single_dataset(ds)

    out_json = REPORT_DIR / "data_quality_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\nAudit complete! Saved detailed findings to: {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
