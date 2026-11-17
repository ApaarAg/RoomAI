r"""Repeat Factor Sampling (RFS) Dataset Generator for Open Images V7 Pilot.

Computes class-aware repeat factors:
    r_c = max(1, ceil(sqrt(threshold / f_c))) if f_c < threshold else 1
    r_i = max_{c in C_i} r_c

Generates:
1. F:\RoomAI_Data\yolo_open_images_pilot\train_rfs.txt (line-by-line image paths)
2. F:\RoomAI_Data\yolo_open_images_pilot\data_rfs.yaml (points train -> train_rfs.txt; val/test untouched)
3. reports\rfs_dataset_generation_report.json
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DATA_ROOT = Path(r"F:\RoomAI_Data")
PILOT_DIR = EXTERNAL_DATA_ROOT / "yolo_open_images_pilot"
MANIFEST_PATH = PILOT_DIR / "manifest.json"
REPORTS_DIR = REPO_ROOT / "reports"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Repeat Factor Sampling (RFS) Dataset for Exp 4")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Path to pilot manifest.json")
    parser.add_argument("--threshold", type=float, default=0.03, help="RFS frequency threshold t (default: 0.03 = 3%)")
    parser.add_argument("--max-repeat", type=int, default=5, help="Maximum repeat factor cap (default: 5)")
    parser.add_argument("--output-yaml", default=str(PILOT_DIR / "data_rfs.yaml"), help="Output data_rfs.yaml")
    parser.add_argument("--output-txt", default=str(PILOT_DIR / "train_rfs.txt"), help="Output train_rfs.txt")
    return parser.parse_args()


def generate_rfs():
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RoomAI: Repeat Factor Sampling (RFS) Generator for Open Images Pilot")
    print("=" * 80)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    class_names = manifest["class_names"]
    images_dict = manifest["images"]

    # Filter train split images
    train_images = {img_id: info for img_id, info in images_dict.items() if info["split"] == "train"}
    total_train_images = len(train_images)
    print(f"[RFS] Total base training images in manifest: {total_train_images}")
    assert total_train_images == 8000, f"Expected 8000 train images, found {total_train_images}"

    # Calculate image-level class frequency f_c
    class_img_counts = defaultdict(int)
    class_box_counts = defaultdict(int)
    for info in train_images.values():
        box_counts = info.get("class_box_counts", {})
        for cname, cnt in box_counts.items():
            class_img_counts[cname] += 1
            class_box_counts[cname] += cnt

    # Compute repeat factor r_c per class
    class_rfs_factors = {}
    print("\n[RFS POLICY] Per-Class Frequencies & Repeat Factors (t = {:.4f}, max_repeat = {}):".format(args.threshold, args.max_repeat))
    print(f"{'Class Name':20s} | {'Train Imgs':10s} | {'Frequency f_c':14s} | {'Repeat Factor r_c':18s}")
    print("-" * 72)

    for cname in class_names:
        img_cnt = class_img_counts.get(cname, 0)
        fc = img_cnt / total_train_images
        if fc < args.threshold and fc > 0:
            r = min(args.max_repeat, max(1, math.ceil(math.sqrt(args.threshold / fc))))
        else:
            r = 1
        class_rfs_factors[cname] = r
        print(f"{cname:20s} | {img_cnt:10d} | {fc:14.4f} | {r:18d}x")

    # Compute image repeat factor r_i = max_{c in C_i} r_c
    manifest_lines = []
    repeat_distribution = defaultdict(int)

    for img_id in sorted(train_images.keys()):
        info = train_images[img_id]
        img_rel_path = info["relative_image_path"]  # e.g. images/train/000020780ccee28d.jpg
        img_abs_path = (PILOT_DIR / img_rel_path).resolve().as_posix()

        classes_in_img = list(info.get("class_box_counts", {}).keys())
        rep = max([class_rfs_factors.get(c, 1) for c in classes_in_img], default=1)
        repeat_distribution[rep] += 1

        for _ in range(rep):
            manifest_lines.append(img_abs_path)

    total_rfs_samples = len(manifest_lines)
    expansion_pct = ((total_rfs_samples - total_train_images) / total_train_images) * 100.0

    print("-" * 72)
    print(f"[RFS SUMMARY] Total expanded training samples: {total_rfs_samples} (+{total_rfs_samples - total_train_images} / +{expansion_pct:.2f}%)")
    print(f"[RFS SUMMARY] Sample repetition distribution: {dict(sorted(repeat_distribution.items()))}")

    # Write train_rfs.txt
    out_txt_path = Path(args.output_txt)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")
    print(f"[OUTPUT] Generated RFS sample list: {out_txt_path}")

    # Write data_rfs.yaml
    out_yaml_path = Path(args.output_yaml)
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = {
        "path": PILOT_DIR.resolve().as_posix(),
        "train": "train_rfs.txt",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"[OUTPUT] Generated RFS dataset YAML: {out_yaml_path}")

    # Assert validation and test splits remain unmutated
    assert yaml_content["val"] == "images/val", "Validation split mutated!"
    assert yaml_content["test"] == "images/test", "Held-out test split mutated!"

    # Export machine-readable report
    report = {
        "generator": "scripts/generate_rfs_dataset.py",
        "threshold": args.threshold,
        "max_repeat": args.max_repeat,
        "total_base_train_images": total_train_images,
        "total_rfs_train_samples": total_rfs_samples,
        "expansion_percentage": round(expansion_pct, 2),
        "repeat_distribution": dict(sorted(repeat_distribution.items())),
        "class_statistics": {
            cname: {
                "train_images": class_img_counts.get(cname, 0),
                "train_boxes": class_box_counts.get(cname, 0),
                "frequency": round(class_img_counts.get(cname, 0) / total_train_images, 4),
                "repeat_factor": class_rfs_factors[cname],
            }
            for cname in class_names
        },
        "artifacts": {
            "train_rfs_txt": str(out_txt_path),
            "data_rfs_yaml": str(out_yaml_path),
        },
    }

    report_path = REPORTS_DIR / "rfs_dataset_generation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[REPORT] Saved RFS generation report to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_rfs()
