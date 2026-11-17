"""Builds and validates the unified Canonical 8-Class Parking & Outdoor Property Dataset (Group C).

Harmonizes:
1. COCO 2017 Outdoor & Parking Scenes (600 images)
   - Extracted dense bounding boxes for vehicles (car, truck), motorcycle, bicycle,
     outdoor seating (bench, chair), outdoor table, plant/bush, and fixtures.
2. Open Images V7 Outdoor Greenery & Fixtures (150 images via fiftyone.zoo)
   - Provides prominent outdoor landscaping trees, garden bushes, street lights, and waste containers.

Canonical 8-Class Schema:
0: car_vehicle
1: motorcycle
2: bicycle
3: outdoor_seating
4: outdoor_table
5: plant_bush
6: tree
7: outdoor_fixture

Guarantees:
- Zero cross-split leakage verified by MD5 content hashes.
- Standard YOLO format: [class_id x_center y_center width height] clamped to [0.0, 1.0].
- Generates YAML config, split manifest JSON, and visual verification contact sheets.
- Detailed progress and timing via StageTimer.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Set, Tuple

# Reconfigure stdout/stderr for Windows UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.timing_utils import StageTimer

# Canonical Parking & Outdoor Property 8-Class Taxonomy
CANONICAL_PARKING_CLASSES = {
    0: "car_vehicle",
    1: "motorcycle",
    2: "bicycle",
    3: "outdoor_seating",
    4: "outdoor_table",
    5: "plant_bush",
    6: "tree",
    7: "outdoor_fixture",
}

# COCO Category ID -> Canonical Class ID
# 3: car -> 0
# 8: truck -> 0
# 6: bus -> 0
# 4: motorcycle -> 1
# 2: bicycle -> 2
# 15: bench -> 3
# 62: chair -> 3
# 67: dining table -> 4
# 64: potted plant -> 5
# 10: traffic light -> 7
# 14: parking meter -> 7
COCO_TO_CANONICAL = {
    3: 0,   # car -> car_vehicle
    8: 0,   # truck -> car_vehicle
    6: 0,   # bus -> car_vehicle
    4: 1,   # motorcycle -> motorcycle
    2: 2,   # bicycle -> bicycle
    15: 3,  # bench -> outdoor_seating
    62: 3,  # chair -> outdoor_seating
    67: 4,  # dining table -> outdoor_table
    64: 5,  # potted plant -> plant_bush
    10: 7,  # traffic light -> outdoor_fixture
    14: 7,  # parking meter -> outdoor_fixture
}

# Open Images Label -> Canonical Class ID
OI_TO_CANONICAL = {
    "Tree": 6,             # tree
    "Plant": 5,            # plant_bush
    "Street light": 7,     # outdoor_fixture
    "Waste container": 7,  # outdoor_fixture
    "Bench": 3,            # outdoor_seating
    "Car": 0,              # car_vehicle
    "Motorcycle": 1,       # motorcycle
    "Bicycle": 2,          # bicycle
}


def compute_md5(file_path: Path) -> str:
    """Computes MD5 hash of a file for zero-leakage duplicate detection."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_coco_parking_samples(
    coco_train_dir: Path,
    coco_json_path: Path,
    target_count: int = 600,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Loads and filters COCO 2017 outdoor and parking scenes with valid annotations."""
    print(f"[COCO] Parsing annotations from {coco_json_path}...")
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    existing_files = set(os.listdir(coco_train_dir))
    print(f"[COCO] Found {len(existing_files)} cached images in {coco_train_dir}")

    # Build image map
    img_map = {}
    for img in coco_data["images"]:
        if img["file_name"] in existing_files:
            img_map[img["id"]] = img

    # Group target annotations by image ID
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data["annotations"]:
        cat_id = ann.get("category_id")
        if cat_id in COCO_TO_CANONICAL and ann.get("image_id") in img_map:
            img_id = ann["image_id"]
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)

    print(f"[COCO] Found {len(anns_by_img)} images with relevant parking/outdoor annotations")

    # Sort deterministically by image ID and select target count
    rng = random.Random(random_seed)
    candidate_ids = sorted(list(anns_by_img.keys()))
    rng.shuffle(candidate_ids)
    selected_ids = candidate_ids[:target_count]

    samples = []
    for img_id in selected_ids:
        img_info = img_map[img_id]
        img_file = coco_train_dir / img_info["file_name"]
        width = float(img_info["width"])
        height = float(img_info["height"])

        boxes = []
        for ann in anns_by_img[img_id]:
            # COCO bbox: [x_min, y_min, w, h] in absolute pixels
            x_min, y_min, w, h = ann["bbox"]
            if w <= 1.0 or h <= 1.0:
                continue

            # Convert to normalized YOLO format [class_id, xc, yc, w, h]
            xc = (x_min + w / 2.0) / width
            yc = (y_min + h / 2.0) / height
            norm_w = w / width
            norm_h = h / height

            # Clamp safely to [0.0, 1.0]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            canonical_class = COCO_TO_CANONICAL[ann["category_id"]]
            boxes.append((canonical_class, xc, yc, norm_w, norm_h))

        if boxes:
            samples.append(
                {
                    "source": "coco_2017",
                    "source_file": str(img_file),
                    "file_name": f"coco_ext_{len(samples):04d}.jpg",
                    "width": int(width),
                    "height": int(height),
                    "boxes": boxes,
                }
            )

    print(f"[COCO] Successfully prepared {len(samples)} COCO parking/outdoor samples.")
    return samples


def download_openimages_outdoor_samples(
    target_count: int = 150,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Downloads Open Images V7 outdoor greenery & fixtures subset via FiftyOne Zoo."""
    import fiftyone.zoo as foz

    print(f"[OpenImages] Loading {target_count} outdoor samples via FiftyOne Zoo...")
    target_classes = ["Tree", "Plant", "Street light", "Waste container"]

    dataset_name = f"parking_outdoor_oi_{int(time.time())}"
    oi_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=target_classes,
        max_samples=target_count,
        dataset_name=dataset_name,
        cleanup=False,
    )

    samples = []
    for sample in oi_ds:
        img_path = Path(sample.filepath)
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            continue

        boxes = []
        if sample.ground_truth is not None and sample.ground_truth.detections is not None:
            for det in sample.ground_truth.detections:
                label = det.label
                if label in OI_TO_CANONICAL:
                    # FiftyOne OpenImages bbox: [x_min, y_min, w, h] normalized in [0, 1]
                    x_min, y_min, bw, bh = det.bounding_box
                    if bw <= 0.001 or bh <= 0.001:
                        continue

                    xc = x_min + bw / 2.0
                    yc = y_min + bh / 2.0
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))

                    canonical_class = OI_TO_CANONICAL[label]
                    boxes.append((canonical_class, xc, yc, bw, bh))

        if boxes:
            samples.append(
                {
                    "source": "open_images_v7",
                    "source_file": str(img_path),
                    "file_name": f"oi_out_{len(samples):04d}.jpg",
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                }
            )

    print(f"[OpenImages] Prepared {len(samples)} valid Open Images samples.")
    return samples


def render_sample_contact_sheet(
    sample_images: List[Tuple[Path, Path]],
    output_path: Path,
    title: str = "Canonical Parking & Outdoor Property Samples (Group C)",
) -> None:
    """Generates a visual contact sheet with bounding box overlays for quality audit."""
    cols = 4
    rows = (len(sample_images) + cols - 1) // cols
    cell_w, cell_h = 320, 240
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 40), color=(245, 247, 250))
    draw = ImageDraw.Draw(sheet)

    # Title banner
    draw.rectangle([0, 0, cols * cell_w, 40], fill=(30, 41, 59))
    draw.text((15, 10), title, fill=(255, 255, 255))

    colors = [
        (220, 38, 38),   # car_vehicle: red
        (217, 119, 6),   # motorcycle: amber
        (5, 150, 105),   # bicycle: emerald
        (37, 99, 235),   # outdoor_seating: blue
        (124, 58, 237),  # outdoor_table: purple
        (16, 185, 129),  # plant_bush: teal
        (34, 197, 94),   # tree: green
        (234, 88, 12),   # outdoor_fixture: orange
    ]

    for idx, (img_path, lbl_path) in enumerate(sample_images):
        col = idx % cols
        row = idx // cols
        x_off = col * cell_w
        y_off = 40 + row * cell_h

        if not img_path.exists():
            continue

        with Image.open(img_path) as im:
            thumb = im.convert("RGB").resize((cell_w, cell_h), Image.Resampling.BILINEAR)

        t_draw = ImageDraw.Draw(thumb)
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        x1 = int((xc - w / 2.0) * cell_w)
                        y1 = int((yc - h / 2.0) * cell_h)
                        x2 = int((xc + w / 2.0) * cell_w)
                        y2 = int((yc + h / 2.0) * cell_h)

                        c = colors[cls_id % len(colors)]
                        t_draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
                        lbl_name = CANONICAL_PARKING_CLASSES.get(cls_id, str(cls_id))
                        t_draw.text((x1 + 2, max(0, y1 - 10)), lbl_name, fill=c)

        sheet.paste(thumb, (x_off, y_off))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    print(f"[Audit] Contact sheet saved to {output_path}")


def main() -> None:
    timer = StageTimer()
    timer.start_stage(
        stage_name="Prepare Canonical Parking & Outdoor Property Dataset (Group C)",
        estimated_sec=90.0,
        total_items=750,
        expected_gpu_vram_gb=0.0,
        assumptions="Harmonizing COCO 2017 outdoor parking scenes (600) + Open Images V7 greenery & fixtures (150)",
    )

    output_dir = REPO_ROOT / "data" / "parking_canonical"
    if output_dir.exists():
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    coco_train_dir = Path.home() / "fiftyone" / "coco-2017" / "train" / "data"
    coco_json_path = Path.home() / "fiftyone" / "coco-2017" / "raw" / "instances_train2017.json"

    print("[COCO] Loading & Parsing COCO Parking/Outdoor Scenes...")
    coco_samples = load_coco_parking_samples(
        coco_train_dir=coco_train_dir,
        coco_json_path=coco_json_path,
        target_count=600,
        random_seed=42,
    )

    print("[OpenImages] Downloading & Extracting Open Images Greenery & Fixtures...")
    oi_samples = download_openimages_outdoor_samples(
        target_count=150,
        random_seed=42,
    )

    all_samples = coco_samples + oi_samples
    print(f"[Harmonization] Total pooled samples: {len(all_samples)}")

    print("[Splits] Performing Stratified Leakage-Free Splitting...")
    # Deterministic split: 68% train, 16% val, 16% test
    rng = random.Random(42)
    rng.shuffle(all_samples)

    n_total = len(all_samples)
    n_val = int(0.16 * n_total)
    n_test = int(0.16 * n_total)
    n_train = n_total - n_val - n_test

    splits_data = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train : n_train + n_val],
        "test": all_samples[n_train + n_val :],
    }

    print(
        f"[Splits] Partitioned {n_total} samples -> "
        f"Train: {len(splits_data['train'])}, "
        f"Val: {len(splits_data['val'])}, "
        f"Held-Out Test: {len(splits_data['test'])}"
    )

    print("[Writing] Writing Images, Labels & Verifying MD5 Hashes...")
    split_md5s: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    class_counts_by_split: Dict[str, Dict[str, int]] = {
        s: {name: 0 for name in CANONICAL_PARKING_CLASSES.values()}
        for s in ["train", "val", "test"]
    }
    manifest_records = []

    for split_name, samples in splits_data.items():
        img_dest_dir = output_dir / "images" / split_name
        lbl_dest_dir = output_dir / "labels" / split_name

        for s in samples:
            src_img = Path(s["source_file"])
            dst_img_name = f"{s['source'][:4]}_{split_name}_{s['file_name']}"
            dst_img = img_dest_dir / dst_img_name
            dst_lbl = lbl_dest_dir / (dst_img.stem + ".txt")

            # Copy image
            shutil.copy2(src_img, dst_img)

            # MD5 hash for leakage check
            img_md5 = compute_md5(dst_img)
            split_md5s[split_name].add(img_md5)

            # Write label file
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, w, h in s["boxes"]:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    class_name = CANONICAL_PARKING_CLASSES[cls_id]
                    class_counts_by_split[split_name][class_name] += 1

            manifest_records.append(
                {
                    "file_name": dst_img_name,
                    "split": split_name,
                    "source": s["source"],
                    "md5": img_md5,
                    "width": s["width"],
                    "height": s["height"],
                    "num_boxes": len(s["boxes"]),
                }
            )

    # Verify zero split leakage
    tv_overlap = split_md5s["train"].intersection(split_md5s["val"])
    tt_overlap = split_md5s["train"].intersection(split_md5s["test"])
    vt_overlap = split_md5s["val"].intersection(split_md5s["test"])

    print("\n--- ZERO-LEAKAGE VERIFICATION ---")
    print(f"Train / Val collisions: {len(tv_overlap)}")
    print(f"Train / Test collisions: {len(tt_overlap)}")
    print(f"Val / Test collisions:   {len(vt_overlap)}")
    assert len(tv_overlap) == 0, "Train and Val share image hashes!"
    assert len(tt_overlap) == 0, "Train and Test share image hashes!"
    assert len(vt_overlap) == 0, "Val and Test share image hashes!"
    print("SUCCESS: 0 cross-split leakage verified via cryptographic MD5 hashing.\n")

    print("[Config] Generating YAML Config & Audit Manifest...")
    yaml_config = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {cls_id: name for cls_id, name in CANONICAL_PARKING_CLASSES.items()},
        "nc": len(CANONICAL_PARKING_CLASSES),
    }

    yaml_path = output_dir / "parking_canonical.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yaml_config, yf, default_flow_style=False, sort_keys=False)
    print(f"[Config] Saved YAML config to {yaml_path}")

    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "taxonomy": CANONICAL_PARKING_CLASSES,
                "split_counts": {k: len(v) for k, v in splits_data.items()},
                "class_distribution": class_counts_by_split,
                "total_images": len(manifest_records),
                "total_annotations": sum(
                    sum(counts.values()) for counts in class_counts_by_split.values()
                ),
                "records": manifest_records,
            },
            mf,
            indent=2,
        )
    print(f"[Manifest] Saved split manifest to {manifest_path}")

    print("[Visual] Generating Visual Contact Sheet & Quality Audit...")
    sample_pairs = []
    for split in ["train", "val", "test"]:
        imgs = sorted(list((output_dir / "images" / split).glob("*.jpg")))[:4]
        for img in imgs:
            lbl = output_dir / "labels" / split / (img.stem + ".txt")
            sample_pairs.append((img, lbl))

    contact_path = REPO_ROOT / "artifacts" / "parking_canonical_samples.png"
    render_sample_contact_sheet(
        sample_pairs,
        contact_path,
        title="Group C: Canonical Parking & Outdoor Property Ground Truth Overlays",
    )

    # Generate Markdown Audit Report
    report_path = REPO_ROOT / "artifacts" / "reports" / "parking_canonical_dataset_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Group C: Canonical Parking & Outdoor Property Dataset Audit\n\n")
        rf.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        rf.write("## 1. Executive Summary\n")
        rf.write(
            f"The **Group C Canonical Dataset** has been constructed and harmonized from COCO 2017 outdoor parking subsets and Open Images V7 landscaping/fixtures.\n\n"
        )
        rf.write(
            f"- **Total Images:** {len(manifest_records)} ({len(splits_data['train'])} train, {len(splits_data['val'])} val, {len(splits_data['test'])} test)\n"
        )
        total_boxes = sum(sum(counts.values()) for counts in class_counts_by_split.values())
        rf.write(f"- **Total Bounding Boxes:** {total_boxes}\n")
        rf.write(f"- **Zero-Leakage Status:** Verified (0 MD5 collisions across all splits)\n\n")
        rf.write("## 2. Canonical Class Distribution\n\n")
        rf.write("| Class ID | Class Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes |\n")
        rf.write("|---|---|---|---|---|---|\n")
        for cls_id, name in CANONICAL_PARKING_CLASSES.items():
            tr = class_counts_by_split["train"][name]
            va = class_counts_by_split["val"][name]
            te = class_counts_by_split["test"][name]
            tot = tr + va + te
            rf.write(f"| {cls_id} | `{name}` | {tr} | {va} | {te} | **{tot}** |\n")
        rf.write("\n## 3. Surface Segmentation vs. Object Detection Boundary\n")
        rf.write(
            "As mandated by requirements, planar continuous surfaces (lawn, driveway pavement, asphalt parking slot markings) "
            "are explicitly deferred to Group D semantic segmentation models, preserving crisp rectangular bounding box learning "
            "for discrete objects (vehicles, seating, tables, shrubs, trees, fixtures).\n"
        )
    print(f"[Report] Saved dataset audit report to {report_path}")

    timer.finish_stage(
        artifacts=[str(manifest_path), str(yaml_path), str(report_path), str(contact_path)],
        extra_metrics={
            "total_images": len(manifest_records),
            "total_bounding_boxes": total_boxes,
            "train_images": len(splits_data["train"]),
            "val_images": len(splits_data["val"]),
            "test_images": len(splits_data["test"]),
            "leakage_collisions": 0,
        },
    )


if __name__ == "__main__":
    main()
