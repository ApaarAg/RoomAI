"""Converts COCO train and val annotations into a YOLO furniture dataset."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

TARGET_CLASSES = [
    "chair",
    "couch",
    "bed",
    "dining table",
    "potted plant",
    "tv",
    "toilet",
    "refrigerator",
    "oven",
    "sink",
]

COCO_NAME_TO_YOLO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

TRAIN_IMG_DIR = Path(r"c:\Users\apaar\fiftyone\coco-2017\train\data")
TRAIN_ANN_FILE = Path(r"c:\Users\apaar\fiftyone\coco-2017\raw\instances_train2017.json")

VAL_IMG_DIR = Path(r"c:\Users\apaar\Desktop\RoomAI\val2017")
VAL_ANN_FILE = Path(r"c:\Users\apaar\fiftyone\coco-2017\raw\instances_val2017.json")

OUTPUT_DATA_DIR = Path(r"c:\Users\apaar\Desktop\RoomAI\data\furniture")


def build_split(
    ann_file: Path,
    src_img_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    max_images: int = 400,
    allowed_filenames: Set[str] = None,
) -> int:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ann_file}...")
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    target_cat_ids = {c["id"]: c["name"] for c in coco["categories"] if c["name"] in COCO_NAME_TO_YOLO_ID}

    img_id_to_info = {img["id"]: img for img in coco["images"]}
    img_id_to_anns = {}

    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid in target_cat_ids and ann.get("iscrowd", 0) == 0:
            iid = ann["image_id"]
            if iid not in img_id_to_anns:
                img_id_to_anns[iid] = []
            img_id_to_anns[iid].append(ann)

    # Sort images by number of furniture annotations (densest rooms first)
    sorted_iids = sorted(img_id_to_anns.keys(), key=lambda iid: len(img_id_to_anns[iid]), reverse=True)

    count = 0
    for iid in sorted_iids:
        img_info = img_id_to_info[iid]
        fname = img_info["file_name"]

        if allowed_filenames is not None and fname not in allowed_filenames:
            continue

        src_img = src_img_dir / fname
        if not src_img.exists():
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        # Generate YOLO label lines: <cls_id> <x_c> <y_c> <w> <h>
        label_lines = []
        for ann in img_id_to_anns[iid]:
            cname = target_cat_ids[ann["category_id"]]
            yolo_cls = COCO_NAME_TO_YOLO_ID[cname]

            # bbox in COCO: [x, y, w, h]
            bx, by, bw, bh = ann["bbox"]
            xc = (bx + bw / 2.0) / img_w
            yc = (by + bh / 2.0) / img_h
            w_norm = bw / img_w
            h_norm = bh / img_h

            # Clip
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            label_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")

        if not label_lines:
            continue

        # Copy image or create hardlink
        dst_img = out_img_dir / fname
        dst_lbl = out_lbl_dir / (Path(fname).stem + ".txt")

        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            lf.write("\n".join(label_lines) + "\n")

        count += 1
        if count >= max_images:
            break

    return count


def main():
    print("Preparing dedicated Room Furniture dataset for fine-tuning experiment...")
    train_count = build_split(
        TRAIN_ANN_FILE,
        TRAIN_IMG_DIR,
        OUTPUT_DATA_DIR / "images" / "train",
        OUTPUT_DATA_DIR / "labels" / "train",
        max_images=300,  # High-quality pilot and fine-tuning size
    )
    print(f"Created {train_count} training samples from train2017.")

    # Now split val2017 into 100 validation and 100 separate untouched test samples
    with open(VAL_ANN_FILE, "r", encoding="utf-8") as f:
        coco_val = json.load(f)
    val_files = [img["file_name"] for img in coco_val["images"]]
    val_split = set(val_files[:200])
    test_split = set(val_files[200:400])

    val_count = build_split(
        VAL_ANN_FILE,
        VAL_IMG_DIR,
        OUTPUT_DATA_DIR / "images" / "val",
        OUTPUT_DATA_DIR / "labels" / "val",
        max_images=100,
        allowed_filenames=val_split,
    )
    print(f"Created {val_count} validation samples from val2017.")

    test_count = build_split(
        VAL_ANN_FILE,
        VAL_IMG_DIR,
        OUTPUT_DATA_DIR / "images" / "test",
        OUTPUT_DATA_DIR / "labels" / "test",
        max_images=100,
        allowed_filenames=test_split,
    )
    print(f"Created {test_count} untouched test samples from val2017.")

    # Create dataset.yaml
    yaml_content = f"""path: {OUTPUT_DATA_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, cname in enumerate(TARGET_CLASSES):
        yaml_content += f"  {idx}: {cname}\n"

    yaml_path = OUTPUT_DATA_DIR / "furniture.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nCreated dataset YAML configuration at: {yaml_path}")


if __name__ == "__main__":
    main()
