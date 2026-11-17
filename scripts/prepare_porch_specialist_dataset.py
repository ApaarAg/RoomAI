"""Phase 15 Stage B: Prepare Dedicated Porch Specialist Dataset.

Builds data/exterior_specialist_porch/ with:
- Positive images (all 361 porch images from training split, 509 porch boxes converted to class 0)
- Hard negative images (360 negative images containing ground-floor facades, doors, columns, balconies)
- Validation split (all 154 validation images, with 24 porch boxes mapped to class 0; 132 negative images)
- dataset.yaml (nc: 1, names: ['porch'])
- manifest.json recording image hashes, box dimensions, and hard-negative selections.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
SRC_DATA_DIR = REPO_ROOT / "data" / "exterior_b6_expanded"
DST_DATA_DIR = REPO_ROOT / "data" / "exterior_specialist_porch"

PORCH_SRC_CID = 6  # Canonical ID for porch in 10-class dataset


def create_porch_dataset():
    print("=== Phase 15 Stage B: Preparing Porch Specialist Dataset ===")

    train_img_dst = DST_DATA_DIR / "images" / "train"
    train_lbl_dst = DST_DATA_DIR / "labels" / "train"
    val_img_dst = DST_DATA_DIR / "images" / "val"
    val_lbl_dst = DST_DATA_DIR / "labels" / "val"

    for d in [train_img_dst, train_lbl_dst, val_img_dst, val_lbl_dst]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Process Training Split
    src_train_imgs = list((SRC_DATA_DIR / "images" / "train").glob("*.jpg"))
    pos_train_records = []
    neg_train_candidates = []

    for img_p in src_train_imgs:
        lbl_p = SRC_DATA_DIR / "labels" / "train" / f"{img_p.stem}.txt"
        porch_boxes = []
        other_cids = set()
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid == PORCH_SRC_CID:
                        w, h = float(parts[3]), float(parts[4])
                        # Filter corrupt / full-frame annotation artifacts
                        if w < 0.01 or h < 0.01 or (w > 0.98 and h > 0.98):
                            continue
                        porch_boxes.append((float(parts[1]), float(parts[2]), w, h))
                    else:
                        other_cids.add(cid)

        if porch_boxes:
            pos_train_records.append((img_p, porch_boxes))
        else:
            # Score negative relevance: prioritize images with doors (3), columns (7), facades (1), balconies (5)
            relevance_score = 0
            if 3 in other_cids: relevance_score += 3  # doors
            if 7 in other_cids: relevance_score += 2  # columns
            if 5 in other_cids: relevance_score += 2  # balconies
            if 1 in other_cids: relevance_score += 1  # facades
            neg_train_candidates.append((relevance_score, img_p))

    # Sort negatives by relevance score descending
    neg_train_candidates.sort(key=lambda x: x[0], reverse=True)
    selected_negatives = [x[1] for x in neg_train_candidates[:360]]

    print(f"Positive training images with verified porch: {len(pos_train_records)}")
    print(f"Selected hard-negative training images: {len(selected_negatives)}")

    manifest = {"train": [], "val": []}
    total_pos_train_boxes = 0

    # Write positive train
    for img_p, boxes in pos_train_records:
        dst_img = train_img_dst / img_p.name
        dst_lbl = train_lbl_dst / f"{img_p.stem}.txt"
        if not dst_img.exists():
            shutil.copy2(img_p, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                total_pos_train_boxes += 1
        manifest["train"].append({
            "image": img_p.name,
            "type": "positive",
            "box_count": len(boxes),
        })

    # Write negative train (empty label files)
    for img_p in selected_negatives:
        dst_img = train_img_dst / img_p.name
        dst_lbl = train_lbl_dst / f"{img_p.stem}.txt"
        if not dst_img.exists():
            shutil.copy2(img_p, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            pass  # empty file signifies negative background
        manifest["train"].append({
            "image": img_p.name,
            "type": "hard_negative",
            "box_count": 0,
        })

    # 2. Process Validation Split (All 154 validation images)
    src_val_imgs = list((SRC_DATA_DIR / "images" / "val").glob("*.jpg"))
    total_pos_val_boxes = 0

    for img_p in src_val_imgs:
        lbl_p = SRC_DATA_DIR / "labels" / "val" / f"{img_p.stem}.txt"
        porch_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid == PORCH_SRC_CID:
                        porch_boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

        dst_img = val_img_dst / img_p.name
        dst_lbl = val_lbl_dst / f"{img_p.stem}.txt"
        if not dst_img.exists():
            shutil.copy2(img_p, dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            for b in porch_boxes:
                f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                total_pos_val_boxes += 1

        manifest["val"].append({
            "image": img_p.name,
            "type": "positive" if porch_boxes else "negative",
            "box_count": len(porch_boxes),
        })

    print(f"Total training images in specialist: {len(manifest['train'])} (Boxes: {total_pos_train_boxes})")
    print(f"Total validation images in specialist: {len(manifest['val'])} (Boxes: {total_pos_val_boxes})")

    # 3. Create dataset.yaml
    dataset_yaml_content = f"""path: {DST_DATA_DIR}
train: images/train
val: images/val
nc: 1
names:
  0: porch
"""
    yaml_path = DST_DATA_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(dataset_yaml_content)

    # 4. Save manifest
    manifest_path = DST_DATA_DIR / "porch_specialist_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SAVED] Dataset config saved to {yaml_path}")
    print(f"[SAVED] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    create_porch_dataset()
