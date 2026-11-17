"""Builds and validates the unified Canonical 10-Class Exterior Dataset (Group B).

Harmonizes:
1. CMP Facade Database (606 images: 378 train, 114 val, 114 test)
   - Extracted dense bounding boxes for facade, window, door, roof/cornice, balcony,
     pillar, shutter/blind, and molding/deco via connected component analysis.
2. Open Images V7 Exterior Subset (120 natural photography images: 80 train, 20 val, 20 test)
   - Provides building/house envelopes and natural porch structures.

Canonical 10-Class Schema:
0: building
1: facade
2: window
3: door
4: roof_cornice
5: balcony
6: porch
7: pillar_column
8: shutter_blind
9: molding_deco

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

# Canonical Exterior 10-Class Taxonomy
CANONICAL_EXTERIOR_CLASSES = {
    0: "building",
    1: "facade",
    2: "window",
    3: "door",
    4: "roof_cornice",
    5: "balcony",
    6: "porch",
    7: "pillar_column",
    8: "shutter_blind",
    9: "molding_deco",
}

# CMP Facade raw mask index -> Canonical class ID
# 1: background (ignore)
# 2: facade -> 1 (facade)
# 3: window -> 2 (window)
# 4: door -> 3 (door)
# 5: cornice -> 4 (roof_cornice)
# 6: sill -> 9 (molding_deco)
# 7: balcony -> 5 (balcony)
# 8: blind -> 8 (shutter_blind)
# 9: pillar -> 7 (pillar_column)
# 10: deco -> 9 (molding_deco)
# 11: molding -> 9 (molding_deco)
# 12: shop -> 3 (door/entrance)
CMP_TO_CANONICAL = {
    2: 1,   # facade
    3: 2,   # window
    4: 3,   # door
    5: 4,   # cornice -> roof_cornice
    6: 9,   # sill -> molding_deco
    7: 5,   # balcony
    8: 8,   # blind -> shutter_blind
    9: 7,   # pillar -> pillar_column
    10: 9,  # deco -> molding_deco
    11: 9,  # molding -> molding_deco
    12: 3,  # shop -> door
}

# Open Images class name -> Canonical class ID
OPEN_IMAGES_TO_CANONICAL = {
    "Building": 0,
    "House": 0,
    "Window": 2,
    "Door": 3,
    "Porch": 6,
}

OUTPUT_DIR = REPO_ROOT / "data" / "exterior_canonical"


def compute_md5(image_bytes: bytes) -> str:
    """Computes MD5 hash of raw image bytes."""
    return hashlib.md5(image_bytes).hexdigest()


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, val))


def extract_cmp_boxes(mask: np.ndarray) -> List[Tuple[int, float, float, float, float]]:
    """Extracts normalized YOLO bounding boxes from CMP segmentation mask.
    
    Returns:
        List of (class_id, x_center, y_center, width, height)
    """
    h, w = mask.shape
    boxes = []
    
    for raw_id, canonical_id in CMP_TO_CANONICAL.items():
        bin_mask = (mask == raw_id).astype(np.uint8)
        if not np.any(bin_mask):
            continue
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            # Minimum dimension and area filters to eliminate single-pixel noise
            if area < 36 or bw < 5 or bh < 5:
                continue
            
            # Normalize to YOLO format
            xc = clamp((bx + bw / 2.0) / w)
            yc = clamp((by + bh / 2.0) / h)
            nw = clamp(bw / float(w))
            nh = clamp(bh / float(h))
            
            if nw > 0.005 and nh > 0.005:
                boxes.append((canonical_id, xc, yc, nw, nh))
                
    return boxes


def main():
    timer = StageTimer()
    timer.start_stage(
        stage_name="Prepare Exterior Canonical Dataset (Group B)",
        estimated_sec=120.0,
        total_items=726,
        expected_gpu_vram_gb=0.0,
        assumptions="Harmonizing CMP Facade (606 images) + Open Images V7 (120 images) into 10 canonical classes",
    )
    
    # 1. Setup Directories
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    split_hashes: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    class_counts: Dict[str, Dict[int, int]] = {
        "train": {i: 0 for i in CANONICAL_EXTERIOR_CLASSES},
        "val": {i: 0 for i in CANONICAL_EXTERIOR_CLASSES},
        "test": {i: 0 for i in CANONICAL_EXTERIOR_CLASSES},
    }
    manifest_records: List[Dict[str, Any]] = []

    # 2. Process CMP Facade Dataset
    print("\n[STAGE 1/3] Loading CMP Facade dataset from Hugging Face cache...")
    from datasets import load_dataset
    
    cmp_splits = {
        "train": "train",
        "val": "eval",
        "test": "test"
    }
    
    cmp_item_idx = 0
    for yolo_split, hf_split in cmp_splits.items():
        print(f" -> Processing CMP Facade '{hf_split}' -> '{yolo_split}' ...")
        ds = load_dataset("Xpitfire/cmp_facade", split=hf_split)
        
        for idx, sample in enumerate(ds):
            img_bytes = sample["pixel_values"]["bytes"]
            lbl_bytes = sample["label"]["bytes"]
            
            md5_hash = compute_md5(img_bytes)
            split_hashes[yolo_split].add(md5_hash)
            
            # Load images
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            mask = np.array(Image.open(io.BytesIO(lbl_bytes)))
            
            # Extract boxes
            boxes = extract_cmp_boxes(mask)
            
            # Save image and label
            filename = f"cmp_{yolo_split}_{idx:04d}"
            img_out_path = OUTPUT_DIR / "images" / yolo_split / f"{filename}.jpg"
            lbl_out_path = OUTPUT_DIR / "labels" / yolo_split / f"{filename}.txt"
            
            img.save(img_out_path, format="JPEG", quality=95)
            
            with open(lbl_out_path, "w", encoding="utf-8") as f:
                for cid, xc, yc, w, h in boxes:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    class_counts[yolo_split][cid] += 1
                    
            manifest_records.append({
                "source": "CMP_Facade",
                "original_split": hf_split,
                "target_split": yolo_split,
                "filename": f"{filename}.jpg",
                "width": img.width,
                "height": img.height,
                "md5": md5_hash,
                "num_boxes": len(boxes),
            })
            
            cmp_item_idx += 1
            if cmp_item_idx % 50 == 0:
                timer.update(
                    completed_items=cmp_item_idx,
                    step_name=f"CMP {cmp_item_idx}/606 ({yolo_split})"
                )
                
    print(f"CMP Facade complete: 606 images converted.")

    # 3. Process Open Images V7 Exterior Subset
    print("\n[STAGE 2/3] Loading Open Images V7 exterior subset via FiftyOne...")
    import fiftyone.zoo as foz
    
    # Load 120 samples from validation split
    oi_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Building", "House", "Porch", "Window", "Door"],
        max_samples=120,
        seed=1337,
        shuffle=True,
    )
    
    # Partition Open Images deterministic split: 80 train, 20 val, 20 test
    oi_samples = list(oi_ds)
    random.seed(42)
    random.shuffle(oi_samples)
    
    oi_split_map = {}
    for i, s in enumerate(oi_samples):
        if i < 80:
            oi_split_map[s.id] = "train"
        elif i < 100:
            oi_split_map[s.id] = "val"
        else:
            oi_split_map[s.id] = "test"
            
    oi_idx = 0
    for sample in oi_samples:
        yolo_split = oi_split_map[sample.id]
        img_path = Path(sample.filepath)
        
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        md5_hash = compute_md5(img_bytes)
        split_hashes[yolo_split].add(md5_hash)
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        im_w, im_h = img.size
        
        # Parse detections
        boxes = []
        if sample.ground_truth is not None and sample.ground_truth.detections is not None:
            for det in sample.ground_truth.detections:
                label_name = det.label
                if label_name in OPEN_IMAGES_TO_CANONICAL:
                    cid = OPEN_IMAGES_TO_CANONICAL[label_name]
                    # FiftyOne bbox is [x_top_left, y_top_left, width, height]
                    bx, by, bw, bh = det.bounding_box
                    xc = clamp(bx + bw / 2.0)
                    yc = clamp(by + bh / 2.0)
                    nw = clamp(bw)
                    nh = clamp(bh)
                    if nw > 0.005 and nh > 0.005:
                        boxes.append((cid, xc, yc, nw, nh))
                        
        filename = f"oi_ext_{yolo_split}_{oi_idx:04d}"
        img_out_path = OUTPUT_DIR / "images" / yolo_split / f"{filename}.jpg"
        lbl_out_path = OUTPUT_DIR / "labels" / yolo_split / f"{filename}.txt"
        
        img.save(img_out_path, format="JPEG", quality=95)
        
        with open(lbl_out_path, "w", encoding="utf-8") as f:
            for cid, xc, yc, w, h in boxes:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                class_counts[yolo_split][cid] += 1
                
        manifest_records.append({
            "source": "OpenImagesV7",
            "original_id": sample.id,
            "target_split": yolo_split,
            "filename": f"{filename}.jpg",
            "width": im_w,
            "height": im_h,
            "md5": md5_hash,
            "num_boxes": len(boxes),
        })
        
        oi_idx += 1
        timer.update(
            completed_items=606 + oi_idx,
            step_name=f"OpenImages {oi_idx}/120 ({yolo_split})"
        )

    print(f"Open Images subset complete: 120 images converted.")

    # 4. Leakage Verification
    print("\n[STAGE 3/3] Auditing split leakage and bounding box integrity...")
    train_val_leak = split_hashes["train"].intersection(split_hashes["val"])
    train_test_leak = split_hashes["train"].intersection(split_hashes["test"])
    val_test_leak = split_hashes["val"].intersection(split_hashes["test"])
    
    total_leakage = len(train_val_leak) + len(train_test_leak) + len(val_test_leak)
    if total_leakage > 0:
        raise ValueError(f"FATAL: Cross-split leakage detected! Collisions: Train-Val={len(train_val_leak)}, Train-Test={len(train_test_leak)}, Val-Test={len(val_test_leak)}")
    print("[LEAKAGE CHECK] PASSED: Exactly 0 image hash collisions across Train, Val, and Test splits.")

    # 5. Write Manifest and YAML
    manifest_path = OUTPUT_DIR / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "RoomAI Exterior Canonical (Group B)",
            "classes": CANONICAL_EXTERIOR_CLASSES,
            "total_images": len(manifest_records),
            "splits": {
                "train": len([r for r in manifest_records if r["target_split"] == "train"]),
                "val": len([r for r in manifest_records if r["target_split"] == "val"]),
                "test": len([r for r in manifest_records if r["target_split"] == "test"]),
            },
            "class_distribution": class_counts,
            "records": manifest_records,
        }, f, indent=2)
    print(f"Wrote split manifest to {manifest_path}")

    # Write YOLO dataset configuration YAML
    yaml_config = {
        "path": str(OUTPUT_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in CANONICAL_EXTERIOR_CLASSES.items()},
    }
    yaml_path = OUTPUT_DIR / "exterior_canonical.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, sort_keys=False)
    print(f"Wrote YOLO configuration to {yaml_path}")

    # 6. Generate Visual Contact Sheet
    contact_sheet_path = REPO_ROOT / "artifacts" / "exterior_canonical_samples.png"
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pick 4 diverse sample images (2 from CMP, 2 from OpenImages)
    sample_records = [
        r for r in manifest_records if r["source"] == "CMP_Facade" and r["num_boxes"] > 5
    ][:2] + [
        r for r in manifest_records if r["source"] == "OpenImagesV7" and r["num_boxes"] > 0
    ][:2]
    
    sample_images = []
    class_palette = {
        0: (255, 99, 71),    # building (Tomato)
        1: (100, 149, 237),  # facade (Cornflower)
        2: (50, 205, 50),    # window (LimeGreen)
        3: (255, 165, 0),    # door (Orange)
        4: (147, 112, 219),  # roof_cornice (Purple)
        5: (255, 215, 0),    # balcony (Gold)
        6: (0, 206, 209),    # porch (DarkTurquoise)
        7: (255, 20, 147),   # pillar_column (DeepPink)
        8: (218, 112, 214),  # shutter_blind (Orchid)
        9: (128, 128, 128),  # molding_deco (Gray)
    }
    
    for r in sample_records:
        split = r["target_split"]
        fn = r["filename"]
        img_p = OUTPUT_DIR / "images" / split / fn
        lbl_p = OUTPUT_DIR / "labels" / split / fn.replace(".jpg", ".txt")
        
        im = Image.open(img_p).convert("RGB")
        draw = ImageDraw.Draw(im)
        
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = (xc - w / 2.0) * im.width
                        y1 = (yc - h / 2.0) * im.height
                        x2 = (xc + w / 2.0) * im.width
                        y2 = (yc + h / 2.0) * im.height
                        color = class_palette.get(cid, (255, 255, 255))
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        cname = CANONICAL_EXTERIOR_CLASSES.get(cid, str(cid))
                        draw.text((x1 + 4, y1 + 4), cname, fill=color)
                        
        im_resized = im.resize((640, 480))
        sample_images.append(im_resized)
        
    if len(sample_images) == 4:
        grid = Image.new("RGB", (1280, 960), color=(20, 20, 20))
        grid.paste(sample_images[0], (0, 0))
        grid.paste(sample_images[1], (640, 0))
        grid.paste(sample_images[2], (0, 480))
        grid.paste(sample_images[3], (640, 480))
        grid.save(contact_sheet_path)
        print(f"Saved visual verification contact sheet to {contact_sheet_path}")

    # 7. Generate Markdown Report
    report_path = REPO_ROOT / "artifacts" / "reports" / "exterior_canonical_dataset_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_imgs = len(manifest_records)
    total_boxes = sum(sum(class_counts[s].values()) for s in ["train", "val", "test"])
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RoomAI Group B (Exterior & Building Components) Dataset Audit\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- **Total Images:** {total_imgs}\n")
        f.write(f"- **Total Annotated Bounding Boxes:** {total_boxes:,}\n")
        f.write(f"- **Train Split:** {len([r for r in manifest_records if r['target_split'] == 'train'])} images\n")
        f.write(f"- **Validation Split:** {len([r for r in manifest_records if r['target_split'] == 'val'])} images\n")
        f.write(f"- **Test Split:** {len([r for r in manifest_records if r['target_split'] == 'test'])} images\n")
        f.write("- **Cross-Split Hash Collisions:** 0 (Zero leakage)\n\n")
        f.write("## 2. Canonical Class Distribution\n\n")
        f.write("| Class ID | Class Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cid in sorted(CANONICAL_EXTERIOR_CLASSES.keys()):
            cname = CANONICAL_EXTERIOR_CLASSES[cid]
            tr = class_counts["train"][cid]
            va = class_counts["val"][cid]
            te = class_counts["test"][cid]
            tot = tr + va + te
            f.write(f"| {cid} | `{cname}` | {tr:,} | {va:,} | {te:,} | {tot:,} |\n")
        f.write("\n## 3. Data Sources\n\n")
        f.write("1. **CMP Facade Database:** 606 rectified architectural facade images with dense component segmentation converted to bounding boxes via 8-way connected component analysis.\n")
        f.write("2. **Open Images V7 Exterior Subset:** 120 natural photography images providing overall building/house frames and outdoor porches.\n")
        f.write(f"\nConfiguration file: `data/exterior_canonical/exterior_canonical.yaml`\n")
    print(f"Wrote dataset audit report to {report_path}")

    timer.finish_stage(
        artifacts=[str(manifest_path), str(yaml_path), str(report_path), str(contact_sheet_path)],
        extra_metrics={
            "total_images": total_imgs,
            "total_bounding_boxes": total_boxes,
            "train_images": len([r for r in manifest_records if r["target_split"] == "train"]),
            "val_images": len([r for r in manifest_records if r["target_split"] == "val"]),
            "test_images": len([r for r in manifest_records if r["target_split"] == "test"]),
            "leakage_collisions": total_leakage,
        }
    )


if __name__ == "__main__":
    main()
