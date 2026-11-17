"""
RoomAI — FastSAM Mask Refinement Strategy Comparison
Compares:
1. Strategy 1: Original Raw FastSAM mask
2. Strategy 2: Bounding-Box Clipped mask (Hard Intersection)
3. Strategy 3: Morphological Connected-Component Soft Refinement
Across representative indoor objects detected by Model A3.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Suppress TensorFlow probing
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.modules["tensorflow"] = None

import torch
from ultralytics import YOLO, FastSAM

REPO_ROOT = Path(__file__).resolve().parent.parent
ADE_ROOT = REPO_ROOT / "archive (2)" / "ADEChallengeData2016"


def soft_refine_mask(raw_mask_bool, box_xyxy, kernel_size=5):
    """
    Morphological refinement:
    1. Keeps only the largest connected component that overlaps with the prompt box center.
    2. Removes detached floating island pixels outside the box.
    3. Smooths boundary jaggedness using morphological opening/closing.
    """
    mask_u8 = (raw_mask_bool * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Morphological close to bridge holes
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    if num_labels <= 1:
        return raw_mask_bool

    # Find component with maximum overlap with the bounding box
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    box_mask = np.zeros_like(closed, dtype=bool)
    box_mask[y1:y2, x1:x2] = True

    best_label = 1
    best_overlap = -1
    for lbl in range(1, num_labels):
        comp = (labels == lbl)
        overlap = np.logical_and(comp, box_mask).sum()
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = lbl

    best_comp = (labels == best_label)
    return best_comp


def main():
    print("===============================================================")
    print("RoomAI FastSAM Mask Refinement Strategy Evaluation")
    print("===============================================================")

    a3 = YOLO(str(REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt"))
    fastsam = FastSAM("FastSAM-s.pt")

    # Select representative validation scenes with diverse objects
    target_samples = [
        {"cat": "bedroom", "img_id": "ADE_val_00000077", "focus": "bed"},
        {"cat": "living_room", "img_id": "ADE_val_00000100", "focus": "sofa"},
        {"cat": "dining_room", "img_id": "ADE_val_00000511", "focus": "chair#stool"},
        {"cat": "kitchen", "img_id": "ADE_val_00000201", "focus": "cabinet#shelf"},
        {"cat": "bathroom", "img_id": "ADE_val_00000301", "focus": "door"},
    ]

    results = []

    for s in target_samples:
        img_path = ADE_ROOT / "images" / "validation" / f"{s['img_id']}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_np = np.array(img)

        det = a3(img_np, imgsz=800, conf=0.25, device="cuda", verbose=False)[0]
        boxes = det.boxes
        if len(boxes) == 0:
            continue

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        classes = [a3.names[int(c)] for c in boxes.cls.cpu().numpy()]

        sam_res = fastsam(img_np, bboxes=boxes_xyxy.tolist(), device="cuda", verbose=False)[0]
        if sam_res.masks is None:
            continue

        masks_data = sam_res.masks.data.cpu().numpy()

        for i, (box, cls_name) in enumerate(zip(boxes_xyxy, classes)):
            if i >= len(masks_data):
                break
            # Upsample mask to orig image size
            mask_raw_resized = cv2.resize(masks_data[i], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0.5

            # Box ROI mask
            x1, y1, x2, y2 = [int(v) for v in box]
            box_mask = np.zeros((orig_h, orig_w), dtype=bool)
            box_mask[y1:y2, x1:x2] = True

            # Strategy 1: Raw FastSAM
            area_raw = int(mask_raw_resized.sum())
            in_box_raw = int(np.logical_and(mask_raw_resized, box_mask).sum())
            out_box_raw = area_raw - in_box_raw
            ratio_raw = float(in_box_raw / area_raw) if area_raw > 0 else 0.0

            # Strategy 2: Hard Box Clipped
            mask_clipped = np.logical_and(mask_raw_resized, box_mask)
            area_clipped = int(mask_clipped.sum())
            ratio_clipped = 1.0 if area_clipped > 0 else 0.0

            # Strategy 3: Morphological Soft Refinement
            mask_soft = soft_refine_mask(mask_raw_resized, box)
            area_soft = int(mask_soft.sum())
            in_box_soft = int(np.logical_and(mask_soft, box_mask).sum())
            out_box_soft = area_soft - in_box_soft
            ratio_soft = float(in_box_soft / area_soft) if area_soft > 0 else 0.0

            # Connected components in raw vs soft
            _, labels_raw = cv2.connectedComponents(mask_raw_resized.astype(np.uint8))
            _, labels_soft = cv2.connectedComponents(mask_soft.astype(np.uint8))
            n_comp_raw = int(labels_raw.max())
            n_comp_soft = int(labels_soft.max())

            res = {
                "image_id": s["img_id"],
                "class_name": cls_name,
                "box_xyxy": [int(v) for v in box],
                "box_area": (x2 - x1) * (y2 - y1),
                "strategy_1_raw": {
                    "total_pixels": area_raw,
                    "pixels_inside_box": in_box_raw,
                    "pixels_outside_box": out_box_raw,
                    "inside_ratio": round(ratio_raw, 4),
                    "connected_components": n_comp_raw,
                },
                "strategy_2_hard_clipped": {
                    "total_pixels": area_clipped,
                    "pixels_inside_box": area_clipped,
                    "pixels_outside_box": 0,
                    "inside_ratio": 1.0,
                    "reduction_vs_raw_pct": round(((area_raw - area_clipped) / area_raw) * 100, 2) if area_raw > 0 else 0.0,
                },
                "strategy_3_soft_refined": {
                    "total_pixels": area_soft,
                    "pixels_inside_box": in_box_soft,
                    "pixels_outside_box": out_box_soft,
                    "inside_ratio": round(ratio_soft, 4),
                    "connected_components": n_comp_soft,
                    "isolated_islands_pruned": n_comp_raw - n_comp_soft,
                },
            }
            results.append(res)
            print(f"Sample: {s['img_id']} | Class: {cls_name:14s} | Box Area: {res['box_area']:6d} px")
            print(f"  Raw FastSAM       : {area_raw:6d} px (Inside Box: {ratio_raw*100:5.1f}%, Fragments: {n_comp_raw})")
            print(f"  Hard Box-Clipped  : {area_clipped:6d} px (100% inside, cut {res['strategy_2_hard_clipped']['reduction_vs_raw_pct']:.1f}% overflow)")
            print(f"  Soft Refined      : {area_soft:6d} px (Inside Box: {ratio_soft*100:5.1f}%, Pruned Islands: {res['strategy_3_soft_refined']['isolated_islands_pruned']})\n")

    out_file = REPO_ROOT / "reports" / "mask_refinement_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Refinement comparison saved to: {out_file}")


if __name__ == "__main__":
    main()

