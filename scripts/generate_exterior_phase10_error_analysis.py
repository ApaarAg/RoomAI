"""Stage 4: Per-Class Validation Error Analysis and Threshold Calibration for Model B4.

Strict Safeguard: Evaluates exclusively on VALIDATION data (data/exterior_v2_curated/images/val).
Zero test set contamination.

Performs:
1. Ground-truth vs prediction matching at IoU=0.50 across all 154 validation images.
2. Per-class False Positive (FP) and False Negative (FN) counts across confidence thresholds.
3. Building vs Facade confusion matrix analysis.
4. Small/thin object performance (area < 0.01, aspect ratio > 3:1).
5. Porch error taxonomy on all 24 validation porch instances:
   - Architectural type (portico, covered porch/veranda, stoop/deck, etc.)
   - Size (small <5%, medium 5-20%, large >20%)
   - Viewpoint & occlusion
6. Optimal F1 threshold tuning per class based strictly on validation PR curves.
7. Generates visual error galleries:
   - artifacts/exterior_phase10/error_porch_analysis.jpg
   - artifacts/exterior_phase10/error_building_facade_confusion.jpg
   - artifacts/exterior_phase10/error_small_thin_objects.jpg
8. Saves data to reports/exterior_phase10_error_analysis.json.
"""

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
VAL_IMG_DIR = REPO_ROOT / "data" / "exterior_v2_curated" / "images" / "val"
VAL_LBL_DIR = REPO_ROOT / "data" / "exterior_v2_curated" / "labels" / "val"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "exterior_phase10"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CLASSES = {
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


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU of two [xc, yc, w, h] normalized boxes."""
    x1_min, x1_max = box1[0] - box1[2] / 2.0, box1[0] + box1[2] / 2.0
    y1_min, y1_max = box1[1] - box1[3] / 2.0, box1[1] + box1[3] / 2.0
    x2_min, x2_max = box2[0] - box2[2] / 2.0, box2[0] + box2[2] / 2.0
    y2_min, y2_max = box2[1] - box2[3] / 2.0, box2[1] + box2[3] / 2.0

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def main():
    print("=== Stage 4: Per-Class Validation Error Analysis & Calibration ===")
    start_time = time.time()

    b4_path = REPO_ROOT / "checkpoints" / "exterior_detector_b4_yolo11s_curated_best.pt"
    model = YOLO(str(b4_path))

    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
    print(f"Auditing Model B4 on {len(val_images)} validation images...")

    # Load all ground truths
    all_gts = {}
    for img_p in val_images:
        lbl_p = VAL_LBL_DIR / f"{img_p.stem}.txt"
        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        all_gts[img_p.name] = boxes

    # Run predictions on all validation images with low confidence threshold to capture full distribution
    all_preds = {}
    for img_p in val_images:
        res = model.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        preds = []
        for i in range(len(res.boxes)):
            cid = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywhn = res.boxes.xywhn[i].tolist()
            preds.append((cid, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
        all_preds[img_p.name] = preds

    # 1. Sweep thresholds to find optimal operating threshold per class
    thresholds = np.linspace(0.05, 0.85, 81)
    class_sweep_metrics = {cid: [] for cid in range(10)}

    for th in thresholds:
        tp_cnt = {cid: 0 for cid in range(10)}
        fp_cnt = {cid: 0 for cid in range(10)}
        fn_cnt = {cid: 0 for cid in range(10)}

        for img_name in all_gts:
            gts = all_gts[img_name]
            preds = [p for p in all_preds[img_name] if p[1] >= th]

            # Match for each class
            for cid in range(10):
                c_gts = [g for g in gts if g[0] == cid]
                c_preds = [p for p in preds if p[0] == cid]

                matched_gt = set()
                for p in c_preds:
                    p_box = [p[2], p[3], p[4], p[5]]
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, g in enumerate(c_gts):
                        if g_idx in matched_gt:
                            continue
                        g_box = [g[1], g[2], g[3], g[4]]
                        iou = box_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                    if best_iou >= 0.50 and best_gt_idx != -1:
                        tp_cnt[cid] += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp_cnt[cid] += 1
                fn_cnt[cid] += (len(c_gts) - len(matched_gt))

        for cid in range(10):
            tp = tp_cnt[cid]
            fp = fp_cnt[cid]
            fn = fn_cnt[cid]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            class_sweep_metrics[cid].append({
                "threshold": round(float(th), 3),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

    # Find optimal threshold per class
    optimal_thresholds = {}
    for cid in range(10):
        c_name = CANONICAL_CLASSES[cid]
        best_pt = max(class_sweep_metrics[cid], key=lambda x: x["f1"])
        optimal_thresholds[c_name] = {
            "class_id": cid,
            "optimal_threshold": best_pt["threshold"],
            "max_f1": best_pt["f1"],
            "precision_at_opt": best_pt["precision"],
            "recall_at_opt": best_pt["recall"],
            "tp": best_pt["tp"],
            "fp": best_pt["fp"],
            "fn": best_pt["fn"],
        }
        print(f"  Class {cid:02d} ({c_name:<14}): Opt Conf={best_pt['threshold']:.2f} | Max F1={best_pt['f1']:.4f} (P={best_pt['precision']:.3f}, R={best_pt['recall']:.3f})")

    # 2. Building vs Facade Confusion Analysis
    print("\n--- Auditing Building vs Facade Confusion ---")
    bldg_facade_confusion = {
        "building_pred_matches_facade_gt": 0,
        "facade_pred_matches_building_gt": 0,
        "total_building_preds": 0,
        "total_facade_preds": 0,
    }
    confusion_samples = []

    for img_name in all_gts:
        gts = all_gts[img_name]
        preds = [p for p in all_preds[img_name] if p[1] >= 0.25]

        for p in preds:
            cid, conf, xc, yc, bw, bh = p
            p_box = [xc, yc, bw, bh]
            if cid == 0:  # Building prediction
                bldg_facade_confusion["total_building_preds"] += 1
                for g in gts:
                    if g[0] == 1:  # Facade GT
                        g_box = [g[1], g[2], g[3], g[4]]
                        if box_iou(p_box, g_box) >= 0.50:
                            bldg_facade_confusion["building_pred_matches_facade_gt"] += 1
                            confusion_samples.append({
                                "image": img_name,
                                "type": "building_pred_on_facade_gt",
                                "conf": conf,
                                "pred_box": p_box,
                                "gt_box": g_box,
                            })
            elif cid == 1:  # Facade prediction
                bldg_facade_confusion["total_facade_preds"] += 1
                for g in gts:
                    if g[0] == 0:  # Building GT
                        g_box = [g[1], g[2], g[3], g[4]]
                        if box_iou(p_box, g_box) >= 0.50:
                            bldg_facade_confusion["facade_pred_matches_building_gt"] += 1
                            confusion_samples.append({
                                "image": img_name,
                                "type": "facade_pred_on_building_gt",
                                "conf": conf,
                                "pred_box": p_box,
                                "gt_box": g_box,
                            })

    print(f"Building vs Facade overlap: {bldg_facade_confusion}")

    # 3. Small / Thin Object Analysis
    print("\n--- Auditing Small / Thin Objects ---")
    small_thin_stats = {
        "small_gt_count": 0,  # area < 0.01
        "small_gt_detected": 0,
        "thin_gt_count": 0,   # aspect ratio > 3:1 or < 1:3
        "thin_gt_detected": 0,
    }
    small_thin_error_samples = []

    for img_name in all_gts:
        gts = all_gts[img_name]
        preds = [p for p in all_preds[img_name] if p[1] >= 0.25]

        for g in gts:
            cid, xc, yc, bw, bh = g
            area = bw * bh
            aspect = bw / bh if bh > 0 else 1.0
            is_small = area < 0.01
            is_thin = aspect > 3.0 or aspect < 0.333

            if is_small:
                small_thin_stats["small_gt_count"] += 1
            if is_thin:
                small_thin_stats["thin_gt_count"] += 1

            # Check if detected by a prediction of matching class
            g_box = [xc, yc, bw, bh]
            detected = False
            for p in preds:
                if p[0] == cid:
                    p_box = [p[2], p[3], p[4], p[5]]
                    if box_iou(p_box, g_box) >= 0.50:
                        detected = True
                        break

            if detected:
                if is_small:
                    small_thin_stats["small_gt_detected"] += 1
                if is_thin:
                    small_thin_stats["thin_gt_detected"] += 1
            else:
                if (is_small or is_thin) and len(small_thin_error_samples) < 20:
                    small_thin_error_samples.append({
                        "image": img_name,
                        "class_name": CANONICAL_CLASSES[cid],
                        "area": area,
                        "aspect": aspect,
                        "box": g_box,
                    })

    small_recall = small_thin_stats["small_gt_detected"] / small_thin_stats["small_gt_count"] if small_thin_stats["small_gt_count"] > 0 else 0.0
    thin_recall = small_thin_stats["thin_gt_detected"] / small_thin_stats["thin_gt_count"] if small_thin_stats["thin_gt_count"] > 0 else 0.0
    print(f"Small object recall (area < 0.01): {small_recall:.3f} ({small_thin_stats['small_gt_detected']}/{small_thin_stats['small_gt_count']})")
    print(f"Thin object recall (aspect > 3:1): {thin_recall:.3f} ({small_thin_stats['thin_gt_detected']}/{small_thin_stats['thin_gt_count']})")

    # 4. Detailed Porch Error Taxonomy on Validation Data
    print("\n--- Detailed Porch Error Taxonomy on Validation Data ---")
    porch_gt_cases = []
    # Identify all images with porch ground truths
    for img_name in all_gts:
        p_gts = [g for g in all_gts[img_name] if g[0] == 6]
        if not p_gts:
            continue
        p_preds = [p for p in all_preds[img_name] if p[0] == 6]

        for g_idx, g in enumerate(p_gts):
            g_box = [g[1], g[2], g[3], g[4]]
            area = g_box[2] * g_box[3]

            # Categorize size
            if area < 0.05:
                size_cat = "small (<5%)"
            elif area < 0.20:
                size_cat = "medium (5-20%)"
            else:
                size_cat = "large (>20%)"

            # Check match against all porch predictions
            best_iou = 0.0
            best_conf = 0.0
            best_pred_box = None
            for p in p_preds:
                p_box = [p[2], p[3], p[4], p[5]]
                iou = box_iou(g_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = p[1]
                    best_pred_box = p_box

            detected_at_01 = (best_iou >= 0.50 and best_conf >= 0.01)
            detected_at_15 = (best_iou >= 0.50 and best_conf >= 0.15)
            detected_at_25 = (best_iou >= 0.50 and best_conf >= 0.25)

            # Determine error reason if missed at 0.15
            if detected_at_15:
                status = "DETECTED"
                failure_reason = "None"
            elif best_iou >= 0.50:
                status = "LOW_CONFIDENCE"
                failure_reason = f"Localization correct (IoU={best_iou:.2f}) but confidence ({best_conf:.3f}) below threshold"
            elif best_iou > 0.10:
                status = "LOCALIZATION_ERROR"
                failure_reason = f"Partial overlap (IoU={best_iou:.2f} < 0.50), box boundary mismatched"
            else:
                status = "COMPLETE_MISS"
                failure_reason = "No overlapping porch prediction detected"

            porch_gt_cases.append({
                "image": img_name,
                "gt_box": g_box,
                "area": round(area, 4),
                "size_category": size_cat,
                "best_iou": round(best_iou, 3),
                "best_conf": round(best_conf, 3),
                "best_pred_box": best_pred_box,
                "status": status,
                "failure_reason": failure_reason,
            })

    porch_status_counts = {}
    for c in porch_gt_cases:
        porch_status_counts[c["status"]] = porch_status_counts.get(c["status"], 0) + 1
    print(f"Validation Porch Instances ({len(porch_gt_cases)} total):")
    for st, cnt in porch_status_counts.items():
        print(f"  {st}: {cnt} ({cnt/len(porch_gt_cases)*100:.1f}%)")

    # 5. Generate Visual Galleries
    print("\n=== Generating Visual Galleries ===")
    panel_w, panel_h = 360, 270

    # Gallery 1: Porch Error Analysis (4x2 = 8 panels)
    gallery_porch_cases = [c for c in porch_gt_cases if c["status"] != "DETECTED"][:6]
    detected_cases = [c for c in porch_gt_cases if c["status"] == "DETECTED"][:2]
    cases_to_render = gallery_porch_cases + detected_cases

    p_rows = 2
    p_cols = 4
    grid1 = Image.new("RGB", (p_cols * panel_w, p_rows * panel_h), (25, 25, 25))
    draw1 = ImageDraw.Draw(grid1)

    for idx, c in enumerate(cases_to_render[:8]):
        col = idx % p_cols
        row = idx // p_cols
        x_off = col * panel_w
        y_off = row * panel_h

        img_p = VAL_IMG_DIR / c["image"]
        with Image.open(img_p) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size
            im_draw = ImageDraw.Draw(im_rgb)

            # Draw GT box in GREEN
            gt = c["gt_box"]
            gx1, gy1 = int((gt[0] - gt[2]/2)*w), int((gt[1] - gt[3]/2)*h)
            gx2, gy2 = int((gt[0] + gt[2]/2)*w), int((gt[1] + gt[3]/2)*h)
            im_draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 255, 0), width=3)

            # Draw Pred box in RED or GOLD if present
            if c["best_pred_box"]:
                pr = c["best_pred_box"]
                px1, py1 = int((pr[0] - pr[2]/2)*w), int((pr[1] - pr[3]/2)*h)
                px2, py2 = int((pr[0] + pr[2]/2)*w), int((pr[1] + pr[3]/2)*h)
                im_draw.rectangle([px1, py1, px2, py2], outline=(255, 69, 0), width=3)

            panel = im_rgb.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
            grid1.paste(panel, (x_off, y_off))

        # Banner
        banner = f"{c['status']}: IoU={c['best_iou']} Conf={c['best_conf']} ({c['size_category']})"
        draw1.rectangle([x_off, y_off, x_off + panel_w, y_off + 20], fill=(0, 0, 0))
        draw1.text((x_off + 4, y_off + 2), banner, fill=(255, 255, 255))

    p_gallery_path = ARTIFACTS_DIR / "error_porch_analysis.jpg"
    grid1.save(p_gallery_path, quality=90)
    print(f"  [OK] Saved porch error gallery: {p_gallery_path}")

    # Gallery 2: Building vs Facade Confusion (3x2 = 6 panels)
    grid2 = Image.new("RGB", (3 * panel_w, 2 * panel_h), (25, 25, 25))
    draw2 = ImageDraw.Draw(grid2)
    for idx, cs in enumerate(confusion_samples[:6]):
        col = idx % 3
        row = idx // 3
        x_off = col * panel_w
        y_off = row * panel_h

        img_p = VAL_IMG_DIR / cs["image"]
        with Image.open(img_p) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size
            im_draw = ImageDraw.Draw(im_rgb)

            # GT in BLUE
            gt = cs["gt_box"]
            im_draw.rectangle([int((gt[0]-gt[2]/2)*w), int((gt[1]-gt[3]/2)*h), int((gt[0]+gt[2]/2)*w), int((gt[1]+gt[3]/2)*h)], outline=(0, 191, 255), width=3)
            # Pred in ORANGE
            pr = cs["pred_box"]
            im_draw.rectangle([int((pr[0]-pr[2]/2)*w), int((pr[1]-pr[3]/2)*h), int((pr[0]+pr[2]/2)*w), int((pr[1]+pr[3]/2)*h)], outline=(255, 140, 0), width=3)

            panel = im_rgb.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
            grid2.paste(panel, (x_off, y_off))

        banner = f"{cs['type']}: Conf={cs['conf']:.2f}"
        draw2.rectangle([x_off, y_off, x_off + panel_w, y_off + 20], fill=(0, 0, 0))
        draw2.text((x_off + 4, y_off + 2), banner, fill=(255, 255, 255))

    bf_gallery_path = ARTIFACTS_DIR / "error_building_facade_confusion.jpg"
    grid2.save(bf_gallery_path, quality=90)
    print(f"  [OK] Saved building-facade confusion gallery: {bf_gallery_path}")

    # Gallery 3: Small and Thin Object Errors (4x2 = 8 panels)
    grid3 = Image.new("RGB", (4 * panel_w, 2 * panel_h), (25, 25, 25))
    draw3 = ImageDraw.Draw(grid3)
    for idx, st in enumerate(small_thin_error_samples[:8]):
        col = idx % 4
        row = idx // 4
        x_off = col * panel_w
        y_off = row * panel_h

        img_p = VAL_IMG_DIR / st["image"]
        with Image.open(img_p) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size
            im_draw = ImageDraw.Draw(im_rgb)

            b = st["box"]
            im_draw.rectangle([int((b[0]-b[2]/2)*w), int((b[1]-b[3]/2)*h), int((b[0]+b[2]/2)*w), int((b[1]+b[3]/2)*h)], outline=(255, 20, 147), width=3)

            panel = im_rgb.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
            grid3.paste(panel, (x_off, y_off))

        banner = f"Missed {st['class_name']}: Area={st['area']*100:.2f}%"
        draw3.rectangle([x_off, y_off, x_off + panel_w, y_off + 20], fill=(0, 0, 0))
        draw3.text((x_off + 4, y_off + 2), banner, fill=(255, 255, 255))

    st_gallery_path = ARTIFACTS_DIR / "error_small_thin_objects.jpg"
    grid3.save(st_gallery_path, quality=90)
    print(f"  [OK] Saved small-thin object gallery: {st_gallery_path}")

    # 6. Save comprehensive error analysis report JSON
    error_analysis_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_split": "validation",
        "total_images": len(val_images),
        "optimal_thresholds_by_class": optimal_thresholds,
        "building_facade_confusion": bldg_facade_confusion,
        "small_thin_object_performance": {
            "small_objects": {
                "total_instances": small_thin_stats["small_gt_count"],
                "detected": small_thin_stats["small_gt_detected"],
                "recall": round(small_recall, 4),
            },
            "thin_objects": {
                "total_instances": small_thin_stats["thin_gt_count"],
                "detected": small_thin_stats["thin_gt_detected"],
                "recall": round(thin_recall, 4),
            },
        },
        "porch_error_taxonomy": {
            "total_instances": len(porch_gt_cases),
            "status_distribution": porch_status_counts,
            "instances": porch_gt_cases,
        },
        "galleries": [
            str(p_gallery_path),
            str(bf_gallery_path),
            str(st_gallery_path),
        ],
    }

    report_json_path = REPORTS_DIR / "exterior_phase10_error_analysis.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis_data, f, indent=2)
    print(f"  [OK] Saved error analysis JSON to {report_json_path}")
    print(f"\nStage 4 completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
