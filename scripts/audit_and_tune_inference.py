"""Audit, Error Diagnosis, and Inference Tuning Script for RoomAI Object Detection.

Performs:
1. Model loading & taxonomy compatibility audit.
2. Ground-truth vs prediction evaluation across validation set (554 images).
3. Error categorization:
   - False Positives (background vs classification confusion vs duplicate predictions).
   - False Negatives (missed objects).
   - Localization Errors (0.1 <= IoU < 0.5).
4. Threshold & fusion parameter tuning on validation data:
   - Confidence threshold sweep (0.20 to 0.50)
   - Intra-model NMS and WBF IoU threshold sweep (0.40 to 0.60)
   - Cross-class duplicate suppression analysis
   - Calibration and model weight tuning
5. Diagnostic visualization generator:
   - Side-by-side visual panels: GT, A1, A2, Ensemble WBF, and Error Classification.
6. CUDA-synchronized timing profiling:
   - Loading (cold vs warm), Preprocessing, Inference, Fusion, Rendering, Total.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.detection_model import CustomObjectDetector, get_cached_yolo
from schemas.responses import DetectionResponse
from services.ensemble_service import EnsembleDetector, calculate_iou
from utils.timing_utils import StageTimer, format_duration

CANONICAL_CLASSES = {
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
    11: "window",
}

CLASS_TO_ID = {v: k for k, v in CANONICAL_CLASSES.items()}


def load_ground_truth(labels_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Loads YOLO ground truth annotations from text files."""
    gt_map: Dict[str, List[Dict[str, Any]]] = {}
    for txt_file in labels_dir.glob("*.txt"):
        stem = txt_file.stem
        boxes = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0.0, xc - w / 2.0)
                    y1 = max(0.0, yc - h / 2.0)
                    x2 = min(1.0, xc + w / 2.0)
                    y2 = min(1.0, yc + h / 2.0)
                    boxes.append({
                        "class_id": cid,
                        "label": CANONICAL_CLASSES.get(cid, f"class_{cid}"),
                        "box": (x1, y1, x2, y2),
                    })
        gt_map[stem] = boxes
    return gt_map


def evaluate_detections(
    preds_by_stem: Dict[str, List[Dict[str, Any]]],
    gt_by_stem: Dict[str, List[Dict[str, Any]]],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics including error breakdown."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    localization_errors = 0
    class_confusion_errors = 0
    duplicate_fps = 0
    background_fps = 0

    per_class_stats = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
        for cid in CANONICAL_CLASSES.keys()
    }

    confusion_matrix = np.zeros((13, 13), dtype=int)  # 12 classes + 1 background

    for stem, gt_boxes in gt_by_stem.items():
        preds = preds_by_stem.get(stem, [])
        for gt in gt_boxes:
            cid = gt["class_id"]
            if cid in per_class_stats:
                per_class_stats[cid]["gt"] += 1

        matched_gt_indices = set()
        sorted_preds = sorted(preds, key=lambda x: x.get("confidence", 0.0), reverse=True)

        for pred in sorted_preds:
            p_cid = pred["class_id"]
            p_box = pred["box"]
            if p_cid in per_class_stats:
                per_class_stats[p_cid]["pred"] += 1

            best_iou = 0.0
            best_gt_idx = -1
            best_any_class_iou = 0.0
            best_any_class_gt_idx = -1

            # Match against GT
            for gt_idx, gt in enumerate(gt_boxes):
                iou = calculate_iou(p_box, gt["box"])
                if iou > best_any_class_iou:
                    best_any_class_iou = iou
                    best_any_class_gt_idx = gt_idx

                if gt["class_id"] == p_cid and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= iou_thresh:
                if best_gt_idx not in matched_gt_indices:
                    matched_gt_indices.add(best_gt_idx)
                    total_tp += 1
                    per_class_stats[p_cid]["tp"] += 1
                    confusion_matrix[p_cid, p_cid] += 1
                else:
                    # Duplicate prediction on same GT box
                    duplicate_fps += 1
                    total_fp += 1
                    per_class_stats[p_cid]["fp"] += 1
            else:
                total_fp += 1
                per_class_stats[p_cid]["fp"] += 1
                if best_any_class_gt_idx >= 0 and best_any_class_iou >= iou_thresh:
                    # High IoU but different class -> Classification confusion
                    class_confusion_errors += 1
                    gt_cid = gt_boxes[best_any_class_gt_idx]["class_id"]
                    confusion_matrix[p_cid, gt_cid] += 1
                elif best_iou >= 0.10:
                    # Localization error (0.10 <= IoU < 0.50)
                    localization_errors += 1
                    confusion_matrix[p_cid, 12] += 1
                else:
                    background_fps += 1
                    confusion_matrix[p_cid, 12] += 1

        unmatched = len(gt_boxes) - len(matched_gt_indices)
        total_fn += unmatched
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt_indices:
                cid = gt["class_id"]
                if cid in per_class_stats:
                    per_class_stats[cid]["fn"] += 1
                confusion_matrix[12, cid] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class_res = {}
    for cid, s in per_class_stats.items():
        c_p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        c_r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
        per_class_res[CANONICAL_CLASSES[cid]] = {
            "gt_count": s["gt"],
            "pred_count": s["pred"],
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1": round(c_f1, 4),
        }

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "localization_errors": localization_errors,
        "class_confusion_errors": class_confusion_errors,
        "duplicate_fps": duplicate_fps,
        "background_fps": background_fps,
        "per_class": per_class_res,
        "confusion_matrix": confusion_matrix.tolist(),
    }


def render_diagnostic_panel(
    image: Image.Image,
    gt_boxes: List[Dict[str, Any]],
    a1_boxes: List[Dict[str, Any]],
    a2_boxes: List[Dict[str, Any]],
    ens_boxes: List[Dict[str, Any]],
    scene_name: str,
    output_path: Path,
) -> None:
    """Renders a 5-panel diagnostic visualization."""
    color_palette = [
        "#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6",
        "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16", "#06B6D4", "#A855F7"
    ]

    def draw_boxes(img: Image.Image, boxes: List[Dict[str, Any]], header: str, is_error_view: bool = False) -> Image.Image:
        canvas = img.copy().convert("RGB")
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size

        # Top banner
        draw.rectangle([0, 0, w, 28], fill="#0F172A")
        draw.text((10, 6), f"{header} ({len(boxes)} boxes)", fill="#F8FAFC")

        for item in boxes:
            cid = item["class_id"]
            conf = item.get("confidence", 1.0)
            color = color_palette[cid % len(color_palette)]
            if is_error_view:
                color = item.get("error_color", color)

            x1, y1, x2, y2 = item["box"]
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)

            draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
            label_txt = f"{item['label']} {conf:.2f}" if "confidence" in item else item["label"]
            badge_w = len(label_txt) * 7 + 8
            badge_h = 16
            top_y = py1 - badge_h if py1 >= 28 + badge_h else py1
            bot_y = top_y + badge_h
            draw.rectangle([px1, top_y, px1 + badge_w, bot_y], fill=color)
            draw.text((px1 + 4, top_y + 1), label_txt, fill="white")

        return canvas

    # Build error view for ensemble
    error_items = []
    # Identify TPs, FPs, and duplicates
    matched_gt = set()
    for item in sorted(ens_boxes, key=lambda x: x.get("confidence", 0.0), reverse=True):
        p_box = item["box"]
        p_cid = item["class_id"]
        best_iou = 0.0
        best_gt = -1
        best_any_iou = 0.0
        for g_idx, g in enumerate(gt_boxes):
            iou = calculate_iou(p_box, g["box"])
            if iou > best_any_iou:
                best_any_iou = iou
            if g["class_id"] == p_cid and iou > best_iou:
                best_iou = iou
                best_gt = g_idx

        copy_item = dict(item)
        if best_gt >= 0 and best_iou >= 0.50:
            if best_gt not in matched_gt:
                matched_gt.add(best_gt)
                copy_item["error_color"] = "#10B981"  # Green = TP
                copy_item["label"] = f"TP: {item['label']}"
            else:
                copy_item["error_color"] = "#EC4899"  # Magenta = Duplicate FP
                copy_item["label"] = f"DUP: {item['label']}"
        elif best_any_iou >= 0.50:
            copy_item["error_color"] = "#EF4444"  # Red = Class confusion FP
            copy_item["label"] = f"CONFUSE: {item['label']}"
        else:
            copy_item["error_color"] = "#F97316"  # Orange = False Positive
            copy_item["label"] = f"FP: {item['label']}"
        error_items.append(copy_item)

    # Add FNs
    for g_idx, g in enumerate(gt_boxes):
        if g_idx not in matched_gt:
            fn_item = dict(g)
            fn_item["error_color"] = "#EAB308"  # Yellow = Missed GT (FN)
            fn_item["label"] = f"FN (Missed): {g['label']}"
            error_items.append(fn_item)

    panel_gt = draw_boxes(image, gt_boxes, "1. Ground Truth")
    panel_a1 = draw_boxes(image, a1_boxes, "2. Model A1 (YOLO11n)")
    panel_a2 = draw_boxes(image, a2_boxes, "3. Model A2 (YOLO11s)")
    panel_ens = draw_boxes(image, ens_boxes, "4. Ensemble WBF (Original)")
    panel_err = draw_boxes(image, error_items, "5. Error Diagnostic (Green=TP, Red=FP, Magenta=DUP, Yellow=FN)", is_error_view=True)

    tw, th = 480, 360
    grid = Image.new("RGB", (tw * 3, th * 2), color="#0F172A")
    grid.paste(panel_gt.resize((tw, th)), (0, 0))
    grid.paste(panel_a1.resize((tw, th)), (tw, 0))
    grid.paste(panel_a2.resize((tw, th)), (tw * 2, 0))
    grid.paste(panel_ens.resize((tw, th)), (0, th))
    grid.paste(panel_err.resize((tw, th)), (tw, th))

    # Info card on remaining slot
    card = Image.new("RGB", (tw, th), color="#1E293B")
    draw_card = ImageDraw.Draw(card)
    draw_card.rectangle([0, 0, tw, 28], fill="#334155")
    draw_card.text((10, 6), f"Scene: {scene_name}", fill="#38BDF8")

    lines = [
        f"Ground Truth Objects: {len(gt_boxes)}",
        f"Model A1 Detections: {len(a1_boxes)}",
        f"Model A2 Detections: {len(a2_boxes)}",
        f"Ensemble WBF Boxes: {len(ens_boxes)}",
        "",
        "Diagnostic Key:",
        "  - Green   : True Positive (IoU >= 0.50)",
        "  - Red     : Class Confusion FP (Overlap with other class)",
        "  - Magenta : Duplicate FP (Same object detected twice)",
        "  - Orange  : Background / Localization FP",
        "  - Yellow  : False Negative (Ground Truth missed)",
    ]
    for i, line in enumerate(lines):
        draw_card.text((15, 45 + i * 22), line, fill="#F1F5F9")

    grid.paste(card, (tw * 2, th))
    grid.save(output_path, format="PNG")


def run_comprehensive_audit():
    print("=================================================================")
    print("STAGE 1 & 2 AUDIT & DIAGNOSTICS: ROOMAI INDOOR OBJECT DETECTION")
    print("=================================================================")

    val_images_dir = REPO_ROOT / "data" / "indoor_canonical" / "images" / "val"
    val_labels_dir = REPO_ROOT / "data" / "indoor_canonical" / "labels" / "val"
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = reports_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    gt_by_stem = load_ground_truth(val_labels_dir)
    val_img_paths = sorted(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))
    print(f">> Loaded {len(val_img_paths)} validation images and {sum(len(v) for v in gt_by_stem.values())} GT boxes.")

    # 1. Checkpoint & class mapping verification
    print("\n--- 1. Checkpoint & Taxonomy Verification ---")
    ckpts = {
        "Baseline": REPO_ROOT / "checkpoints" / "deepfurniture_yolo11n_finetune_best.pt",
        "Model A1": REPO_ROOT / "checkpoints" / "indoor_detector_a1_yolo11n_pilot_best.pt",
        "Model A2": REPO_ROOT / "checkpoints" / "indoor_detector_a2_yolo11s_pilot_best.pt",
    }
    for name, p in ckpts.items():
        if not p.exists():
            print(f"[ERROR] Missing {name}: {p}")
        else:
            y_m = get_cached_yolo(str(p), "cuda:0")
            print(f"  * {name:10s} : {p.name} | classes={len(y_m.names)} | task={y_m.task}")

    # 2. Timing benchmark (synchronized CUDA)
    print("\n--- 2. Synchronized CUDA Timing Profile ---")
    dummy_img = Image.new("RGB", (640, 640), color="white")
    m_a1 = CustomObjectDetector({"checkpoint_path": str(ckpts["Model A1"]), "device": "cuda:0"})
    m_a2 = CustomObjectDetector({"checkpoint_path": str(ckpts["Model A2"]), "device": "cuda:0"})

    # Cold start load time
    t_cold_a1_start = time.perf_counter()
    m_a1.load_model()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_load_a1_ms = (time.perf_counter() - t_cold_a1_start) * 1000.0

    t_cold_a2_start = time.perf_counter()
    m_a2.load_model()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_load_a2_ms = (time.perf_counter() - t_cold_a2_start) * 1000.0

    # Warmup
    for _ in range(5):
        m_a1.predict(dummy_img)
        m_a2.predict(dummy_img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Steady-state latencies
    lat_a1 = []
    lat_a2 = []
    lat_wbf = []
    ens_orig = EnsembleDetector(
        models=[m_a1, m_a2],
        model_names=["Model A1", "Model A2"],
        fusion_iou_threshold=0.50,
        single_model_factor=0.88,
        consensus_bonus=0.06,
        fusion_method="wbf",
    )

    test_sample_img = Image.open(val_img_paths[0]).convert("RGB")
    for _ in range(20):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = m_a1.predict(test_sample_img, confidence_threshold=0.35)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_a1.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = m_a2.predict(test_sample_img, confidence_threshold=0.35)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_a2.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = ens_orig.predict(test_sample_img, confidence_threshold=0.35)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_wbf.append((time.perf_counter() - t0) * 1000.0)

    print(f"  * Cold start Model A1 load: {cold_load_a1_ms:.2f} ms")
    print(f"  * Cold start Model A2 load: {cold_load_a2_ms:.2f} ms")
    print(f"  * Steady-state Model A1 : mean={np.mean(lat_a1):.2f} ms | p50={np.median(lat_a1):.2f} ms | p95={np.percentile(lat_a1, 95):.2f} ms")
    print(f"  * Steady-state Model A2 : mean={np.mean(lat_a2):.2f} ms | p50={np.median(lat_a2):.2f} ms | p95={np.percentile(lat_a2, 95):.2f} ms")
    print(f"  * Steady-state Ensemble : mean={np.mean(lat_wbf):.2f} ms | p50={np.median(lat_wbf):.2f} ms | p95={np.percentile(lat_wbf, 95):.2f} ms")

    # 3. Collect predictions on validation set for Model A1, Model A2, and Baseline
    print("\n--- 3. Collecting Predictions on Validation Set (554 images) ---")
    preds_a1_by_stem = {}
    preds_a2_by_stem = {}

    t_eval_start = time.time()
    for idx, img_path in enumerate(val_img_paths):
        stem = img_path.stem
        img = Image.open(img_path).convert("RGB")

        # Run at candidate threshold 0.15 so we can evaluate any threshold downstream
        r_a1 = m_a1.predict(img, confidence_threshold=0.15)
        r_a2 = m_a2.predict(img, confidence_threshold=0.15)

        preds_a1_by_stem[stem] = [
            {"class_id": d.class_id, "label": d.label, "confidence": d.confidence, "box": (d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max)}
            for d in r_a1.detections
        ]
        preds_a2_by_stem[stem] = [
            {"class_id": d.class_id, "label": d.label, "confidence": d.confidence, "box": (d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max)}
            for d in r_a2.detections
        ]

        if (idx + 1) % 100 == 0 or (idx + 1) == len(val_img_paths):
            print(f"  [{idx+1}/{len(val_img_paths)}] images processed ({(time.time()-t_eval_start):.1f}s)...")

    # 4. Generate Visualizations for Key Representative Scenes
    print("\n--- 4. Generating Visual Panels for Representative Scenes ---")
    # Identify sample scenes with sofas, chairs, windows, lamps, plants, decorations
    target_stems = []
    for stem, gts in gt_by_stem.items():
        cids = set(g["class_id"] for g in gts)
        # Looking for living rooms with sofa, chair, window, plant, lamp
        if 6 in cids and 2 in cids and len(gts) >= 6:
            target_stems.append(stem)
            if len(target_stems) >= 4:
                break

    for i, stem in enumerate(target_stems):
        img_file = val_images_dir / f"{stem}.jpg"
        if not img_file.exists():
            img_file = val_images_dir / f"{stem}.png"
        img = Image.open(img_file).convert("RGB")
        gts = gt_by_stem.get(stem, [])

        a1_boxes = [d for d in preds_a1_by_stem[stem] if d["confidence"] >= 0.35]
        a2_boxes = [d for d in preds_a2_by_stem[stem] if d["confidence"] >= 0.35]

        # Standard ensemble prediction
        ens_resp = ens_orig.predict(img, confidence_threshold=0.35)
        ens_boxes = [
            {"class_id": d.class_id, "label": d.label, "confidence": d.confidence, "box": (d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max)}
            for d in ens_resp.detections
        ]

        out_panel = vis_dir / f"diagnostic_scene_{i+1}_{stem[:12]}.png"
        render_diagnostic_panel(
            img, gts, a1_boxes, a2_boxes, ens_boxes, f"Validation Scene {i+1} ({stem})", out_panel
        )
        print(f"  Saved visual panel: {out_panel.name}")

    # 5. Evaluate individual models at baseline threshold 0.35
    print("\n--- 5. Evaluating Models at Default Threshold (0.35) ---")
    filtered_a1 = {
        stem: [d for d in items if d["confidence"] >= 0.35]
        for stem, items in preds_a1_by_stem.items()
    }
    filtered_a2 = {
        stem: [d for d in items if d["confidence"] >= 0.35]
        for stem, items in preds_a2_by_stem.items()
    }

    eval_a1_035 = evaluate_detections(filtered_a1, gt_by_stem, iou_thresh=0.50)
    eval_a2_035 = evaluate_detections(filtered_a2, gt_by_stem, iou_thresh=0.50)

    print(f"Model A1 (YOLO11n) @ 0.35: Precision={eval_a1_035['precision']:.4f}, Recall={eval_a1_035['recall']:.4f}, F1={eval_a1_035['f1']:.4f}")
    print(f"  TP={eval_a1_035['tp']}, FP={eval_a1_035['fp']}, FN={eval_a1_035['fn']}")
    print(f"  Class Confusion FPs: {eval_a1_035['class_confusion_errors']}, Duplicate FPs: {eval_a1_035['duplicate_fps']}, Background FPs: {eval_a1_035['background_fps']}")

    print(f"Model A2 (YOLO11s) @ 0.35: Precision={eval_a2_035['precision']:.4f}, Recall={eval_a2_035['recall']:.4f}, F1={eval_a2_035['f1']:.4f}")
    print(f"  TP={eval_a2_035['tp']}, FP={eval_a2_035['fp']}, FN={eval_a2_035['fn']}")
    print(f"  Class Confusion FPs: {eval_a2_035['class_confusion_errors']}, Duplicate FPs: {eval_a2_035['duplicate_fps']}, Background FPs: {eval_a2_035['background_fps']}")

    # 6. Evaluate Original Ensemble WBF across the validation set
    print("\n--- 6. Evaluating Baseline Ensemble WBF across Validation Set ---")
    ens_orig_preds = {}
    for stem in val_img_paths:
        s = stem.stem
        a1_items = preds_a1_by_stem[s]
        a2_items = preds_a2_by_stem[s]
        fused = ens_orig.fuse_predictions([a1_items, a2_items], confidence_threshold=0.35)
        ens_orig_preds[s] = [
            {"class_id": d.class_id, "label": d.label, "confidence": d.confidence, "box": (d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max)}
            for d in fused
        ]

    eval_ens_orig = evaluate_detections(ens_orig_preds, gt_by_stem, iou_thresh=0.50)
    print(f"Ensemble WBF (Original) @ 0.35: Precision={eval_ens_orig['precision']:.4f}, Recall={eval_ens_orig['recall']:.4f}, F1={eval_ens_orig['f1']:.4f}")
    print(f"  TP={eval_ens_orig['tp']}, FP={eval_ens_orig['fp']}, FN={eval_ens_orig['fn']}")
    print(f"  Class Confusion FPs: {eval_ens_orig['class_confusion_errors']}, Duplicate FPs: {eval_ens_orig['duplicate_fps']}, Background FPs: {eval_ens_orig['background_fps']}")

    # Save initial audit data to json
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timing": {
            "cold_load_a1_ms": round(cold_load_a1_ms, 2),
            "cold_load_a2_ms": round(cold_load_a2_ms, 2),
            "mean_a1_ms": round(float(np.mean(lat_a1)), 2),
            "mean_a2_ms": round(float(np.mean(lat_a2)), 2),
            "mean_wbf_ms": round(float(np.mean(lat_wbf)), 2),
        },
        "model_a1_035": eval_a1_035,
        "model_a2_035": eval_a2_035,
        "ensemble_orig_035": eval_ens_orig,
    }
    with open(reports_dir / "audit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print("\n>> Initial audit data exported to reports/audit_metrics.json")
    
    # 7. Run Stage 3: Inference and Fusion Tuning
    print("\n=================================================================")
    print("STAGE 3: TUNING INFERENCE, NMS, CROSS-CLASS RESOLUTION & WBF")
    print("=================================================================")
    tuning_results = run_tuning_grid(preds_a1_by_stem, preds_a2_by_stem, gt_by_stem, m_a1, m_a2)
    
    with open(reports_dir / "tuning_metrics.json", "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)
    print("\n>> Complete tuning metrics exported to reports/tuning_metrics.json")

    return preds_a1_by_stem, preds_a2_by_stem, gt_by_stem


def resolve_cross_class_conflicts(
    items: List[Dict[str, Any]], cross_iou_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """Suppresses lower-confidence overlapping detections across different classes when bounding boxes are nearly identical."""
    if not items or cross_iou_threshold <= 0:
        return items

    sorted_items = sorted(items, key=lambda x: x["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for item in sorted_items:
        suppress = False
        for k in kept:
            # Different class conflict
            if item["class_id"] != k["class_id"]:
                iou = calculate_iou(item["box"], k["box"])
                if iou >= cross_iou_threshold:
                    suppress = True
                    break
        if not suppress:
            kept.append(item)

    return kept


def run_tuning_grid(
    preds_a1: Dict[str, List[Dict[str, Any]]],
    preds_a2: Dict[str, List[Dict[str, Any]]],
    gt: Dict[str, List[Dict[str, Any]]],
    m_a1: CustomObjectDetector,
    m_a2: CustomObjectDetector,
) -> Dict[str, Any]:
    """Sweeps confidence thresholds, IoU thresholds, fusion methods, and conflict resolution."""
    stems = list(preds_a1.keys())

    # 1. Confidence threshold sweep for single models
    print("\n[Sweep 1]: Confidence threshold sweep on single detectors...")
    conf_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    single_model_sweeps = {"model_a1": {}, "model_a2": {}}

    for conf in conf_thresholds:
        f_a1 = {s: [d for d in items if d["confidence"] >= conf] for s, items in preds_a1.items()}
        f_a2 = {s: [d for d in items if d["confidence"] >= conf] for s, items in preds_a2.items()}

        e_a1 = evaluate_detections(f_a1, gt)
        e_a2 = evaluate_detections(f_a2, gt)

        single_model_sweeps["model_a1"][str(conf)] = {
            "precision": e_a1["precision"], "recall": e_a1["recall"], "f1": e_a1["f1"],
            "tp": e_a1["tp"], "fp": e_a1["fp"], "fn": e_a1["fn"]
        }
        single_model_sweeps["model_a2"][str(conf)] = {
            "precision": e_a2["precision"], "recall": e_a2["recall"], "f1": e_a2["f1"],
            "tp": e_a2["tp"], "fp": e_a2["fp"], "fn": e_a2["fn"]
        }
        print(f"  Conf={conf:.2f} | A1: P={e_a1['precision']:.3f} R={e_a1['recall']:.3f} F1={e_a1['f1']:.3f} | A2: P={e_a2['precision']:.3f} R={e_a2['recall']:.3f} F1={e_a2['f1']:.3f}")

    # 2. Ensemble Parameter Sweep: Weights & Fusion IoU
    print("\n[Sweep 2]: Ensemble parameter grid (weights, WBF IoU, single factor)...")
    ensemble_sweeps = {}

    fusion_configs = [
        {"method": "wbf", "weights": [1.0, 1.0], "iou": 0.50, "factor": 0.88, "bonus": 0.06, "name": "WBF Original [1.0, 1.0] iou=0.50"},
        {"method": "wbf", "weights": [1.0, 1.0], "iou": 0.55, "factor": 0.88, "bonus": 0.06, "name": "WBF iou=0.55 factor=0.88"},
        {"method": "wbf", "weights": [1.0, 1.0], "iou": 0.45, "factor": 0.88, "bonus": 0.06, "name": "WBF iou=0.45 factor=0.88"},
        {"method": "wbf", "weights": [0.8, 1.2], "iou": 0.50, "factor": 0.88, "bonus": 0.06, "name": "WBF Weighted-A2 [0.8, 1.2] iou=0.50"},
        {"method": "wbf", "weights": [0.7, 1.3], "iou": 0.50, "factor": 0.85, "bonus": 0.08, "name": "WBF High-Cap A2 [0.7, 1.3] iou=0.50"},
        {"method": "nms", "weights": [1.0, 1.0], "iou": 0.50, "factor": 1.00, "bonus": 0.00, "name": "Class-Aware NMS iou=0.50"},
        {"method": "nms", "weights": [0.8, 1.2], "iou": 0.45, "factor": 1.00, "bonus": 0.00, "name": "Class-Aware NMS [0.8, 1.2] iou=0.45"},
    ]

    best_f1 = 0.0
    best_config_name = ""

    for cfg in fusion_configs:
        ens = EnsembleDetector(
            models=[m_a1, m_a2],
            model_names=["Model A1", "Model A2"],
            model_weights=cfg["weights"],
            fusion_iou_threshold=cfg["iou"],
            single_model_factor=cfg["factor"],
            consensus_bonus=cfg["bonus"],
            fusion_method=cfg["method"],
        )

        for conf in [0.30, 0.35, 0.40]:
            fused_preds = {}
            for s in stems:
                fused = ens.fuse_predictions([preds_a1[s], preds_a2[s]], confidence_threshold=conf)
                fused_preds[s] = [
                    {"class_id": d.class_id, "label": d.label, "confidence": d.confidence, "box": (d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max)}
                    for d in fused
                ]

            ev = evaluate_detections(fused_preds, gt)
            key = f"{cfg['name']} @ conf={conf:.2f}"
            ensemble_sweeps[key] = {
                "config": cfg, "conf": conf,
                "precision": ev["precision"], "recall": ev["recall"], "f1": ev["f1"],
                "tp": ev["tp"], "fp": ev["fp"], "fn": ev["fn"],
                "class_confusions": ev["class_confusion_errors"],
                "duplicate_fps": ev["duplicate_fps"]
            }
            print(f"  {key:48s} | P={ev['precision']:.4f} R={ev['recall']:.4f} F1={ev['f1']:.4f} (TP={ev['tp']}, FP={ev['fp']}, Dup={ev['duplicate_fps']})")

            if ev["f1"] > best_f1:
                best_f1 = ev["f1"]
                best_config_name = key

    # 3. Test Cross-Class Conflict Resolution on best WBF configuration
    print("\n[Sweep 3]: Testing Cross-Class Conflict Suppression on WBF...")
    cross_sweeps = {}
    best_wbf_ens = EnsembleDetector(
        models=[m_a1, m_a2],
        model_names=["Model A1", "Model A2"],
        model_weights=[0.8, 1.2],
        fusion_iou_threshold=0.50,
        single_model_factor=0.88,
        consensus_bonus=0.06,
        fusion_method="wbf",
    )

    for cross_iou in [0.70, 0.75, 0.80, 0.85]:
        for conf in [0.30, 0.35, 0.40]:
            cleaned_preds = {}
            for s in stems:
                fused = best_wbf_ens.fuse_predictions([preds_a1[s], preds_a2[s]], confidence_threshold=conf)
                fused_dicts = [
                    {"class_id": d.class_id, "label": d.label, "confidence": d.confidence, "box": (d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max)}
                    for d in fused
                ]
                resolved = resolve_cross_class_conflicts(fused_dicts, cross_iou_threshold=cross_iou)
                cleaned_preds[s] = resolved

            ev_clean = evaluate_detections(cleaned_preds, gt)
            k = f"WBF [0.8, 1.2] + Cross-Conflict (IoU={cross_iou:.2f}) @ conf={conf:.2f}"
            cross_sweeps[k] = {
                "cross_iou": cross_iou, "conf": conf,
                "precision": ev_clean["precision"], "recall": ev_clean["recall"], "f1": ev_clean["f1"],
                "tp": ev_clean["tp"], "fp": ev_clean["fp"], "fn": ev_clean["fn"],
                "class_confusions": ev_clean["class_confusion_errors"],
                "duplicate_fps": ev_clean["duplicate_fps"],
                "per_class": ev_clean["per_class"]
            }
            print(f"  {k:60s} | P={ev_clean['precision']:.4f} R={ev_clean['recall']:.4f} F1={ev_clean['f1']:.4f} (TP={ev_clean['tp']}, FP={ev_clean['fp']}, Dup={ev_clean['duplicate_fps']})")

    return {
        "single_model_sweeps": single_model_sweeps,
        "ensemble_sweeps": ensemble_sweeps,
        "cross_class_sweeps": cross_sweeps,
        "best_baseline_wbf": best_config_name,
        "best_baseline_wbf_f1": best_f1,
    }


if __name__ == "__main__":
    run_comprehensive_audit()

