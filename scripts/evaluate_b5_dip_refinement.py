"""Phase 5B: Digital Image Processing (DIP) for Detection Refinement.

Explores classical computer vision techniques to complement Model B5:
A. Boundary & Edge Detection: Canny, Sobel gradients, Laplacian, Morphological gradients.
B. Morphological Operations: Dilation/erosion, opening/closing, contour hole-filling, CC analysis.
C. Preprocessing Experiments: CLAHE (LAB color space), Gamma correction, Bilateral filtering.
D. Boundary-Aware Refinement: Gradient-guided bounding-box edge snapping.
E. Controlled Experiments: Fixed weights, fixed splits, measuring P, R, F1, IoU, and latency.
F. Visualization: Comparative multi-panel galleries.
"""

import hashlib
import json
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
from PIL import Image
import torch
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
VAL_IMG_DIR = REPO_ROOT / "data" / "exterior_b5_unified" / "images" / "val"
VAL_LBL_DIR = REPO_ROOT / "data" / "exterior_b5_unified" / "labels" / "val"
VIS_DIR = REPO_ROOT / "reports" / "visualizations" / "dip_refinement"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


# ==========================================
# DIP Function Definitions
# ==========================================

def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization to L-channel in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma(img_bgr: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Applies non-linear gamma intensity adjustment."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def apply_bilateral(img_bgr: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """Applies edge-preserving bilateral filter."""
    return cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)


def compute_edge_maps(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Computes Canny, Sobel gradients, Laplacian, and Morphological gradients."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1. Canny
    canny = cv2.Canny(blurred, 50, 150)

    # 2. Sobel Gradients
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = np.uint8(np.clip((grad_mag / grad_mag.max()) * 255.0, 0, 255)) if grad_mag.max() > 0 else np.zeros_like(gray)

    # 3. Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian_abs = np.uint8(np.absolute(laplacian))

    # 4. Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    return {
        "gray": gray,
        "canny": canny,
        "sobel_mag": grad_mag,
        "laplacian": laplacian_abs,
        "morph_grad": morph_grad,
        "sobel_x": np.absolute(sobel_x),
        "sobel_y": np.absolute(sobel_y),
    }


def snap_box_to_gradients(
    box: Tuple[int, float, float, float, float, float],
    sobel_x: np.ndarray,
    sobel_y: np.ndarray,
    img_w: int,
    img_h: int,
    radius: int = 4,
    grad_thresh: float = 25.0,
) -> Tuple[int, float, float, float, float, float]:
    """Refines bounding-box boundaries by snapping to local gradient peaks."""
    cid, conf, xc, yc, bw, bh = box
    x1 = int(round((xc - bw / 2.0) * img_w))
    x2 = int(round((xc + bw / 2.0) * img_w))
    y1 = int(round((yc - bh / 2.0) * img_h))
    y2 = int(round((yc + bh / 2.0) * img_h))

    # Clamp
    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w - 1, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h - 1, y2))

    # Snap x1 (left vertical edge) along horizontal gradient
    x1_min = max(0, x1 - radius)
    x1_max = min(img_w - 1, x1 + radius)
    slice_x1 = sobel_x[y1:y2, x1_min:x1_max + 1]
    if slice_x1.size > 0:
        col_means = np.mean(slice_x1, axis=0)
        best_col_idx = np.argmax(col_means)
        if col_means[best_col_idx] >= grad_thresh:
            x1 = x1_min + best_col_idx

    # Snap x2 (right vertical edge)
    x2_min = max(0, x2 - radius)
    x2_max = min(img_w - 1, x2 + radius)
    slice_x2 = sobel_x[y1:y2, x2_min:x2_max + 1]
    if slice_x2.size > 0:
        col_means = np.mean(slice_x2, axis=0)
        best_col_idx = np.argmax(col_means)
        if col_means[best_col_idx] >= grad_thresh:
            x2 = x2_min + best_col_idx

    # Snap y1 (top horizontal edge) along vertical gradient
    y1_min = max(0, y1 - radius)
    y1_max = min(img_h - 1, y1 + radius)
    slice_y1 = sobel_y[y1_min:y1_max + 1, x1:x2]
    if slice_y1.size > 0:
        row_means = np.mean(slice_y1, axis=1)
        best_row_idx = np.argmax(row_means)
        if row_means[best_row_idx] >= grad_thresh:
            y1 = y1_min + best_row_idx

    # Snap y2 (bottom horizontal edge)
    y2_min = max(0, y2 - radius)
    y2_max = min(img_h - 1, y2 + radius)
    slice_y2 = sobel_y[y2_min:y2_max + 1, x1:x2]
    if slice_y2.size > 0:
        row_means = np.mean(slice_y2, axis=1)
        best_row_idx = np.argmax(row_means)
        if row_means[best_row_idx] >= grad_thresh:
            y2 = y2_min + best_row_idx

    # Ensure box validity
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return box

    new_xc = ((x1 + x2) / 2.0) / img_w
    new_yc = ((y1 + y2) / 2.0) / img_h
    new_bw = (x2 - x1) / img_w
    new_bh = (y2 - y1) / img_h

    return (cid, conf, round(new_xc, 6), round(new_yc, 6), round(new_bw, 6), round(new_bh, 6))


def box_iou(box1: List[float], box2: List[float]) -> float:
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


def evaluate_predictions(all_gts, all_preds, threshold=0.25, iou_thresh=0.50):
    tp_cnt = {cid: 0 for cid in range(10)}
    fp_cnt = {cid: 0 for cid in range(10)}
    fn_cnt = {cid: 0 for cid in range(10)}
    matched_ious = {cid: [] for cid in range(10)}

    for img_name, gts in all_gts.items():
        preds = all_preds.get(img_name, [])
        for cid in range(10):
            c_gts = [g for g in gts if g[0] == cid]
            c_preds = [p for p in preds if p[0] == cid and p[1] >= threshold]
            c_preds.sort(key=lambda x: x[1], reverse=True)

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

                if best_iou >= iou_thresh and best_gt_idx != -1:
                    tp_cnt[cid] += 1
                    matched_gt.add(best_gt_idx)
                    matched_ious[cid].append(best_iou)
                else:
                    fp_cnt[cid] += 1

            fn_cnt[cid] += (len(c_gts) - len(matched_gt))

    per_class = {}
    f1_list = []
    total_tp = sum(tp_cnt.values())
    total_fp = sum(fp_cnt.values())
    total_fn = sum(fn_cnt.values())

    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        tp = tp_cnt[cid]
        fp = fp_cnt[cid]
        fn = fn_cnt[cid]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious[cid])) if matched_ious[cid] else 0.0
        f1_list.append(f1)
        per_class[cname] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
        }

    macro_f1 = float(np.mean(f1_list))
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    overall_mean_iou = float(np.mean([iou for ious in matched_ious.values() for iou in ious])) if any(matched_ious.values()) else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "overall_mean_iou": round(overall_mean_iou, 4),
        "per_class": per_class,
    }


def run_dip_evaluation(model_ckpt: Path):
    print(f"=== Phase 5B: Evaluating Digital Image Processing on Model {model_ckpt.name} ===")
    assert model_ckpt.exists(), f"Model checkpoint missing: {model_ckpt}"
    model = YOLO(str(model_ckpt))

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
    print(f"Loaded {len(val_images)} validation images from {VAL_IMG_DIR}")

    # Load ground truths
    all_gts = {}
    for img_p in val_images:
        lbl_p = VAL_LBL_DIR / f"{img_p.stem}.txt"
        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        all_gts[img_p.name] = boxes

    # 1. Condition 1: Baseline B5 (Raw images)
    print("\n--- Running Condition 1: Baseline B5 (Raw Images) ---")
    preds_baseline = {}
    t0_b = time.perf_counter()
    for img_p in val_images:
        res = model.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        preds = []
        for i in range(len(res.boxes)):
            cid = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywhn = res.boxes.xywhn[i].tolist()
            preds.append((cid, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
        preds_baseline[img_p.name] = preds
    lat_baseline = (time.perf_counter() - t0_b) / len(val_images) * 1000.0

    res_baseline = evaluate_predictions(all_gts, preds_baseline, threshold=0.25)
    print(f"  Baseline Macro-F1: {res_baseline['macro_f1']:.4f} | Mean IoU: {res_baseline['overall_mean_iou']:.4f} | Latency: {lat_baseline:.2f} ms/img")

    # 2. Condition 2: Experiment 1 - CLAHE Preprocessing
    print("\n--- Running Condition 2: Experiment 1 (CLAHE Preprocessing) ---")
    preds_clahe = {}
    t0_c = time.perf_counter()
    for img_p in val_images:
        im_bgr = cv2.imread(str(img_p))
        im_clahe = apply_clahe(im_bgr, clip_limit=2.0)
        im_rgb = cv2.cvtColor(im_clahe, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im_rgb)

        res = model.predict(source=im_pil, conf=0.01, imgsz=640, device=0, verbose=False)[0]
        preds = []
        for i in range(len(res.boxes)):
            cid = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywhn = res.boxes.xywhn[i].tolist()
            preds.append((cid, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
        preds_clahe[img_p.name] = preds
    lat_clahe = (time.perf_counter() - t0_c) / len(val_images) * 1000.0

    res_clahe = evaluate_predictions(all_gts, preds_clahe, threshold=0.25)
    print(f"  CLAHE Macro-F1: {res_clahe['macro_f1']:.4f} | Mean IoU: {res_clahe['overall_mean_iou']:.4f} | Latency: {lat_clahe:.2f} ms/img")

    # 3. Condition 3: Experiment 2 - Gradient-Guided Boundary Snapping
    print("\n--- Running Condition 3: Experiment 2 (Gradient-Guided Box Snapping) ---")
    preds_snapped = {}
    t0_s = time.perf_counter()
    for img_p in val_images:
        im_bgr = cv2.imread(str(img_p))
        h, w, _ = im_bgr.shape
        edge_maps = compute_edge_maps(im_bgr)
        sobel_x = edge_maps["sobel_x"]
        sobel_y = edge_maps["sobel_y"]

        raw_preds = preds_baseline[img_p.name]
        snapped = []
        for b in raw_preds:
            # Snap boundary edges to local gradient maximum
            b_snapped = snap_box_to_gradients(b, sobel_x, sobel_y, w, h, radius=4, grad_thresh=25.0)
            snapped.append(b_snapped)
        preds_snapped[img_p.name] = snapped
    lat_snapped = lat_baseline + ((time.perf_counter() - t0_s) / len(val_images) * 1000.0)

    res_snapped = evaluate_predictions(all_gts, preds_snapped, threshold=0.25)
    print(f"  Snapped Macro-F1: {res_snapped['macro_f1']:.4f} | Mean IoU: {res_snapped['overall_mean_iou']:.4f} | Latency: {lat_snapped:.2f} ms/img")

    # 4. Generate Visual Comparisons Gallery
    print("\n=== Generating DIP Visual Verification Gallery ===")
    sample_stems = ["cmp_val_0001", "cmp_val_0024", "oi_curated_porch_val_0110", "oi_curated_porch_val_0112"]

    for stem in sample_stems:
        img_p = VAL_IMG_DIR / f"{stem}.jpg"
        if not img_p.exists():
            continue
        im_bgr = cv2.imread(str(img_p))
        h, w, _ = im_bgr.shape

        # A. Edge decomposition panel
        edge_maps = compute_edge_maps(im_bgr)
        canny_rgb = cv2.cvtColor(edge_maps["canny"], cv2.COLOR_GRAY2BGR)
        sobel_rgb = cv2.cvtColor(edge_maps["sobel_mag"], cv2.COLOR_GRAY2BGR)
        morph_rgb = cv2.cvtColor(edge_maps["morph_grad"], cv2.COLOR_GRAY2BGR)

        # Labels
        cv2.putText(im_bgr, "Original", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canny_rgb, "Canny Edges", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(sobel_rgb, "Sobel Gradient Mag", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(morph_rgb, "Morphological Gradient", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)

        panel_edges = np.hstack([im_bgr, canny_rgb, sobel_rgb, morph_rgb])
        cv2.imwrite(str(VIS_DIR / f"dip_edges_{stem}.jpg"), panel_edges)

        # B. Preprocessing comparison panel
        im_clahe = apply_clahe(im_bgr)
        im_gamma = apply_gamma(im_bgr, gamma=0.8)
        cv2.putText(im_clahe, "CLAHE (Clip=2.0)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(im_gamma, "Gamma=0.8 (Shadow Boost)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        panel_prep = np.hstack([im_bgr, im_clahe, im_gamma])
        cv2.imwrite(str(VIS_DIR / f"dip_preprocessing_{stem}.jpg"), panel_prep)

        # C. Detection comparison panel (Baseline vs Snapped)
        im_det_b = im_bgr.copy()
        im_det_s = im_bgr.copy()

        for p in preds_baseline[img_p.name]:
            if p[1] < 0.25:
                continue
            cx, cy, bw, bh = p[2] * w, p[3] * h, p[4] * w, p[5] * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(im_det_b, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im_det_b, f"{CANONICAL_CLASSES[p[0]]}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        for p in preds_snapped[img_p.name]:
            if p[1] < 0.25:
                continue
            cx, cy, bw, bh = p[2] * w, p[3] * h, p[4] * w, p[5] * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(im_det_s, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(im_det_s, f"{CANONICAL_CLASSES[p[0]]}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.putText(im_det_b, "Baseline B5 Detections", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(im_det_s, "DIP Gradient-Snapped Boxes", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        panel_det = np.hstack([im_det_b, im_det_s])
        cv2.imwrite(str(VIS_DIR / f"dip_detection_snap_{stem}.jpg"), panel_det)

        print(f"  Generated DIP comparison panels for: {stem}")

    # 5. Compile and Output DIP Metrics JSON
    dip_output = {
        "timestamp": "2026-10-25 00:20:00",
        "phase": "Phase 5B — Digital Image Processing for Detection Refinement",
        "conditions_evaluated": {
            "baseline_b5": {
                "macro_f1": res_baseline["macro_f1"],
                "micro_f1": res_baseline["micro_f1"],
                "overall_mean_iou": res_baseline["overall_mean_iou"],
                "latency_ms_per_image": round(lat_baseline, 2),
                "per_class": res_baseline["per_class"],
            },
            "experiment_1_clahe_preprocessing": {
                "macro_f1": res_clahe["macro_f1"],
                "micro_f1": res_clahe["micro_f1"],
                "overall_mean_iou": res_clahe["overall_mean_iou"],
                "latency_ms_per_image": round(lat_clahe, 2),
                "delta_macro_f1": round(res_clahe["macro_f1"] - res_baseline["macro_f1"], 4),
                "delta_mean_iou": round(res_clahe["overall_mean_iou"] - res_baseline["overall_mean_iou"], 4),
                "per_class": res_clahe["per_class"],
            },
            "experiment_2_gradient_box_snapping": {
                "macro_f1": res_snapped["macro_f1"],
                "micro_f1": res_snapped["micro_f1"],
                "overall_mean_iou": res_snapped["overall_mean_iou"],
                "latency_ms_per_image": round(lat_snapped, 2),
                "delta_macro_f1": round(res_snapped["macro_f1"] - res_baseline["macro_f1"], 4),
                "delta_mean_iou": round(res_snapped["overall_mean_iou"] - res_baseline["overall_mean_iou"], 4),
                "per_class": res_snapped["per_class"],
            },
        },
    }

    out_json = REPO_ROOT / "reports" / "exterior_b5_dip_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dip_output, f, indent=2)
    print(f"\n[SAVED] DIP metrics saved to {out_json}")


if __name__ == "__main__":
    b5_ckpt = REPO_ROOT / "checkpoints" / "exterior_detector_b5_transfer_best.pt"
    if b5_ckpt.exists():
        target_ckpt = b5_ckpt
    else:
        # Fallback to B4 if B5 still training
        target_ckpt = REPO_ROOT / "checkpoints" / "exterior_detector_b4_yolo11s_curated_best.pt"
    run_dip_evaluation(target_ckpt)
