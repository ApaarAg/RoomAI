"""Phase 17: Final Holdout Evaluation and Specialist Selection.

Evaluates on the quarantined 154 test scenes:
1. Model B2 Production Default (checkpoint: exterior_detector_b2_yolo11s_best.pt)
2. Model B6 Expanded Candidate (checkpoint: exterior_detector_b6_expanded_best.pt)
3. Model B6 + CLAHE (Luminance L-channel, clipLimit=2.0, tileGridSize=(8, 8))

Protocol:
- Frozen evaluation threshold: conf = 0.25, iou_thresh = 0.50
- PR-curve integration for AP50 and AP50-95 (IoUs 0.50:0.05:0.95)
- Synchronized CUDA latency (Model-Only vs Full Pipeline, with diverse geometry warm-up)
- Peak allocated and reserved GPU memory
- Scene-level bootstrap (B=1000 resamples) for 95% Confidence Intervals
- Granular error taxonomy analysis across classes
- Visual error gallery generation in reports/phase17_final_test_error_gallery/
- Saves reports/phase17_final_test_metrics.json
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.detection_model import CustomObjectDetector

TEST_IMG_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "test"
TEST_LBL_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "labels" / "test"

B2_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
B6_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two sets of boxes in (xc, yc, w, h) normalized format."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2.0
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2.0
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2.0
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2.0

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2.0
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2.0
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2.0
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2.0

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union_area = area1[:, None] + area2[None, :] - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0)


def apply_clahe(img: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    """Applies adaptive histogram equalization to L channel in LAB color space."""
    np_img = np.array(img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes all-point interpolated Average Precision."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_class_aps(all_gts: Dict[str, List], all_preds: Dict[str, List], iou_thresholds: List[float]) -> Tuple[List[float], List[float]]:
    """Fast vectorized AP computation across all 10 classes and multiple IoU thresholds."""
    ap50_list = []
    ap50_95_list = []

    for cid in range(10):
        # Total positive GTs for this class
        n_pos = sum(sum(1 for g in gts if g[0] == cid) for gts in all_gts.values())
        if n_pos == 0:
            ap50_list.append(0.0)
            ap50_95_list.append(0.0)
            continue

        # Gather all predictions >= 0.05 for class cid
        preds_all = []
        for img_name, preds in all_preds.items():
            for p in preds:
                if p[0] == cid and p[1] >= 0.05:
                    preds_all.append((img_name, p[1], [p[2], p[3], p[4], p[5]]))

        if not preds_all:
            ap50_list.append(0.0)
            ap50_95_list.append(0.0)
            continue

        # Sort descending by confidence
        preds_all.sort(key=lambda x: x[1], reverse=True)
        nd = len(preds_all)

        # Precompute per-image GT boxes and prediction mappings
        img_gt_boxes = {}
        for img_name, gts in all_gts.items():
            boxes = [g[1:5] for g in gts if g[0] == cid]
            img_gt_boxes[img_name] = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

        # Precompute IoU matrices for each prediction against image GTs
        pred_iou_with_gts = []
        for img_name, conf, p_box in preds_all:
            g_boxes = img_gt_boxes[img_name]
            if len(g_boxes) > 0:
                p_arr = np.array([p_box], dtype=np.float32)
                ious = box_iou_matrix(p_arr, g_boxes)[0]
            else:
                ious = np.zeros(0, dtype=np.float32)
            pred_iou_with_gts.append((img_name, ious))

        # Evaluate at each IoU threshold
        aps_for_class = []
        for iou_th in iou_thresholds:
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            matched_gts = {img_name: set() for img_name in all_gts.keys()}

            for idx, (img_name, ious) in enumerate(pred_iou_with_gts):
                if len(ious) > 0:
                    best_iou = 0.0
                    best_g_idx = -1
                    for g_idx, iou_val in enumerate(ious):
                        if g_idx not in matched_gts[img_name] and iou_val > best_iou:
                            best_iou = iou_val
                            best_g_idx = g_idx

                    if best_iou >= iou_th and best_g_idx != -1:
                        tp[idx] = 1.0
                        matched_gts[img_name].add(best_g_idx)
                    else:
                        fp[idx] = 1.0
                else:
                    fp[idx] = 1.0

            cumsum_fp = np.cumsum(fp)
            cumsum_tp = np.cumsum(tp)
            recalls = cumsum_tp / n_pos
            precisions = cumsum_tp / np.maximum(cumsum_tp + cumsum_fp, np.finfo(np.float64).eps)
            aps_for_class.append(compute_ap(recalls, precisions))

        ap50_list.append(round(float(aps_for_class[0]), 4))
        ap50_95_list.append(round(float(np.mean(aps_for_class)), 4))

    return ap50_list, ap50_95_list


def evaluate_fixed_threshold(all_gts: Dict[str, List], all_preds: Dict[str, List], threshold: float = 0.25, iou_thresh: float = 0.50):
    """Computes exact TP, FP, FN, P, R, F1 at fixed evaluation threshold (conf=0.25, IoU=0.50)."""
    tp_cnt = {cid: 0 for cid in range(10)}
    fp_cnt = {cid: 0 for cid in range(10)}
    fn_cnt = {cid: 0 for cid in range(10)}
    matched_ious = {cid: [] for cid in range(10)}

    scene_counts = {}

    for img_name, gts in all_gts.items():
        preds = all_preds.get(img_name, [])
        s_tp = {cid: 0 for cid in range(10)}
        s_fp = {cid: 0 for cid in range(10)}
        s_fn = {cid: 0 for cid in range(10)}

        for cid in range(10):
            c_gts = [g for g in gts if g[0] == cid]
            c_preds = [p for p in preds if p[0] == cid and p[1] >= threshold]
            c_preds.sort(key=lambda x: x[1], reverse=True)

            if c_gts and c_preds:
                p_arr = np.array([[p[2], p[3], p[4], p[5]] for p in c_preds], dtype=np.float32)
                g_arr = np.array([g[1:5] for g in c_gts], dtype=np.float32)
                iou_mat = box_iou_matrix(p_arr, g_arr)

                matched_gt = set()
                for p_idx in range(len(c_preds)):
                    ious = iou_mat[p_idx]
                    best_iou = 0.0
                    best_g_idx = -1
                    for g_idx in range(len(c_gts)):
                        if g_idx not in matched_gt and ious[g_idx] > best_iou:
                            best_iou = ious[g_idx]
                            best_g_idx = g_idx
                    if best_iou >= iou_thresh and best_g_idx != -1:
                        tp_cnt[cid] += 1
                        s_tp[cid] += 1
                        matched_gt.add(best_g_idx)
                        matched_ious[cid].append(float(best_iou))
                    else:
                        fp_cnt[cid] += 1
                        s_fp[cid] += 1
                fn_val = len(c_gts) - len(matched_gt)
                fn_cnt[cid] += fn_val
                s_fn[cid] += fn_val
            else:
                fp_cnt[cid] += len(c_preds)
                s_fp[cid] += len(c_preds)
                fn_cnt[cid] += len(c_gts)
                s_fn[cid] += len(c_gts)

        scene_counts[img_name] = {"tp": s_tp, "fp": s_fp, "fn": s_fn}

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
        gt_count = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious[cid])) if matched_ious[cid] else 0.0
        f1_list.append(f1)
        per_class[cname] = {
            "gt_count": gt_count,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
        }

    macro_f1 = float(np.mean(f1_list))
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class,
        "scene_counts": scene_counts,
    }


def run_fast_bootstrap(scene_counts: Dict[str, Dict], n_bootstraps: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Blazing fast scene-level bootstrap using precomputed per-scene TP/FP/FN vectors."""
    rng = np.random.default_rng(seed)
    scene_keys = list(scene_counts.keys())
    n_scenes = len(scene_keys)

    # Convert scene counts to integer matrices: shape (n_scenes, 10)
    tp_mat = np.array([[scene_counts[k]["tp"][cid] for cid in range(10)] for k in scene_keys], dtype=np.int32)
    fp_mat = np.array([[scene_counts[k]["fp"][cid] for cid in range(10)] for k in scene_keys], dtype=np.int32)
    fn_mat = np.array([[scene_counts[k]["fn"][cid] for cid in range(10)] for k in scene_keys], dtype=np.int32)

    # Sample scene indices with replacement: shape (n_bootstraps, n_scenes)
    sampled_indices = rng.choice(n_scenes, size=(n_bootstraps, n_scenes), replace=True)

    macro_f1s = np.zeros(n_bootstraps, dtype=np.float32)
    micro_f1s = np.zeros(n_bootstraps, dtype=np.float32)
    porch_f1s = np.zeros(n_bootstraps, dtype=np.float32)

    for b in range(n_bootstraps):
        idx = sampled_indices[b]
        b_tp = tp_mat[idx].sum(axis=0)
        b_fp = fp_mat[idx].sum(axis=0)
        b_fn = fn_mat[idx].sum(axis=0)

        p = np.where(b_tp + b_fp > 0, b_tp / np.maximum(b_tp + b_fp, 1e-9), 0.0)
        r = np.where(b_tp + b_fn > 0, b_tp / np.maximum(b_tp + b_fn, 1e-9), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)

        macro_f1s[b] = float(np.mean(f1))
        porch_f1s[b] = float(f1[6])  # Class 6: porch

        tot_tp = b_tp.sum()
        tot_fp = b_fp.sum()
        tot_fn = b_fn.sum()
        mp = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        mr = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        micro_f1s[b] = float(2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0)

    return {
        "macro_f1_ci95": [round(float(np.percentile(macro_f1s, 2.5)), 4), round(float(np.percentile(macro_f1s, 97.5)), 4)],
        "micro_f1_ci95": [round(float(np.percentile(micro_f1s, 2.5)), 4), round(float(np.percentile(micro_f1s, 97.5)), 4)],
        "porch_f1_ci95": [round(float(np.percentile(porch_f1s, 2.5)), 4), round(float(np.percentile(porch_f1s, 97.5)), 4)],
    }


def analyze_error_taxonomy(all_gts: Dict[str, List], all_preds: Dict[str, List]) -> Dict[str, Any]:
    """Granular error taxonomy breakdown by class on the test set."""
    taxonomy = {}

    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        n_gt = sum(sum(1 for g in gts if g[0] == cid) for gts in all_gts.values())

        missed_fn = 0
        low_conf_fn = 0
        loc_failure_cnt = 0
        class_confusion_cnt = 0
        bg_fp_cnt = 0

        for img_name, gts in all_gts.items():
            preds = all_preds.get(img_name, [])
            c_gts = [g for g in gts if g[0] == cid]
            if not c_gts:
                continue

            g_arr = np.array([g[1:5] for g in c_gts], dtype=np.float32)

            for g_idx, g_box in enumerate(g_arr):
                high_conf_preds = [p for p in preds if p[0] == cid and p[1] >= 0.25]
                matched_tp = False
                if high_conf_preds:
                    p_arr = np.array([p[2:6] for p in high_conf_preds], dtype=np.float32)
                    ious = box_iou_matrix(np.array([g_box]), p_arr)[0]
                    if np.any(ious >= 0.50):
                        matched_tp = True

                if matched_tp:
                    continue

                # Check low confidence detection (0.10 <= conf < 0.25)
                low_conf_preds = [p for p in preds if p[0] == cid and 0.10 <= p[1] < 0.25]
                matched_low = False
                if low_conf_preds:
                    p_arr = np.array([p[2:6] for p in low_conf_preds], dtype=np.float32)
                    ious = box_iou_matrix(np.array([g_box]), p_arr)[0]
                    if np.any(ious >= 0.50):
                        matched_low = True

                if matched_low:
                    low_conf_fn += 1
                    continue

                # Check localization failure (0.10 <= IoU < 0.50)
                matched_loc = False
                if high_conf_preds:
                    p_arr = np.array([p[2:6] for p in high_conf_preds], dtype=np.float32)
                    ious = box_iou_matrix(np.array([g_box]), p_arr)[0]
                    if np.any((ious >= 0.10) & (ious < 0.50)):
                        matched_loc = True

                if matched_loc:
                    loc_failure_cnt += 1
                    continue

                missed_fn += 1

        # Audit False Positives
        for img_name, preds in all_preds.items():
            gts = all_gts.get(img_name, [])
            c_preds = [p for p in preds if p[0] == cid and p[1] >= 0.25]
            if not c_preds:
                continue

            c_gts = [g for g in gts if g[0] == cid]
            other_gts = [g for g in gts if g[0] != cid]

            g_same_arr = np.array([g[1:5] for g in c_gts], dtype=np.float32) if c_gts else np.zeros((0, 4), dtype=np.float32)
            g_other_arr = np.array([g[1:5] for g in other_gts], dtype=np.float32) if other_gts else np.zeros((0, 4), dtype=np.float32)

            for p in c_preds:
                p_arr = np.array([[p[2], p[3], p[4], p[5]]], dtype=np.float32)

                # Check if TP
                if len(g_same_arr) > 0 and np.any(box_iou_matrix(p_arr, g_same_arr)[0] >= 0.50):
                    continue

                # Check class confusion (overlap with another class >= 0.35)
                if len(g_other_arr) > 0 and np.any(box_iou_matrix(p_arr, g_other_arr)[0] >= 0.35):
                    class_confusion_cnt += 1
                else:
                    bg_fp_cnt += 1

        taxonomy[cname] = {
            "total_gt": n_gt,
            "missed_objects_fn": missed_fn,
            "low_confidence_fn": low_conf_fn,
            "localization_failure_fn": loc_failure_cnt,
            "class_confusion_fp": class_confusion_cnt,
            "background_fp": bg_fp_cnt,
        }

    return taxonomy


def main():
    print("=" * 85)
    print("RoomAI Phase 17: Final Quarantined Test Evaluation & Specialist Selection")
    print("=" * 85)

    # 1. Preflight Verification
    print("\n1. Preflight Verification...")
    test_images = sorted(list(TEST_IMG_DIR.glob("*.jpg")))
    test_labels = sorted(list(TEST_LBL_DIR.glob("*.txt")))
    print(f"  Quarantined Test Images: {len(test_images)} in {TEST_IMG_DIR}")
    print(f"  Quarantined Test Labels: {len(test_labels)} in {TEST_LBL_DIR}")
    assert len(test_images) == 154, f"Expected 154 test images, got {len(test_images)}"
    assert len(test_labels) == 154, f"Expected 154 test labels, got {len(test_labels)}"

    b2_hash = hashlib.md5(B2_CKPT.read_bytes()).hexdigest()
    b6_hash = hashlib.md5(B6_CKPT.read_bytes()).hexdigest()
    print(f"  B2 Checkpoint: {B2_CKPT.name} | MD5: {b2_hash}")
    print(f"  B6 Checkpoint: {B6_CKPT.name} | MD5: {b6_hash}")
    assert b2_hash == "14a931dd0adabc16cbd7b28ed1d17f66", "B2 hash mismatch!"
    assert b6_hash == "f5ceea56b0610ea14b647014e48152dc", "B6 hash mismatch!"

    all_gts = {}
    for img_p in test_images:
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        all_gts[img_p.name] = boxes

    total_test_boxes = sum(len(b) for b in all_gts.values())
    print(f"  Total ground truth boxes in test set: {total_test_boxes}")

    configurations = [
        ("Config 1: Model B2 Production Default", B2_CKPT, False),
        ("Config 2: Model B6 Expanded Candidate", B6_CKPT, False),
        ("Config 3: Model B6 + CLAHE (Luminance L)", B6_CKPT, True),
    ]

    results_all = {}
    all_predictions_map = {}
    timing_benchmarks = {}

    gallery_dir = REPO_ROOT / "reports" / "phase17_final_test_error_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    porch_test_imgs = [img_p for img_p in test_images if any(g[0] == 6 for g in all_gts[img_p.name])][:4]

    iou_thresholds = [round(0.50 + 0.05 * k, 2) for k in range(10)]

    for cfg_name, ckpt_p, use_clahe in configurations:
        print(f"\n" + "-" * 85)
        print(f"Evaluating {cfg_name} on Quarantined Test Set...")
        print("-" * 85)

        detector = CustomObjectDetector({"checkpoint_path": str(ckpt_p), "device": "cuda:0", "confidence_threshold": 0.05})
        detector.load_model()
        torch.cuda.synchronize()

        # Warmup on diverse aspect ratios
        for w_p in test_images[:10]:
            with Image.open(w_p) as w_im:
                im_w = w_im.convert("RGB")
                if use_clahe:
                    im_w = apply_clahe(im_w)
                detector.predict(im_w, confidence_threshold=0.25)
        torch.cuda.synchronize()

        # Run synchronized benchmark and collect predictions
        torch.cuda.reset_peak_memory_stats(0)
        model_latencies = []
        pipeline_latencies = []
        preds_dict = {}

        for img_p in test_images:
            t_pipe_0 = time.perf_counter()
            with Image.open(img_p) as raw_im:
                im = raw_im.convert("RGB")
                if use_clahe:
                    im = apply_clahe(im)

                torch.cuda.synchronize()
                t_model_0 = time.perf_counter()
                resp = detector.predict(im, confidence_threshold=0.05)
                torch.cuda.synchronize()
                t_model_1 = time.perf_counter()

            t_pipe_1 = time.perf_counter()

            model_latencies.append((t_model_1 - t_model_0) * 1000.0)
            pipeline_latencies.append((t_pipe_1 - t_pipe_0) * 1000.0)

            img_preds = []
            for d in resp.detections:
                bx = d.box
                xc = (bx.x_min + bx.x_max) / 2.0
                yc = (bx.y_min + bx.y_max) / 2.0
                bw = bx.x_max - bx.x_min
                bh = bx.y_max - bx.y_min
                img_preds.append((d.class_id, d.confidence, xc, yc, bw, bh))
            preds_dict[img_p.name] = img_preds

        all_predictions_map[cfg_name] = preds_dict

        peak_vram_alloc_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
        peak_vram_res_mb = round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2)

        # 1. Compute Fixed Threshold Metrics (conf=0.25, IoU=0.50)
        fixed_metrics = evaluate_fixed_threshold(all_gts, preds_dict, threshold=0.25, iou_thresh=0.50)

        # 2. Compute Vectorized AP50 and AP50-95
        ap50_list, ap50_95_list = evaluate_class_aps(all_gts, preds_dict, iou_thresholds)
        mAP50 = float(np.mean(ap50_list))
        mAP50_95 = float(np.mean(ap50_95_list))

        # Add APs to per_class dict
        for cid in range(10):
            cname = CANONICAL_CLASSES[cid]
            fixed_metrics["per_class"][cname]["ap50"] = ap50_list[cid]
            fixed_metrics["per_class"][cname]["ap50_95"] = ap50_95_list[cid]

        fixed_metrics["mAP50"] = round(mAP50, 4)
        fixed_metrics["mAP50_95"] = round(mAP50_95, 4)

        # 3. Fast Scene-Level Bootstrap
        boot_cis = run_fast_bootstrap(fixed_metrics["scene_counts"], n_bootstraps=1000, seed=42)

        # 4. Error Taxonomy
        tax = analyze_error_taxonomy(all_gts, preds_dict)

        # Remove raw scene_counts from JSON output
        scene_counts = fixed_metrics.pop("scene_counts")

        timing_benchmarks[cfg_name] = {
            "model_only_latency_ms": {
                "median": round(float(np.median(model_latencies)), 2),
                "p90": round(float(np.percentile(model_latencies, 90)), 2),
                "p95": round(float(np.percentile(model_latencies, 95)), 2),
                "mean": round(float(np.mean(model_latencies)), 2),
            },
            "full_pipeline_latency_ms": {
                "median": round(float(np.median(pipeline_latencies)), 2),
                "p90": round(float(np.percentile(pipeline_latencies, 90)), 2),
                "p95": round(float(np.percentile(pipeline_latencies, 95)), 2),
                "mean": round(float(np.mean(pipeline_latencies)), 2),
            },
            "peak_vram_allocated_mb": peak_vram_alloc_mb,
            "peak_vram_reserved_mb": peak_vram_res_mb,
            "steady_state_fps": round(1000.0 / float(np.median(model_latencies)), 1),
        }

        results_all[cfg_name] = {
            "checkpoint": ckpt_p.name,
            "use_clahe": use_clahe,
            "metrics": fixed_metrics,
            "bootstrap_ci95": boot_cis,
            "error_taxonomy": tax,
            "timing": timing_benchmarks[cfg_name],
        }

        pm = fixed_metrics["per_class"]["porch"]
        print(f"\n  Results for {cfg_name}:")
        print(f"    Macro-F1 : {fixed_metrics['macro_f1']:.4f} (95% CI: [{boot_cis['macro_f1_ci95'][0]}, {boot_cis['macro_f1_ci95'][1]}])")
        print(f"    Micro-F1 : {fixed_metrics['micro_f1']:.4f} (95% CI: [{boot_cis['micro_f1_ci95'][0]}, {boot_cis['micro_f1_ci95'][1]}])")
        print(f"    mAP50    : {fixed_metrics['mAP50']:.4f} | mAP50-95: {fixed_metrics['mAP50_95']:.4f}")
        print(f"    Porch    : F1={pm['f1']:.4f} (P={pm['precision']:.4f}, R={pm['recall']:.4f}) | TP={pm['tp']}, FP={pm['fp']}, FN={pm['fn']}")
        print(f"    Latency  : Model Median={timing_benchmarks[cfg_name]['model_only_latency_ms']['median']} ms | Pipeline Median={timing_benchmarks[cfg_name]['full_pipeline_latency_ms']['median']} ms")
        print(f"    VRAM     : Allocated={peak_vram_alloc_mb} MB | Reserved={peak_vram_res_mb} MB")

    # Generate Gallery Visualizations
    print("\nGenerating final test visual error gallery...")
    cfg_short_tags = {
        "Config 1: Model B2 Production Default": "b2_prod",
        "Config 2: Model B6 Expanded Candidate": "b6_cand",
        "Config 3: Model B6 + CLAHE (Luminance L)": "b6_clahe",
    }
    for sample_p in porch_test_imgs:
        for cfg_name, ckpt_p, use_clahe in configurations:
            tag = cfg_short_tags[cfg_name]
            with Image.open(sample_p) as raw_im:
                vis_im = raw_im.convert("RGB")
                if use_clahe:
                    vis_im = apply_clahe(vis_im)
                w_i, h_i = vis_im.size
                draw = ImageDraw.Draw(vis_im)

                # Ground truth in green
                for g in all_gts[sample_p.name]:
                    if g[0] == 6:  # porch
                        xmin = int((g[1] - g[3] / 2.0) * w_i)
                        ymin = int((g[2] - g[4] / 2.0) * h_i)
                        xmax = int((g[1] + g[3] / 2.0) * w_i)
                        ymax = int((g[2] + g[4] / 2.0) * h_i)
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00", width=4)
                        draw.text((xmin + 4, ymin + 4), "GT: porch", fill="#00FF00")

                # Predictions
                c_preds = all_predictions_map[cfg_name].get(sample_p.name, [])
                for p in c_preds:
                    if p[0] == 6 and p[1] >= 0.20:
                        xmin = int((p[2] - p[4] / 2.0) * w_i)
                        ymin = int((p[3] - p[5] / 2.0) * h_i)
                        xmax = int((p[2] + p[4] / 2.0) * w_i)
                        ymax = int((p[3] + p[5] / 2.0) * h_i)
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF00FF", width=3)
                        draw.text((xmin + 4, ymax - 18), f"porch {p[1]:.2f}", fill="#FFFF00")

                out_p = gallery_dir / f"{tag}_{sample_p.stem}.jpg"
                vis_im.save(out_p, quality=90)
                print(f"  Saved {out_p.name}")

    # Save reports/phase17_final_test_metrics.json
    out_json_path = REPO_ROOT / "reports" / "phase17_final_test_metrics.json"
    final_payload = {
        "metadata": {
            "test_manifest_count": len(test_images),
            "total_ground_truth_boxes": total_test_boxes,
            "checkpoints": {
                "b2": {"path": str(B2_CKPT), "md5": b2_hash},
                "b6": {"path": str(B6_CKPT), "md5": b6_hash},
            },
            "evaluation_protocol": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.50,
                "clahe_params": {"clip_limit": 2.0, "tile_grid_size": [8, 8], "space": "LAB_Luminance"},
                "bootstrap_resamples": 1000,
            },
        },
        "results": results_all,
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[SAVED] Final Test Metrics saved to: {out_json_path}")

    # Comparative Summary Table
    print("\n" + "=" * 135)
    print(f"{'Configuration':<40} | {'Macro-F1 [95% CI]':<26} | {'mAP50':<8} | {'mAP50-95':<9} | {'Porch F1':<10} | {'Model Med':<10} | {'Pipe Med':<10} | {'Alloc VRAM':<10}")
    print("-" * 135)
    for cfg_name, d in results_all.items():
        mf1 = d["metrics"]["macro_f1"]
        ci = d["bootstrap_ci95"]["macro_f1_ci95"]
        ci_str = f"{mf1:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        m50 = d["metrics"]["mAP50"]
        m5095 = d["metrics"]["mAP50_95"]
        pf1 = d["metrics"]["per_class"]["porch"]["f1"]
        m_med = d["timing"]["model_only_latency_ms"]["median"]
        p_med = d["timing"]["full_pipeline_latency_ms"]["median"]
        vram = d["timing"]["peak_vram_allocated_mb"]
        print(f"{cfg_name:<40} | {ci_str:<26} | {m50:<8.4f} | {m5095:<9.4f} | {pf1:<10.4f} | {m_med:<10.2f} | {p_med:<10.2f} | {vram:<10.1f}")
    print("=" * 135)


if __name__ == "__main__":
    main()
