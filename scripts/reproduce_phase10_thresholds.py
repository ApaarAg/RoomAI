"""Stage A: Reproduce Phase 10 Baseline Metrics for Model B4 on Validation Data.

Evaluates Model B4 on data/exterior_v2_curated/images/val (154 images, 8504 annotations):
1. Uniform operating threshold = 0.25
2. Phase 10 candidate per-class thresholds:
   - building: 0.26
   - facade: 0.31
   - window: 0.46
   - door: 0.22
   - roof_cornice: 0.44
   - balcony: 0.24
   - porch: 0.17
   - pillar_column: 0.23
   - shutter_blind: 0.17
   - molding_deco: 0.24

Outputs precision, recall, and F1 per class and computes macro-F1 and micro-F1.
"""

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

import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
VAL_IMG_DIR = REPO_ROOT / "data" / "exterior_v2_curated" / "images" / "val"
VAL_LBL_DIR = REPO_ROOT / "data" / "exterior_v2_curated" / "labels" / "val"

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

PHASE10_THRESHOLDS = {
    0: 0.26,  # building
    1: 0.31,  # facade
    2: 0.46,  # window
    3: 0.22,  # door
    4: 0.44,  # roof_cornice
    5: 0.24,  # balcony
    6: 0.17,  # porch
    7: 0.23,  # pillar_column
    8: 0.17,  # shutter_blind
    9: 0.24,  # molding_deco
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


def evaluate_predictions(
    all_gts: Dict[str, List[Tuple[int, float, float, float, float]]],
    all_preds: Dict[str, List[Tuple[int, float, float, float, float, float]]],
    thresholds: Dict[int, float],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Matches predictions against ground truth using greedy highest-IoU matching."""
    tp_cnt = {cid: 0 for cid in range(10)}
    fp_cnt = {cid: 0 for cid in range(10)}
    fn_cnt = {cid: 0 for cid in range(10)}

    for img_name, gts in all_gts.items():
        preds = all_preds.get(img_name, [])

        for cid in range(10):
            th = thresholds[cid]
            c_gts = [g for g in gts if g[0] == cid]
            c_preds = [p for p in preds if p[0] == cid and p[1] >= th]

            # Sort predictions descending by confidence
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
                else:
                    fp_cnt[cid] += 1

            fn_cnt[cid] += (len(c_gts) - len(matched_gt))

    per_class = {}
    f1_list = []
    total_tp = sum(tp_cnt.values())
    total_fp = sum(fp_cnt.values())
    total_fn = sum(fn_cnt.values())

    for cid in range(10):
        c_name = CANONICAL_CLASSES[cid]
        tp = tp_cnt[cid]
        fp = fp_cnt[cid]
        fn = fn_cnt[cid]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1)
        per_class[c_name] = {
            "class_id": cid,
            "threshold": thresholds[cid],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
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
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_class": per_class,
    }


def main():
    print("=== Stage A: Reproduce Baseline Metrics for Model B4 ===")
    b4_path = REPO_ROOT / "checkpoints" / "exterior_detector_b4_yolo11s_curated_best.pt"
    model = YOLO(str(b4_path))

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
                    if len(parts) == 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        all_gts[img_p.name] = boxes

    # Predict at low confidence (0.01) to allow sweeping any threshold >= 0.01
    print("Running base inference at conf=0.01...")
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

    # 1. Uniform threshold = 0.25
    uniform_025 = {cid: 0.25 for cid in range(10)}
    res_uniform = evaluate_predictions(all_gts, all_preds, uniform_025)

    # 2. Phase 10 per-class thresholds
    res_per_class = evaluate_predictions(all_gts, all_preds, PHASE10_THRESHOLDS)

    print("\n--- RESULTS COMPARISON ---")
    print(f"{'Class':<15} | {'Uniform 0.25 F1':<15} (P / R) | {'Phase 10 Per-Class F1':<22} (P / R) | {'Delta F1':<10}")
    print("-" * 75)
    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        u = res_uniform["per_class"][cname]
        p10 = res_per_class["per_class"][cname]
        diff = p10["f1"] - u["f1"]
        print(f"{cname:<15} | {u['f1']:.4f} ({u['precision']:.3f}/{u['recall']:.3f}) | {p10['f1']:.4f} [th={p10['threshold']:.2f}] ({p10['precision']:.3f}/{p10['recall']:.3f}) | {diff:+.4f}")

    print("-" * 75)
    print(f"{'Macro-F1':<15} | {res_uniform['macro_f1']:.4f}                          | {res_per_class['macro_f1']:.4f}                         | {res_per_class['macro_f1'] - res_uniform['macro_f1']:+.4f}")
    print(f"{'Micro-F1':<15} | {res_uniform['micro_f1']:.4f}                          | {res_per_class['micro_f1']:.4f}                         | {res_per_class['micro_f1'] - res_uniform['micro_f1']:+.4f}")

    # Check porch reproduction specifically
    p10_porch = res_per_class["per_class"]["porch"]
    print(f"\nPhase 10 Porch Reproduction Check:")
    print(f"  Target from Phase 10: F1=0.3922, Precision=0.370, Recall=0.417")
    print(f"  Reproduced in Script: F1={p10_porch['f1']:.4f}, Precision={p10_porch['precision']:.3f}, Recall={p10_porch['recall']:.3f}")
    assert abs(p10_porch['f1'] - 0.3922) < 0.001, f"Mismatch in porch F1: {p10_porch['f1']}"
    print("  [OK] Porch baseline reproduced with exact precision!")


if __name__ == "__main__":
    main()
