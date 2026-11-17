"""Phase 3: Establish a Reliable Baseline for B2, B3, and B4 on the Unified Evaluation Split.

Evaluates B2, B3, and B4 on data/exterior_b5_unified/dataset.yaml (split='val', 154 images, 8504 annotations).
Uses identical protocol:
- Resolution: 640px
- IoU matching threshold: 0.50
- Confidence operating threshold: 0.25
- Confidence-independent AP50 and AP50-95 from Ultralytics val
- Ground-truth instance count, TP, FP, FN, Precision, Recall, F1 per class
- Macro-F1 (unweighted mean across 10 classes) and Micro-F1 (aggregated over all instances)
"""

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
DATASET_YAML = REPO_ROOT / "data" / "exterior_b5_unified" / "dataset.yaml"
VAL_IMG_DIR = REPO_ROOT / "data" / "exterior_b5_unified" / "images" / "val"
VAL_LBL_DIR = REPO_ROOT / "data" / "exterior_b5_unified" / "labels" / "val"

CHECKPOINTS = {
    "B2 (Production)": REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt",
    "B3 (Rollback)": REPO_ROOT / "checkpoints" / "exterior_detector_b3_yolo11s_rfs_best.pt",
    "B4 (Experimental)": REPO_ROOT / "checkpoints" / "exterior_detector_b4_yolo11s_curated_best.pt",
}

EXPECTED_HASHES = {
    "B2 (Production)": "14a931dd0adabc16cbd7b28ed1d17f66",
    "B3 (Rollback)": "2c44291fa642aba3893db001520deba0",
    "B4 (Experimental)": "d2710fba760713e68ea306e029ab651c",
}

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


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
        gt_count = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1)
        per_class[cname] = {
            "gt_count": gt_count,
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
    print("=== Phase 3: Benchmarking B2, B3, and B4 Baselines ===\n")

    # 1. Verify Checkpoints
    for name, p in CHECKPOINTS.items():
        with open(p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        assert h == EXPECTED_HASHES[name], f"Mismatch for {name}: {h}"
        print(f"  [VERIFIED] {name}: {h}")

    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
    print(f"\nLoaded {len(val_images)} validation images from {VAL_IMG_DIR}")

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

    results = {}

    for model_name, ckpt_p in CHECKPOINTS.items():
        print(f"\n--- Evaluating {model_name} ---")
        model = YOLO(str(ckpt_p))

        # Ultralytics val for AP50 and AP50-95
        val_res = model.val(
            data=str(DATASET_YAML),
            split="val",
            batch=8,
            imgsz=640,
            device=0,
            verbose=False,
        )
        ap50 = {CANONICAL_CLASSES[i]: round(float(val_res.box.ap50[i]), 4) for i in range(10)}
        ap = {CANONICAL_CLASSES[i]: round(float(val_res.box.ap[i]), 4) for i in range(10)}
        map50 = round(float(val_res.box.map50), 4)
        map95 = round(float(val_res.box.map), 4)

        # Base inference at conf=0.01 for operating threshold eval
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

        op_eval = evaluate_predictions(all_gts, all_preds, threshold=0.25)

        results[model_name] = {
            "checkpoint": str(ckpt_p.name),
            "mAP50": map50,
            "mAP50_95": map95,
            "macro_f1": op_eval["macro_f1"],
            "micro_f1": op_eval["micro_f1"],
            "micro_precision": op_eval["micro_precision"],
            "micro_recall": op_eval["micro_recall"],
            "ap50_per_class": ap50,
            "ap_per_class": ap,
            "per_class_op": op_eval["per_class"],
        }

        print(f"  mAP50: {map50:.4f} | mAP50-95: {map95:.4f} | Macro-F1 (conf=0.25): {op_eval['macro_f1']:.4f}")

    # Output comparison table
    print("\n" + "=" * 105)
    print(f"{'Class':<15} | {'GT':<5} | {'B2 F1 (AP50)':<18} | {'B3 F1 (AP50)':<18} | {'B4 F1 (AP50)':<18}")
    print("-" * 105)
    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        gt = results["B2 (Production)"]["per_class_op"][cname]["gt_count"]
        b2_f1 = results["B2 (Production)"]["per_class_op"][cname]["f1"]
        b2_ap = results["B2 (Production)"]["ap50_per_class"][cname]
        b3_f1 = results["B3 (Rollback)"]["per_class_op"][cname]["f1"]
        b3_ap = results["B3 (Rollback)"]["ap50_per_class"][cname]
        b4_f1 = results["B4 (Experimental)"]["per_class_op"][cname]["f1"]
        b4_ap = results["B4 (Experimental)"]["ap50_per_class"][cname]
        print(f"{cname:<15} | {gt:<5d} | {b2_f1:.4f} ({b2_ap:.4f})     | {b3_f1:.4f} ({b3_ap:.4f})     | {b4_f1:.4f} ({b4_ap:.4f})")

    print("-" * 105)
    print(f"{'Macro-F1':<15} | {'8504':<5} | {results['B2 (Production)']['macro_f1']:.4f} (mAP={results['B2 (Production)']['mAP50']:.4f}) | {results['B3 (Rollback)']['macro_f1']:.4f} (mAP={results['B3 (Rollback)']['mAP50']:.4f}) | {results['B4 (Experimental)']['macro_f1']:.4f} (mAP={results['B4 (Experimental)']['mAP50']:.4f})")
    print(f"{'Micro-F1':<15} | {'8504':<5} | {results['B2 (Production)']['micro_f1']:.4f}              | {results['B3 (Rollback)']['micro_f1']:.4f}              | {results['B4 (Experimental)']['micro_f1']:.4f}")
    print("=" * 105)

    out_json = REPO_ROOT / "reports" / "exterior_b5_baseline_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline audit metrics to {out_json}")


if __name__ == "__main__":
    main()
