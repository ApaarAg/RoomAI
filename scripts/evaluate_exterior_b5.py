"""Phase 7: Locked Test Set Evaluation for Model B5 vs Baseline B2.

Evaluates Model B5 and Baseline B2 on the locked test split:
  data/exterior_b5_unified/dataset.yaml (split='test', 154 images, 9826 annotations).

Protocol:
- Resolution: 640px
- IoU threshold: 0.50
- Confidence threshold: 0.25 uniform
- AP50 and AP50-95 (confidence-independent via Ultralytics val)
- Class-level metrics: GT count, TP, FP, FN, Precision, Recall, F1, Mean IoU
- Aggregate metrics: Macro-F1, Micro-F1, Mean IoU
- Latency: Warmed CUDA model-only vs end-to-end inference across 50 images
- Saves: reports/exterior_b5_metrics.json
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
DATASET_YAML = REPO_ROOT / "data" / "exterior_b5_unified" / "dataset.yaml"
TEST_IMG_DIR = REPO_ROOT / "data" / "exterior_b5_unified" / "images" / "test"
TEST_LBL_DIR = REPO_ROOT / "data" / "exterior_b5_unified" / "labels" / "test"

B2_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
B5_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b5_transfer_best.pt"

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
        gt_count = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mean_iou = float(np.mean(matched_ious[cid])) if matched_ious[cid] else 0.0
        f1_list.append(f1)
        per_class[cname] = {
            "gt_count": gt_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
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
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_class": per_class,
    }


def measure_latencies(model: YOLO, img_paths: List[Path], num_runs: int = 50) -> Dict[str, float]:
    """Measures warmed CUDA model-only vs end-to-end inference latency."""
    subset = img_paths[:num_runs]

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(source=dummy, imgsz=640, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 1. Model-only latency (pre-loaded torch tensor)
    model_times = []
    for p in subset:
        im = cv2.imread(str(p))
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(im_rgb).to(0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = torch.nn.functional.interpolate(tensor, size=(640, 640), mode="bilinear", align_corners=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(source=tensor, imgsz=640, device=0, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. Pipeline latency (file read -> BGR2RGB -> predict -> bbox extract)
    pipe_times = []
    for p in subset:
        t0 = time.perf_counter()
        res = model.predict(source=str(p), imgsz=640, device=0, verbose=False)[0]
        boxes = res.boxes.xywhn.tolist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pipe_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "model_only_latency_ms": round(float(np.mean(model_times)), 2),
        "pipeline_latency_ms": round(float(np.mean(pipe_times)), 2),
    }


def main():
    print("=== Phase 7: Locked Test Set Evaluation (B2 vs B5) ===")

    # Safeguard check on B2
    with open(B2_CKPT, "rb") as f:
        h2 = hashlib.md5(f.read()).hexdigest()
    assert h2 == "14a931dd0adabc16cbd7b28ed1d17f66", f"B2 modified: {h2}"
    print(f"  [VERIFIED] B2 Checkpoint: {h2}")

    assert B5_CKPT.exists(), f"B5 Checkpoint missing at {B5_CKPT}"
    with open(B5_CKPT, "rb") as f:
        h5 = hashlib.md5(f.read()).hexdigest()
    print(f"  [FOUND] B5 Checkpoint: {h5}")

    test_images = sorted(list(TEST_IMG_DIR.glob("*.jpg")))
    print(f"\nLoaded {len(test_images)} locked test images from {TEST_IMG_DIR}")

    # Load ground truths
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

    models_to_evaluate = {
        "B2 (Production)": B2_CKPT,
        "B5 (Candidate)": B5_CKPT,
    }

    eval_results = {}

    for name, ckpt_p in models_to_evaluate.items():
        print(f"\n--- Evaluating {name} on Locked Test Split ---")
        model = YOLO(str(ckpt_p))

        # 1. Ultralytics val for AP50 and AP50-95
        print("  Running Ultralytics validation...")
        val_res = model.val(
            data=str(DATASET_YAML),
            split="test",
            batch=8,
            imgsz=640,
            device=0,
            verbose=False,
        )
        ap50 = {CANONICAL_CLASSES[i]: round(float(val_res.box.ap50[i]), 4) for i in range(10)}
        ap = {CANONICAL_CLASSES[i]: round(float(val_res.box.ap[i]), 4) for i in range(10)}
        map50 = round(float(val_res.box.map50), 4)
        map95 = round(float(val_res.box.map), 4)

        # 2. Operating threshold evaluation at conf=0.25
        print("  Running inference across 154 test images...")
        all_preds = {}
        for img_p in test_images:
            res = model.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
            preds = []
            for i in range(len(res.boxes)):
                cid = int(res.boxes.cls[i].item())
                conf = float(res.boxes.conf[i].item())
                xywhn = res.boxes.xywhn[i].tolist()
                preds.append((cid, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
            all_preds[img_p.name] = preds

        op_res = evaluate_predictions(all_gts, all_preds, threshold=0.25)

        # 3. Latency measurement
        print("  Measuring latency over 50 test images...")
        lat_res = measure_latencies(model, test_images, num_runs=50)

        eval_results[name] = {
            "checkpoint": str(ckpt_p.name),
            "md5": hashlib.md5(open(ckpt_p, "rb").read()).hexdigest(),
            "mAP50": map50,
            "mAP50_95": map95,
            "macro_f1": op_res["macro_f1"],
            "micro_f1": op_res["micro_f1"],
            "micro_precision": op_res["micro_precision"],
            "micro_recall": op_res["micro_recall"],
            "overall_mean_iou": op_res["overall_mean_iou"],
            "model_only_latency_ms": lat_res["model_only_latency_ms"],
            "pipeline_latency_ms": lat_res["pipeline_latency_ms"],
            "ap50_per_class": ap50,
            "ap_per_class": ap,
            "per_class": op_res["per_class"],
        }
        print(f"  mAP50: {map50:.4f} | mAP50-95: {map95:.4f} | Macro-F1: {op_res['macro_f1']:.4f} | Model Latency: {lat_res['model_only_latency_ms']:.2f} ms")

    # Comparative Summary
    print("\n" + "=" * 105)
    print(f"{'Class':<15} | {'GT':<5} | {'B2 F1 (AP50)':<18} | {'B5 F1 (AP50)':<18} | {'F1 Delta':<10}")
    print("-" * 105)
    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        gt = eval_results["B2 (Production)"]["per_class"][cname]["gt_count"]
        b2_f1 = eval_results["B2 (Production)"]["per_class"][cname]["f1"]
        b2_ap = eval_results["B2 (Production)"]["ap50_per_class"][cname]
        b5_f1 = eval_results["B5 (Candidate)"]["per_class"][cname]["f1"]
        b5_ap = eval_results["B5 (Candidate)"]["ap50_per_class"][cname]
        delta_f1 = b5_f1 - b2_f1
        print(f"{cname:<15} | {gt:<5d} | {b2_f1:.4f} ({b2_ap:.4f})     | {b5_f1:.4f} ({b5_ap:.4f})     | {delta_f1:+.4f}")

    print("-" * 105)
    b2_m = eval_results["B2 (Production)"]
    b5_m = eval_results["B5 (Candidate)"]
    print(f"{'Macro-F1':<15} | {'9826':<5} | {b2_m['macro_f1']:.4f} (mAP={b2_m['mAP50']:.4f}) | {b5_m['macro_f1']:.4f} (mAP={b5_m['mAP50']:.4f}) | {b5_m['macro_f1'] - b2_m['macro_f1']:+.4f}")
    print(f"{'Micro-F1':<15} | {'9826':<5} | {b2_m['micro_f1']:.4f}              | {b5_m['micro_f1']:.4f}              | {b5_m['micro_f1'] - b2_m['micro_f1']:+.4f}")
    print(f"{'Mean IoU':<15} | {'9826':<5} | {b2_m['overall_mean_iou']:.4f}              | {b5_m['overall_mean_iou']:.4f}              | {b5_m['overall_mean_iou'] - b2_m['overall_mean_iou']:+.4f}")
    print(f"{'Latency':<15} | {'50 img':<5} | {b2_m['model_only_latency_ms']:.2f} ms            | {b5_m['model_only_latency_ms']:.2f} ms            | {b5_m['model_only_latency_ms'] - b2_m['model_only_latency_ms']:+.2f} ms")
    print("=" * 105)

    # Save final metrics JSON
    metrics_json = REPO_ROOT / "reports" / "exterior_b5_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[SAVED] Locked test metrics saved to {metrics_json}")


if __name__ == "__main__":
    main()
