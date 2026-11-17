"""Audit and reproduce Phase 20 / Model B7 metrics on the authorized 154 validation scenes."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

B6_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
B7_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b7_pilot_best.pt"
VAL_DIR = REPO_ROOT / "data" / "pilot_b7_dataset" / "images" / "val"
LBL_DIR = REPO_ROOT / "data" / "pilot_b7_dataset" / "labels" / "val"
DATASET_YAML = REPO_ROOT / "data" / "pilot_b7_dataset" / "dataset.yaml"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def box_iou(box1, box2):
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


def evaluate_scenes(all_gts, all_preds, threshold=0.25, iou_thresh=0.50):
    scene_metrics = {}
    for img_name, gts in all_gts.items():
        preds = all_preds.get(img_name, [])
        tp = {cid: 0 for cid in range(10)}
        fp = {cid: 0 for cid in range(10)}
        fn = {cid: 0 for cid in range(10)}

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
                    tp[cid] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[cid] += 1

            fn[cid] += (len(c_gts) - len(matched_gt))

        scene_metrics[img_name] = {"tp": tp, "fp": fp, "fn": fn}
    return scene_metrics


def aggregate(scene_metrics):
    tp_tot = {cid: 0 for cid in range(10)}
    fp_tot = {cid: 0 for cid in range(10)}
    fn_tot = {cid: 0 for cid in range(10)}

    for s in scene_metrics.values():
        for cid in range(10):
            tp_tot[cid] += s["tp"][cid]
            fp_tot[cid] += s["fp"][cid]
            fn_tot[cid] += s["fn"][cid]

    per_class = {}
    f1_list = []
    tot_tp = sum(tp_tot.values())
    tot_fp = sum(fp_tot.values())
    tot_fn = sum(fn_tot.values())

    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        tp = tp_tot[cid]
        fp = fp_tot[cid]
        fn = fn_tot[cid]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1)
        per_class[cname] = {
            "gt": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)
        }

    macro_f1 = float(np.mean(f1_list))
    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class
    }


def compute_bootstrap(scenes_b6, scenes_b7, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    keys = list(scenes_b6.keys())
    n = len(keys)

    diff_macro, diff_micro, diff_porch = [], [], []
    for _ in range(n_boot):
        sample = rng.choice(keys, size=n, replace=True)
        m6 = aggregate({k: scenes_b6[k] for k in sample})
        m7 = aggregate({k: scenes_b7[k] for k in sample})
        diff_macro.append(m7["macro_f1"] - m6["macro_f1"])
        diff_micro.append(m7["micro_f1"] - m6["micro_f1"])
        diff_porch.append(m7["per_class"]["porch"]["f1"] - m6["per_class"]["porch"]["f1"])

    return {
        "macro_f1_delta_mean": round(float(np.mean(diff_macro)), 4),
        "macro_f1_ci95": [round(float(np.percentile(diff_macro, 2.5)), 4), round(float(np.percentile(diff_macro, 97.5)), 4)],
        "micro_f1_delta_mean": round(float(np.mean(diff_micro)), 4),
        "micro_f1_ci95": [round(float(np.percentile(diff_micro, 2.5)), 4), round(float(np.percentile(diff_micro, 97.5)), 4)],
        "porch_f1_delta_mean": round(float(np.mean(diff_porch)), 4),
        "porch_f1_ci95": [round(float(np.percentile(diff_porch, 2.5)), 4), round(float(np.percentile(diff_porch, 97.5)), 4)],
    }


def run_benchmark_and_audit():
    val_images = sorted(list(VAL_DIR.glob("*.jpg")))
    print(f"Loaded {len(val_images)} validation scenes.")

    all_gts = {}
    for img_p in val_images:
        lbl_p = LBL_DIR / f"{img_p.stem}.txt"
        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        all_gts[img_p.name] = boxes

    # 1. Evaluate B6
    print("\nEvaluating Model B6...")
    m6 = YOLO(str(B6_CKPT))
    # Warmup
    for img_p in val_images[:10]:
        _ = m6.predict(source=str(img_p), imgsz=640, device=0, verbose=False)

    latencies_b6 = []
    preds_b6 = {}
    for img_p in val_images:
        t0 = time.perf_counter()
        res = m6.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000
        latencies_b6.append(dt)
        preds = []
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywh = res.boxes.xywhn[i].tolist()
            preds.append((cls_id, conf, xywh[0], xywh[1], xywh[2], xywh[3]))
        preds_b6[img_p.name] = preds

    val_res_b6 = m6.val(data=str(DATASET_YAML), split="val", batch=8, imgsz=640, device=0, verbose=False)
    scenes_b6 = evaluate_scenes(all_gts, preds_b6, threshold=0.25, iou_thresh=0.50)
    metrics_b6 = aggregate(scenes_b6)
    metrics_b6["mAP50"] = round(float(val_res_b6.box.map50), 4)
    metrics_b6["mAP50_95"] = round(float(val_res_b6.box.map), 4)
    metrics_b6["median_latency_ms"] = round(float(np.median(latencies_b6)), 2)
    metrics_b6["p95_latency_ms"] = round(float(np.percentile(latencies_b6, 95)), 2)

    # 2. Evaluate B7
    print("\nEvaluating Model B7...")
    m7 = YOLO(str(B7_CKPT))
    # Warmup
    for img_p in val_images[:10]:
        _ = m7.predict(source=str(img_p), imgsz=640, device=0, verbose=False)

    latencies_b7 = []
    preds_b7 = {}
    for img_p in val_images:
        t0 = time.perf_counter()
        res = m7.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000
        latencies_b7.append(dt)
        preds = []
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywh = res.boxes.xywhn[i].tolist()
            preds.append((cls_id, conf, xywh[0], xywh[1], xywh[2], xywh[3]))
        preds_b7[img_p.name] = preds

    val_res_b7 = m7.val(data=str(DATASET_YAML), split="val", batch=8, imgsz=640, device=0, verbose=False)
    scenes_b7 = evaluate_scenes(all_gts, preds_b7, threshold=0.25, iou_thresh=0.50)
    metrics_b7 = aggregate(scenes_b7)
    metrics_b7["mAP50"] = round(float(val_res_b7.box.map50), 4)
    metrics_b7["mAP50_95"] = round(float(val_res_b7.box.map), 4)
    metrics_b7["median_latency_ms"] = round(float(np.median(latencies_b7)), 2)
    metrics_b7["p95_latency_ms"] = round(float(np.percentile(latencies_b7, 95)), 2)

    # 3. Bootstrap
    print("\nRunning Bootstrap analysis (1000 resamples)...")
    boot = compute_bootstrap(scenes_b6, scenes_b7, n_boot=1000, seed=42)

    output = {
        "b6": metrics_b6,
        "b7": metrics_b7,
        "bootstrap": boot,
    }

    out_p = REPO_ROOT / "reports" / "phase22_reproduced_audit_metrics.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== REPRODUCED RESULTS ===")
    print(f"B6 Macro-F1: {metrics_b6['macro_f1']} | Micro-F1: {metrics_b6['micro_f1']} | mAP50: {metrics_b6['mAP50']} | Latency median: {metrics_b6['median_latency_ms']} ms")
    print(f"B7 Macro-F1: {metrics_b7['macro_f1']} | Micro-F1: {metrics_b7['micro_f1']} | mAP50: {metrics_b7['mAP50']} | Latency median: {metrics_b7['median_latency_ms']} ms")
    print(f"Porch F1: B6={metrics_b6['per_class']['porch']['f1']} -> B7={metrics_b7['per_class']['porch']['f1']}")
    print(f"Bootstrap 95% CI: Macro={boot['macro_f1_ci95']}, Micro={boot['micro_f1_ci95']}, Porch={boot['porch_f1_ci95']}")
    print(f"Saved audit output to {out_p}")


if __name__ == "__main__":
    run_benchmark_and_audit()

