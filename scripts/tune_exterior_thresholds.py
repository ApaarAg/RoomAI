"""Tuning & Comparison Script for RoomAI Exterior Detectors: Baseline B2 vs Candidate B3.

Evaluates confidence thresholds and NMS-IoU parameters STRICTLY on the validation split,
identifies 3 operating points (High Precision, Balanced F1, High Recall), tests generalization
on the held-out test split, and benchmarks latency and peak GPU memory.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_YAML = REPO_ROOT / "data" / "exterior_canonical" / "exterior_canonical.yaml"
OUTPUT_DIR = REPO_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "exterior_error_analysis"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
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

MODELS = [
    {
        "id": "model_b2_baseline",
        "name": "Model B2 (YOLO11s Baseline)",
        "checkpoint": REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt",
    },
    {
        "id": "model_b3_candidate",
        "name": "Model B3 (YOLO11s RFS Candidate)",
        "checkpoint": REPO_ROOT / "checkpoints" / "exterior_detector_b3_yolo11s_rfs_best.pt",
    },
]

CONF_RANGE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
IOU_RANGE = [0.40, 0.45, 0.50, 0.60]


def run_val_sweep(model_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs a threshold sweep on the validation split and identifies 3 operating points."""
    model = YOLO(str(model_path))
    sweep_results = []
    
    best_f1_val = -1.0
    best_f1_op = None
    
    for conf in CONF_RANGE:
        for iou in IOU_RANGE:
            res = model.val(
                data=str(CANONICAL_YAML),
                split="val",
                batch=8,
                imgsz=640,
                conf=conf,
                iou=iou,
                device=0 if torch.cuda.is_available() else "cpu",
                plots=False,
                verbose=False,
                save_json=False,
            )
            mp = float(res.box.mp)
            mr = float(res.box.mr)
            f1 = (2 * mp * mr) / (mp + mr + 1e-6) if (mp + mr) > 0 else 0.0
            map50 = float(res.box.map50)
            map50_95 = float(res.box.map)
            
            entry = {
                "conf": round(conf, 2),
                "iou": round(iou, 2),
                "precision": round(mp, 4),
                "recall": round(mr, 4),
                "f1": round(f1, 4),
                "map50": round(map50, 4),
                "map50_95": round(map50_95, 4),
            }
            sweep_results.append(entry)
            if f1 > best_f1_val:
                best_f1_val = f1
                best_f1_op = entry

    # High precision operating point: highest precision among points with recall >= 0.35
    hp_candidates = [e for e in sweep_results if e["recall"] >= 0.35]
    if not hp_candidates:
        hp_candidates = sweep_results
    hp_op = max(hp_candidates, key=lambda x: x["precision"])

    # Balanced F1 operating point
    balanced_op = best_f1_op

    # High recall operating point: highest recall among points with precision >= 0.40
    hr_candidates = [e for e in sweep_results if e["precision"] >= 0.40]
    if not hr_candidates:
        hr_candidates = sweep_results
    hr_op = max(hr_candidates, key=lambda x: x["recall"])

    operating_points = {
        "high_precision": hp_op,
        "balanced_f1": balanced_op,
        "high_recall": hr_op,
    }
    return sweep_results, operating_points


def evaluate_operating_point_on_test(model_path: Path, op: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates a validation-chosen operating point on the held-out test split."""
    model = YOLO(str(model_path))
    res = model.val(
        data=str(CANONICAL_YAML),
        split="test",
        batch=8,
        imgsz=640,
        conf=op["conf"],
        iou=op["iou"],
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        verbose=False,
        save_json=False,
    )
    mp = float(res.box.mp)
    mr = float(res.box.mr)
    f1 = (2 * mp * mr) / (mp + mr + 1e-6) if (mp + mr) > 0 else 0.0
    map50 = float(res.box.map50)
    map50_95 = float(res.box.map)
    return {
        "applied_conf": op["conf"],
        "applied_iou": op["iou"],
        "test_precision": round(mp, 4),
        "test_recall": round(mr, 4),
        "test_f1": round(f1, 4),
        "test_map50": round(map50, 4),
        "test_map50_95": round(map50_95, 4),
    }


def benchmark_latency(model_path: Path, num_warmup: int = 15, num_timed: int = 50) -> Dict[str, float]:
    """Measures precise inference latency (mean, median, p95) and peak VRAM."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    
    # Dummy image tensor
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))

    # Warm-up
    for _ in range(num_warmup):
        _ = model.predict(dummy_input, device=device, verbose=False)

    latencies_ms = []
    for _ in range(num_timed):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 * 1024), 2)

    return {
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "median_latency_ms": round(float(np.median(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "peak_vram_mb": peak_vram_mb,
    }


def generate_visual_comparisons(
    model_b2_path: Path,
    model_b3_path: Path,
    target_samples: List[str] = [
        "oi_ext_test_0106",
        "oi_ext_test_0113",
        "cmp_b0021",
        "cmp_b0042",
        "cmp_b0085",
    ],
):
    """Generates side-by-side visual comparisons: Ground Truth vs B2 vs B3."""
    test_img_dir = REPO_ROOT / "data" / "exterior_canonical" / "images" / "test"
    test_lbl_dir = REPO_ROOT / "data" / "exterior_canonical" / "labels" / "test"
    
    model_b2 = YOLO(str(model_b2_path))
    model_b3 = YOLO(str(model_b3_path))
    
    device = 0 if torch.cuda.is_available() else "cpu"

    for stem in target_samples:
        img_path = test_img_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = test_img_dir / f"{stem}.png"
        if not img_path.exists():
            continue

        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            continue
        h, w = raw_img.shape[:2]

        # 1. Ground truth visualization
        gt_img = raw_img.copy()
        lbl_path = test_lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        color = (0, 255, 0)
                        cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            gt_img,
                            CLASS_NAMES.get(cid, str(cid)),
                            (x1, max(y1 - 4, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                        )
        cv2.putText(gt_img, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # 2. Model B2 Predictions (conf=0.25)
        pred_b2 = model_b2.predict(raw_img, conf=0.25, device=device, verbose=False)[0]
        b2_img = raw_img.copy()
        if pred_b2.boxes is not None:
            for box in pred_b2.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                c_name = CLASS_NAMES.get(cls_id, str(cls_id))
                color = (0, 165, 255)
                cv2.rectangle(b2_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(
                    b2_img,
                    f"{c_name} {conf:.2f}",
                    (xyxy[0], max(xyxy[1] - 4, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
        cv2.putText(b2_img, "B2 Baseline", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

        # 3. Model B3 Predictions (conf=0.25)
        pred_b3 = model_b3.predict(raw_img, conf=0.25, device=device, verbose=False)[0]
        b3_img = raw_img.copy()
        if pred_b3.boxes is not None:
            for box in pred_b3.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                c_name = CLASS_NAMES.get(cls_id, str(cls_id))
                color = (255, 105, 180)
                cv2.rectangle(b3_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(
                    b3_img,
                    f"{c_name} {conf:.2f}",
                    (xyxy[0], max(xyxy[1] - 4, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
        cv2.putText(b3_img, "B3 Candidate (RFS)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 105, 180), 2)

        # Side-by-side concatenation
        combined = np.hstack([gt_img, b2_img, b3_img])
        out_path = ARTIFACTS_DIR / f"comparison_b2_vs_b3_{stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"[VISUAL COMPARISON] Saved: {out_path.name}")


def main():
    print("=" * 80)
    print("RoomAI Exterior Detector Optimization: Baseline B2 vs Candidate B3")
    print("=" * 80)

    report_data = {}

    for m in MODELS:
        m_id = m["id"]
        m_name = m["name"]
        ckpt = m["checkpoint"]
        if not ckpt.exists():
            print(f"[SKIP] Checkpoint {ckpt} not found.")
            continue

        print(f"\n--- Analyzing {m_name} ({ckpt.name}) ---")
        
        # 1. Validation threshold sweep
        sweep_results, operating_points = run_val_sweep(ckpt)
        
        # 2. Evaluate operating points on held-out test split
        test_evals = {}
        for op_name, op_val in operating_points.items():
            test_res = evaluate_operating_point_on_test(ckpt, op_val)
            test_evals[op_name] = test_res
            print(f"  [{op_name.upper()} OP (conf={op_val['conf']}, iou={op_val['iou']})] "
                  f"VAL F1: {op_val['f1']:.4f} -> TEST P: {test_res['test_precision']:.4f}, "
                  f"R: {test_res['test_recall']:.4f}, F1: {test_res['test_f1']:.4f}, "
                  f"mAP50: {test_res['test_map50']:.4f}")

        # 3. Latency & VRAM benchmark
        bench_metrics = benchmark_latency(ckpt)
        print(f"  [BENCHMARK] Latency: {bench_metrics['mean_latency_ms']} ms (median: {bench_metrics['median_latency_ms']} ms), Peak VRAM: {bench_metrics['peak_vram_mb']} MB")

        report_data[m_id] = {
            "name": m_name,
            "checkpoint": str(ckpt.name),
            "validation_operating_points": operating_points,
            "test_evaluations": test_evals,
            "benchmark": bench_metrics,
            "sweep_sample": sweep_results[:5],
        }

    # Save machine readable metrics
    out_json = OUTPUT_DIR / "exterior_optimization_thresholds_and_latency.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[REPORT] Saved optimization metrics to: {out_json}")

    # Generate visual comparisons if both checkpoints exist
    b2_ckpt = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
    b3_ckpt = REPO_ROOT / "checkpoints" / "exterior_detector_b3_yolo11s_rfs_best.pt"
    if b2_ckpt.exists() and b3_ckpt.exists():
        print("\n--- Generating Visual Prediction Comparisons ---")
        generate_visual_comparisons(b2_ckpt, b3_ckpt)


if __name__ == "__main__":
    main()
