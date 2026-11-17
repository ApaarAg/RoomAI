"""Phase 16: CLAHE Verification, Benchmark Integrity, and Final Architecture Evaluation.

Executes:
1. Exact reproduction of B6 baseline (no preprocessing).
2. Controlled CLAHE ablation:
   - Condition 1: B6 Baseline (No preprocessing)
   - Condition 2: B6 + CLAHE (Luminance L-channel, clipLimit=2.0, tileGridSize=(8,8))
   - Condition 3: B6 + Global Histogram Equalization on Luminance (Global HE)
   - Condition 4: B6 + Gamma Correction (gamma=0.8 on Luminance)
3. Latency & VRAM Benchmark with synchronized CUDA timing on:
   - B2 General Only (Production Default)
   - B6 General Only (Expanded Candidate)
   - B6 + CLAHE
   - Cascaded Specialist (B6 + Porch Specialist)
   - Parallel Ensemble (B6 + Porch Specialist)
4. Saves:
   - reports/phase16_metrics.json
   - reports/phase16_latency_benchmark.json
   - Visual error gallery in reports/phase16_clahe_error_gallery/
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
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
from services.specialist_ensemble_service import SpecialistEnsembleDetector, box_iou_xyxy

VAL_IMG_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "val"
VAL_LBL_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "labels" / "val"

B2_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
B6_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
PORCH_SPEC_CKPT = REPO_ROOT / "checkpoints" / "exterior_specialist_porch_yolo11s_best.pt"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def box_iou_xywh(box1: List[float], box2: List[float]) -> float:
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


def evaluate_predictions(all_gts: Dict[str, List], all_preds: Dict[str, List], threshold: float = 0.25, iou_thresh: float = 0.50):
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
                    iou = box_iou_xywh(p_box, g_box)
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
    overall_mean_iou = float(np.mean([iou for ious in matched_ious.values() for iou in ious])) if any(matched_ious.values()) else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "overall_mean_iou": round(overall_mean_iou, 4),
        "per_class": per_class,
    }


# Image Pre-Processing Transforms
def apply_clahe(img: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    """Applies adaptive histogram equalization to L channel in LAB color space."""
    np_img = np.array(img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def apply_global_he(img: Image.Image) -> Image.Image:
    """Applies standard global histogram equalization to L channel in LAB space."""
    np_img = np.array(img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    eq_l = cv2.equalizeHist(l)
    merged = cv2.merge((eq_l, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def apply_gamma(img: Image.Image, gamma: float = 0.8) -> Image.Image:
    """Applies gamma correction (gamma < 1 lifts shadows) to L channel in LAB space."""
    np_img = np.array(img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    l_gamma = cv2.LUT(l, table)
    merged = cv2.merge((l_gamma, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def main():
    print("=" * 80)
    print("RoomAI Phase 16: CLAHE Verification & Benchmark Integrity Audit")
    print("=" * 80)

    # 1. Load validation manifest and ground truth
    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
    print(f"Validation Manifest: {len(val_images)} images loaded from {VAL_IMG_DIR}")

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

    total_gt_boxes = sum(len(b) for b in all_gts.values())
    print(f"Total ground truth boxes: {total_gt_boxes}")

    # Checkpoint hashes
    ckpt_hashes = {}
    for p, name in [(B2_CKPT, "B2"), (B6_CKPT, "B6"), (PORCH_SPEC_CKPT, "Porch Specialist")]:
        assert p.exists(), f"Missing checkpoint: {p}"
        md5_h = hashlib.md5(p.read_bytes()).hexdigest()
        ckpt_hashes[name] = {"path": str(p), "md5": md5_h, "size": p.stat().st_size}
        print(f"[{name}] Checkpoint verified: {p.name} | MD5: {md5_h}")

    # -------------------------------------------------------------
    # STAGE B & C: ABLATION EXPERIMENTS (Metrics)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE B & C: Controlled CLAHE & Preprocessing Ablation")
    print("=" * 80)

    # Preload B6 model once for exact consistency
    t_load_0 = time.perf_counter()
    b6_detector = CustomObjectDetector({"checkpoint_path": str(B6_CKPT), "device": "cuda:0", "confidence_threshold": 0.01})
    b6_detector.load_model()
    torch.cuda.synchronize()
    b6_load_time_ms = round((time.perf_counter() - t_load_0) * 1000.0, 2)

    # Preload Porch Specialist for cascaded / parallel comparisons
    spec_detector = CustomObjectDetector({"checkpoint_path": str(PORCH_SPEC_CKPT), "device": "cuda:0", "confidence_threshold": 0.01})
    spec_detector.load_model()
    torch.cuda.synchronize()

    # Define Ablation Conditions
    ablation_conditions = [
        ("Cond 1: B6 Baseline (No Preproc)", None),
        ("Cond 2: B6 + CLAHE (Luminance, clip=2.0)", lambda im: apply_clahe(im, clip_limit=2.0)),
        ("Cond 3: B6 + Global HE (Luminance)", apply_global_he),
        ("Cond 4: B6 + Gamma Correction (gamma=0.8)", lambda im: apply_gamma(im, gamma=0.8)),
    ]

    ablation_metrics = {}
    ablation_predictions = {}

    gallery_dir = REPO_ROOT / "reports" / "phase16_clahe_error_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    sample_gallery_imgs = [img_p for img_p in val_images if any(g[0] == 6 for g in all_gts[img_p.name])][:4]

    for cond_name, preproc_fn in ablation_conditions:
        print(f"\nEvaluating {cond_name}...")
        preds_dict = {}

        # Warmup
        dummy = Image.new("RGB", (640, 640))
        for _ in range(5):
            b6_detector.predict(dummy, confidence_threshold=0.25)
        torch.cuda.synchronize()

        for img_p in val_images:
            with Image.open(img_p) as raw_im:
                im = raw_im.convert("RGB")
                if preproc_fn is not None:
                    im = preproc_fn(im)

                resp = b6_detector.predict(im, confidence_threshold=0.25)
                preds = []
                for d in resp.detections:
                    bx = d.box
                    xc = (bx.x_min + bx.x_max) / 2.0
                    yc = (bx.y_min + bx.y_max) / 2.0
                    bw = bx.x_max - bx.x_min
                    bh = bx.y_max - bx.y_min
                    preds.append((d.class_id, d.confidence, xc, yc, bw, bh))
                preds_dict[img_p.name] = preds

        ablation_predictions[cond_name] = preds_dict
        res = evaluate_predictions(all_gts, preds_dict, threshold=0.25, iou_thresh=0.50)
        ablation_metrics[cond_name] = res

        porch_m = res["per_class"]["porch"]
        print(f"  Macro-F1: {res['macro_f1']:.4f} | Micro-F1: {res['micro_f1']:.4f}")
        print(f"  Porch: TP={porch_m['tp']}, FP={porch_m['fp']}, FN={porch_m['fn']} | P={porch_m['precision']:.4f}, R={porch_m['recall']:.4f}, F1={porch_m['f1']:.4f}")

    # Generate Gallery Visualizations comparing Baseline vs CLAHE vs Global HE
    cond_key_map = {
        "baseline": "Cond 1: B6 Baseline (No Preproc)",
        "clahe": "Cond 2: B6 + CLAHE (Luminance, clip=2.0)",
        "global_he": "Cond 3: B6 + Global HE (Luminance)",
    }
    for sample_p in sample_gallery_imgs:
        for cond_tag, preproc_fn in [
            ("baseline", None),
            ("clahe", lambda im: apply_clahe(im, clip_limit=2.0)),
            ("global_he", apply_global_he),
        ]:
            with Image.open(sample_p) as raw_im:
                vis_im = raw_im.convert("RGB")
                if preproc_fn is not None:
                    vis_im = preproc_fn(vis_im)
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

                # Predictions
                c_preds = ablation_predictions[cond_key_map[cond_tag]][sample_p.name]
                for p in c_preds:
                    if p[0] == 6 and p[1] >= 0.20:
                        xmin = int((p[2] - p[4] / 2.0) * w_i)
                        ymin = int((p[3] - p[5] / 2.0) * h_i)
                        xmax = int((p[2] + p[4] / 2.0) * w_i)
                        ymax = int((p[3] + p[5] / 2.0) * h_i)
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF00FF", width=3)
                        draw.text((xmin + 4, ymin + 4), f"porch {p[1]:.2f}", fill="#FFFF00")

                out_p = gallery_dir / f"{cond_tag}_{sample_p.stem}.jpg"
                vis_im.save(out_p, quality=90)

    # -------------------------------------------------------------
    # STAGE D: CORRECTED LATENCY & VRAM BENCHMARK
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE D: Corrected Latency & GPU Memory Benchmark (Synchronized)")
    print("=" * 80)

    # Retained Candidate Configurations to Benchmark
    candidates = [
        ("B2 General Only (Production Default)", B2_CKPT, None, "general_only", False),
        ("B6 General Only (Expanded Candidate)", B6_CKPT, None, "general_only", False),
        ("B6 + CLAHE (Luminance clip=2.0)", B6_CKPT, None, "general_only", True),
        ("Cascaded Specialist (B6 + Porch Specialist)", B6_CKPT, PORCH_SPEC_CKPT, "cascaded", False),
        ("Parallel Ensemble (B6 + Porch Specialist)", B6_CKPT, PORCH_SPEC_CKPT, "parallel_ensemble", False),
    ]

    benchmark_results = {}

    for cand_name, g_ckpt, s_ckpt, mode, use_clahe in candidates:
        print(f"\nBenchmarking: {cand_name}...")

        # 1. Measure Cold Load Time
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t_ld_0 = time.perf_counter()

        g_mod = CustomObjectDetector({"checkpoint_path": str(g_ckpt), "device": "cuda:0", "confidence_threshold": 0.01})
        g_mod.load_model()
        s_mod = None
        if s_ckpt:
            s_mod = CustomObjectDetector({"checkpoint_path": str(s_ckpt), "device": "cuda:0", "confidence_threshold": 0.01})
            s_mod.load_model()
        torch.cuda.synchronize()
        cold_load_time_ms = round((time.perf_counter() - t_ld_0) * 1000.0, 2)

        ens = SpecialistEnsembleDetector(
            general_model=g_mod,
            specialist_model=s_mod,
            mode=mode,
            general_conf_thresh=0.25,
            specialist_conf_thresh=0.20,
        )

        # 2. Comprehensive Warm-Up on actual diverse image shapes (10 images)
        print("  Warming up CUDA kernels on diverse image geometries...")
        for w_img_p in val_images[:10]:
            with Image.open(w_img_p) as w_im:
                im_w = w_im.convert("RGB")
                if use_clahe:
                    im_w = apply_clahe(im_w)
                ens.predict(im_w, confidence_threshold=0.25)
        torch.cuda.synchronize()

        # 3. Synchronized Inference Timing across all 154 images
        model_only_latencies = []
        pipeline_latencies = []

        torch.cuda.reset_peak_memory_stats(0)

        for img_p in val_images:
            # Full Pipeline Timing (including image loading & preprocessing)
            t_pipe_0 = time.perf_counter()
            with Image.open(img_p) as raw_im:
                im = raw_im.convert("RGB")
                if use_clahe:
                    im = apply_clahe(im)

                # Pure Model-Only Timing (with CUDA synchronization)
                torch.cuda.synchronize()
                t_model_0 = time.perf_counter()
                resp = ens.predict(im, confidence_threshold=0.25)
                torch.cuda.synchronize()
                t_model_end = time.perf_counter()

            t_pipe_end = time.perf_counter()

            model_only_latencies.append((t_model_end - t_model_0) * 1000.0)
            pipeline_latencies.append((t_pipe_end - t_pipe_0) * 1000.0)

        peak_vram_alloc_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
        peak_vram_res_mb = round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2)

        b_entry = {
            "candidate": cand_name,
            "cold_load_time_ms": cold_load_time_ms,
            "peak_vram_allocated_mb": peak_vram_alloc_mb,
            "peak_vram_reserved_mb": peak_vram_res_mb,
            "model_only_latency_ms": {
                "median": round(float(np.median(model_only_latencies)), 2),
                "p90": round(float(np.percentile(model_only_latencies, 90)), 2),
                "p95": round(float(np.percentile(model_only_latencies, 95)), 2),
                "min": round(float(np.min(model_only_latencies)), 2),
                "max": round(float(np.max(model_only_latencies)), 2),
                "mean": round(float(np.mean(model_only_latencies)), 2),
            },
            "full_pipeline_latency_ms": {
                "median": round(float(np.median(pipeline_latencies)), 2),
                "p90": round(float(np.percentile(pipeline_latencies, 90)), 2),
                "p95": round(float(np.percentile(pipeline_latencies, 95)), 2),
                "min": round(float(np.min(pipeline_latencies)), 2),
                "max": round(float(np.max(pipeline_latencies)), 2),
                "mean": round(float(np.mean(pipeline_latencies)), 2),
            },
            "steady_state_fps": round(1000.0 / float(np.median(model_only_latencies)), 1),
        }
        benchmark_results[cand_name] = b_entry

        print(f"  Cold Load: {cold_load_time_ms} ms | VRAM Allocated: {peak_vram_alloc_mb} MB")
        print(f"  Model-Only Latency: Median={b_entry['model_only_latency_ms']['median']} ms | P90={b_entry['model_only_latency_ms']['p90']} ms | P95={b_entry['model_only_latency_ms']['p95']} ms")
        print(f"  Full Pipeline:     Median={b_entry['full_pipeline_latency_ms']['median']} ms | P90={b_entry['full_pipeline_latency_ms']['p90']} ms | P95={b_entry['full_pipeline_latency_ms']['p95']} ms")

    # Save reports/phase16_metrics.json
    out_metrics_path = REPO_ROOT / "reports" / "phase16_metrics.json"
    full_metrics_payload = {
        "metadata": {
            "validation_manifest_count": len(val_images),
            "total_ground_truth_boxes": total_gt_boxes,
            "checkpoints": ckpt_hashes,
        },
        "ablation_metrics": ablation_metrics,
    }
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics_payload, f, indent=2)
    print(f"\n[SAVED] Phase 16 Metrics saved to: {out_metrics_path}")

    # Save reports/phase16_latency_benchmark.json
    out_latency_path = REPO_ROOT / "reports" / "phase16_latency_benchmark.json"
    with open(out_latency_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"[SAVED] Phase 16 Latency Benchmark saved to: {out_latency_path}")

    # Print Comparative Summary
    print("\n" + "=" * 125)
    print(f"{'Condition / Architecture':<42} | {'Macro-F1':<9} | {'Porch F1':<9} | {'Porch P/R':<15} | {'Model Med':<10} | {'Model P95':<10} | {'Alloc VRAM':<10}")
    print("-" * 125)
    for cname, m in ablation_metrics.items():
        pm = m["per_class"]["porch"]
        print(f"{cname:<42} | {m['macro_f1']:<9.4f} | {pm['f1']:<9.4f} | {pm['precision']:.3f} / {pm['recall']:.3f}    | {'-':<10} | {'-':<10} | {'-':<10}")
    for bname, bm in benchmark_results.items():
        print(f"{bname:<42} | {'-':<9} | {'-':<9} | {'-':<15} | {bm['model_only_latency_ms']['median']:<10.2f} | {bm['model_only_latency_ms']['p95']:<10.2f} | {bm['peak_vram_allocated_mb']:<10.1f}")
    print("=" * 125)


if __name__ == "__main__":
    main()
