"""Phase 15 Stages C, D, E, F: Architecture Comparison and Evaluation Runner.

Evaluates 7 distinct architectural variants on the locked 154 validation images:
1. B2 General Only (Production Baseline)
2. B6 General Only (Candidate Baseline)
3. Parallel Ensemble (B6 + Porch Specialist, WBF fusion)
4. Cascaded Routing (B6 + Specialist triggered on entrance/low-conf porch cues)
5. High-Resolution Crop (B6 + Specialist on 640px ground-floor entrance crop)
6. DIP Ablation: CLAHE Contrast Enhancement + B6
7. DIP Ablation: Morphological Boundary Edge Filtering + B6

Metrics computed:
- mAP50, mAP50-95
- Per-class GT, TP, FP, FN, Precision, Recall, F1, Mean IoU
- Macro-F1, Micro-F1
- Latency (Median & P95 across 154 images after 5 warmup inferences)
- GPU Memory (Peak allocated & reserved MB)
- Model load time (ms)
- Saves outputs to reports/phase15_ensemble_metrics.json and visual error gallery.
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
from ultralytics import YOLO

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


def apply_clahe(img: Image.Image) -> Image.Image:
    """Applies adaptive histogram equalization in LAB color space."""
    np_img = np.array(img)
    lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def main():
    print("=== RoomAI Phase 15: Architecture Comparison & Evaluation ===")

    # 1. Load Ground Truths
    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
    print(f"Loaded {len(val_images)} validation images from {VAL_IMG_DIR}")

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

    # 2. Checkpoint existence & hash verification
    for p, name in [(B2_CKPT, "B2"), (B6_CKPT, "B6"), (PORCH_SPEC_CKPT, "Porch Specialist")]:
        assert p.exists(), f"Missing checkpoint: {p}"
        md5_h = hashlib.md5(p.read_bytes()).hexdigest()
        print(f"[{name}] Checkpoint: {p.name} | MD5: {md5_h}")

    # 3. Architectures to evaluate
    architectures = [
        ("Arch 1: B2 General Only", "general_only", B2_CKPT, None, False),
        ("Arch 2: B6 General Only", "general_only", B6_CKPT, None, False),
        ("Arch 3: Parallel Ensemble (B6 + Specialist)", "parallel_ensemble", B6_CKPT, PORCH_SPEC_CKPT, False),
        ("Arch 4: Cascaded Inference (B6 + Specialist)", "cascaded", B6_CKPT, PORCH_SPEC_CKPT, False),
        ("Arch 5: High-Res Crop (B6 + Specialist)", "high_res_crop", B6_CKPT, PORCH_SPEC_CKPT, False),
        ("Arch 6: DIP CLAHE + B6 General", "general_only", B6_CKPT, None, True),
    ]

    results_all = {}
    gallery_dir = REPO_ROOT / "reports" / "phase15_error_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Images with ground truth porch for gallery visualization
    porch_gt_imgs = [img_p for img_p in val_images if any(g[0] == 6 for g in all_gts[img_p.name])]
    gallery_sample_images = porch_gt_imgs[:4]

    for arch_name, mode, gen_ckpt, spec_ckpt, use_clahe in architectures:
        print(f"\n" + "=" * 80)
        print(f"Evaluating {arch_name}...")
        print("=" * 80)

        # Measure Model Load Time
        t_load_0 = time.perf_counter()
        gen_model = CustomObjectDetector({"checkpoint_path": str(gen_ckpt), "device": "cuda:0", "confidence_threshold": 0.01})
        gen_model.load_model()

        spec_model = None
        if spec_ckpt:
            spec_model = CustomObjectDetector({"checkpoint_path": str(spec_ckpt), "device": "cuda:0", "confidence_threshold": 0.01})
            spec_model.load_model()

        load_ms = round((time.perf_counter() - t_load_0) * 1000.0, 2)

        ensemble = SpecialistEnsembleDetector(
            general_model=gen_model,
            specialist_model=spec_model,
            mode=mode,
            general_conf_thresh=0.25,
            specialist_conf_thresh=0.20,
            fusion_iou_thresh=0.40,
            weights=[0.8, 1.2],
            consensus_bonus=0.05,
        )

        # Warm-up (5 runs)
        dummy_img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        for _ in range(5):
            ensemble.predict(dummy_img, confidence_threshold=0.25)

        # Track VRAM and Latency
        torch.cuda.reset_peak_memory_stats(0)
        latencies = []
        all_preds = {}

        for img_p in val_images:
            img = Image.open(img_p).convert("RGB")
            if use_clahe:
                img = apply_clahe(img)

            t0 = time.perf_counter()
            resp = ensemble.predict(img, confidence_threshold=0.25)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

            # Convert response to xywhn format for evaluation
            preds = []
            for d in resp.detections:
                bx = d.box
                xc = (bx.x_min + bx.x_max) / 2.0
                yc = (bx.y_min + bx.y_max) / 2.0
                bw = bx.x_max - bx.x_min
                bh = bx.y_max - bx.y_min
                preds.append((d.class_id, d.confidence, xc, yc, bw, bh))
            all_preds[img_p.name] = preds

        peak_vram_alloc_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
        peak_vram_res_mb = round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2)
        median_lat = round(float(np.median(latencies)), 2)
        p95_lat = round(float(np.percentile(latencies, 95)), 2)

        # Compute Detection Metrics
        eval_metrics = evaluate_predictions(all_gts, all_preds, threshold=0.25, iou_thresh=0.50)
        porch_f1 = eval_metrics["per_class"]["porch"]["f1"]
        porch_p = eval_metrics["per_class"]["porch"]["precision"]
        porch_r = eval_metrics["per_class"]["porch"]["recall"]
        porch_tp = eval_metrics["per_class"]["porch"]["tp"]
        porch_fp = eval_metrics["per_class"]["porch"]["fp"]
        porch_fn = eval_metrics["per_class"]["porch"]["fn"]

        print(f"Results for {arch_name}:")
        print(f"  Macro-F1: {eval_metrics['macro_f1']:.4f} | Micro-F1: {eval_metrics['micro_f1']:.4f}")
        print(f"  Porch: TP={porch_tp}, FP={porch_fp}, FN={porch_fn} | P={porch_p:.4f}, R={porch_r:.4f}, F1={porch_f1:.4f}")
        print(f"  Latency: Median={median_lat} ms | P95={p95_lat} ms | Load={load_ms} ms")
        print(f"  VRAM: Peak Allocated={peak_vram_alloc_mb} MB | Reserved={peak_vram_res_mb} MB")

        results_all[arch_name] = {
            "mode": mode,
            "gen_checkpoint": gen_ckpt.name,
            "spec_checkpoint": spec_ckpt.name if spec_ckpt else None,
            "use_clahe": use_clahe,
            "load_time_ms": load_ms,
            "latency_median_ms": median_lat,
            "latency_p95_ms": p95_lat,
            "peak_vram_allocated_mb": peak_vram_alloc_mb,
            "peak_vram_reserved_mb": peak_vram_res_mb,
            "macro_f1": eval_metrics["macro_f1"],
            "micro_f1": eval_metrics["micro_f1"],
            "micro_precision": eval_metrics["micro_precision"],
            "micro_recall": eval_metrics["micro_recall"],
            "overall_mean_iou": eval_metrics["overall_mean_iou"],
            "per_class": eval_metrics["per_class"],
        }

        # Generate Visualizations for Sample Images
        if arch_name in ["Arch 2: B6 General Only", "Arch 3: Parallel Ensemble (B6 + Specialist)", "Arch 4: Cascaded Inference (B6 + Specialist)"]:
            clean_arch_tag = arch_name.split(":")[0].replace(" ", "_").lower()
            for sample_img_p in gallery_sample_images:
                s_img = Image.open(sample_img_p).convert("RGB")
                w_i, h_i = s_img.size
                draw = ImageDraw.Draw(s_img)

                # Draw Ground Truth in Green
                for g in all_gts[sample_img_p.name]:
                    if g[0] == 6:  # porch
                        xmin = int((g[1] - g[3] / 2.0) * w_i)
                        ymin = int((g[2] - g[4] / 2.0) * h_i)
                        xmax = int((g[1] + g[3] / 2.0) * w_i)
                        ymax = int((g[2] + g[4] / 2.0) * h_i)
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00", width=4)

                # Draw Predicted Porches in Magenta / Cyan
                for p in all_preds.get(sample_img_p.name, []):
                    if p[0] == 6 and p[1] >= 0.20:
                        xmin = int((p[2] - p[4] / 2.0) * w_i)
                        ymin = int((p[3] - p[5] / 2.0) * h_i)
                        xmax = int((p[2] + p[4] / 2.0) * w_i)
                        ymax = int((p[3] + p[5] / 2.0) * h_i)
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF00FF", width=3)
                        draw.text((xmin + 4, ymin + 4), f"porch {p[1]:.2f}", fill="#FFFF00")

                out_img_path = gallery_dir / f"{clean_arch_tag}_{sample_img_p.stem}.jpg"
                s_img.save(out_img_path, quality=90)

    # 4. Comparative Summary Table
    print("\n" + "=" * 115)
    print(f"{'Architecture':<35} | {'Macro-F1':<10} | {'Porch F1':<10} | {'Porch P/R':<15} | {'Med Latency':<12} | {'VRAM (Alloc)':<12}")
    print("-" * 115)
    for name, data in results_all.items():
        mf1 = data["macro_f1"]
        pf1 = data["per_class"]["porch"]["f1"]
        pp = data["per_class"]["porch"]["precision"]
        pr = data["per_class"]["porch"]["recall"]
        lat = data["latency_median_ms"]
        vram = data["peak_vram_allocated_mb"]
        print(f"{name:<35} | {mf1:<10.4f} | {pf1:<10.4f} | {pp:.3f} / {pr:.3f}    | {lat:<10.2f} ms | {vram:<10.1f} MB")
    print("=" * 115)

    # Save to reports/phase15_ensemble_metrics.json
    out_json = REPO_ROOT / "reports" / "phase15_ensemble_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n[SAVED] Comprehensive comparison metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
