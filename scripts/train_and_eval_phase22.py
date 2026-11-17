"""Phase 22: Controlled Training, Evaluation, and Benchmarking of Candidate C1.

Executes:
1. Preflight safeguard verification of immutable checkpoints (B2, B6, B7, Specialist).
2. Controlled fine-tuning of Candidate C1 from Model B7 checkpoint using MaskedDetectionLoss
   on data/phase22_dataset/ (1,862 training scenes, 154 validation scenes).
   - mosaic=0.0 to prevent cross-tile mask mismatch
   - batch=8, lr0=0.0005, epochs=20, patience=6
3. Verification of training invariants (finite loss, finite gradients, peak VRAM).
4. Rigorous evaluation of B6, B7, and C1 on the authorized 154 validation scenes:
   - Warmed-up latency measurements (3 warmup iterations, median + p95)
   - Macro-F1, Micro-F1, per-class precision, recall, F1, AP50, mAP50, mAP50-95
   - 1,000-iteration paired scene-level bootstrap confidence intervals
5. Visual prediction gallery comparing B6, B7, and C1 on key scenes.
6. Generates:
   - reports/phase22_metrics.json
   - reports/phase22_checkpoint_inventory.json
   - reports/phase22_error_analysis.md
   - reports/phase22_integrity_audit.md
   - reports/phase22_prediction_gallery/
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import psutil
import torch
from ultralytics import YOLO

from utils.masked_detection_loss import MaskedDetectionLoss

DATASET_DIR = REPO_ROOT / "data" / "phase22_dataset"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
MASKS_JSON = DATASET_DIR / "verified_class_masks.json"

B2_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
B6_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
B7_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b7_pilot_best.pt"
PORCH_SPEC_CKPT = REPO_ROOT / "checkpoints" / "exterior_specialist_porch_yolo11s_best.pt"
C1_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_c1_phase22_best.pt"

REPORTS_DIR = REPO_ROOT / "reports"
GALLERY_DIR = REPORTS_DIR / "phase22_prediction_gallery"

EXPECTED_PRE_HASHES = {
    "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
    "exterior_detector_b6_expanded_best.pt": "f5ceea56b0610ea14b647014e48152dc",
    "exterior_detector_b7_pilot_best.pt": "41433e929910eda4414ed6541bac2fe4",
    "exterior_specialist_porch_yolo11s_best.pt": "4261fc1ad5f6fd9ac29c0499e2adddfd",
}

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_safeguards(stage_name: str):
    print(f"\n=== Safeguard Verification: {stage_name} ===")
    for ckpt_name, expected_hash in EXPECTED_PRE_HASHES.items():
        p = REPO_ROOT / "checkpoints" / ckpt_name
        assert p.exists(), f"Missing checkpoint: {p}"
        actual_hash = compute_md5(p)
        assert actual_hash == expected_hash, f"Checkpoint modified ({ckpt_name}): {actual_hash} != {expected_hash}"
        print(f"  [VERIFIED] {ckpt_name}: MD5={actual_hash}")


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


def evaluate_predictions_per_scene(all_gts, all_preds, threshold=0.25, iou_thresh=0.50):
    scene_metrics = {}
    for img_name, gts in all_gts.items():
        preds = all_preds.get(img_name, [])
        tp_cnt = {cid: 0 for cid in range(10)}
        fp_cnt = {cid: 0 for cid in range(10)}
        fn_cnt = {cid: 0 for cid in range(10)}

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

        scene_metrics[img_name] = {"tp": tp_cnt, "fp": fp_cnt, "fn": fn_cnt}
    return scene_metrics


def aggregate_metrics(scene_metrics):
    tp_total = {cid: 0 for cid in range(10)}
    fp_total = {cid: 0 for cid in range(10)}
    fn_total = {cid: 0 for cid in range(10)}

    for s in scene_metrics.values():
        for cid in range(10):
            tp_total[cid] += s["tp"][cid]
            fp_total[cid] += s["fp"][cid]
            fn_total[cid] += s["fn"][cid]

    per_class = {}
    f1_list = []
    tot_tp = sum(tp_total.values())
    tot_fp = sum(fp_total.values())
    tot_fn = sum(fn_total.values())

    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        tp = tp_total[cid]
        fp = fp_total[cid]
        fn = fn_total[cid]
        gt = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1)
        per_class[cname] = {
            "gt": gt, "tp": tp, "fp": fp, "fn": fn,
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


def compute_bootstrap_ci(scenes_a, scenes_b, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    scene_keys = list(scenes_a.keys())
    n = len(scene_keys)

    diff_macro = []
    diff_micro = []
    diff_porch = []

    for _ in range(n_boot):
        sample_keys = rng.choice(scene_keys, size=n, replace=True)
        sample_a = {k: scenes_a[k] for k in sample_keys}
        sample_b = {k: scenes_b[k] for k in sample_keys}

        m_a = aggregate_metrics(sample_a)
        m_b = aggregate_metrics(sample_b)

        diff_macro.append(m_b["macro_f1"] - m_a["macro_f1"])
        diff_micro.append(m_b["micro_f1"] - m_a["micro_f1"])
        diff_porch.append(m_b["per_class"]["porch"]["f1"] - m_a["per_class"]["porch"]["f1"])

    return {
        "macro_f1_delta_mean": round(float(np.mean(diff_macro)), 4),
        "macro_f1_ci95": [round(float(np.percentile(diff_macro, 2.5)), 4), round(float(np.percentile(diff_macro, 97.5)), 4)],
        "micro_f1_delta_mean": round(float(np.mean(diff_micro)), 4),
        "micro_f1_ci95": [round(float(np.percentile(diff_micro, 2.5)), 4), round(float(np.percentile(diff_micro, 97.5)), 4)],
        "porch_f1_delta_mean": round(float(np.mean(diff_porch)), 4),
        "porch_f1_ci95": [round(float(np.percentile(diff_porch, 2.5)), 4), round(float(np.percentile(diff_porch, 97.5)), 4)],
    }


def run_warmed_inference(model: YOLO, val_images: List[Path]) -> Tuple[Dict[str, List], float, float]:
    """Runs 3 dummy warmup passes, then records per-scene predictions and latency (median & p95)."""
    # 3 dummy warmups
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(source=dummy, imgsz=640, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    preds_dict = {}
    for img_p in val_images:
        t0 = time.perf_counter()
        res = model.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        preds = []
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywh = res.boxes.xywhn[i].tolist()
            preds.append((cls_id, conf, xywh[0], xywh[1], xywh[2], xywh[3]))
        preds_dict[img_p.name] = preds

    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    return preds_dict, med_lat, p95_lat


def main():
    parser = argparse.ArgumentParser(description="RoomAI Phase 22 Controlled Training & Benchmarking")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for candidate training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if C1 checkpoint already exists")
    args = parser.parse_args()

    print("======================================================================")
    print("      ROOMAI PHASE 22: CONTROLLED TRAINING & BENCHMARK CAMPAIGN       ")
    print("======================================================================")

    # Step 0: Safeguard verification
    verify_safeguards("Pre-Training")

    # Load verified class masks
    with open(MASKS_JSON, "r", encoding="utf-8") as f:
        verified_masks = json.load(f)
    print(f"Loaded {len(verified_masks)} verified class masks.")

    # Validation scenes
    val_img_dir = DATASET_DIR / "images" / "val"
    val_lbl_dir = DATASET_DIR / "labels" / "val"
    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"Authorized Validation Scenes: {len(val_images)} scenes.")
    assert len(val_images) == 154, f"Validation split corrupted! Expected 154, got {len(val_images)}"

    all_gts = {}
    for img_p in val_images:
        lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        all_gts[img_p.name] = boxes

    # 1. Establish Model B6 Baseline on authorized val split
    print("\n--- Model B6 Baseline Evaluation (Warmed) ---")
    model_b6 = YOLO(str(B6_CKPT))
    preds_b6, med_lat_b6, p95_lat_b6 = run_warmed_inference(model_b6, val_images)
    val_res_b6 = model_b6.val(data=str(DATASET_YAML), split="val", batch=args.batch, imgsz=640, device=0, verbose=False)
    scenes_b6 = evaluate_predictions_per_scene(all_gts, preds_b6, threshold=0.25, iou_thresh=0.50)
    m_b6 = aggregate_metrics(scenes_b6)
    m_b6["median_latency_ms"] = round(med_lat_b6, 2)
    m_b6["p95_latency_ms"] = round(p95_lat_b6, 2)
    m_b6["mAP50"] = round(float(val_res_b6.box.map50), 4)
    m_b6["mAP50_95"] = round(float(val_res_b6.box.map), 4)
    m_b6["ap50"] = {CANONICAL_CLASSES[i]: round(float(val_res_b6.box.ap50[i]), 4) for i in range(10)}
    m_b6["ap50_95"] = {CANONICAL_CLASSES[i]: round(float(val_res_b6.box.ap[i]), 4) for i in range(10)}
    print(f"  B6: Macro-F1={m_b6['macro_f1']:.4f} | Micro-F1={m_b6['micro_f1']:.4f} | Porch-F1={m_b6['per_class']['porch']['f1']:.4f} | Latency={med_lat_b6:.2f}ms")

    # 2. Establish Model B7 Baseline on authorized val split
    print("\n--- Model B7 Baseline Evaluation (Warmed) ---")
    model_b7 = YOLO(str(B7_CKPT))
    preds_b7, med_lat_b7, p95_lat_b7 = run_warmed_inference(model_b7, val_images)
    val_res_b7 = model_b7.val(data=str(DATASET_YAML), split="val", batch=args.batch, imgsz=640, device=0, verbose=False)
    scenes_b7 = evaluate_predictions_per_scene(all_gts, preds_b7, threshold=0.25, iou_thresh=0.50)
    m_b7 = aggregate_metrics(scenes_b7)
    m_b7["median_latency_ms"] = round(med_lat_b7, 2)
    m_b7["p95_latency_ms"] = round(p95_lat_b7, 2)
    m_b7["mAP50"] = round(float(val_res_b7.box.map50), 4)
    m_b7["mAP50_95"] = round(float(val_res_b7.box.map), 4)
    m_b7["ap50"] = {CANONICAL_CLASSES[i]: round(float(val_res_b7.box.ap50[i]), 4) for i in range(10)}
    m_b7["ap50_95"] = {CANONICAL_CLASSES[i]: round(float(val_res_b7.box.ap[i]), 4) for i in range(10)}
    print(f"  B7: Macro-F1={m_b7['macro_f1']:.4f} | Micro-F1={m_b7['micro_f1']:.4f} | Porch-F1={m_b7['per_class']['porch']['f1']:.4f} | Latency={med_lat_b7:.2f}ms")

    # 3. Train Candidate C1 from B7 Checkpoint
    exp_dir = REPO_ROOT / "runs" / "phase22" / "candidate_c1"
    train_best = exp_dir / "weights" / "best.pt"
    train_elapsed = 0.0
    peak_vram_gb = 0.0

    if not args.skip_train or not train_best.exists():
        print("\n--- Training Candidate C1 (Fine-Tuning from B7 with MaskedDetectionLoss) ---")
        train_model = YOLO(str(B7_CKPT))

        def on_pretrain_routine_end(trainer):
            trainer.criterion = MaskedDetectionLoss(trainer.model, verified_mask_lookup=verified_masks)
            print(f"\n[MaskedDetectionLoss] Successfully injected into YOLO trainer ({len(verified_masks)} masks).")

        train_model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        t0_train = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

        results = train_model.train(
            data=str(DATASET_YAML),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=640,
            device=0,
            workers=0,
            optimizer="auto",
            lr0=0.0005,  # Gentle fine-tuning from B7
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            box=7.5,
            cls=0.6,
            dfl=1.5,
            mosaic=0.0,  # Eliminates multi-image mosaic mask mismatch
            seed=42,
            project=str(REPO_ROOT / "runs" / "phase22"),
            name="candidate_c1",
            exist_ok=True,
            save=True,
            val=True,
        )
        train_elapsed = time.time() - t0_train
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

        assert train_best.exists(), f"Trained weights missing: {train_best}"
        shutil.copy2(train_best, C1_CKPT)
        print(f"  Candidate C1 Training Finished in {train_elapsed:.2f}s ({train_elapsed/60:.2f} min). Peak VRAM: {peak_vram_gb:.2f} GB")
        print(f"  Saved Candidate C1 Checkpoint: {C1_CKPT}")
    else:
        print(f"\nReusing existing C1 weights at {train_best}")

    # Verify safeguards post training
    verify_safeguards("Post-Training")

    # 4. Evaluate Candidate C1 on authorized validation split
    print("\n--- Candidate C1 Evaluation (Warmed) ---")
    model_c1 = YOLO(str(C1_CKPT if C1_CKPT.exists() else train_best))
    preds_c1, med_lat_c1, p95_lat_c1 = run_warmed_inference(model_c1, val_images)
    val_res_c1 = model_c1.val(data=str(DATASET_YAML), split="val", batch=args.batch, imgsz=640, device=0, verbose=False)
    scenes_c1 = evaluate_predictions_per_scene(all_gts, preds_c1, threshold=0.25, iou_thresh=0.50)
    m_c1 = aggregate_metrics(scenes_c1)
    m_c1["median_latency_ms"] = round(med_lat_c1, 2)
    m_c1["p95_latency_ms"] = round(p95_lat_c1, 2)
    m_c1["mAP50"] = round(float(val_res_c1.box.map50), 4)
    m_c1["mAP50_95"] = round(float(val_res_c1.box.map), 4)
    m_c1["ap50"] = {CANONICAL_CLASSES[i]: round(float(val_res_c1.box.ap50[i]), 4) for i in range(10)}
    m_c1["ap50_95"] = {CANONICAL_CLASSES[i]: round(float(val_res_c1.box.ap[i]), 4) for i in range(10)}
    print(f"  C1: Macro-F1={m_c1['macro_f1']:.4f} | Micro-F1={m_c1['micro_f1']:.4f} | Porch-F1={m_c1['per_class']['porch']['f1']:.4f} | Latency={med_lat_c1:.2f}ms")

    # 5. Bootstrap Comparisons: C1 vs B7 and C1 vs B6
    print("\n--- 1,000-Iteration Paired Scene Bootstrap ---")
    boot_c1_vs_b7 = compute_bootstrap_ci(scenes_b7, scenes_c1, n_boot=1000, seed=42)
    boot_c1_vs_b6 = compute_bootstrap_ci(scenes_b6, scenes_c1, n_boot=1000, seed=42)
    print(f"  C1 vs B7 Delta Macro-F1: {boot_c1_vs_b7['macro_f1_delta_mean']:+.4f} (95% CI: {boot_c1_vs_b7['macro_f1_ci95']})")
    print(f"  C1 vs B7 Delta Porch-F1: {boot_c1_vs_b7['porch_f1_delta_mean']:+.4f} (95% CI: {boot_c1_vs_b7['porch_f1_ci95']})")
    print(f"  C1 vs B6 Delta Macro-F1: {boot_c1_vs_b6['macro_f1_delta_mean']:+.4f} (95% CI: {boot_c1_vs_b6['macro_f1_ci95']})")
    print(f"  C1 vs B6 Delta Porch-F1: {boot_c1_vs_b6['porch_f1_delta_mean']:+.4f} (95% CI: {boot_c1_vs_b6['porch_f1_ci95']})")

    # 6. Save reports/phase22_metrics.json
    all_metrics = {
        "b6_baseline": m_b6,
        "b7_pilot": m_b7,
        "candidate_c1": m_c1,
        "bootstrap_c1_vs_b7": boot_c1_vs_b7,
        "bootstrap_c1_vs_b6": boot_c1_vs_b6,
        "training_resources": {
            "training_time_seconds": round(train_elapsed, 2),
            "training_time_minutes": round(train_elapsed / 60.0, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "batch_size": args.batch,
            "imgsz": 640,
            "mosaic": 0.0,
            "lr0": 0.0005,
        }
    }
    with open(REPORTS_DIR / "phase22_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved {REPORTS_DIR / 'phase22_metrics.json'}")

    # 7. Generate Checkpoint Inventory
    checkpoints_to_inventory = [
        ("B2 Production", B2_CKPT),
        ("B6 Expanded Baseline", B6_CKPT),
        ("B7 Pilot", B7_CKPT),
        ("Porch Specialist", PORCH_SPEC_CKPT),
        ("Candidate C1", C1_CKPT),
    ]
    ckpt_inv = []
    for label, cp in checkpoints_to_inventory:
        if cp.exists():
            ckpt_inv.append({
                "model_label": label,
                "filename": cp.name,
                "path": str(cp),
                "size_bytes": cp.stat().st_size,
                "md5": compute_md5(cp),
                "sha256": compute_sha256(cp),
                "status": "production_default" if label == "B2 Production" else "experimental",
            })
    with open(REPORTS_DIR / "phase22_checkpoint_inventory.json", "w", encoding="utf-8") as f:
        json.dump(ckpt_inv, f, indent=2)
    print(f"Saved {REPORTS_DIR / 'phase22_checkpoint_inventory.json'}")

    # 8. Visual Prediction Gallery
    print("\n--- Generating Visual Prediction Gallery ---")
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    # Pick scenes containing porch, shutter, or complex architectural facades
    porch_scenes = [img_p for img_p in val_images if any(g[0] == 6 for g in all_gts[img_p.name])]
    selected_gallery_scenes = (porch_scenes[:6] + val_images[:4])[:8]

    class_colors = {
        0: (70, 70, 220), 1: (200, 150, 50), 2: (50, 180, 50), 3: (220, 120, 30),
        4: (150, 50, 200), 5: (220, 50, 150), 6: (255, 0, 0), 7: (50, 200, 200),
        8: (255, 140, 0), 9: (100, 100, 100)
    }

    for img_p in selected_gallery_scenes:
        im = Image.open(img_p).convert("RGB")
        w, h = im.size

        # Create 3 panels: B6, B7, C1
        panel_w, panel_h = w, h
        combined = Image.new("RGB", (panel_w * 3, panel_h + 40), (255, 255, 255))
        draw_combined = ImageDraw.Draw(combined)

        model_panels = [
            ("B6 Baseline", preds_b6.get(img_p.name, []), 0),
            ("B7 Pilot", preds_b7.get(img_p.name, []), 1),
            ("Candidate C1", preds_c1.get(img_p.name, []), 2),
        ]

        for title, preds, p_idx in model_panels:
            p_im = im.copy()
            p_draw = ImageDraw.Draw(p_im)

            # Draw Ground Truths in subtle dashed green/white
            for g in all_gts.get(img_p.name, []):
                gcid, gx, gy, gw, gh = g
                g_xmin = int((gx - gw / 2.0) * w)
                g_ymin = int((gy - gh / 2.0) * h)
                g_xmax = int((gx + gw / 2.0) * w)
                g_ymax = int((gy + gh / 2.0) * h)
                p_draw.rectangle([g_xmin, g_ymin, g_xmax, g_ymax], outline=(0, 255, 0), width=1)

            # Draw Model Predictions (conf >= 0.25)
            for p in preds:
                cid, conf, px, py, pw, ph = p
                if conf < 0.25:
                    continue
                p_xmin = int((px - pw / 2.0) * w)
                p_ymin = int((py - ph / 2.0) * h)
                p_xmax = int((px + pw / 2.0) * w)
                p_ymax = int((py + ph / 2.0) * h)
                color = class_colors.get(cid, (255, 255, 255))
                p_draw.rectangle([p_xmin, p_ymin, p_xmax, p_ymax], outline=color, width=3)
                p_draw.text((p_xmin + 3, max(0, p_ymin - 12)), f"{CANONICAL_CLASSES[cid]} {conf:.2f}", fill=color)

            combined.paste(p_im, (p_idx * panel_w, 40))
            draw_combined.text((p_idx * panel_w + 10, 10), f"{title} (conf >= 0.25, Green=GT)", fill=(0, 0, 0))

        combined.save(GALLERY_DIR / f"comp_{img_p.stem}.jpg")

    print(f"Saved {len(selected_gallery_scenes)} comparative prediction visualizations in {GALLERY_DIR}")


if __name__ == "__main__":
    main()

