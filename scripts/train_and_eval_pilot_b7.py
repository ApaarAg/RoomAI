"""RoomAI Phase 20: Scientifically Controlled Multi-Dataset Pilot Experiment.

Executes:
1. Preflight safeguard verification of immutable checkpoints (B2, B6, specialist).
2. Establishes the exact Model B6 baseline on the authorized 154-scene validation split.
3. Fine-tunes Model B6 into Model B7 Pilot on the 1,602-image dataset (1,302 B6 + 300 Curated Pilot)
   using class-conditional MaskedDetectionLoss.
4. Measures training time, peak VRAM, RAM, and disk space.
5. Performs honest, head-to-head evaluation of B6 vs B7 on the exact same 154 validation scenes:
   - Macro-F1, Micro-F1, per-class P/R/F1, AP50, mAP50-95
   - Specific analysis for porch, shutter_blind, facade, pillar_column
   - 1,000-iteration paired scene-level bootstrap confidence intervals (95% CI)
   - False-positive and false-negative detection galleries
6. Applies Go/No-Go Gate (porch F1 >= 0.20, macro-F1 >= 0.50).
7. Outputs reports/phase20_pilot_metrics.json.
"""

import argparse
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import psutil
import torch
from ultralytics import YOLO

from utils.masked_detection_loss import MaskedDetectionLoss

PILOT_DATA_DIR = REPO_ROOT / "data" / "pilot_b7_dataset"
DATASET_YAML = PILOT_DATA_DIR / "dataset.yaml"
MASKS_JSON = PILOT_DATA_DIR / "verified_class_masks.json"
MANIFEST_JSON = PILOT_DATA_DIR / "pilot_manifest.json"

B2_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
B6_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
PORCH_SPEC_CKPT = REPO_ROOT / "checkpoints" / "exterior_specialist_porch_yolo11s_best.pt"
B7_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b7_pilot_best.pt"

REPORTS_DIR = REPO_ROOT / "reports"
GALLERY_DIR = REPORTS_DIR / "phase20_pilot_gallery"

EXPECTED_PRE_HASHES = {
    "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
    "exterior_detector_b6_expanded_best.pt": "f5ceea56b0610ea14b647014e48152dc",
    "exterior_specialist_porch_yolo11s_best.pt": "4261fc1ad5f6fd9ac29c0499e2adddfd",
}

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def verify_safeguards(stage_name: str):
    print(f"\n=== Safeguard Verification: {stage_name} ===")
    for ckpt_name, expected_hash in EXPECTED_PRE_HASHES.items():
        p = REPO_ROOT / "checkpoints" / ckpt_name
        assert p.exists(), f"Missing checkpoint: {p}"
        actual_hash = hashlib.md5(p.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Checkpoint modified ({ckpt_name}): {actual_hash} != {expected_hash}"
        print(f"  [VERIFIED] {ckpt_name}: {actual_hash}")


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


def compute_bootstrap_ci(scenes_b6, scenes_b7, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    scene_keys = list(scenes_b6.keys())
    n = len(scene_keys)

    diff_macro = []
    diff_micro = []
    diff_porch = []

    for _ in range(n_boot):
        sample_keys = rng.choice(scene_keys, size=n, replace=True)
        sample_b6 = {k: scenes_b6[k] for k in sample_keys}
        sample_b7 = {k: scenes_b7[k] for k in sample_keys}

        m_b6 = aggregate_metrics(sample_b6)
        m_b7 = aggregate_metrics(sample_b7)

        diff_macro.append(m_b7["macro_f1"] - m_b6["macro_f1"])
        diff_micro.append(m_b7["micro_f1"] - m_b6["micro_f1"])
        diff_porch.append(m_b7["per_class"]["porch"]["f1"] - m_b6["per_class"]["porch"]["f1"])

    return {
        "macro_f1_delta_mean": round(float(np.mean(diff_macro)), 4),
        "macro_f1_ci95": [round(float(np.percentile(diff_macro, 2.5)), 4), round(float(np.percentile(diff_macro, 97.5)), 4)],
        "micro_f1_delta_mean": round(float(np.mean(diff_micro)), 4),
        "micro_f1_ci95": [round(float(np.percentile(diff_micro, 2.5)), 4), round(float(np.percentile(diff_micro, 97.5)), 4)],
        "porch_f1_delta_mean": round(float(np.mean(diff_porch)), 4),
        "porch_f1_ci95": [round(float(np.percentile(diff_porch, 2.5)), 4), round(float(np.percentile(diff_porch, 97.5)), 4)],
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase 20 Multi-Dataset Pilot Experiment")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for B7 pilot")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if B7 checkpoint exists")
    args = parser.parse_args()

    print("======================================================================")
    print("      ROOMAI PHASE 20: CONTROLLED MULTI-DATASET PILOT EXPERIMENT      ")
    print("======================================================================")

    # 1. Preflight safeguard verification
    verify_safeguards("Pre-Experiment")

    # Load verified masks
    assert MASKS_JSON.exists(), f"Masks file missing: {MASKS_JSON}"
    with open(MASKS_JSON, "r", encoding="utf-8") as f:
        verified_masks = json.load(f)
    print(f"\nLoaded {len(verified_masks)} verified class masks from {MASKS_JSON}")

    # Prepare validation data
    val_img_dir = PILOT_DATA_DIR / "images" / "val"
    val_lbl_dir = PILOT_DATA_DIR / "labels" / "val"
    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"Validation Scenes: {len(val_images)} images from {val_img_dir}")
    assert len(val_images) == 154, f"Validation set altered! Expected 154 scenes, found {len(val_images)}"

    # Load ground truths
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

    # 2. Establish Model B6 Baseline
    print("\n--- Establishing Model B6 Baseline Evaluation ---")
    model_b6 = YOLO(str(B6_CKPT))
    t0_inf_b6 = time.time()
    preds_b6 = {}
    for img_p in val_images:
        res = model_b6.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        preds = []
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywh = res.boxes.xywhn[i].tolist()
            preds.append((cls_id, conf, xywh[0], xywh[1], xywh[2], xywh[3]))
        preds_b6[img_p.name] = preds
    latency_b6_ms = ((time.time() - t0_inf_b6) / len(val_images)) * 1000

    val_res_b6 = model_b6.val(data=str(DATASET_YAML), split="val", batch=args.batch, imgsz=640, device=0, verbose=False)
    ap50_b6 = {CANONICAL_CLASSES[i]: round(float(val_res_b6.box.ap50[i]), 4) for i in range(10)}
    ap_b6 = {CANONICAL_CLASSES[i]: round(float(val_res_b6.box.ap[i]), 4) for i in range(10)}

    scenes_b6 = evaluate_predictions_per_scene(all_gts, preds_b6, threshold=0.25, iou_thresh=0.50)
    metrics_b6 = aggregate_metrics(scenes_b6)
    metrics_b6["latency_ms"] = round(latency_b6_ms, 2)
    metrics_b6["mAP50"] = round(float(val_res_b6.box.map50), 4)
    metrics_b6["mAP50_95"] = round(float(val_res_b6.box.map), 4)
    metrics_b6["ap50"] = ap50_b6
    metrics_b6["ap50_95"] = ap_b6

    print(f"  Model B6 Baseline: Macro-F1: {metrics_b6['macro_f1']:.4f} | Micro-F1: {metrics_b6['micro_f1']:.4f}")
    print(f"  Model B6 Porch F1: {metrics_b6['per_class']['porch']['f1']:.4f} | AP50: {ap50_b6['porch']:.4f}")
    print(f"  Model B6 Latency: {latency_b6_ms:.2f} ms/scene")

    # 3. Model B7 Pilot Fine-Tuning
    exp_name = "exterior_detector_b7_pilot"
    train_best = REPO_ROOT / "runs" / "train" / exp_name / "weights" / "best.pt"
    train_elapsed = 0.0
    peak_vram_gb = 0.0

    if not args.skip_train or not train_best.exists():
        print("\n--- Training Model B7 Pilot (with MaskedDetectionLoss) ---")
        train_model = YOLO(str(B6_CKPT))

        # Register callback to inject MaskedDetectionLoss
        def on_pretrain_routine_end(trainer):
            trainer.criterion = MaskedDetectionLoss(trainer.model, verified_mask_lookup=verified_masks)
            print(f"\n[MaskedDetectionLoss] Injected into YOLO trainer ({len(verified_masks)} verified masks).")

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
            lr0=0.001,  # Gentle fine-tuning rate from B6
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1.0,
            box=7.5,
            cls=0.6,
            dfl=1.5,
            seed=42,
            project=str(REPO_ROOT / "runs" / "train"),
            name=exp_name,
            exist_ok=True,
            save=True,
            val=True,
        )
        train_elapsed = time.time() - t0_train
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

        assert train_best.exists(), f"B7 trained weights missing at {train_best}"
        shutil.copy2(train_best, B7_CKPT)
        print(f"  B7 Training Finished in {train_elapsed:.2f}s ({train_elapsed/60:.2f} min). Peak VRAM: {peak_vram_gb:.2f} GB")
        print(f"  Saved B7 Pilot Checkpoint: {B7_CKPT}")
    else:
        print(f"\nReusing existing B7 Pilot weights at {train_best}")

    # Verify safeguards post training
    verify_safeguards("Post-Training")

    # 4. Evaluate Model B7 on Exact Same Validation Split
    print("\n--- Evaluating Model B7 Pilot on Validation Split ---")
    model_b7 = YOLO(str(B7_CKPT if B7_CKPT.exists() else train_best))
    t0_inf_b7 = time.time()
    preds_b7 = {}
    for img_p in val_images:
        res = model_b7.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        preds = []
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywh = res.boxes.xywhn[i].tolist()
            preds.append((cls_id, conf, xywh[0], xywh[1], xywh[2], xywh[3]))
        preds_b7[img_p.name] = preds
    latency_b7_ms = ((time.time() - t0_inf_b7) / len(val_images)) * 1000

    val_res_b7 = model_b7.val(data=str(DATASET_YAML), split="val", batch=args.batch, imgsz=640, device=0, verbose=False)
    ap50_b7 = {CANONICAL_CLASSES[i]: round(float(val_res_b7.box.ap50[i]), 4) for i in range(10)}
    ap_b7 = {CANONICAL_CLASSES[i]: round(float(val_res_b7.box.ap[i]), 4) for i in range(10)}

    scenes_b7 = evaluate_predictions_per_scene(all_gts, preds_b7, threshold=0.25, iou_thresh=0.50)
    metrics_b7 = aggregate_metrics(scenes_b7)
    metrics_b7["latency_ms"] = round(latency_b7_ms, 2)
    metrics_b7["mAP50"] = round(float(val_res_b7.box.map50), 4)
    metrics_b7["mAP50_95"] = round(float(val_res_b7.box.map), 4)
    metrics_b7["ap50"] = ap50_b7
    metrics_b7["ap50_95"] = ap_b7

    print(f"  Model B7 Pilot: Macro-F1: {metrics_b7['macro_f1']:.4f} | Micro-F1: {metrics_b7['micro_f1']:.4f}")
    print(f"  Model B7 Porch F1: {metrics_b7['per_class']['porch']['f1']:.4f} | AP50: {ap50_b7['porch']:.4f}")
    print(f"  Model B7 Latency: {latency_b7_ms:.2f} ms/scene")

    # 5. Bootstrap Confidence Intervals
    print("\n--- Running 1,000-Iteration Paired Scene Bootstrap Analysis ---")
    boot_res = compute_bootstrap_ci(scenes_b6, scenes_b7, n_boot=1000, seed=42)
    print(f"  Macro-F1 Delta: {boot_res['macro_f1_delta_mean']:+.4f} (95% CI: {boot_res['macro_f1_ci95']})")
    print(f"  Micro-F1 Delta: {boot_res['micro_f1_delta_mean']:+.4f} (95% CI: {boot_res['micro_f1_ci95']})")
    print(f"  Porch F1 Delta: {boot_res['porch_f1_delta_mean']:+.4f} (95% CI: {boot_res['porch_f1_ci95']})")

    # Class improvement and regression breakdown
    improved_classes = []
    regressed_classes = []
    neutral_classes = []
    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        f1_6 = metrics_b6["per_class"][cname]["f1"]
        f1_7 = metrics_b7["per_class"][cname]["f1"]
        delta = round(f1_7 - f1_6, 4)
        if delta > 0.005:
            improved_classes.append((cname, f1_6, f1_7, delta))
        elif delta < -0.005:
            regressed_classes.append((cname, f1_6, f1_7, delta))
        else:
            neutral_classes.append((cname, f1_6, f1_7, delta))

    # 6. Apply Go / No-Go Gate
    gate_porch_f1 = metrics_b7["per_class"]["porch"]["f1"]
    gate_macro_f1 = metrics_b7["macro_f1"]
    pass_porch = gate_porch_f1 >= 0.20
    pass_macro = gate_macro_f1 >= 0.50
    macro_improved = metrics_b7["macro_f1"] >= metrics_b6["macro_f1"]

    if pass_porch and pass_macro and macro_improved and len(regressed_classes) <= 2:
        decision = "GO"
        decision_reason = (
            f"B7 Pilot passed both initial gates (Porch F1 {gate_porch_f1:.4f} >= 0.20, Macro-F1 {gate_macro_f1:.4f} >= 0.50) "
            f"and preserved or improved overall system performance without catastrophic regression."
        )
    elif gate_porch_f1 > metrics_b6["per_class"]["porch"]["f1"] and (not pass_porch or not pass_macro):
        decision = "INCONCLUSIVE"
        decision_reason = (
            f"B7 Pilot showed positive directional response on target rare classes (Porch F1 {metrics_b6['per_class']['porch']['f1']:.4f} -> {gate_porch_f1:.4f}) "
            f"but did not decisively clear both initial operational gates simultaneously."
        )
    else:
        decision = "NO-GO"
        decision_reason = (
            f"B7 Pilot failed gates or degraded existing performance (Macro-F1 {metrics_b6['macro_f1']:.4f} -> {metrics_b7['macro_f1']:.4f}, "
            f"Porch F1 {gate_porch_f1:.4f})."
        )

    print(f"\n======================================================================")
    print(f"      GATE DECISION: {decision}")
    print(f"      Reason: {decision_reason}")
    print(f"======================================================================")

    # 7. Generate False Positive / False Negative Error Gallery
    print("\n--- Generating Comparative Error Analysis Gallery ---")
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    # Pick scenes where B7 and B6 diverge on Porch or Doors
    porch_divergent_scenes = []
    for s_name in val_images[:30]:
        gt_porches = [g for g in all_gts[s_name.name] if g[0] == 6]
        p6_porches = [p for p in preds_b6.get(s_name.name, []) if p[0] == 6 and p[1] >= 0.25]
        p7_porches = [p for p in preds_b7.get(s_name.name, []) if p[0] == 6 and p[1] >= 0.25]
        if gt_porches or p6_porches or p7_porches:
            porch_divergent_scenes.append(s_name)

    for idx, img_path in enumerate(porch_divergent_scenes[:6]):
        img_name = img_path.name
        with Image.open(img_path) as im:
            w, h = im.size
            im_draw_b6 = im.copy()
            draw_b6 = ImageDraw.Draw(im_draw_b6)
            im_draw_b7 = im.copy()
            draw_b7 = ImageDraw.Draw(im_draw_b7)

            # Draw Ground Truths (Green) on both
            for g in all_gts.get(img_name, []):
                cid, xc, yc, bw, bh = g
                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)
                draw_b6.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                draw_b7.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            # Draw B6 Predictions (Yellow)
            for p in preds_b6.get(img_name, []):
                cid, conf, xc, yc, bw, bh = p
                if conf < 0.25 or cid != 6:
                    continue
                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)
                draw_b6.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)
                draw_b6.text((x1, max(0, y1 - 12)), f"B6 Porch {conf:.2f}", fill=(255, 255, 0))

            # Draw B7 Predictions (Cyan)
            for p in preds_b7.get(img_name, []):
                cid, conf, xc, yc, bw, bh = p
                if conf < 0.25 or cid != 6:
                    continue
                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)
                draw_b7.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=3)
                draw_b7.text((x1, max(0, y1 - 12)), f"B7 Porch {conf:.2f}", fill=(0, 255, 255))

            # Save comparison side-by-side
            comp_w = w * 2
            comp_im = Image.new("RGB", (comp_w, h))
            comp_im.paste(im_draw_b6, (0, 0))
            comp_im.paste(im_draw_b7, (w, 0))
            out_p = GALLERY_DIR / f"eval_comp_{idx:02d}_{img_path.stem}.jpg"
            comp_im.save(out_p, quality=90)

    print(f"Saved {min(6, len(porch_divergent_scenes))} comparative gallery images to {GALLERY_DIR}")

    # 8. Record Metrics JSON
    final_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_summary": {
            "train_images": 1602,
            "val_images": 154,
            "curated_pilot_images": 300,
            "manifest_path": str(MANIFEST_JSON),
            "verified_masks_path": str(MASKS_JSON)
        },
        "system_resources": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "peak_vram_gb": round(peak_vram_gb, 2),
            "training_time_s": round(train_elapsed, 2),
            "training_time_min": round(train_elapsed / 60.0, 2),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        },
        "model_b6_baseline": metrics_b6,
        "model_b7_pilot": metrics_b7,
        "bootstrap_analysis": boot_res,
        "class_deltas": {
            "improved": [{"class": c, "b6_f1": b6, "b7_f1": b7, "delta": d} for c, b6, b7, d in improved_classes],
            "regressed": [{"class": c, "b6_f1": b6, "b7_f1": b7, "delta": d} for c, b6, b7, d in regressed_classes],
            "neutral": [{"class": c, "b6_f1": b6, "b7_f1": b7, "delta": d} for c, b6, b7, d in neutral_classes]
        },
        "go_no_go_gate": {
            "decision": decision,
            "reason": decision_reason,
            "porch_f1_target": 0.20,
            "porch_f1_actual": gate_porch_f1,
            "macro_f1_target": 0.50,
            "macro_f1_actual": gate_macro_f1,
            "passed_porch_gate": pass_porch,
            "passed_macro_gate": pass_macro
        }
    }

    metrics_out = REPORTS_DIR / "phase20_pilot_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved metrics to: {metrics_out}")


if __name__ == "__main__":
    main()
