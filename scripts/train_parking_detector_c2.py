"""Trains and evaluates Parking & Outdoor Property Detector Model C2 (YOLO11s) on the Canonical 8-Class Dataset.

Architecture:
- Base Weights: COCO pre-trained yolo11s.pt (9.43M parameters, higher spatial resolution)
- Dataset: data/parking_canonical/parking_canonical.yaml (8 canonical classes)
- Checkpoint output: checkpoints/parking_detector_c2_yolo11s_pilot_best.pt
- Comprehensive validation & test split evaluation
- Instrumentation with StageTimer and peak GPU VRAM tracking.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Reconfigure stdout/stderr for Windows UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.timing_utils import StageTimer, format_duration

DATASET_YAML = REPO_ROOT / "data" / "parking_canonical" / "parking_canonical.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
BASE_CHECKPOINT = REPO_ROOT / "yolo11s.pt"


def run_training(
    base_model_path: Path = BASE_CHECKPOINT,
    epochs: int = 15,
    patience: int = 5,
    batch_size: int = 4,
    imgsz: int = 640,
    device: str = "0",
    is_pilot: bool = False,
    run_name: str = "parking_detector_c2_yolo11s",
) -> Dict[str, Any]:
    """Trains Model C2 on the canonical 8-class parking/outdoor dataset with StageTimer reporting."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if is_pilot:
        run_name = f"{run_name}_pilot"
        epochs = 2
        patience = 2

    run_dir = REPO_ROOT / "runs" / "train" / run_name

    timer = StageTimer()
    estimated_duration_sec = epochs * 24.0
    timer.start_stage(
        stage_name=f"Model C2 (YOLO11s Parking/Outdoor) {'Pilot' if is_pilot else 'Full'} Training",
        estimated_sec=estimated_duration_sec,
        total_items=epochs,
        expected_gpu_vram_gb=2.4,
        assumptions=f"{epochs} epochs on 510 training images, batch_size={batch_size}, imgsz={imgsz}",
    )

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        except Exception:
            pass

    model = YOLO(str(base_model_path))

    t0 = time.time()
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=2,
        project=str(REPO_ROOT / "runs" / "train"),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        amp=False,
        verbose=True,
    )
    actual_train_sec = time.time() - t0

    peak_gpu_mb = 0.0
    if torch.cuda.is_available():
        try:
            peak_gpu_mb = round(torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 * 1024), 2)
        except Exception:
            pass

    # Save best checkpoint
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    src_pt = best_pt if best_pt.exists() else last_pt
    dest_pt = CHECKPOINTS_DIR / f"{run_name}_best.pt"

    if src_pt.exists():
        shutil.copy2(src_pt, dest_pt)
        print(f"[CHECKPOINT] Saved best weights: {dest_pt}")

    # Evaluate on validation split
    val_model = YOLO(str(dest_pt if dest_pt.exists() else src_pt))
    print(f"\n[EVAL] Evaluating {dest_pt.name} on Validation split...")
    val_results = val_model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        workers=2,
        half=False,
    )

    # Evaluate on held-out Test split
    print(f"\n[EVAL] Evaluating {dest_pt.name} on Held-Out Test split (120 images)...")
    test_results = val_model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        workers=2,
        half=False,
    )

    val_map50 = round(float(val_results.box.map50), 4)
    val_map = round(float(val_results.box.map), 4)
    val_p = round(float(val_results.box.mp), 4)
    val_r = round(float(val_results.box.mr), 4)

    test_map50 = round(float(test_results.box.map50), 4)
    test_map = round(float(test_results.box.map), 4)
    test_p = round(float(test_results.box.mp), 4)
    test_r = round(float(test_results.box.mr), 4)
    test_f1 = round(2 * test_p * test_r / (test_p + test_r + 1e-8), 4)

    # Per-class test performance
    class_names = getattr(test_results, "names", {})
    per_class_test = {}
    if hasattr(test_results.box, "maps") and len(test_results.box.maps) > 0:
        for idx, cls_map in enumerate(test_results.box.maps):
            cname = class_names.get(idx, f"class_{idx}")
            per_class_test[cname] = round(float(cls_map), 4)

    metrics_payload = {
        "model_name": "Model C2 (YOLO11s Parking/Outdoor)",
        "run_name": run_name,
        "is_pilot": is_pilot,
        "checkpoint_path": str(dest_pt),
        "actual_train_sec": round(actual_train_sec, 2),
        "peak_gpu_vram_mb": peak_gpu_mb,
        "val_metrics": {
            "mAP50": val_map50,
            "mAP50_95": val_map,
            "precision": val_p,
            "recall": val_r,
        },
        "test_metrics": {
            "mAP50": test_map50,
            "mAP50_95": test_map,
            "precision": test_p,
            "recall": test_r,
            "f1": test_f1,
            "per_class_mAP50": per_class_test,
        },
    }

    results_json = run_dir / ("pilot_results.json" if is_pilot else "finetune_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    timer.finish_stage(
        artifacts=[str(dest_pt), str(results_json)],
        extra_metrics={
            "val_mAP50": val_map50,
            "test_mAP50": test_map50,
            "test_F1": test_f1,
            "peak_vram_mb": peak_gpu_mb,
            "duration": format_duration(actual_train_sec),
        },
    )

    print(f"\n================ MODEL C2 TRAINING SUMMARY ================")
    print(f"Checkpoint Saved : {dest_pt}")
    print(f"Training Time    : {format_duration(actual_train_sec)} ({actual_train_sec:.1f}s)")
    print(f"Peak GPU VRAM    : {peak_gpu_mb} MB ({peak_gpu_mb / 1024:.2f} GB)")
    print(f"Val mAP@0.50     : {val_map50:.4f}")
    print(f"Test mAP@0.50    : {test_map50:.4f}")
    print(f"Test Precision   : {test_p:.4f} | Recall: {test_r:.4f} | F1: {test_f1:.4f}")
    print("Per-Class Test mAP50:")
    for cname, cmap in per_class_test.items():
        print(f"  - {cname:18s}: {cmap:.4f}")
    print("============================================================\n")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model C2 (YOLO11s Parking/Outdoor)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for full training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--pilot", action="store_true", default=False, help="Run 2-epoch pilot run")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        is_pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
