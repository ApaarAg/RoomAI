"""Trains and evaluates Exterior Detector Model B1 (YOLO11n) on the Canonical 10-Class Exterior Dataset.

Architecture:
- Base Weights: COCO pre-trained yolo11n.pt
- Dataset: data/exterior_canonical/exterior_canonical.yaml (10 canonical classes)
- Checkpoint output: checkpoints/exterior_detector_b1_yolo11n_best.pt
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

DATASET_YAML = REPO_ROOT / "data" / "exterior_canonical" / "exterior_canonical.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
BASE_CHECKPOINT = REPO_ROOT / "yolo11n.pt"


def run_training(
    base_model_path: Path = BASE_CHECKPOINT,
    epochs: int = 15,
    patience: int = 5,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "0",
    is_pilot: bool = False,
    run_name: str = "exterior_detector_b1_yolo11n",
) -> Dict[str, Any]:
    """Trains Model B1 on the canonical 10-class exterior dataset with StageTimer reporting."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if is_pilot:
        run_name = f"{run_name}_pilot"
        epochs = 2
        patience = 2

    run_dir = REPO_ROOT / "runs" / "train" / run_name

    timer = StageTimer()
    estimated_duration_sec = epochs * 8.0
    timer.start_stage(
        stage_name=f"Model B1 (YOLO11n Exterior) {'Pilot' if is_pilot else 'Full'} Training",
        estimated_sec=estimated_duration_sec,
        total_items=epochs,
        expected_gpu_vram_gb=1.8,
        assumptions=f"{epochs} epochs on 458 training images, batch_size={batch_size}, imgsz={imgsz}",
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
        half=False,
    )

    # Evaluate on held-out Test split
    print(f"\n[EVAL] Evaluating {dest_pt.name} on Held-Out Test split (134 images)...")
    test_results = val_model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
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

    # Per-class metrics on test split
    per_class_test = {}
    class_names = test_results.names
    ap_class_indices = getattr(test_results.box, "ap_class_index", [])
    for i, c_id in enumerate(ap_class_indices):
        name = class_names.get(int(c_id), f"class_{c_id}")
        p_val = round(float(test_results.box.p[i]), 4) if hasattr(test_results.box, "p") and len(test_results.box.p) > i else 0.0
        r_val = round(float(test_results.box.r[i]), 4) if hasattr(test_results.box, "r") and len(test_results.box.r) > i else 0.0
        ap50 = round(float(test_results.box.ap50[i]), 4) if hasattr(test_results.box, "ap50") and len(test_results.box.ap50) > i else 0.0
        ap_val = round(float(test_results.box.maps[int(c_id)]), 4) if hasattr(test_results.box, "maps") and len(test_results.box.maps) > int(c_id) else 0.0
        per_class_test[name] = {
            "class_id": int(c_id),
            "precision": p_val,
            "recall": r_val,
            "mAP50": ap50,
            "mAP50_95": ap_val,
        }
    for c_id, name in class_names.items():
        if name not in per_class_test:
            per_class_test[name] = {
                "class_id": int(c_id),
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50_95": 0.0,
                "note": "Class not present in test split ground truth",
            }

    eval_summary = {
        "model_name": f"Exterior Detector Model B1 ({'Pilot' if is_pilot else 'Full'})",
        "architecture": "YOLO11n",
        "base_model": str(base_model_path),
        "epochs_requested": epochs,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "train_duration_sec": round(actual_train_sec, 2),
        "sec_per_epoch": round(actual_train_sec / epochs, 2),
        "peak_gpu_mem_mb": peak_gpu_mb,
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
        },
        "per_class_test_metrics": per_class_test,
        "checkpoint_path": str(dest_pt),
    }

    eval_json_path = run_dir / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    timer.finish_stage(
        artifacts=[str(dest_pt), str(eval_json_path)],
        extra_metrics={
            "actual_sec_per_epoch": f"{eval_summary['sec_per_epoch']}s",
            "val_mAP50": val_map50,
            "test_mAP50": test_map50,
            "test_mAP50_95": test_map,
            "test_precision": test_p,
            "test_recall": test_r,
            "peak_gpu_vram_mb": peak_gpu_mb,
        },
    )

    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Train Exterior Detector Model B1 (YOLO11n)")
    parser.add_argument("--pilot", action="store_true", help="Run 2-epoch pilot for profiling and ETA estimation")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for full training (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Target CUDA device (default: 0)")
    args = parser.parse_args()

    run_training(
        base_model_path=BASE_CHECKPOINT,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        is_pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
