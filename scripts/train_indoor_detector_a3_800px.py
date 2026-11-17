"""Trains and evaluates Indoor Detector Model A3 (YOLO11s at 800px Higher Resolution).

Experiment B: Controlled intervention testing higher-resolution training at 800 pixels (imgsz=800)
to improve small-object feature retention on chair#stool (42.3% small) and fine decorations.

Protocol:
- Base weights: checkpoints/indoor_detector_a2_yolo11s_pilot_best.pt (identical base weights as A2)
- Architecture: YOLO11s (9.4M params)
- Resolution: imgsz=800 (controlled intervention: +56.25% pixel expansion)
- Batch size: 4 (safe for 8GB VRAM)
- Epochs: 6, patience: 3, seed: 42, amp: False, workers: 2
- Checkpoint output: checkpoints/indoor_detector_a3_yolo11s_800px_best.pt
- Checkpoints A1, A2, and DeepFurn rollback are NEVER modified.
- Full StageTimer instrumentation and peak GPU VRAM tracking.
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

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATASET_YAML = REPO_ROOT / "data" / "indoor_canonical" / "indoor_canonical.yaml"
BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "indoor_detector_a2_yolo11s_pilot_best.pt"

CLASS_NAMES = {
    0: "cabinet#shelf",
    1: "table",
    2: "chair#stool",
    3: "lamp",
    4: "door",
    5: "bed",
    6: "sofa",
    7: "plant",
    8: "decoration",
    9: "curtain",
    10: "home-appliance",
    11: "window",
}


def train_and_eval_a3(
    base_model_path: Path = BASE_CHECKPOINT,
    epochs: int = 6,
    patience: int = 3,
    batch_size: int = 4,
    imgsz: int = 800,
    device: str = "0",
    seed: int = 42,
    run_name: str = "indoor_detector_a3_yolo11s_800px",
) -> Dict[str, Any]:
    """Trains Model A3 (YOLO11s at 800px) and evaluates on validation and held-out test splits."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if not base_model_path.exists():
        base_model_path = REPO_ROOT / "yolo11s.pt"

    run_dir = REPO_ROOT / "runs" / "train" / run_name

    timer = StageTimer()
    estimated_duration_sec = epochs * 140.0  # ~840s estimated
    timer.start_stage(
        stage_name="Model A3 (YOLO11s 800px Indoor) Controlled Training",
        estimated_sec=estimated_duration_sec,
        total_items=epochs,
        expected_gpu_vram_gb=4.2,
        assumptions=f"{epochs} epochs on 2985 indoor train images, batch_size={batch_size}, imgsz={imgsz}, seed={seed}",
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
        seed=seed,
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

    # Save best checkpoint to dedicated A3 path (never overwrite A2!)
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    src_pt = best_pt if best_pt.exists() else last_pt
    dest_pt = CHECKPOINTS_DIR / f"{run_name}_best.pt"

    if src_pt.exists():
        shutil.copy2(src_pt, dest_pt)
        print(f"\n[CHECKPOINT] Successfully saved candidate weights to: {dest_pt}")
    else:
        raise FileNotFoundError(f"Neither best.pt nor last.pt found in {run_dir / 'weights'}")

    # Evaluate on untouched validation split at 800px
    val_model = YOLO(str(dest_pt))
    print(f"\n[EVAL] Evaluating candidate A3 on Validation split (554 images, imgsz={imgsz})...")
    val_results = val_model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        workers=2,
        half=False,
    )

    # Evaluate on held-out Test split (150 images, imgsz={imgsz})
    print(f"\n[EVAL] Evaluating candidate A3 on Held-Out Test split (150 images, imgsz={imgsz})...")
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
    val_f1 = round((2 * val_p * val_r) / (val_p + val_r + 1e-6) if (val_p + val_r) > 0 else 0.0, 4)

    test_map50 = round(float(test_results.box.map50), 4)
    test_map = round(float(test_results.box.map), 4)
    test_p = round(float(test_results.box.mp), 4)
    test_r = round(float(test_results.box.mr), 4)
    test_f1 = round((2 * test_p * test_r) / (test_p + test_r + 1e-6) if (test_p + test_r) > 0 else 0.0, 4)

    # Extract per-class test metrics
    per_class_test = {}
    class_names = test_results.names
    ap_class_indices = getattr(test_results.box, "ap_class_index", [])
    for i, c_id in enumerate(ap_class_indices):
        name = class_names.get(int(c_id), f"class_{c_id}")
        p_val = round(float(test_results.box.p[i]), 4) if hasattr(test_results.box, "p") and len(test_results.box.p) > i else 0.0
        r_val = round(float(test_results.box.r[i]), 4) if hasattr(test_results.box, "r") and len(test_results.box.r) > i else 0.0
        ap50 = round(float(test_results.box.ap50[i]), 4) if hasattr(test_results.box, "ap50") and len(test_results.box.ap50) > i else 0.0
        ap_val = round(float(test_results.box.maps[int(c_id)]), 4) if hasattr(test_results.box, "maps") and len(test_results.box.maps) > int(c_id) else 0.0
        f1_c = round((2 * p_val * r_val) / (p_val + r_val + 1e-6) if (p_val + r_val) > 0 else 0.0, 4)
        per_class_test[name] = {
            "class_id": int(c_id),
            "precision": p_val,
            "recall": r_val,
            "f1": f1_c,
            "mAP50": ap50,
            "mAP50_95": ap_val,
        }
    for c_id, name in class_names.items():
        if name not in per_class_test:
            per_class_test[name] = {
                "class_id": int(c_id),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mAP50": 0.0,
                "mAP50_95": 0.0,
            }

    eval_summary = {
        "model_name": "Indoor Detector Model A3 (YOLO11s 800px)",
        "architecture": "YOLO11s",
        "base_model": str(base_model_path),
        "intervention": "Higher Resolution Training (imgsz=800, batch=4)",
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "seed": seed,
        "train_duration_sec": round(actual_train_sec, 2),
        "sec_per_epoch": round(actual_train_sec / epochs, 2),
        "peak_gpu_mem_mb": peak_gpu_mb,
        "val_metrics": {
            "mAP50": val_map50,
            "mAP50_95": val_map,
            "precision": val_p,
            "recall": val_r,
            "f1": val_f1,
        },
        "test_metrics": {
            "mAP50": test_map50,
            "mAP50_95": test_map,
            "precision": test_p,
            "recall": test_r,
            "f1": test_f1,
        },
        "per_class_test_metrics": per_class_test,
        "checkpoint_path": str(dest_pt),
    }

    eval_json_path = REPO_ROOT / "reports" / "indoor_a3_800px_eval.json"
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
            "test_f1": test_f1,
            "peak_gpu_vram_mb": peak_gpu_mb,
        },
    )

    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Train Indoor Detector Model A3 (800px)")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=800, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="Target CUDA device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base-checkpoint", type=str, default=str(BASE_CHECKPOINT), help="Base checkpoint path")
    args = parser.parse_args()

    train_and_eval_a3(
        base_model_path=Path(args.base_checkpoint),
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
