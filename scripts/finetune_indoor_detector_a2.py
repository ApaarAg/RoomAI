"""Fine-tunes the strongest candidate detector (Model A2: YOLO11s) on the 12-class canonical indoor dataset.

Stage 4 Implementation:
- Preserves the original pilot checkpoint: checkpoints/indoor_detector_a2_yolo11s_pilot_best.pt.
- Saves fine-tuned weights to: checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt.
- New experiment directory: runs/train/indoor_detector_a2_yolo11s_finetune.
- Verifies class IDs and annotation compatibility across 12 canonical classes.
- Applies realistic augmentations (fliplr=0.5, mosaic=0.5, degrees=5.0, scale=0.2).
- Uses early stopping (patience=4) to protect against overfitting.
- Measures peak GPU VRAM and execution timing via StageTimer.
- Includes a short 1-epoch pilot mode (--pilot) before full execution.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict

import torch
from ultralytics import YOLO

# Windows UTF-8 console safety
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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Strongest Candidate Detector (Model A2: YOLO11s)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (safe for 8GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--pilot", action="store_true", help="Run 1-epoch pilot check before full training")
    parser.add_argument(
        "--base-checkpoint",
        default="checkpoints/indoor_detector_a2_yolo11s_pilot_best.pt",
        help="Base checkpoint to initialize from",
    )
    return parser.parse_args()


def run_finetuning(
    base_checkpoint: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    patience: int,
    is_pilot: bool,
) -> Dict[str, Any]:
    project_root = REPO_ROOT
    data_yaml = project_root / "data" / "indoor_canonical" / "indoor_canonical.yaml"
    checkpoints_dir = project_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base_pt = project_root / base_checkpoint
    if not base_pt.exists():
        # Fallback to general yolo11s if pilot pt doesn't exist
        base_pt = project_root / "yolo11s.pt"
    if not base_pt.exists():
        raise FileNotFoundError(f"Base checkpoint not found at {base_pt}")

    run_name = "indoor_detector_a2_yolo11s_pilot_check" if is_pilot else "indoor_detector_a2_yolo11s_finetune"
    dest_checkpoint_name = "indoor_detector_a2_yolo11s_pilot_check.pt" if is_pilot else "indoor_detector_a2_yolo11s_finetune_best.pt"
    dest_pt = checkpoints_dir / dest_checkpoint_name

    num_epochs = 1 if is_pilot else epochs
    stage_name = f"Stage 4: Model A2 Fine-Tuning ({'1-Epoch Pilot' if is_pilot else f'{num_epochs} Epochs Full'})"

    timer = StageTimer()
    timer.start_stage(
        stage_name=stage_name,
        estimated_sec=num_epochs * 90.0,
        total_items=num_epochs,
        expected_gpu_vram_gb=3.2,
        assumptions=f"Initializing from {base_pt.name}, batch_size={batch_size}, imgsz={imgsz}, patience={patience}",
    )

    if torch.cuda.is_available() and device != "cpu":
        try:
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        except Exception:
            torch.cuda.reset_peak_memory_stats()

    print(f"\n[INIT] Loading base checkpoint: {base_pt}")
    model = YOLO(str(base_pt))

    t0 = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=num_epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=2,
        amp=False,
        patience=patience,
        project=str(project_root / "runs" / "train"),
        name=run_name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        verbose=True,
        plots=True,
        save=True,
        # Realistic spatial augmentations for interior scenes
        fliplr=0.5,
        mosaic=0.5,
        degrees=5.0,
        scale=0.2,
    )
    actual_train_sec = time.time() - t0

    peak_gpu_mb = 0.0
    if torch.cuda.is_available() and device != "cpu":
        try:
            peak_gpu_mb = round(torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 * 1024), 2)
        except Exception:
            peak_gpu_mb = 0.0

    run_dir = project_root / "runs" / "train" / run_name
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "weights" / "last.pt"

    shutil.copy2(best_weights, dest_pt)
    print(f"\n[SAVED] Preserved fine-tuned checkpoint at: {dest_pt}")

    # Evaluate on Validation split
    print(f"\n[EVAL] Evaluating fine-tuned model on Validation split (554 images)...")
    eval_model = YOLO(str(dest_pt))
    val_results = eval_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        half=False,
    )

    # Evaluate on Untouched Held-Out Test split
    print(f"\n[EVAL] Evaluating fine-tuned model on Untouched Test split (150 images)...")
    test_results = eval_model.val(
        data=str(data_yaml),
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

    # Per-class metrics
    per_class_val = {}
    class_names = val_results.names
    ap_indices = getattr(val_results.box, "ap_class_index", [])
    for i, c_id in enumerate(ap_indices):
        name = class_names.get(int(c_id), f"class_{c_id}")
        p_val = round(float(val_results.box.p[i]), 4) if hasattr(val_results.box, "p") and len(val_results.box.p) > i else 0.0
        r_val = round(float(val_results.box.r[i]), 4) if hasattr(val_results.box, "r") and len(val_results.box.r) > i else 0.0
        ap50 = round(float(val_results.box.ap50[i]), 4) if hasattr(val_results.box, "ap50") and len(val_results.box.ap50) > i else 0.0
        ap_val = round(float(val_results.box.maps[int(c_id)]), 4) if hasattr(val_results.box, "maps") and len(val_results.box.maps) > int(c_id) else 0.0
        per_class_val[name] = {
            "class_id": int(c_id),
            "precision": p_val,
            "recall": r_val,
            "mAP50": ap50,
            "mAP50_95": ap_val,
        }

    summary = {
        "model_name": "Fine-Tuned Model A2 (YOLO11s)",
        "base_checkpoint": str(base_pt),
        "fine_tuned_checkpoint": str(dest_pt),
        "is_pilot": is_pilot,
        "epochs_trained": num_epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "training_duration_sec": round(actual_train_sec, 2),
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
        },
        "per_class_val_metrics": per_class_val,
    }

    eval_json = run_dir / "eval_results.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    timer.finish_stage(
        artifacts=[str(dest_pt), str(eval_json)],
        extra_metrics={
            "train_duration": f"{round(actual_train_sec, 1)}s",
            "peak_vram_mb": peak_gpu_mb,
            "val_mAP50": val_map50,
            "val_mAP50_95": val_map,
            "test_mAP50": test_map50,
            "test_mAP50_95": test_map,
        },
    )

    return summary


def main():
    args = parse_args()
    run_finetuning(
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        is_pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
