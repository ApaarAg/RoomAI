"""Model A2 Training & Evaluation Script (YOLO11s complementary architecture).

Trains an independently initialized, higher-capacity YOLO11s detector on
data/indoor_canonical/indoor_canonical.yaml to provide architectural complementarity
with the YOLO11n Model A1.

Features:
- StageTimer integration with ETA, elapsed time, and peak GPU VRAM tracking.
- Fixed seed (42), early stopping, and automatic best-checkpoint preservation.
- Comprehensive evaluation on both validation (12 classes) and untouched test (150 images).
- Clean JSON summary export and terminal audit logging.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

from utils.timing_utils import StageTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Indoor Detector Model A2 (YOLO11s)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of full training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (safe for 8GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--pilot", action="store_true", help="Run 2-epoch pilot run to measure throughput")
    return parser.parse_args()


def run_training(epochs: int, batch_size: int, imgsz: int, device: str, patience: int, is_pilot: bool):
    stage_name = f"Indoor Detector Model A2 (YOLO11s) {'Pilot' if is_pilot else 'Full Training'}"
    run_name = "indoor_detector_a2_yolo11s_pilot" if is_pilot else "indoor_detector_a2_yolo11s"
    
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "data" / "indoor_canonical" / "indoor_canonical.yaml"
    base_model_path = project_root / "yolo11s.pt"
    checkpoints_dir = project_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {data_yaml}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model checkpoint not found at {base_model_path}")

    # Estimate timing
    # YOLO11s has ~9.4M params (approx 1.5x slower per epoch than YOLO11n on RTX 4060)
    est_sec_per_epoch = 90.0
    est_total_duration = epochs * est_sec_per_epoch

    timer = StageTimer()
    timer.start_stage(
        stage_name=stage_name,
        estimated_sec=est_total_duration,
        total_items=epochs,
        expected_gpu_vram_gb=3.5,
        assumptions=f"YOLO11s COCO pre-trained, {epochs} epochs, batch_size={batch_size}, imgsz={imgsz}",
    )

    # Reset CUDA stats
    if torch.cuda.is_available() and device != "cpu":
        try:
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        except Exception:
            torch.cuda.reset_peak_memory_stats()

    model = YOLO(str(base_model_path))

    # Launch Ultralytics training
    t0 = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
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
    )
    actual_train_sec = time.time() - t0

    # Measure peak GPU memory
    peak_gpu_mb = 0.0
    if torch.cuda.is_available() and device != "cpu":
        try:
            peak_gpu_mb = round(torch.cuda.max_memory_allocated(torch.device("cuda:0")) / (1024 * 1024), 2)
        except Exception:
            peak_gpu_mb = 0.0

    # Locate best checkpoint
    run_dir = project_root / "runs" / "train" / run_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    dest_checkpoint_name = "indoor_detector_a2_yolo11s_pilot_best.pt" if is_pilot else "indoor_detector_a2_yolo11s_best.pt"
    dest_pt = checkpoints_dir / dest_checkpoint_name
    shutil.copy2(best_pt, dest_pt)
    print(f"\n[OK] Copied best checkpoint to: {dest_pt}")

    # Evaluate on Validation split (12 classes)
    print(f"\n[EVAL] Evaluating {dest_checkpoint_name} on Validation split...")
    best_model = YOLO(str(dest_pt))
    val_results = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        half=False,
    )

    # Evaluate on untouched held-out Test split (150 images)
    print(f"\n[EVAL] Evaluating {dest_checkpoint_name} on Untouched Test split (150 images)...")
    test_results = best_model.val(
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
        "model_name": f"Indoor Detector Model A2 ({'Pilot' if is_pilot else 'Full'})",
        "architecture": "YOLO11s",
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
    args = parse_args()
    epochs = 2 if args.pilot else args.epochs
    run_training(
        epochs=epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        is_pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
