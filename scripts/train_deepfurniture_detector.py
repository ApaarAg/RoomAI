"""Train and fine-tune Ultralytics YOLO model on the DeepFurniture dataset."""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "data" / "deepfurniture" / "deepfurniture.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def train_deepfurniture(
    base_model: str = "checkpoints/deepfurniture_yolo11n_pilot_best.pt",
    epochs: int = 30,
    patience: int = 7,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "0",
    run_name: str = "deepfurniture_yolo11n_finetune",
) -> Path:
    """Fine-tunes YOLO on DeepFurniture annotations and logs comprehensive performance."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = REPO_ROOT / "runs" / "train" / run_name

    print(f"\n========================================================")
    print(f" DeepFurniture YOLO Fine-Tuning Pipeline")
    print(f"========================================================")
    print(f" Base model:       {base_model}")
    print(f" Dataset config:   {DEFAULT_YAML}")
    print(f" Epochs:           {epochs}")
    print(f" Early stopping:   patience={patience}")
    print(f" Batch size:       {batch_size}")
    print(f" Image size:       {imgsz}")
    print(f" Run name:         {run_name}")
    print(f" Output run dir:   {run_dir}")
    print(f" Target device:    {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"========================================================\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    model = YOLO(base_model)

    t0 = time.time()
    results = model.train(
        data=str(DEFAULT_YAML),
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
    train_duration_sec = time.time() - t0

    peak_gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)

    print(f"\nTraining completed in {train_duration_sec:.2f}s ({train_duration_sec / 60:.2f} min).")
    if peak_gpu_mem_mb > 0:
        print(f"Peak GPU VRAM allocated: {peak_gpu_mem_mb} MB ({peak_gpu_mem_mb / 1024:.2f} GB)")

    # Locate best weights from this specific run (selected via validation split)
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    dest_weights = CHECKPOINTS_DIR / f"{run_name}_best.pt"

    source_weights = best_weights if best_weights.exists() else last_weights
    if source_weights.exists():
        shutil.copy2(source_weights, dest_weights)
        print(f"\nSaved best model-selected checkpoint to: {dest_weights}")
    else:
        print(f"\nWarning: Could not find checkpoint file in {run_dir / 'weights'}")

    # Evaluate validation split
    eval_model = YOLO(str(dest_weights if dest_weights.exists() else source_weights))
    print("\nRunning final evaluation on Validation Split (for model selection verification)...")
    val_results = eval_model.val(
        data=str(DEFAULT_YAML),
        split="val",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        half=False,
    )

    # Evaluate on untouched test split (ONLY after final model selection)
    print("\nRunning final evaluation on UNTOUCHED Test Split...")
    test_results = eval_model.val(
        data=str(DEFAULT_YAML),
        split="test",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        half=False,
    )

    val_map50 = float(val_results.box.map50)
    val_map50_95 = float(val_results.box.map)
    val_precision = float(val_results.box.p.mean()) if hasattr(val_results.box.p, "mean") else float(val_results.box.mp)
    val_recall = float(val_results.box.r.mean()) if hasattr(val_results.box.r, "mean") else float(val_results.box.mr)

    test_map50 = float(test_results.box.map50)
    test_map50_95 = float(test_results.box.map)
    test_precision = float(test_results.box.p.mean()) if hasattr(test_results.box.p, "mean") else float(test_results.box.mp)
    test_recall = float(test_results.box.r.mean()) if hasattr(test_results.box.r, "mean") else float(test_results.box.mr)

    # Per-class metrics from test set
    per_class_metrics = {}
    class_names = test_results.names
    for idx, cname in class_names.items():
        p_val = float(test_results.box.p[idx]) if idx < len(test_results.box.p) else 0.0
        r_val = float(test_results.box.r[idx]) if idx < len(test_results.box.r) else 0.0
        m50_val = float(test_results.box.all_ap[idx, 0]) if hasattr(test_results.box, "all_ap") and idx < test_results.box.all_ap.shape[0] else 0.0
        m_val = float(test_results.box.maps[idx]) if idx < len(test_results.box.maps) else 0.0
        per_class_metrics[cname] = {
            "class_id": idx,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "mAP50": round(m50_val, 4),
            "mAP50_95": round(m_val, 4),
        }

    # Extract Confusion Matrix if available
    confusion_dict = {}
    if hasattr(test_results, "confusion_matrix") and test_results.confusion_matrix is not None:
        try:
            matrix = test_results.confusion_matrix.matrix
            confusion_dict["shape"] = list(matrix.shape)
            # Focus on chair#stool (2), sofa (6), home-appliance (10)
            target_ids = {"chair#stool": 2, "sofa": 6, "home-appliance": 10}
            confusion_patterns = {}
            for target_name, tid in target_ids.items():
                if tid < matrix.shape[0]:
                    row = matrix[tid]
                    top_confusions = []
                    for other_id in range(matrix.shape[1]):
                        if other_id != tid and row[other_id] > 0:
                            other_name = class_names.get(other_id, "background")
                            top_confusions.append((other_name, int(row[other_id])))
                    top_confusions.sort(key=lambda x: x[1], reverse=True)
                    confusion_patterns[target_name] = {
                        "correct_detections": int(row[tid]),
                        "misclassified_as": top_confusions[:3],
                    }
            confusion_dict["patterns"] = confusion_patterns
        except Exception as e:
            confusion_dict["error"] = str(e)

    # Read training curves from results.csv
    results_csv = run_dir / "results.csv"
    training_history = []
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k.strip(): float(v.strip()) for k, v in row.items() if v.strip()}
                training_history.append(clean_row)

    summary = {
        "model_name": f"DeepFurniture YOLO Fine-Tune ({run_name})",
        "base_model": base_model,
        "dataset": "Dalby123/DeepFurniture",
        "epochs_requested": epochs,
        "epochs_trained": len(training_history),
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "train_duration_sec": round(train_duration_sec, 2),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "val_metrics": {
            "mAP50": round(val_map50, 4),
            "mAP50_95": round(val_map50_95, 4),
            "precision": round(val_precision, 4),
            "recall": round(val_recall, 4),
        },
        "test_metrics": {
            "mAP50": round(test_map50, 4),
            "mAP50_95": round(test_map50_95, 4),
            "precision": round(test_precision, 4),
            "recall": round(test_recall, 4),
        },
        "per_class_test_metrics": per_class_metrics,
        "confusion_analysis": confusion_dict,
        "checkpoint_path": str(dest_weights),
    }

    eval_json = run_dir / "finetune_results.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========================================================")
    print(" Validation Split Performance (Model Selection):")
    print(f"   mAP@0.50:       {val_map50:.4f}")
    print(f"   mAP@0.50:0.95:  {val_map50_95:.4f}")
    print(f"   Precision:      {val_precision:.4f}")
    print(f"   Recall:         {val_recall:.4f}")
    print("--------------------------------------------------------")
    print(" Untouched Test Split Performance (Final Evaluation):")
    print(f"   mAP@0.50:       {test_map50:.4f}")
    print(f"   mAP@0.50:0.95:  {test_map50_95:.4f}")
    print(f"   Precision:      {test_precision:.4f}")
    print(f"   Recall:         {test_recall:.4f}")
    print(f"   Saved report:   {eval_json}")
    print("========================================================\n")

    return dest_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on DeepFurniture")
    parser.add_argument("--base", type=str, default="checkpoints/deepfurniture_yolo11n_pilot_best.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--name", type=str, default="deepfurniture_yolo11n_finetune", help="Run directory name")
    args = parser.parse_args()

    train_deepfurniture(
        base_model=args.base,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        run_name=args.name,
    )
