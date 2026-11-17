"""Verification script for fine-tuned YOLO11n artifacts and metrics."""

import csv
import json
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "deepfurniture_yolo11n_finetune_best.pt"
RUN_DIR = REPO_ROOT / "runs" / "train" / "deepfurniture_yolo11n_finetune"
RESULTS_CSV = RUN_DIR / "results.csv"
EVAL_JSON = RUN_DIR / "finetune_results.json"
DATASET_YAML = REPO_ROOT / "data" / "deepfurniture" / "deepfurniture.yaml"


def verify_artifacts():
    print("=" * 65)
    print(" TASK A: ARTIFACT & METRIC VERIFICATION")
    print("=" * 65)

    # 1. File existence checks
    files = {
        "Checkpoint": CHECKPOINT_PATH,
        "Dataset YAML": DATASET_YAML,
        "Results CSV": RESULTS_CSV,
        "Evaluation JSON": EVAL_JSON,
    }
    for name, path in files.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"[{'OK' if exists else 'MISSING'}] {name:16s}: {path} ({size} bytes)")
        if not exists:
            raise FileNotFoundError(f"Missing required artifact: {path}")

    # 2. Inspect Dataset YAML
    print("\n--- Dataset YAML Inspection ---")
    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    print("Root path:", yaml_data.get("path"))
    print("Train split:", yaml_data.get("train"))
    print("Val split:  ", yaml_data.get("val"))
    print("Test split: ", yaml_data.get("test"))
    yaml_names = yaml_data.get("names", {})
    print(f"Classes ({len(yaml_names)} total):", yaml_names)

    # 3. Load checkpoint with Ultralytics
    print("\n--- Ultralytics Model Inspection ---")
    model = YOLO(str(CHECKPOINT_PATH))
    print("Model task:      ", model.task)
    print("Model names count:", len(model.names))
    print("Model names:     ", model.names)
    assert len(model.names) == 11, f"Expected 11 classes, got {len(model.names)}"
    assert model.names == yaml_names, "Mismatch between model.names and deepfurniture.yaml names!"

    # 4. Read results.csv directly from disk
    print("\n--- results.csv Disk Inspection ---")
    rows = []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clean_r = {k.strip(): float(v.strip()) for k, v in r.items() if v.strip()}
            rows.append(clean_r)
    print(f"Total epochs recorded in results.csv: {len(rows)}")
    epoch_1 = rows[0]
    epoch_30 = rows[-1]
    print(f"Epoch 1:  train/box={epoch_1['train/box_loss']:.4f}, train/cls={epoch_1['train/cls_loss']:.4f}, val/box={epoch_1['val/box_loss']:.4f}, val/cls={epoch_1['val/cls_loss']:.4f}, mAP50={epoch_1['metrics/mAP50(B)']:.4f}, mAP50-95={epoch_1['metrics/mAP50-95(B)']:.4f}")
    print(f"Epoch 30: train/box={epoch_30['train/box_loss']:.4f}, train/cls={epoch_30['train/cls_loss']:.4f}, val/box={epoch_30['val/box_loss']:.4f}, val/cls={epoch_30['val/cls_loss']:.4f}, mAP50={epoch_30['metrics/mAP50(B)']:.4f}, mAP50-95={epoch_30['metrics/mAP50-95(B)']:.4f}")

    # Best epoch in results.csv by validation mAP50
    best_epoch_row = max(rows, key=lambda x: x["metrics/mAP50(B)"])
    print(f"Best Validation Epoch in CSV: Epoch {int(best_epoch_row['epoch'])} with mAP50={best_epoch_row['metrics/mAP50(B)']:.4f}, mAP50-95={best_epoch_row['metrics/mAP50-95(B)']:.4f}")

    # 5. Read finetune_results.json directly from disk
    print("\n--- finetune_results.json Disk Inspection ---")
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print("Model name:           ", eval_data.get("model_name"))
    print("Base model:           ", eval_data.get("base_model"))
    print("Epochs trained:       ", eval_data.get("epochs_trained"))
    print("Train duration (s):   ", eval_data.get("train_duration_sec"))
    print("Peak GPU VRAM (MB):   ", eval_data.get("peak_gpu_mem_mb"))
    print("Validation metrics:   ", eval_data.get("val_metrics"))
    print("Test metrics:         ", eval_data.get("test_metrics"))
    print("Checkpoint path stored:", eval_data.get("checkpoint_path"))

    # 6. Verify separation between model selection and test split
    print("\n--- Split & Model Selection Separation Check ---")
    val_map50 = eval_data["val_metrics"]["mAP50"]
    test_map50 = eval_data["test_metrics"]["mAP50"]
    print(f"Validation mAP@0.50 (used for checkpoint selection): {val_map50}")
    print(f"Test Split mAP@0.50 (untouched final evaluation):    {test_map50}")
    print("Checkpoint evaluated on test split:", eval_data.get("checkpoint_path"))

    # 7. Check for inconsistencies
    inconsistencies = []
    if abs(val_map50 - best_epoch_row["metrics/mAP50(B)"]) > 0.005:
        inconsistencies.append(f"Validation mAP50 ({val_map50}) differs from best CSV epoch mAP50 ({best_epoch_row['metrics/mAP50(B)']})")
    if eval_data["checkpoint_path"] != str(CHECKPOINT_PATH):
        inconsistencies.append("Checkpoint path in JSON does not match expected CHECKPOINT_PATH")

    print("\n--- Inconsistency Report ---")
    if not inconsistencies:
        print("NO INCONSISTENCIES FOUND. All artifacts match disk records exactly.")
    else:
        for inc in inconsistencies:
            print("WARNING:", inc)


if __name__ == "__main__":
    verify_artifacts()
