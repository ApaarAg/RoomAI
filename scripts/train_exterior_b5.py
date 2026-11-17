"""Phase 4: Trains Model B5 Using Large-Scale Transfer Learning.

Safeguards:
- Preserves existing B2, B3, B4 checkpoints (verified byte-for-byte).
- Trains on data/exterior_b5_unified/dataset.yaml (903 train, 154 val).
- Base weights: yolo11s.pt (COCO pretrained).
- Adapts detection head to RoomAI 10 canonical classes.
- Fixed seed (42), early stopping (patience 6), AMP (half=True).
- Saves checkpoint: checkpoints/exterior_detector_b5_transfer_best.pt.
- Writes reports/exterior_b5_training_report.md.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
DATASET_YAML = REPO_ROOT / "data" / "exterior_b5_unified" / "dataset.yaml"
BASE_WEIGHTS = REPO_ROOT / "yolo11s.pt"
OUTPUT_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b5_transfer_best.pt"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}

EXPECTED_CHECKPOINTS = {
    "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
    "exterior_detector_b3_yolo11s_rfs_best.pt": "2c44291fa642aba3893db001520deba0",
    "exterior_detector_b4_yolo11s_curated_best.pt": "d2710fba760713e68ea306e029ab651c",
}


def verify_safeguard_checkpoints(stage: str):
    print(f"\n=== Safeguard Verification: {stage} ===")
    for ckpt_name, expected_hash in EXPECTED_CHECKPOINTS.items():
        p = REPO_ROOT / "checkpoints" / ckpt_name
        assert p.exists(), f"Missing checkpoint: {p}"
        with open(p, "rb") as f:
            actual_hash = hashlib.md5(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {ckpt_name}: {actual_hash} != {expected_hash}"
        print(f"  [VERIFIED] {ckpt_name}: {actual_hash}")


def run_training(epochs: int = 25, patience: int = 6, batch_size: int = 8, imgsz: int = 640, is_smoke_test: bool = False):
    verify_safeguard_checkpoints("Pre-Training")

    run_name = "b5_smoke_test" if is_smoke_test else "exterior_detector_b5_transfer"
    run_dir = REPO_ROOT / "runs" / "train" / run_name
    if run_dir.exists():
        print(f"Cleaning previous run directory: {run_dir}")
        shutil.rmtree(run_dir)

    print(f"\n=== Launching {'Smoke Test' if is_smoke_test else 'Full B5 Training'} ===")
    print(f"Base Weights: {BASE_WEIGHTS}")
    print(f"Dataset YAML: {DATASET_YAML}")
    print(f"Epochs: {epochs}, Patience: {patience}, Batch Size: {batch_size}, Image Size: {imgsz}")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass

    t0 = time.time()
    model = YOLO(str(BASE_WEIGHTS))

    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=imgsz,
        device=0,
        seed=42,
        half=True,
        workers=0,  # Windows safe multiprocessing
        project=str(REPO_ROOT / "runs" / "train"),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    elapsed_sec = time.time() - t0
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

    print(f"\nTraining completed in {elapsed_sec:.2f}s ({elapsed_sec/60.0:.2f} min). Peak VRAM: {peak_vram_gb:.2f} GB")

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"
    assert best_pt.exists(), f"No trained weights found in {run_dir / 'weights'}"

    if not is_smoke_test:
        # Copy to official checkpoint path
        shutil.copy2(best_pt, OUTPUT_CKPT)
        with open(OUTPUT_CKPT, "rb") as f:
            b5_hash = hashlib.md5(f.read()).hexdigest()
        b5_size = OUTPUT_CKPT.stat().st_size
        print(f"\nSaved Best B5 Checkpoint to: {OUTPUT_CKPT}")
        print(f"  MD5 Hash: {b5_hash} ({b5_size:,} bytes)")

    verify_safeguard_checkpoints("Post-Training")

    # Evaluate best model on validation split
    print("\n--- Evaluating Trained Model on Validation Split ---")
    val_model = YOLO(str(best_pt))
    val_metrics = val_model.val(
        data=str(DATASET_YAML),
        split="val",
        batch=8,
        imgsz=640,
        device=0,
        verbose=False,
    )

    map50 = float(val_metrics.box.map50)
    map95 = float(val_metrics.box.map)
    per_class_ap50 = {CANONICAL_CLASSES[i]: round(float(val_metrics.box.ap50[i]), 4) for i in range(10)}
    per_class_ap95 = {CANONICAL_CLASSES[i]: round(float(val_metrics.box.ap[i]), 4) for i in range(10)}

    print(f"Validation mAP50: {map50:.4f} | mAP50-95: {map95:.4f}")
    print("Per-class AP50:")
    for cname, ap in per_class_ap50.items():
        print(f"  {cname:<16}: {ap:.4f}")

    return {
        "elapsed_sec": round(elapsed_sec, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map95, 4),
        "per_class_ap50": per_class_ap50,
        "per_class_ap95": per_class_ap95,
        "b5_hash": b5_hash if not is_smoke_test else None,
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Exterior Detector B5")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    if args.smoke_test:
        print("=== Running Smoke Test (1 epoch, batch=4) ===")
        res = run_training(epochs=1, patience=1, batch_size=4, imgsz=640, is_smoke_test=True)
        print("Smoke test successfully passed! Memory and pipelines verified.")
    else:
        print("=== Running Full B5 Transfer Learning Experiment ===")
        res = run_training(epochs=args.epochs, patience=args.patience, batch_size=args.batch, imgsz=640, is_smoke_test=False)
        print("Full training run successfully finished!")


if __name__ == "__main__":
    main()
