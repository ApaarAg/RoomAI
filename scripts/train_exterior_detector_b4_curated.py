"""Trains Exterior Detector Model B4 (YOLO11s) on the Curated Exterior V2 Dataset.

Experiment:
- Base weights: yolo11s.pt (identical COCO base weights as B2 and B3)
- Dataset: data/exterior_v2_curated/exterior_v2_curated.yaml (876 images: 568 train, 154 val, 154 test)
- Resolution: 640px
- Batch size: 4
- Epochs: 20 (patience: 5, early stopping based on validation mAP50)
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- Checkpoint output: checkpoints/exterior_detector_b4_yolo11s_curated_best.pt

Non-negotiable safeguards:
- B2 (checkpoints/exterior_detector_b2_yolo11s_best.pt) and B3 (checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt)
  hashes verified before and after training.
- Active production config (config/model_config.yaml) is NOT modified.
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

# DLL paths for Windows
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
sys.path.insert(0, str(REPO_ROOT))

DATASET_YAML = REPO_ROOT / "data" / "exterior_v2_curated" / "exterior_v2_curated.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
BASE_CHECKPOINT = REPO_ROOT / "yolo11s.pt"

SAFEGUARD_CHECKPOINTS = {
    "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
    "exterior_detector_b3_yolo11s_rfs_best.pt": "2c44291fa642aba3893db001520deba0",
}


def verify_safeguard_checkpoints():
    print("Verifying baseline checkpoints preservation...")
    for ckpt_name, expected_hash in SAFEGUARD_CHECKPOINTS.items():
        p = CHECKPOINTS_DIR / ckpt_name
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        with open(p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h != expected_hash:
            raise ValueError(f"Hash mismatch for {ckpt_name}: expected {expected_hash}, got {h}")
        print(f"  [OK] {ckpt_name}: {h}")


def run_training(
    epochs: int = 20,
    patience: int = 5,
    batch_size: int = 4,
    imgsz: int = 640,
    seed: int = 42,
    device: str = "0",
) -> Dict[str, Any]:
    verify_safeguard_checkpoints()

    run_name = "exterior_detector_b4_curated"
    run_dir = REPO_ROOT / "runs" / "train" / run_name
    if run_dir.exists():
        print(f"Cleaning previous run dir: {run_dir}")
        shutil.rmtree(run_dir)

    print(f"\n=== Training Model B4 on Curated Exterior V2 Dataset ===")
    print(f"Dataset YAML: {DATASET_YAML}")
    print(f"Base weights: {BASE_CHECKPOINT}")
    print(f"Epochs: {epochs}, Patience: {patience}, Batch: {batch_size}, Imgsz: {imgsz}, Seed: {seed}")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass

    start_wall_time = time.time()
    model = YOLO(str(BASE_CHECKPOINT))

    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=2,
        seed=seed,
        project=str(REPO_ROOT / "runs" / "train"),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        amp=False,
        verbose=True,
    )

    elapsed_wall_sec = time.time() - start_wall_time

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        try:
            peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        except Exception:
            pass

    # Best weights path
    best_weights_path = run_dir / "weights" / "best.pt"
    if not best_weights_path.exists():
        raise FileNotFoundError(f"Training failed to produce best weights at {best_weights_path}")

    # Copy to checkpoints/exterior_detector_b4_yolo11s_curated_best.pt
    dest_ckpt_path = CHECKPOINTS_DIR / "exterior_detector_b4_yolo11s_curated_best.pt"
    shutil.copy2(best_weights_path, dest_ckpt_path)

    with open(dest_ckpt_path, "rb") as f:
        b4_hash = hashlib.md5(f.read()).hexdigest()
    b4_bytes = dest_ckpt_path.stat().st_size

    print(f"\n=== Model B4 Training Complete ===")
    print(f"Wall-clock elapsed time: {elapsed_wall_sec:.1f}s ({elapsed_wall_sec / 60.0:.2f} min)")
    print(f"Peak GPU VRAM during training: {peak_vram_mb:.1f} MB ({peak_vram_mb / 1024.0:.2f} GB)")
    print(f"Saved Checkpoint: {dest_ckpt_path}")
    print(f"Checkpoint MD5: {b4_hash} ({b4_bytes:,} bytes)")

    # Post-training safeguard check
    verify_safeguard_checkpoints()

    # Save training metadata
    train_metadata = {
        "model_name": "exterior_detector_b4_yolo11s_curated",
        "dataset": str(DATASET_YAML),
        "base_weights": str(BASE_CHECKPOINT),
        "epochs_requested": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "seed": seed,
        "device": device,
        "elapsed_wall_sec": elapsed_wall_sec,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_path": str(dest_ckpt_path),
        "checkpoint_md5": b4_hash,
        "checkpoint_bytes": b4_bytes,
        "safeguards_verified": True,
    }

    meta_path = REPO_ROOT / "runs" / "train" / run_name / "b4_training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, indent=2)
    print(f"Saved training metadata to {meta_path}")

    return train_metadata


if __name__ == "__main__":
    run_training()
