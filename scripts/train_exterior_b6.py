"""Phase 14 Stage C: Controlled Transfer-Learning Retraining for Model B6.

Trains Model B6 initialized from audited Model B5 weights:
  Base Weights: checkpoints/exterior_detector_b5_transfer_best.pt
  Dataset: data/exterior_b6_expanded/dataset.yaml (154 val images, locked)
  Output: checkpoints/exterior_detector_b6_expanded_best.pt

Safeguards:
- Verifies B2, B3, B4, and B5 checkpoint MD5 hashes before and after training.
- Never touches or uses the locked test split.
- Uses early stopping with patience=6.
- Fixed seed=42 for reproducibility.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
CONFIG_YAML = REPO_ROOT / "config" / "exterior_b6.yaml"
OUTPUT_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"

EXPECTED_PRE_HASHES = {
    "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
    "exterior_detector_b3_yolo11s_rfs_best.pt": "2c44291fa642aba3893db001520deba0",
    "exterior_detector_b4_yolo11s_curated_best.pt": "d2710fba760713e68ea306e029ab651c",
    "exterior_detector_b5_transfer_best.pt": "e12b4181fee244a3569dfa782ba3c3c3",
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


def main():
    parser = argparse.ArgumentParser(description="Train Model B6 with targeted data expansion")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test")
    args = parser.parse_args()

    verify_safeguards("Pre-Training")

    epochs = 1 if args.smoke_test else args.epochs
    patience = 1 if args.smoke_test else args.patience
    exp_name = "exterior_detector_b6_smoke" if args.smoke_test else "exterior_detector_b6_expanded"

    b5_weights = REPO_ROOT / "checkpoints" / "exterior_detector_b5_transfer_best.pt"
    dataset_yaml = REPO_ROOT / "data" / "exterior_b6_expanded" / "dataset.yaml"

    assert b5_weights.exists(), f"B5 base weights missing: {b5_weights}"
    assert dataset_yaml.exists(), f"Dataset YAML missing: {dataset_yaml}"

    print(f"\n=== Launching Model B6 Fine-Tuning ===")
    print(f"Base Weights: {b5_weights}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Epochs: {epochs}, Patience: {patience}, Batch Size: {args.batch}, Image Size: 640")

    model = YOLO(str(b5_weights))

    t0 = time.time()
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        patience=patience,
        batch=args.batch,
        imgsz=640,
        device=0,
        workers=0,
        optimizer="auto",
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0 if not args.smoke_test else 0.0,
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
    elapsed = time.time() - t0

    # Locate best checkpoint
    trained_best = REPO_ROOT / "runs" / "train" / exp_name / "weights" / "best.pt"
    assert trained_best.exists(), f"Trained best.pt missing at {trained_best}"

    # Determine peak VRAM
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f} min). Peak VRAM: {peak_vram_gb:.2f} GB")

    if not args.smoke_test:
        shutil.copy2(trained_best, OUTPUT_CKPT)
        h_best = hashlib.md5(OUTPUT_CKPT.read_bytes()).hexdigest()
        print(f"\nSaved Best B6 Checkpoint to: {OUTPUT_CKPT}")
        print(f"  MD5 Hash: {h_best} ({OUTPUT_CKPT.stat().st_size:,} bytes)")
    else:
        print("\nSmoke test completed successfully!")

    verify_safeguards("Post-Training")

    # Evaluate validation metrics
    print("\n--- Evaluating Model B6 on Validation Split ---")
    val_model = YOLO(str(trained_best))
    val_res = val_model.val(
        data=str(dataset_yaml),
        split="val",
        batch=args.batch,
        imgsz=640,
        device=0,
        verbose=False,
    )
    mAP50 = round(float(val_res.box.map50), 4)
    mAP50_95 = round(float(val_res.box.map), 4)
    print(f"Validation mAP50: {mAP50:.4f} | mAP50-95: {mAP50_95:.4f}")
    print("Per-class AP50:")
    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        ap50 = round(float(val_res.box.ap50[cid]), 4)
        print(f"  {cname:<16}: {ap50:.4f}")

    print("\nModel B6 training process successfully completed!")


if __name__ == "__main__":
    main()
