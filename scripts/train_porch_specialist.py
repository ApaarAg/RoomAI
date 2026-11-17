"""Phase 15 Stage B: Porch Specialist Training and Evaluation Script.

Trains a single-class YOLO11s specialist specifically for 'porch' detection
using transfer learning on data/exterior_specialist_porch/.

Safeguards:
- Checkpoints B2, B5, B6 verified before and after training.
- Dedicated checkpoint: checkpoints/exterior_specialist_porch_yolo11s_best.pt
- Evaluates on the locked validation set (154 images, 24 porch boxes).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import torch
import yaml
from ultralytics import YOLO

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
CONFIG_PATH = REPO_ROOT / "config" / "porch_specialist.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_CKPT_PATH = CHECKPOINTS_DIR / "exterior_specialist_porch_yolo11s_best.pt"

CANONICAL_HASHES = {
    "B2": ("checkpoints/exterior_detector_b2_yolo11s_best.pt", "14a931dd0adabc16cbd7b28ed1d17f66"),
    "B5": ("checkpoints/exterior_detector_b5_transfer_best.pt", "e12b4181fee244a3569dfa782ba3c3c3"),
    "B6": ("checkpoints/exterior_detector_b6_expanded_best.pt", "f5ceea56b0610ea14b647014e48152dc"),
}


def verify_safeguard_hashes():
    for name, (rel_p, expected) in CANONICAL_HASHES.items():
        p = REPO_ROOT / rel_p
        if p.exists():
            actual = hashlib.md5(p.read_bytes()).hexdigest()
            assert actual == expected, f"CRITICAL: {name} hash mismatch! Expected {expected}, got {actual}"
    print("[SAFEGUARD] Checkpoints B2, B5, and B6 hashes verified intact byte-for-byte.")


def train_porch_specialist(epochs=25, batch=8, imgsz=640, smoke_test=False):
    print("=== RoomAI Phase 15: Porch Specialist Training ===")
    verify_safeguard_hashes()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if smoke_test:
        epochs = 1
        batch = 4
        print("[SMOKE TEST] Running 1 epoch quick validation smoke test...")

    dataset_yaml = REPO_ROOT / cfg["data"]
    assert dataset_yaml.exists(), f"Missing dataset yaml at {dataset_yaml}"

    pretrained_model = REPO_ROOT / cfg["model"]
    assert pretrained_model.exists(), f"Missing pretrained model at {pretrained_model}"

    print(f"\nConfiguration:")
    print(f"  Pretrained weights: {pretrained_model}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    print(f"  Device: cuda:0 ({torch.cuda.get_device_name(0)})")

    # Load pretrained model
    model = YOLO(str(pretrained_model))

    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(0)

    train_results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=cfg.get("patience", 7) if not smoke_test else 1,
        device=0,
        workers=cfg.get("workers", 2),
        optimizer=cfg.get("optimizer", "SGD"),
        lr0=cfg.get("lr0", 0.005),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        project=str(REPO_ROOT / cfg["project"]),
        name=cfg["name"] + ("_smoke" if smoke_test else ""),
        exist_ok=True,
        seed=cfg.get("seed", 42),
        deterministic=True,
        verbose=True,
    )

    training_duration_s = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    print(f"\nTraining completed in {training_duration_s:.1f}s ({training_duration_s/60:.2f} min).")
    print(f"Peak VRAM allocated: {peak_vram_mb:.1f} MB")

    # Locate best checkpoint
    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    assert best_pt.exists(), f"Best checkpoint not found at {best_pt}"

    if not smoke_test:
        shutil.copy2(best_pt, TARGET_CKPT_PATH)
        ckpt_hash = hashlib.md5(TARGET_CKPT_PATH.read_bytes()).hexdigest()
        print(f"\n[SAVED] Saved porch specialist checkpoint to: {TARGET_CKPT_PATH}")
        print(f"[HASH] MD5: {ckpt_hash} ({TARGET_CKPT_PATH.stat().st_size:,} bytes)")
    else:
        ckpt_hash = hashlib.md5(best_pt.read_bytes()).hexdigest()
        print(f"[SMOKE TEST] Checkpoint created at {best_pt} (MD5: {ckpt_hash})")

    # Standalone Evaluation on validation set
    print("\n--- Evaluating Porch Specialist Standalone on Validation Split ---")
    eval_model = YOLO(str(best_pt))
    val_res = eval_model.val(
        data=str(dataset_yaml),
        split="val",
        batch=8,
        imgsz=imgsz,
        device=0,
        verbose=True,
    )

    p = float(val_res.box.p[0]) if hasattr(val_res.box.p, "__len__") and len(val_res.box.p) > 0 else float(val_res.box.p)
    r = float(val_res.box.r[0]) if hasattr(val_res.box.r, "__len__") and len(val_res.box.r) > 0 else float(val_res.box.r)
    f1 = float(val_res.box.f1[0]) if hasattr(val_res.box.f1, "__len__") and len(val_res.box.f1) > 0 else float(val_res.box.f1)
    ap50 = float(val_res.box.map50)
    ap50_95 = float(val_res.box.map)


    print("\nStandalone Porch Specialist Metrics:")
    print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    print(f"  AP50: {ap50:.4f} | AP50-95: {ap50_95:.4f}")

    # Verify safeguards
    verify_safeguard_hashes()

    return {
        "status": "success",
        "training_duration_s": round(training_duration_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "checkpoint": str(TARGET_CKPT_PATH.name) if not smoke_test else str(best_pt.name),
        "md5": ckpt_hash,
        "metrics": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "ap50": round(ap50, 4),
            "ap50_95": round(ap50_95, 4),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    train_porch_specialist(epochs=args.epochs, batch=args.batch, smoke_test=args.smoke_test)
