"""Trains and evaluates Exterior Detector Model B3 (YOLO11s with Repeat Factor Sampling).

Experiment: Controlled intervention testing Repeat Factor Sampling (RFS) / Class-Aware Sampling
to improve recall and rare-class representation without test-set contamination.

Protocol:
- Base weights: yolo11s.pt (identical base weights as B2)
- Architecture: YOLO11s (9.4M params)
- Identical splits: Canonical exterior train, val, and held-out test splits.
- Val/Test splits remain strictly unaugmented and untouched.
- Rare classes (porch, building, shutter_blind) oversampled via deterministic RFS manifest.
- Checkpoint output: checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt
- Checkpoint B2 (checkpoints/exterior_detector_b2_yolo11s_best.pt) is NEVER modified.
- Full StageTimer instrumentation and peak GPU VRAM tracking.
"""

import argparse
from collections import defaultdict
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

CANONICAL_YAML = REPO_ROOT / "data" / "exterior_canonical" / "exterior_canonical.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
BASE_CHECKPOINT = REPO_ROOT / "yolo11s.pt"

CLASS_NAMES = {
    0: "building",
    1: "facade",
    2: "window",
    3: "door",
    4: "roof_cornice",
    5: "balcony",
    6: "porch",
    7: "pillar_column",
    8: "shutter_blind",
    9: "molding_deco",
}


def build_rfs_dataset_manifest() -> Path:
    """Builds a deterministic Repeat Factor Sampling (RFS) image list for the training set."""
    train_labels_dir = REPO_ROOT / "data" / "exterior_canonical" / "labels" / "train"
    train_images_dir = REPO_ROOT / "data" / "exterior_canonical" / "images" / "train"
    
    label_files = sorted(list(train_labels_dir.glob("*.txt")))
    
    img_to_classes = defaultdict(set)
    for lf in label_files:
        stem = lf.stem
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    img_to_classes[stem].add(int(parts[0]))

    # Class-aware repeat factor policy based on audited rare classes:
    # porch (1 train image): 15x
    # building (63 train images): 2x
    # shutter_blind (145 train images, recall=19.4%): 2x
    class_repeats = {
        6: 15,  # porch
        0: 2,   # building
        8: 2,   # shutter_blind
    }

    manifest_lines = []
    repeat_stats = defaultdict(int)

    for lf in label_files:
        stem = lf.stem
        # Corresponding image can be .jpg or .png
        img_path = train_images_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = train_images_dir / f"{stem}.png"
        if not img_path.exists():
            continue

        classes = img_to_classes[stem]
        rep = 1
        for c in classes:
            if c in class_repeats:
                rep = max(rep, class_repeats[c])
        
        repeat_stats[rep] += 1
        for _ in range(rep):
            manifest_lines.append(str(img_path.resolve()).replace("\\", "/"))

    rfs_txt_path = REPO_ROOT / "data" / "exterior_canonical" / "train_rfs.txt"
    with open(rfs_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"[RFS MANIFEST] Created {rfs_txt_path} with {len(manifest_lines)} total samples from {len(label_files)} base images.")
    print(f"[RFS MANIFEST] Repetition distribution: {dict(repeat_stats)}")

    # Create dataset yaml pointing to train_rfs.txt
    rfs_yaml_path = REPO_ROOT / "data" / "exterior_canonical" / "exterior_rfs.yaml"
    yaml_content = {
        "path": str((REPO_ROOT / "data" / "exterior_canonical").resolve()).replace("\\", "/"),
        "train": "train_rfs.txt",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    with open(rfs_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"[RFS YAML] Written config: {rfs_yaml_path}")
    return rfs_yaml_path


def train_and_eval_b3(
    base_model_path: Path = BASE_CHECKPOINT,
    epochs: int = 15,
    patience: int = 5,
    batch_size: int = 4,
    imgsz: int = 640,
    device: str = "0",
    seed: int = 42,
    run_name: str = "exterior_detector_b3_yolo11s_rfs",
) -> Dict[str, Any]:
    """Trains Model B3 (YOLO11s RFS) and evaluates on validation and held-out test splits."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    rfs_yaml_path = build_rfs_dataset_manifest()

    run_dir = REPO_ROOT / "runs" / "train" / run_name

    timer = StageTimer()
    estimated_duration_sec = epochs * 18.0  # ~270s estimated
    timer.start_stage(
        stage_name="Model B3 (YOLO11s RFS Exterior) Controlled Training",
        estimated_sec=estimated_duration_sec,
        total_items=epochs,
        expected_gpu_vram_gb=2.6,
        assumptions=f"{epochs} epochs on 679 RFS train samples, batch_size={batch_size}, imgsz={imgsz}, seed={seed}",
    )

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        except Exception:
            pass

    model = YOLO(str(base_model_path))

    t0 = time.time()
    results = model.train(
        data=str(rfs_yaml_path),
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

    # Save best checkpoint to dedicated B3 path (never overwrite B2!)
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    src_pt = best_pt if best_pt.exists() else last_pt
    dest_pt = CHECKPOINTS_DIR / f"{run_name}_best.pt"

    if src_pt.exists():
        shutil.copy2(src_pt, dest_pt)
        print(f"\n[CHECKPOINT] Successfully saved candidate weights to: {dest_pt}")
    else:
        raise FileNotFoundError(f"Neither best.pt nor last.pt found in {run_dir / 'weights'}")

    # Evaluate on untouched validation split
    val_model = YOLO(str(dest_pt))
    print(f"\n[EVAL] Evaluating candidate B3 on Validation split (134 images)...")
    val_results = val_model.val(
        data=str(rfs_yaml_path),
        split="val",
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        workers=2,
        half=False,
    )

    # Evaluate on held-out Test split (134 images)
    print(f"\n[EVAL] Evaluating candidate B3 on Held-Out Test split (134 images)...")
    test_results = val_model.val(
        data=str(rfs_yaml_path),
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
        "model_name": "Exterior Detector Model B3 (YOLO11s RFS)",
        "architecture": "YOLO11s",
        "base_model": str(base_model_path),
        "intervention": "Repeat Factor Sampling (porch 15x, building 2x, shutter_blind 2x)",
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

    eval_json_path = REPO_ROOT / "reports" / "exterior_b3_rfs_eval.json"
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
    parser = argparse.ArgumentParser(description="Train Exterior Detector Model B3 with RFS")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="Target CUDA device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base-checkpoint", type=str, default=str(BASE_CHECKPOINT), help="Base checkpoint path")
    args = parser.parse_args()

    train_and_eval_b3(
        base_model_path=Path(args.base_checkpoint),
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
