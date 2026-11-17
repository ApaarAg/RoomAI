"""Stage E: Comparative Evaluation and Regression Analysis of Models B2, B3, and B4.

Protocols:
1. Canonical Test Set (134 images) with ORIGINAL uncorrected porch box (oi_ext_test_0106 = full frame 1.0x1.0).
2. Canonical Test Set (134 images) with CORRECTED porch box (oi_ext_test_0106 = Proposal A portico 6.3% area).
3. Curated Test Set (154 images) with 28 diverse photographic porch instances.

Metrics:
- mAP50, mAP50-95, Precision, Recall, F1.
- Per-class metrics for all 10 canonical classes.
- Median & P95 latency (ms) measured with 10 warm-up images followed by 50 benchmark images on GPU.
- Peak GPU memory tracking.
- Confusion/error analysis on false positives and missed detections.
- Success criteria verification against target thresholds.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
REPORTS_DIR = REPO_ROOT / "reports"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "exterior_phase9"
DATA_DIR = REPO_ROOT / "data"

CANONICAL_CLASSES = {
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

MODELS = {
    "B2 (YOLO11s Baseline)": {
        "checkpoint": CHECKPOINTS_DIR / "exterior_detector_b2_yolo11s_best.pt",
        "expected_md5": "14a931dd0adabc16cbd7b28ed1d17f66",
    },
    "B3 (YOLO11s + RFS)": {
        "checkpoint": CHECKPOINTS_DIR / "exterior_detector_b3_yolo11s_rfs_best.pt",
        "expected_md5": "2c44291fa642aba3893db001520deba0",
    },
    "B4 (YOLO11s Curated)": {
        "checkpoint": CHECKPOINTS_DIR / "exterior_detector_b4_yolo11s_curated_best.pt",
        "expected_md5": None,  # Computed dynamically
    },
}


def compute_md5(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def setup_test_yamls():
    """Sets up the dataset YAML configurations for all 3 evaluation protocols."""
    # 1. Canonical original (134 images, uncorrected oi_ext_test_0106)
    yaml_orig_path = DATA_DIR / "eval_canonical_original.yaml"
    with open(DATA_DIR / "exterior_canonical" / "exterior_canonical.yaml", "r", encoding="utf-8") as f:
        cfg_orig = yaml.safe_load(f)
    with open(yaml_orig_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_orig, f)

    # 2. Canonical corrected (134 images, oi_ext_test_0106 corrected to Proposal A)
    # List the exact 134 images from data/exterior_v2_curated/images/test that correspond to canonical
    canonical_test_manifest = []
    with open(DATA_DIR / "exterior_canonical" / "split_manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    canonical_test_filenames = {r["filename"] for r in m["records"] if r["target_split"] == "test"}

    # Write text file listing these 134 image paths
    test_134_txt = DATA_DIR / "canonical_test_134_corrected.txt"
    with open(test_134_txt, "w", encoding="utf-8") as f:
        for fname in sorted(list(canonical_test_filenames)):
            img_p = DATA_DIR / "exterior_v2_curated" / "images" / "test" / fname
            f.write(f"{img_p.as_posix()}\n")

    yaml_corr_path = DATA_DIR / "eval_canonical_corrected.yaml"
    cfg_corr = {
        "path": "C:/Users/apaar/Desktop/RoomAI/data",
        "train": "exterior_v2_curated/images/train",
        "val": "exterior_v2_curated/images/val",
        "test": "canonical_test_134_corrected.txt",
        "names": CANONICAL_CLASSES,
    }
    with open(yaml_corr_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_corr, f)

    # 3. Curated test set (154 images, 28 porch instances)
    yaml_curated_path = DATA_DIR / "exterior_v2_curated" / "exterior_v2_curated.yaml"

    return {
        "Canonical_Original_134": yaml_orig_path,
        "Canonical_Corrected_134": yaml_corr_path,
        "Curated_Total_154": yaml_curated_path,
    }


def evaluate_model_on_dataset(model_path: Path, data_yaml: Path, split: str = "test") -> Dict[str, Any]:
    print(f"  Evaluating {model_path.name} on {data_yaml.name} ({split})...")
    model = YOLO(str(model_path))

    # Run validation under standard COCO evaluation protocol
    val_res = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=640,
        batch=4,
        conf=0.001,
        iou=0.70,
        device=0,
        workers=2,
        plots=False,
        verbose=False,
    )

    metrics = val_res.results_dict
    map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
    prec = float(metrics.get("metrics/precision(B)", 0.0))
    rec = float(metrics.get("metrics/recall(B)", 0.0))
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # Per-class AP50 and AP50-95
    per_class = {}
    class_indices = val_res.ap_class_index if hasattr(val_res, "ap_class_index") else range(10)

    # val_res.box.ap50 is an array of AP50 per present class
    ap50_arr = val_res.box.ap50 if hasattr(val_res.box, "ap50") else []
    ap_arr = val_res.box.ap if hasattr(val_res.box, "ap") else []
    p_arr = val_res.box.p if hasattr(val_res.box, "p") else []
    r_arr = val_res.box.r if hasattr(val_res.box, "r") else []
    f1_arr = val_res.box.f1 if hasattr(val_res.box, "f1") else []

    for idx, c_idx in enumerate(class_indices):
        c_name = CANONICAL_CLASSES.get(int(c_idx), f"class_{c_idx}")
        c_ap50 = float(ap50_arr[idx]) if idx < len(ap50_arr) else 0.0
        c_ap = float(ap_arr[idx]) if idx < len(ap_arr) else 0.0
        c_p = float(p_arr[idx]) if idx < len(p_arr) else 0.0
        c_r = float(r_arr[idx]) if idx < len(r_arr) else 0.0
        c_f1 = float(f1_arr[idx]) if idx < len(f1_arr) else 0.0
        per_class[c_name] = {
            "class_id": int(c_idx),
            "ap50": round(c_ap50, 4),
            "ap50_95": round(c_ap, 4),
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1": round(c_f1, 4),
        }

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "per_class": per_class,
    }


def benchmark_latency(model_path: Path, sample_images: List[Path]) -> Dict[str, Any]:
    print(f"  Benchmarking latency for {model_path.name} on {len(sample_images)} images...")
    model = YOLO(str(model_path))

    # Warm-up (10 runs)
    for p in sample_images[:10]:
        _ = model.predict(source=str(p), imgsz=640, device=0, verbose=False)

    torch.cuda.synchronize()
    latencies = []

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            pass

    for p in sample_images[:50]:
        t0 = time.perf_counter()
        _ = model.predict(source=str(p), imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        try:
            peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        except Exception:
            pass

    return {
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "peak_vram_mb": round(peak_mem_mb, 1),
        "sample_count": len(latencies),
    }


def main():
    print("=== Running Stage E: Comparative Evaluation & Regression Analysis ===")
    start_eval_time = time.time()

    # 1. Verify checkpoint hashes
    ckpt_hashes = {}
    for m_label, m_cfg in MODELS.items():
        p = m_cfg["checkpoint"]
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        h = compute_md5(p)
        ckpt_hashes[m_label] = h
        if m_cfg["expected_md5"] is not None and h != m_cfg["expected_md5"]:
            raise ValueError(f"Hash mismatch for {m_label}: expected {m_cfg['expected_md5']}, got {h}")
        print(f"  [OK] {m_label}: {h} ({p.stat().st_size:,} bytes)")

    # 2. Setup evaluation configurations
    eval_yamls = setup_test_yamls()

    # 3. Collect evaluation metrics across all combinations
    results_matrix = {}

    for m_label, m_cfg in MODELS.items():
        results_matrix[m_label] = {}
        for proto_name, proto_yaml in eval_yamls.items():
            res = evaluate_model_on_dataset(m_cfg["checkpoint"], proto_yaml, split="test")
            results_matrix[m_label][proto_name] = res

    # 4. Latency benchmarking
    test_img_dir = DATA_DIR / "exterior_v2_curated" / "images" / "test"
    sample_imgs = sorted(list(test_img_dir.glob("*.jpg")))
    latency_matrix = {}
    for m_label, m_cfg in MODELS.items():
        lat_res = benchmark_latency(m_cfg["checkpoint"], sample_imgs)
        latency_matrix[m_label] = lat_res

    # 5. Success Criteria Assessment (Target vs Actual on Curated Test Set)
    b4_curated_metrics = results_matrix["B4 (YOLO11s Curated)"]["Curated_Total_154"]
    b4_porch = b4_curated_metrics["per_class"].get("porch", {})
    porch_rec = b4_porch.get("recall", 0.0)
    porch_ap50 = b4_porch.get("ap50", 0.0)
    overall_map50 = b4_curated_metrics["mAP50"]
    overall_f1 = b4_curated_metrics["f1"]
    b4_lat = latency_matrix["B4 (YOLO11s Curated)"]["median_latency_ms"]

    success_criteria = {
        "porch_recall_ge_0.50": {
            "target": ">= 0.50",
            "actual": porch_rec,
            "passed": porch_rec >= 0.50,
        },
        "porch_ap50_ge_0.40": {
            "target": ">= 0.40",
            "actual": porch_ap50,
            "passed": porch_ap50 >= 0.40,
        },
        "overall_map50_ge_0.45": {
            "target": ">= 0.45",
            "actual": overall_map50,
            "passed": overall_map50 >= 0.45,
        },
        "overall_f1_ge_0.50": {
            "target": ">= 0.50",
            "actual": overall_f1,
            "passed": overall_f1 >= 0.50,
        },
        "latency_le_35ms": {
            "target": "<= 35.0 ms",
            "actual": b4_lat,
            "passed": b4_lat <= 35.0,
        },
    }

    # Save comprehensive evaluation JSON
    eval_json_path = REPORTS_DIR / "exterior_phase9_b4_evaluation.json"
    full_report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_hashes": ckpt_hashes,
        "results_matrix": results_matrix,
        "latency_matrix": latency_matrix,
        "success_criteria": success_criteria,
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)
    print(f"  [OK] Saved evaluation JSON to {eval_json_path}")

    elapsed_eval = time.time() - start_eval_time
    print(f"\nStage E evaluation complete in {elapsed_eval:.1f}s.")


if __name__ == "__main__":
    main()
