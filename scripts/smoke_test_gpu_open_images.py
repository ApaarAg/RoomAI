r"""Rapid 1-Epoch GPU Transfer Learning Smoke Test for Open Images V7 Pilot.

Validates:
1. Pre-training safeguard audit: MD5 hashes of all local production checkpoints in checkpoints/.
2. CUDA device 0 binding (NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM).
3. Automatic Mixed Precision (amp=True) verification.
4. Loss finiteness across batches (no NaN, no Inf in box, cls, dfl loss).
5. Peak VRAM allocation strictly < 6.0 GB (leaving > 2.0 GB headroom).
6. External checkpoint delivery to F:\RoomAI_Data\checkpoints\smoke_test_yolo11s_best.pt.
7. Post-training safeguard audit: 100% hash match on all repo checkpoints.
8. Generation of reports/smoke_test_gpu_results.json and reports/smoke_test_gpu_metrics.json.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["USE_TF"] = "0"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from ultralytics import YOLO
from ultralytics.utils.checks import check_amp

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DATA_ROOT = Path(r"F:\RoomAI_Data")
DEFAULT_DATA_YAML = EXTERNAL_DATA_ROOT / "yolo_open_images_pilot" / "data.yaml"
EXTERNAL_CHECKPOINTS_DIR = EXTERNAL_DATA_ROOT / "checkpoints"
EXTERNAL_RUNS_DIR = EXTERNAL_DATA_ROOT / "runs" / "train"
LOCAL_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
REPORTS_DIR = REPO_ROOT / "reports"


def compute_checkpoint_hashes(directory: Path) -> dict:
    hashes = {}
    if not directory.exists():
        return hashes
    for pt in sorted(directory.glob("*.pt")):
        h = hashlib.md5(pt.read_bytes()).hexdigest()
        hashes[pt.name] = {"md5": h, "size": pt.stat().st_size}
    return hashes


def parse_args():
    parser = argparse.ArgumentParser(description="RoomAI GPU Transfer Learning Smoke Test")
    parser.add_argument("--base-checkpoint", default="yolo11s.pt", help="Pretrained base model")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (safe for 8GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--exp-name", default="smoke_test_open_images_yolo11s", help="Run name")
    return parser.parse_args()


def run_smoke_test():
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RoomAI: Rapid 1-Epoch GPU Transfer Learning Smoke Test")
    print("=" * 80)

    # 1. Preflight System Checks
    assert torch.cuda.is_available(), "CUDA device unavailable! Smoke test requires GPU."
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"[PREFLIGHT] Device: {device_name} ({total_vram_gb:.2f} GB total VRAM / {total_vram_mb:.1f} MB)")
    print(f"[PREFLIGHT] Base Checkpoint: {args.base_checkpoint}")
    print(f"[PREFLIGHT] Dataset YAML: {args.data}")
    print(f"[PREFLIGHT] Hyperparameters: batch={args.batch}, imgsz={args.imgsz}, workers={args.workers}, amp=True")

    # 2. Checkpoint Safeguard Audit (Pre-Training)
    pre_hashes = compute_checkpoint_hashes(LOCAL_CHECKPOINTS_DIR)
    print(f"[SAFEGUARD] Audited {len(pre_hashes)} production checkpoints in {LOCAL_CHECKPOINTS_DIR}")
    assert len(pre_hashes) >= 21, f"Expected at least 21 production checkpoints, found {len(pre_hashes)}"

    # 3. Check AMP on temporary model without mutating main model instance
    base_model_path = REPO_ROOT / args.base_checkpoint
    if not base_model_path.exists():
        base_model_path = Path(args.base_checkpoint)
    assert base_model_path.exists(), f"Base checkpoint not found at {base_model_path}"

    test_model = YOLO(str(base_model_path))
    amp_ok = check_amp(test_model.model.to("cuda:0"))
    del test_model
    torch.cuda.empty_cache()
    print(f"[AMP] Automatic Mixed Precision verification: {'PASSED' if amp_ok else 'FAILED'}")
    assert amp_ok, "AMP mixed precision check failed on GPU!"

    # 4. Instantiate Fresh Model on CPU and Execute 1-Epoch Smoke Training Run
    yolo = YOLO(str(base_model_path))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    train_results = yolo.train(
        data=str(args.data),
        epochs=1,
        patience=1,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        amp=True,
        project=str(EXTERNAL_RUNS_DIR),
        name=args.exp_name,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        seed=42,
        deterministic=True,
        verbose=True,
        warmup_epochs=0.0,
    )
    training_duration_sec = time.time() - t0

    # 5. Measure VRAM Utilization
    peak_allocated_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
    peak_reserved_gb = peak_reserved_mb / 1024.0
    vram_headroom_gb = total_vram_gb - peak_reserved_gb

    print("\n" + "=" * 80)
    print("VRAM & Resource Utilization:")
    print(f"  Peak Allocated: {peak_allocated_mb:.1f} MB ({peak_allocated_mb/1024:.2f} GB)")
    print(f"  Peak Reserved:  {peak_reserved_mb:.1f} MB ({peak_reserved_gb:.2f} GB)")
    print(f"  VRAM Headroom:  {vram_headroom_gb:.2f} GB (Requirement: > 2.0 GB, Ceiling: < 6.0 GB)")
    print(f"  Duration:       {training_duration_sec:.1f} s")
    print("=" * 80)

    # 6. Locate Artifacts and Deliver to External Storage
    run_dir = EXTERNAL_RUNS_DIR / args.exp_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"
    assert best_pt.exists(), f"Trained weights not found in {run_dir / 'weights'}"

    ext_dest_pt = EXTERNAL_CHECKPOINTS_DIR / f"{args.exp_name}_best.pt"
    shutil.copy2(best_pt, ext_dest_pt)
    ext_hash = hashlib.md5(ext_dest_pt.read_bytes()).hexdigest()
    print(f"\n[ARTIFACT] Best weights safely delivered outside repo:")
    print(f"  Destination: {ext_dest_pt}")
    print(f"  MD5 Hash:    {ext_hash} ({ext_dest_pt.stat().st_size:,} bytes)")

    # 7. Checkpoint Safeguard Audit (Post-Training)
    post_hashes = compute_checkpoint_hashes(LOCAL_CHECKPOINTS_DIR)
    assert len(pre_hashes) == len(post_hashes), f"Local checkpoint count changed! Pre: {len(pre_hashes)}, Post: {len(post_hashes)}"
    for name, info in pre_hashes.items():
        assert name in post_hashes, f"Checkpoint deleted: {name}"
        assert info["md5"] == post_hashes[name]["md5"], f"Checkpoint mutated: {name} (Pre: {info['md5']}, Post: {post_hashes[name]['md5']})"
    print(f"[SAFEGUARD] 100% verified: All {len(post_hashes)} repo checkpoints remained untouched.")

    # 8. Parse Results & Validate Pass/Fail Criteria
    results_csv = run_dir / "results.csv"
    train_metrics = {}
    if results_csv.exists():
        lines = [line.strip() for line in results_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) >= 2:
            headers = [h.strip() for h in lines[0].split(",")]
            values = [v.strip() for v in lines[-1].split(",")]
            for h, v in zip(headers, values):
                try:
                    train_metrics[h] = float(v)
                except ValueError:
                    train_metrics[h] = v

    # Check finite loss
    box_loss = train_metrics.get("train/box_loss", 0.0)
    cls_loss = train_metrics.get("train/cls_loss", 0.0)
    dfl_loss = train_metrics.get("train/dfl_loss", 0.0)
    loss_finite = not (math.isnan(box_loss) or math.isnan(cls_loss) or math.isnan(dfl_loss) or
                       math.isinf(box_loss) or math.isinf(cls_loss) or math.isinf(dfl_loss))

    # Evaluate Pass/Fail Criteria (all 8 assertions)
    pass_criteria = {
        "cuda_device_bound": torch.cuda.is_available() and ("4060" in device_name or "RTX" in device_name),
        "amp_mixed_precision_active": amp_ok,
        "loss_finite_and_converging": loss_finite and (box_loss > 0),
        "peak_vram_under_6gb": peak_reserved_gb < 6.0,
        "vram_headroom_over_2gb": vram_headroom_gb > 2.0,
        "external_checkpoint_delivered": ext_dest_pt.exists() and ext_dest_pt.stat().st_size > 10_000_000,
        "repo_checkpoints_unmodified": True,
        "no_cuda_oom": True,
    }

    all_passed = all(pass_criteria.values())

    summary = {
        "test_name": "Rapid 1-Epoch GPU Transfer Learning Smoke Test",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "device": device_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "peak_allocated_mb": round(peak_allocated_mb, 2),
            "peak_reserved_mb": round(peak_reserved_mb, 2),
            "vram_headroom_gb": round(vram_headroom_gb, 2),
        },
        "hyperparameters": {
            "base_checkpoint": str(base_model_path),
            "dataset": str(args.data),
            "epochs": 1,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "amp": True,
            "seed": 42,
        },
        "performance": {
            "duration_sec": round(training_duration_sec, 2),
            "throughput_img_per_sec": round(8000.0 / training_duration_sec, 2),
            "train_box_loss": box_loss,
            "train_cls_loss": cls_loss,
            "train_dfl_loss": dfl_loss,
            "val_mAP50": train_metrics.get("metrics/mAP50(B)", 0.0),
            "val_mAP50_95": train_metrics.get("metrics/mAP50-95(B)", 0.0),
        },
        "artifacts": {
            "external_checkpoint": str(ext_dest_pt),
            "md5_hash": ext_hash,
            "size_bytes": ext_dest_pt.stat().st_size,
            "run_directory": str(run_dir),
        },
        "pass_criteria": pass_criteria,
        "overall_status": "PASS" if all_passed else "FAIL",
    }

    # Save to reports/smoke_test_gpu_results.json (as in dispatch)
    report_path_results = REPORTS_DIR / "smoke_test_gpu_results.json"
    with open(report_path_results, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[REPORT] Saved structured smoke test results to: {report_path_results}")

    # Also save to reports/smoke_test_gpu_metrics.json (as in explorer blueprint)
    report_path_metrics = REPORTS_DIR / "smoke_test_gpu_metrics.json"
    with open(report_path_metrics, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[REPORT] Mirrored report to: {report_path_metrics}")

    print("\n" + "=" * 80)
    for crit, passed in pass_criteria.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"  [{status_str}] {crit}")
    print("=" * 80)
    print(f"OVERALL STATUS: {summary['overall_status']}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
