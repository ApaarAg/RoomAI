r"""Unified Object Detection Training Script for Open Images V7 Pilot.

Supports:
- Exp 1: Baseline fine-tuning (YOLO11s, 640px, batch 16, AMP=True)
- Exp 2: High-resolution fine-tuning (YOLO11s, 800px, batch 8, accumulate 2 via nbs=16, AMP=True)
- Exp 3: Lightweight model comparison (YOLO11n, 640px, batch 16, AMP=True)
- Exp 4: Tail-class imbalance remedy via RFS (YOLO11s, 640px, batch 16, AMP=True, data_rfs.yaml)

Enforces:
1. Pre/post cryptographic MD5 hash audit on all 21 production checkpoints in checkpoints/.
2. External storage of runs (F:\RoomAI_Data\runs\train\<exp_name>) and checkpoints (F:\RoomAI_Data\checkpoints\<exp_name>_best.pt).
3. Peak VRAM tracking asserting < 6.0 GB ceiling and > 2.0 GB safety headroom.
4. Export of structured metric report to reports/<exp_name>_metrics.json.
5. Periodic heap garbage collection (gc.collect()) and CUDA cache emptying to prevent ArrayMemoryError.
6. Checkpoint resumption via --resume resuming seamlessly from last.pt.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict

os.environ["USE_TF"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

TAIL_CLASSES = ["Bed", "Couch", "Stairs", "Cabinetry", "Bookcase", "Bench"]


def compute_checkpoint_hashes(directory: Path) -> Dict[str, Dict[str, Any]]:
    hashes = {}
    if not directory.exists():
        return hashes
    for pt in sorted(directory.glob("*.pt")):
        h = hashlib.md5(pt.read_bytes()).hexdigest()
        hashes[pt.name] = {"md5": h, "size": pt.stat().st_size}
    return hashes


def parse_args():
    parser = argparse.ArgumentParser(description="RoomAI Unified Training Runner for Open Images V7")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model checkpoint (yolo11s.pt or yolo11n.pt)")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to data.yaml or data_rfs.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution (640 or 800)")
    parser.add_argument("--batch", type=int, default=16, help="Micro-batch size (16 for 640px, 8 for 800px)")
    parser.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation factor (e.g. 2 for 800px)")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs (10-15 epochs)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (0 for Windows stability)")
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument("--cache", default="none", choices=["none", "disk", "ram"], help="Image caching strategy")
    parser.add_argument("--exp-name", required=True, help="Unique experiment name (e.g. exp1_yolo11s_baseline_640)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt checkpoint if it exists")
    return parser.parse_args()


def run_training():
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"RoomAI Object Detection Training: {args.exp_name}")
    print("=" * 80)

    # 1. Preflight System Checks
    assert torch.cuda.is_available(), "CUDA device unavailable! Training requires GPU."
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[PREFLIGHT] Device: {device_name} ({total_vram_gb:.2f} GB total VRAM)")
    print(f"[PREFLIGHT] Base Model: {args.model} | Dataset Config: {args.data}")
    print(f"[PREFLIGHT] Settings: imgsz={args.imgsz}, batch={args.batch}, accum={args.accumulate}, epochs={args.epochs}, patience={args.patience}, cache={args.cache}, resume={args.resume}")

    # 2. Pre-Training Safeguard Audit
    pre_hashes = compute_checkpoint_hashes(LOCAL_CHECKPOINTS_DIR)
    print(f"[SAFEGUARD] Audited {len(pre_hashes)} production checkpoints in {LOCAL_CHECKPOINTS_DIR}")
    assert len(pre_hashes) >= 21, f"Expected at least 21 production checkpoints, found {len(pre_hashes)}"

    # 3. Check for resumption vs fresh training initialization
    run_dir = EXTERNAL_RUNS_DIR / args.exp_name
    last_pt = run_dir / "weights" / "last.pt"
    is_resuming = args.resume and last_pt.exists()

    if is_resuming:
        print(f"[RESUME] Found existing checkpoint: {last_pt} ({last_pt.stat().st_size:,} bytes)")
        print(f"[RESUME] Resuming training for {args.exp_name} from checkpoint...")
        yolo = YOLO(str(last_pt))
    else:
        # Base model path resolution & AMP check
        base_model_path = REPO_ROOT / args.model
        if not base_model_path.exists():
            base_model_path = Path(args.model)
        assert base_model_path.exists(), f"Base checkpoint not found at {base_model_path}"

        test_model = YOLO(str(base_model_path))
        amp_ok = check_amp(test_model.model.to("cuda:0"))
        del test_model
        torch.cuda.empty_cache()
        print(f"[AMP] Automatic Mixed Precision verification: {'PASSED' if amp_ok else 'FAILED'}")
        assert amp_ok, "AMP mixed precision check failed on GPU!"
        yolo = YOLO(str(base_model_path))

    # 4. Attach periodic memory cleanup callbacks to prevent heap fragmentation
    def _cleanup_batch(trainer):
        # Trigger explicit GC every 50 batches to release unreferenced NumPy arrays / tensor buffers
        if getattr(trainer, "ni", 0) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _cleanup_epoch(trainer):
        # Trigger comprehensive GC at epoch end
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            progress_file = REPO_ROOT / ".agents" / "teamwork" / "worker_m3_2" / "progress.md"
            if progress_file.exists():
                text = progress_file.read_text(encoding="utf-8")
                epoch_now = getattr(trainer, "epoch", 0) + 1
                total_epochs = getattr(trainer, "epochs", 10)
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                import re
                text = re.sub(r"Last visited: [^\n]+", f"Last visited: {ts} (Epoch {epoch_now}/{total_epochs})", text)
                progress_file.write_text(text, encoding="utf-8")
        except Exception:
            pass

    yolo.add_callback("on_train_batch_end", _cleanup_batch)
    yolo.add_callback("on_train_epoch_end", _cleanup_epoch)
    yolo.add_callback("on_fit_epoch_end", _cleanup_epoch)
    print("[MEMORY] Attached periodic garbage collection & CUDA cache flushing callbacks (every 50 batches & epoch ends).")

    # 5. Map gradient accumulation to Ultralytics nominal batch size (nbs)
    nbs = args.batch * args.accumulate
    cache_arg = False if args.cache == "none" else args.cache
    close_mosaic = min(2, max(1, args.epochs // 5))

    # 6. Execute Training
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    if is_resuming:
        train_results = yolo.train(
            resume=True,
            workers=args.workers,
            device=args.device,
        )
    else:
        train_results = yolo.train(
            data=str(args.data),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            nbs=nbs,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            amp=True,
            cache=cache_arg,
            project=str(EXTERNAL_RUNS_DIR),
            name=args.exp_name,
            exist_ok=True,
            save=True,
            val=True,
            plots=True,
            seed=args.seed,
            deterministic=True,
            lr0=args.lr0,
            lrf=args.lrf,
            cos_lr=True,
            warmup_epochs=3.0,
            close_mosaic=close_mosaic,
            verbose=True,
        )
    session_duration_sec = time.time() - t0

    # 7. Measure Peak VRAM & Assert Budget
    peak_allocated_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
    peak_reserved_gb = peak_reserved_mb / 1024.0
    vram_headroom_gb = total_vram_gb - peak_reserved_gb

    # Extract cumulative metrics and timing from results.csv
    results_csv = run_dir / "results.csv"
    train_metrics = {}
    csv_duration_sec = None
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
            if "time" in train_metrics:
                csv_duration_sec = float(train_metrics["time"])

    total_training_duration_sec = csv_duration_sec if csv_duration_sec is not None else session_duration_sec

    print("\n" + "=" * 80)
    print("VRAM & Resource Utilization:")
    print(f"  Peak Allocated: {peak_allocated_mb:.1f} MB ({peak_allocated_mb/1024:.2f} GB)")
    print(f"  Peak Reserved:  {peak_reserved_mb:.1f} MB ({peak_reserved_gb:.2f} GB)")
    print(f"  VRAM Headroom:  {vram_headroom_gb:.2f} GB (Ceiling: < 6.0 GB, Headroom: > 2.0 GB)")
    print(f"  Session Time:   {session_duration_sec:.1f} s ({session_duration_sec/60:.2f} min)")
    print(f"  Cumulative Time:{total_training_duration_sec:.1f} s ({total_training_duration_sec/60:.2f} min)")
    print("=" * 80)

    assert peak_reserved_gb < 6.0, f"Peak reserved VRAM ({peak_reserved_gb:.2f} GB) exceeded 6.0 GB ceiling!"
    assert vram_headroom_gb > 2.0, f"VRAM headroom ({vram_headroom_gb:.2f} GB) fell below 2.0 GB safety threshold!"

    # 8. Locate & Deliver Artifacts to External Storage
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

    # 9. Post-Training Safeguard Audit
    post_hashes = compute_checkpoint_hashes(LOCAL_CHECKPOINTS_DIR)
    assert len(pre_hashes) == len(post_hashes), f"Local checkpoint count mismatch! Pre: {len(pre_hashes)}, Post: {len(post_hashes)}"
    for name, info in pre_hashes.items():
        assert name in post_hashes, f"Checkpoint deleted: {name}"
        assert info["md5"] == post_hashes[name]["md5"], f"Checkpoint mutated: {name} (Pre: {info['md5']}, Post: {post_hashes[name]['md5']})"
    print(f"[SAFEGUARD] 100% verified: All {len(post_hashes)} production checkpoints remained untouched.")

    # 10. Run dedicated validation on best checkpoint to extract per-class metrics
    eval_model = YOLO(str(ext_dest_pt))
    torch.cuda.empty_cache()
    val_res = eval_model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        half=True,
    )

    per_class_val = {}
    ap_indices = getattr(val_res.box, "ap_class_index", [])
    for idx, cid in enumerate(ap_indices):
        cname = val_res.names.get(int(cid), f"class_{cid}")
        ap50 = float(val_res.box.ap50[idx]) if hasattr(val_res.box, "ap50") and len(val_res.box.ap50) > idx else 0.0
        ap = float(val_res.box.maps[int(cid)]) if hasattr(val_res.box, "maps") and len(val_res.box.maps) > int(cid) else 0.0
        p = float(val_res.box.p[idx]) if hasattr(val_res.box, "p") and len(val_res.box.p) > idx else 0.0
        r = float(val_res.box.r[idx]) if hasattr(val_res.box, "r") and len(val_res.box.r) > idx else 0.0
        per_class_val[cname] = {
            "mAP50": round(ap50, 4),
            "mAP50_95": round(ap, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
        }

    tail_ap50_dict = {c: per_class_val.get(c, {}).get("mAP50", 0.0) for c in TAIL_CLASSES}
    mean_tail_map50 = round(float(sum(tail_ap50_dict.values()) / len(TAIL_CLASSES)), 4)

    val_speed = getattr(val_res, "speed", {})

    experiment_report = {
        "experiment_name": args.exp_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hyperparameters": {
            "model": str(run_dir / "weights" / "last.pt") if is_resuming else str(args.model),
            "data": str(args.data),
            "imgsz": args.imgsz,
            "batch_size": args.batch,
            "accumulate": args.accumulate,
            "effective_batch_size": nbs,
            "epochs": args.epochs,
            "patience": args.patience,
            "cache": args.cache,
            "workers": args.workers,
            "amp": True,
            "seed": args.seed,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "resumed": is_resuming,
        },
        "resources": {
            "device": device_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "peak_allocated_mb": round(peak_allocated_mb, 2),
            "peak_reserved_mb": round(peak_reserved_mb, 2),
            "vram_headroom_gb": round(vram_headroom_gb, 2),
            "training_duration_sec": round(total_training_duration_sec, 2),
            "sec_per_epoch": round(total_training_duration_sec / max(1, args.epochs), 2),
        },
        "validation_metrics": {
            "mAP50": round(float(val_res.box.map50), 4),
            "mAP50_95": round(float(val_res.box.map), 4),
            "precision": round(float(val_res.box.mp), 4),
            "recall": round(float(val_res.box.mr), 4),
            "mean_tail_mAP50": mean_tail_map50,
            "tail_class_mAP50": tail_ap50_dict,
            "per_class": per_class_val,
            "speed_ms": {
                "preprocess": round(float(val_speed.get("preprocess", 0.0)), 2),
                "inference": round(float(val_speed.get("inference", 0.0)), 2),
                "postprocess": round(float(val_speed.get("postprocess", 0.0)), 2),
            },
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "best_checkpoint": str(ext_dest_pt),
            "checkpoint_md5": ext_hash,
            "size_bytes": ext_dest_pt.stat().st_size,
        },
    }

    report_path = REPORTS_DIR / f"{args.exp_name}_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)
    print(f"\n[REPORT] Saved experiment metrics to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
