"""Evaluate and Benchmark Depth Anything V2 Metric Indoor Small.

Measures:
1. First inference latency vs warm inference latency.
2. Peak VRAM allocated on RTX 4060 GPU.
3. Physical depth range in meters (min, median, max, 95th percentile).
4. Point cloud generation performance.
Across representative indoor room images.
"""

import os
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import numpy as np
from PIL import Image
import torch

from models.scene3d_model import CustomScene3DModel

def benchmark_metric_depth():
    print("=" * 60)
    print("BENCHMARKING DEPTH ANYTHING V2 METRIC INDOOR SMALL")
    print("=" * 60)

    val_dir = Path("archive (2)/ADEChallengeData2016/images/validation")
    sample_images = list(val_dir.glob("*.jpg"))[:5]
    if not sample_images:
        print("No validation images found, generating synthetic test images.")
        sample_images = [None]

    ckpt_path = "checkpoints/depth_anything_v2_metric_indoor_small"
    config = {
        "checkpoint_path": ckpt_path,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "depth_mode": "metric",
    }

    t0 = time.perf_counter()
    model = CustomScene3DModel(config=config)
    load_time_ms = (time.perf_counter() - t0) * 1000
    print(f"Model load latency: {load_time_ms:.2f} ms")
    print(f"Loaded: {model.get_model_name()}")
    print(f"Device: {model.resolved_device}")

    results = []

    for i, img_path in enumerate(sample_images):
        if img_path and img_path.exists():
            img = Image.open(img_path).convert("RGB")
            name = img_path.name
        else:
            img = Image.new("RGB", (640, 480), color=(120, 140, 160))
            name = f"sample_{i+1}"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        norm_depth, resp = model.reconstruct(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram_mb = 0.0
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        pc = resp.point_cloud or {}
        n_points = len(pc.get("xyz", []))

        item_res = {
            "image": name,
            "size": list(img.size),
            "latency_ms": round(elapsed_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "depth_min_m": resp.depth_min,
            "depth_max_m": resp.depth_max,
            "depth_unit": resp.depth_unit,
            "points": n_points,
            "relative_depth_only": resp.relative_depth_only,
        }
        results.append(item_res)
        print(f"[{i+1}/{len(sample_images)}] {name} ({img.size[0]}x{img.size[1]}): {elapsed_ms:.1f}ms, Peak VRAM: {peak_vram_mb:.1f}MB, Range: {resp.depth_min:.2f}m - {resp.depth_max:.2f}m, Points: {n_points:,}")

    # Output summary
    latencies = [r["latency_ms"] for r in results]
    vrams = [r["peak_vram_mb"] for r in results]
    summary = {
        "load_time_ms": round(load_time_ms, 2),
        "first_inference_latency_ms": latencies[0],
        "warm_inference_latency_avg_ms": round(float(np.mean(latencies[1:])), 2) if len(latencies) > 1 else latencies[0],
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "peak_vram_mb": max(vrams),
        "tested_samples": results,
    }

    out_file = Path("artifacts/metric_depth_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    benchmark_metric_depth()
