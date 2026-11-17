"""Comprehensive Evaluation Script for Depth Anything V2 Small in RoomAI.

Evaluates:
  - Numerical integrity (output dimensions, range, NaN/Inf checks)
  - Visual alignment and semantic plausibility on indoor and exterior scenes
  - Hardware benchmarks: Latency (cold, warm, p95, FPS), VRAM usage on RTX 4060
  - CPU fallback robustness and mock mode isolation
  - Visual artifacts saved to artifacts/depth_visualizations/
"""

import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch

from models.scene3d_model import CustomScene3DModel, MockScene3DModel
from services.scene3d_service import Scene3DService
from services.image_service import create_depth_heatmap

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "artifacts" / "depth_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_CUDA = {
    "enabled": True,
    "model_name": "Depth Anything V2 Small",
    "checkpoint_path": "checkpoints/depth_anything_v2_small",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

CONFIG_CPU = {
    "enabled": True,
    "model_name": "Depth Anything V2 Small (CPU Fallback)",
    "checkpoint_path": "checkpoints/depth_anything_v2_small",
    "device": "cpu",
}

TEST_IMAGES = [
    {
        "name": "indoor_living_room",
        "category": "indoor",
        "path": "data/indoor_canonical/images/test/df_L3D121S21ENDIHOOVJYUI5L7SFOS3P3W2888.jpg",
        "notes": "Complex indoor scene with furniture, smooth walls, textured carpet, and distant doorways",
    },
    {
        "name": "exterior_facade",
        "category": "exterior",
        "path": "data/exterior_canonical/images/test/cmp_test_0000.jpg",
        "notes": "Building facade with windows, architectural trim, shadows, and sky backdrop",
    },
]


def evaluate_model():
    print("=" * 70)
    print("RoomAI Monocular Depth Estimation: Depth Anything V2 Small Benchmark")
    print("=" * 70)

    results = {
        "model_name": "Depth Anything V2 Small",
        "backbone": "DINOv2 ViT-Small",
        "parameter_count": 24785089,
        "license": "Apache-2.0",
        "checkpoint": "checkpoints/depth_anything_v2_small",
        "device": CONFIG_CUDA["device"],
        "scene_evaluations": [],
        "latency_benchmark": {},
        "cpu_fallback_benchmark": {},
        "mock_mode_isolation": {},
    }

    # 1. Initialize Real Model
    print(f"\n1. Initializing CustomScene3DModel on {CONFIG_CUDA['device']}...")
    model_cuda = CustomScene3DModel(CONFIG_CUDA)
    assert model_cuda.is_ready(), "Model failed to initialize!"

    # 2. Evaluate Scenes
    for item in TEST_IMAGES:
        img_path = REPO_ROOT / item["path"]
        print(f"\nEvaluating Scene: {item['name']} ({item['category']})")
        print(f"Path: {img_path}")
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        depth_map, response = model_cuda.reconstruct(image)

        # Assertions & Quality Checks
        assert depth_map is not None, "Depth map must not be None"
        assert depth_map.shape == (h, w), f"Expected shape {(h, w)}, got {depth_map.shape}"
        assert not np.isnan(depth_map).any(), "Depth map contains NaN values!"
        assert not np.isinf(depth_map).any(), "Depth map contains Inf values!"
        assert 0.0 <= depth_map.min() <= depth_map.max() <= 1.0, "Depth map not in [0.0, 1.0]"
        assert response.is_simulated is False, "Real model returned is_simulated=True!"

        # Generate & save heatmap
        heatmap = create_depth_heatmap(depth_map)
        heatmap_path = OUTPUT_DIR / f"{item['name']}_depth_heatmap.png"
        heatmap.save(heatmap_path)

        # Save side-by-side comparison
        side_by_side = Image.new("RGB", (w * 2, h))
        side_by_side.paste(image, (0, 0))
        side_by_side.paste(heatmap, (w, 0))
        comp_path = OUTPUT_DIR / f"{item['name']}_comparison.png"
        side_by_side.save(comp_path)

        # Region gradient analysis: bottom 15% (near) vs top 15% (far)
        near_region_mean = float(depth_map[int(h * 0.85):, :].mean())
        far_region_mean = float(depth_map[:int(h * 0.15), :].mean())
        depth_contrast = near_region_mean - far_region_mean

        scene_res = {
            "name": item["name"],
            "category": item["category"],
            "input_resolution": [w, h],
            "output_shape": list(depth_map.shape),
            "inference_time_ms": response.inference_time_ms,
            "peak_vram_mb": response.peak_memory_mb,
            "depth_min": float(depth_map.min()),
            "depth_max": float(depth_map.max()),
            "near_region_mean": round(near_region_mean, 4),
            "far_region_mean": round(far_region_mean, 4),
            "depth_contrast": round(depth_contrast, 4),
            "saved_heatmap": str(heatmap_path.relative_to(REPO_ROOT)),
            "saved_comparison": str(comp_path.relative_to(REPO_ROOT)),
            "is_finite": True,
            "notes": item["notes"],
        }
        results["scene_evaluations"].append(scene_res)
        print(f"  Shape: {depth_map.shape} | Time: {response.inference_time_ms:.1f}ms | VRAM: {response.peak_memory_mb:.1f}MB")
        print(f"  Near Mean: {near_region_mean:.3f} | Far Mean: {far_region_mean:.3f} | Gradient contrast: {depth_contrast:+.3f}")
        print(f"  Saved heatmap to: {heatmap_path}")

    # 3. Latency & Throughput Benchmark (Warmed-up, 20 iterations on 640x480 standard resolution)
    print("\n3. Running 20-iteration warmed latency benchmark on GPU...")
    bench_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    # Warmup
    for _ in range(5):
        _ = model_cuda.reconstruct(bench_img)

    latencies = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    for _ in range(20):
        t0 = time.perf_counter()
        _ = model_cuda.reconstruct(bench_img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    peak_mem = (
        torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    results["latency_benchmark"] = {
        "test_resolution": [640, 480],
        "iterations": 20,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "steady_fps": round(1000.0 / float(np.mean(latencies)), 1),
        "peak_vram_mb": round(peak_mem, 1),
    }
    print(f"  Mean Latency: {results['latency_benchmark']['mean_latency_ms']} ms")
    print(f"  Median Latency: {results['latency_benchmark']['median_latency_ms']} ms")
    print(f"  P95 Latency: {results['latency_benchmark']['p95_latency_ms']} ms")
    print(f"  Steady FPS: {results['latency_benchmark']['steady_fps']} FPS")
    print(f"  Peak VRAM: {results['latency_benchmark']['peak_vram_mb']} MB")

    # 4. CPU Fallback Verification
    print("\n4. Testing safe CPU fallback...")
    model_cpu = CustomScene3DModel(CONFIG_CPU)
    t0 = time.perf_counter()
    cpu_depth, cpu_resp = model_cpu.reconstruct(bench_img)
    cpu_time = (time.perf_counter() - t0) * 1000.0

    assert cpu_resp.device == "cpu"
    assert cpu_resp.is_simulated is False
    assert cpu_depth.shape == (480, 640)
    results["cpu_fallback_benchmark"] = {
        "status": "Verified",
        "device": "cpu",
        "latency_ms": round(cpu_time, 2),
        "output_shape": list(cpu_depth.shape),
    }
    print(f"  CPU Fallback Verified! Latency: {cpu_time:.1f} ms, Device: {cpu_resp.device}")

    # 5. Mock Mode Isolation Test
    print("\n5. Testing Mock Mode isolation...")
    mock_model = MockScene3DModel()
    mock_depth, mock_resp = mock_model.reconstruct(bench_img)
    assert mock_resp.is_simulated is True
    assert mock_resp.device == "cpu"
    results["mock_mode_isolation"] = {
        "mock_is_simulated": mock_resp.is_simulated,
        "real_is_simulated": False,
        "isolation_verified": True,
    }
    print("  Mock Mode Isolation Verified (mock is_simulated=True, real is_simulated=False)")

    # Save summary JSON
    summary_path = REPO_ROOT / "reports" / "depth_evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation summary written to: {summary_path}")


if __name__ == "__main__":
    evaluate_model()
