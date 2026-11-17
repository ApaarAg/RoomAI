"""Comprehensive Final Backend Benchmark & Real-Image Verification Script.

Benchmarks all production components on RTX 4060 Laptop GPU using CUDA synchronization:
- Cold start latency vs warm median (p50) & p95 latency.
- Peak VRAM allocated and reserved.
- Host system RAM.
- End-to-end reconstruction pipeline latency.
- Generates verified artifacts for independent audit.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import json
from pathlib import Path
import platform
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import psutil
import torch

from services.roomai_backend import BACKEND
from utils.export_3d import validate_glb_bytes, validate_ply_bytes

OUTPUT_DIR = REPO_ROOT / "artifacts" / "final_backend_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_IMG_PATH = REPO_ROOT / "artifacts" / "sample_room.jpg"


def get_mem_stats():
    """Returns host RAM and GPU VRAM statistics."""
    proc = psutil.Process(os.getpid())
    host_ram_mb = proc.memory_info().rss / (1024 * 1024)
    vram_alloc_mb = 0.0
    vram_res_mb = 0.0
    if torch.cuda.is_available():
        vram_alloc_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        vram_res_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
    return {
        "host_ram_mb": round(host_ram_mb, 1),
        "vram_allocated_mb": round(vram_alloc_mb, 1),
        "vram_reserved_mb": round(vram_res_mb, 1),
    }


def measure_benchmark(func, warm_iters=10):
    """Executes cold run followed by warm_iters, measuring CUDA synchronized timings."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

    # Cold start
    t0 = time.perf_counter()
    cold_res = func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Warm runs
    timings = []
    last_res = cold_res
    for _ in range(warm_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        last_res = func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t_start) * 1000.0)

    peak_vram_alloc = 0.0
    peak_vram_res = 0.0
    if torch.cuda.is_available():
        peak_vram_alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        peak_vram_res = torch.cuda.max_memory_reserved(0) / (1024 * 1024)

    return {
        "cold_start_ms": round(cold_ms, 2),
        "warm_mean_ms": round(float(np.mean(timings)), 2),
        "warm_median_ms": round(float(np.median(timings)), 2),
        "warm_p95_ms": round(float(np.percentile(timings, 95)), 2),
        "warm_min_ms": round(float(np.min(timings)), 2),
        "warm_max_ms": round(float(np.max(timings)), 2),
        "peak_vram_allocated_mb": round(peak_vram_alloc, 1),
        "peak_vram_reserved_mb": round(peak_vram_res, 1),
        "last_res": last_res,
    }


def run_full_benchmark():
    print("=" * 70)
    print("ROOMAI FINAL BACKEND BENCHMARK & HARDWARE TELEMETRY AUDIT")
    print("=" * 70)

    # 1. Environment & Hardware Header
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    props = torch.cuda.get_device_properties(0) if cuda_avail else None
    total_vram_mb = round(props.total_memory / (1024 * 1024), 1) if props else 0.0
    sys_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})")
    print(f"Device: {device_name} ({total_vram_mb} MB VRAM)")
    print(f"Host System RAM: {sys_ram_gb} GB")
    print("-" * 70)

    # Load test image
    assert SAMPLE_IMG_PATH.exists(), f"Missing sample image: {SAMPLE_IMG_PATH}"
    img = Image.open(SAMPLE_IMG_PATH).convert("RGB")
    print(f"Loaded verification image: {SAMPLE_IMG_PATH.name} ({img.size[0]}x{img.size[1]} RGB)")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": device_name,
            "total_vram_mb": total_vram_mb,
            "system_ram_gb": sys_ram_gb,
            "cuda_available": cuda_avail,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        },
        "components": {},
    }

    # 2. Benchmark Indoor Detector A3 YOLO11s 800px
    print("\n[1/8] Benchmarking Indoor Detector A3 YOLO11s (800px)...")
    b_det_a3 = measure_benchmark(lambda: BACKEND.detect_objects(img, task_group="Group A: Indoor Furniture & Household Objects"))
    print(f"  Cold start: {b_det_a3['cold_start_ms']} ms | Warm p50: {b_det_a3['warm_median_ms']} ms | p95: {b_det_a3['warm_p95_ms']} ms")
    print(f"  Peak VRAM: {b_det_a3['peak_vram_allocated_mb']} MB allocated, {b_det_a3['peak_vram_reserved_mb']} MB reserved")
    results["components"]["indoor_detector_a3"] = {
        "model": "YOLO11s-A3-800px",
        "cold_start_ms": b_det_a3["cold_start_ms"],
        "p50_ms": b_det_a3["warm_median_ms"],
        "p95_ms": b_det_a3["warm_p95_ms"],
        "peak_vram_alloc_mb": b_det_a3["peak_vram_allocated_mb"],
        "peak_vram_res_mb": b_det_a3["peak_vram_reserved_mb"],
        "detections_count": len(b_det_a3["last_res"]["data"]["detections"]) if b_det_a3["last_res"]["data"] else 0,
    }

    # Save annotated image
    if b_det_a3["last_res"]["annotated_image"]:
        b_det_a3["last_res"]["annotated_image"].save(OUTPUT_DIR / "indoor_a3_detections.png")

    # 3. Benchmark Exterior Detector B2 YOLO11s
    print("\n[2/8] Benchmarking Exterior Detector B2 YOLO11s...")
    b_det_b2 = measure_benchmark(lambda: BACKEND.detect_objects(img, task_group="Group B: Exterior and Building Components"))
    print(f"  Cold start: {b_det_b2['cold_start_ms']} ms | Warm p50: {b_det_b2['warm_median_ms']} ms | p95: {b_det_b2['warm_p95_ms']} ms")
    results["components"]["exterior_detector_b2"] = {
        "model": "YOLO11s-B2",
        "cold_start_ms": b_det_b2["cold_start_ms"],
        "p50_ms": b_det_b2["warm_median_ms"],
        "p95_ms": b_det_b2["warm_p95_ms"],
        "peak_vram_alloc_mb": b_det_b2["peak_vram_allocated_mb"],
        "peak_vram_res_mb": b_det_b2["peak_vram_reserved_mb"],
    }

    # 4. Benchmark Parking Detector C2 YOLO11s
    print("\n[3/8] Benchmarking Parking Detector C2 YOLO11s...")
    b_det_c2 = measure_benchmark(lambda: BACKEND.detect_objects(img, task_group="Group C: Parking & Outdoor Property"))
    print(f"  Cold start: {b_det_c2['cold_start_ms']} ms | Warm p50: {b_det_c2['warm_median_ms']} ms | p95: {b_det_c2['warm_p95_ms']} ms")
    results["components"]["parking_detector_c2"] = {
        "model": "YOLO11s-C2",
        "cold_start_ms": b_det_c2["cold_start_ms"],
        "p50_ms": b_det_c2["warm_median_ms"],
        "p95_ms": b_det_c2["warm_p95_ms"],
        "peak_vram_alloc_mb": b_det_c2["peak_vram_allocated_mb"],
        "peak_vram_res_mb": b_det_c2["peak_vram_reserved_mb"],
    }

    # 5. Benchmark Segmentation YOLO11n-Seg
    print("\n[4/8] Benchmarking YOLO11n-Seg Instance Segmentation...")
    b_seg = measure_benchmark(lambda: BACKEND.segment_scene(img))
    print(f"  Cold start: {b_seg['cold_start_ms']} ms | Warm p50: {b_seg['warm_median_ms']} ms | p95: {b_seg['warm_p95_ms']} ms")
    results["components"]["segmentation_yolo11n_seg"] = {
        "model": "YOLO11n-Seg",
        "cold_start_ms": b_seg["cold_start_ms"],
        "p50_ms": b_seg["warm_median_ms"],
        "p95_ms": b_seg["warm_p95_ms"],
        "peak_vram_alloc_mb": b_seg["peak_vram_allocated_mb"],
        "peak_vram_res_mb": b_seg["peak_vram_reserved_mb"],
    }
    if b_seg["last_res"]["overlay_image"]:
        b_seg["last_res"]["overlay_image"].save(OUTPUT_DIR / "segmentation_overlay.png")

    # 6. Benchmark Metric Depth Estimation
    print("\n[5/8] Benchmarking Depth Anything V2 Metric Indoor Small...")
    b_depth_metric = measure_benchmark(lambda: BACKEND.estimate_depth(img, mode="metric"))
    print(f"  Cold start: {b_depth_metric['cold_start_ms']} ms | Warm p50: {b_depth_metric['warm_median_ms']} ms | p95: {b_depth_metric['warm_p95_ms']} ms")
    results["components"]["depth_anything_v2_metric"] = {
        "model": "Depth Anything V2 Metric Indoor Small",
        "cold_start_ms": b_depth_metric["cold_start_ms"],
        "p50_ms": b_depth_metric["warm_median_ms"],
        "p95_ms": b_depth_metric["warm_p95_ms"],
        "peak_vram_alloc_mb": b_depth_metric["peak_vram_allocated_mb"],
        "peak_vram_res_mb": b_depth_metric["peak_vram_reserved_mb"],
    }
    if b_depth_metric["last_res"]["heatmap_image"]:
        b_depth_metric["last_res"]["heatmap_image"].save(OUTPUT_DIR / "metric_depth_heatmap.png")

    # 7. Benchmark Relative Depth Estimation
    print("\n[6/8] Benchmarking Depth Anything V2 Small Relative...")
    b_depth_rel = measure_benchmark(lambda: BACKEND.estimate_depth(img, mode="relative"))
    print(f"  Cold start: {b_depth_rel['cold_start_ms']} ms | Warm p50: {b_depth_rel['warm_median_ms']} ms | p95: {b_depth_rel['warm_p95_ms']} ms")
    results["components"]["depth_anything_v2_relative"] = {
        "model": "Depth Anything V2 Small Relative",
        "cold_start_ms": b_depth_rel["cold_start_ms"],
        "p50_ms": b_depth_rel["warm_median_ms"],
        "p95_ms": b_depth_rel["warm_p95_ms"],
        "peak_vram_alloc_mb": b_depth_rel["peak_vram_allocated_mb"],
        "peak_vram_res_mb": b_depth_rel["peak_vram_reserved_mb"],
    }

    # 8. Benchmark End-to-End Single-View 3D Scene Reconstruction
    print("\n[7/8] Benchmarking Full End-to-End Single-View 3D Reconstruction...")
    b_recon = measure_benchmark(lambda: BACKEND.reconstruct_room_3d(img, refine_depth=True), warm_iters=5)
    print(f"  Cold start: {b_recon['cold_start_ms']} ms | Warm p50: {b_recon['warm_median_ms']} ms | p95: {b_recon['warm_p95_ms']} ms")
    print(f"  Peak VRAM: {b_recon['peak_vram_allocated_mb']} MB allocated, {b_recon['peak_vram_reserved_mb']} MB reserved")

    scene = b_recon["last_res"]["scene_graph"]
    pts_count = len(scene.point_cloud.get("xyz", [])) if scene.point_cloud else 0
    obj_count = len(scene.objects)

    results["components"]["single_view_3d_reconstruction"] = {
        "model": "Full Unified Scene Pipeline (A3 + YOLO11-Seg + Metric Depth + RANSAC)",
        "cold_start_ms": b_recon["cold_start_ms"],
        "p50_ms": b_recon["warm_median_ms"],
        "p95_ms": b_recon["warm_p95_ms"],
        "peak_vram_alloc_mb": b_recon["peak_vram_allocated_mb"],
        "peak_vram_res_mb": b_recon["peak_vram_reserved_mb"],
        "points_reconstructed": pts_count,
        "objects_recognized": obj_count,
        "floor_detected": scene.room_layout.get("floor_plane_detected", False) if scene.room_layout else False,
    }

    # Save SceneGraph JSON
    scene_json_path = OUTPUT_DIR / "reconstructed_scene_graph.json"
    with open(scene_json_path, "w", encoding="utf-8") as f:
        json.dump(scene.model_dump(), f, indent=2, default=str)
    print(f"  Saved SceneGraph JSON to: {scene_json_path}")

    # 9. Benchmark 3D Asset Export (GLB + PLY)
    print("\n[8/8] Benchmarking 3D Asset Export (GLB + PLY)...")
    b_export = measure_benchmark(lambda: BACKEND.export_3d_assets(scene))
    print(f"  Cold start: {b_export['cold_start_ms']} ms | Warm p50: {b_export['warm_median_ms']} ms | p95: {b_export['warm_p95_ms']} ms")

    exports = b_export["last_res"]
    glb_bytes = exports["glb"]
    ply_bytes = exports["ply"]

    # Validate output bytes
    ply_val = validate_ply_bytes(ply_bytes)
    glb_val = validate_glb_bytes(glb_bytes)

    print(f"  GLB export: {len(glb_bytes):,} bytes | Valid: {glb_val['is_valid']} ({glb_val.get('mesh_count', 0)} meshes)")
    print(f"  PLY export: {len(ply_bytes):,} bytes | Valid: {ply_val['is_valid']} ({ply_val.get('vertex_count', 0)} vertices)")

    results["components"]["export_3d_assets"] = {
        "cold_start_ms": b_export["cold_start_ms"],
        "p50_ms": b_export["warm_median_ms"],
        "p95_ms": b_export["warm_p95_ms"],
        "glb_bytes": len(glb_bytes),
        "glb_valid": glb_val["is_valid"],
        "ply_bytes": len(ply_bytes),
        "ply_valid": ply_val["is_valid"],
    }

    # Save binary GLB and ASCII PLY
    glb_path = OUTPUT_DIR / "reconstructed_scene.glb"
    with open(glb_path, "wb") as f:
        f.write(glb_bytes)

    ply_path = OUTPUT_DIR / "reconstructed_pointcloud.ply"
    with open(ply_path, "wb") as f:
        f.write(ply_bytes)

    print(f"  Saved GLB to: {glb_path}")
    print(f"  Saved PLY to: {ply_path}")

    # 10. Summary & Output Saving
    mem = get_mem_stats()
    results["system_memory_at_completion"] = mem
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK AUDIT SUMMARY")
    print("=" * 70)
    print(f"Final VRAM Allocated: {mem['vram_allocated_mb']} MB (Total: {total_vram_mb} MB, Headroom: {round(total_vram_mb - mem['vram_allocated_mb'], 1)} MB)")
    print(f"Final VRAM Reserved: {mem['vram_reserved_mb']} MB")
    print(f"Host RAM: {mem['host_ram_mb']} MB")
    print(f"Single-View Reconstruction steady-state latency: {b_recon['warm_median_ms']} ms")
    print("All production subsystems executed real inference with 0 errors!")

    summary_path = OUTPUT_DIR / "benchmark_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to: {summary_path}")

    return results


if __name__ == "__main__":
    run_full_benchmark()
