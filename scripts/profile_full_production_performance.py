"""Comprehensive production performance profiling across all 5 RoomAI pipelines."""

import gc
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch
import yaml

from services.detection_service import DetectionService
from services.segmentation_service import SegmentationService
from services.scene3d_service import Scene3DService

def profile_pipeline():
    with open(REPO_ROOT / "config" / "model_config.yaml", "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=== RoomAI Production Performance Profiler (Device: {device}) ===")

    # Test images for each domain
    indoor_img_path = REPO_ROOT / "data" / "deepfurniture" / "images" / "test"
    ext_img_path = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "val"
    parking_img_path = REPO_ROOT / "data" / "parking_canonical" / "images" / "test"

    indoor_file = list(indoor_img_path.glob("*.jpg"))[0]
    ext_file = list(ext_img_path.glob("*.jpg"))[0]
    parking_file = list(parking_img_path.glob("*.jpg"))[0]

    img_indoor = Image.open(indoor_file).convert("RGB")
    img_ext = Image.open(ext_file).convert("RGB")
    img_parking = Image.open(parking_file).convert("RGB")

    results = {}

    pipelines = [
        ("indoor_detection_a3", "detection", model_config.get("detection", {}), img_indoor),
        ("exterior_detection_b2", "exterior_detection", model_config.get("exterior_detection", {}), img_ext),
        ("parking_detection_c2", "parking_detection", model_config.get("parking_detection", {}), img_parking),
        ("segmentation_yolo11seg", "segmentation", model_config.get("segmentation", {}), img_indoor),
        ("relative_depth_anything_v2", "scene3d", model_config.get("scene3d", {}), img_indoor),
    ]

    for name, domain, cfg, test_img in pipelines:
        print(f"\n--- Profiling {name} ---")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 1. Cold start time (service instantiation + first run / load)
        t_cold_start = time.perf_counter()
        if domain == "segmentation":
            svc = SegmentationService(mock_mode=False, config=cfg)
        elif domain == "scene3d":
            svc = Scene3DService(mock_mode=False, config=cfg)
        else:
            svc = DetectionService(mock_mode=False, config=cfg)

        cold_res = svc.run(test_img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cold_time_ms = (time.perf_counter() - t_cold_start) * 1000.0

        # 2. Warmup run
        svc.run(test_img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 3. Multi-pass latency evaluation (10 passes)
        latencies = []
        inf_times = []
        pre_times = []
        post_times = []
        save_times = []

        for p in range(10):
            t0 = time.perf_counter()
            out = svc.run(test_img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_pass_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(total_pass_ms)

            # Extract detailed timings if available
            tm = out.get("timing_metrics", {})
            t_inf = tm.get("per_model_inference_ms") or out.get("timing", {}).get("inference_time_ms", 0.0)
            t_pre = tm.get("preprocess_time_ms", 0.0)
            t_render = tm.get("rendering_time_ms", 0.0)
            t_save = tm.get("saving_time_ms", 0.0)

            inf_times.append(t_inf)
            pre_times.append(t_pre)
            post_times.append(t_render)
            save_times.append(t_save)

        # Memory stats
        alloc_mb = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
        peak_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
        res_mb = torch.cuda.memory_reserved(0) / (1024**2) if torch.cuda.is_available() else 0.0

        p50 = float(np.median(latencies))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        fps = round(1000.0 / p50, 2) if p50 > 0 else 0.0

        results[name] = {
            "model_name": cfg.get("model_name", name),
            "checkpoint": cfg.get("checkpoint_path", ""),
            "cold_start_ms": round(cold_time_ms, 2),
            "warm_median_p50_ms": round(p50, 2),
            "warm_p90_ms": round(p90, 2),
            "warm_p95_ms": round(p95, 2),
            "warm_p99_ms": round(p99, 2),
            "throughput_fps": fps,
            "median_forward_ms": round(float(np.median(inf_times)), 2),
            "median_preprocess_ms": round(float(np.median(pre_times)), 2),
            "median_rendering_ms": round(float(np.median(post_times)), 2),
            "median_saving_ms": round(float(np.median(save_times)), 2),
            "vram_allocated_mb": round(alloc_mb, 2),
            "vram_peak_mb": round(peak_mb, 2),
            "vram_reserved_mb": round(res_mb, 2),
            "passes_raw_ms": [round(x, 2) for x in latencies],
        }

        print(f"  Cold start: {cold_time_ms:.1f}ms | Warm P50: {p50:.1f}ms | FPS: {fps} | VRAM Peak: {peak_mb:.1f}MB")

    # Save summary
    out_path = REPO_ROOT / "reports" / "production_audit_performance_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PASS] Performance benchmark completed and saved to {out_path}")

if __name__ == "__main__":
    profile_pipeline()
