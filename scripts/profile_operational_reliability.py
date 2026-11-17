"""Operational reliability and multi-pass latency profiling for RoomAI."""

import gc
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
import yaml

from models.registry import GLOBAL_MODEL_REGISTRY
from services.detection_service import DetectionService
from services.scene3d_service import Scene3DService

with open(REPO_ROOT / "config" / "model_config.yaml", "r", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)

# Use test image
test_img_path = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "val" / "cmp_val_0001.jpg"
if not test_img_path.exists():
    test_img_path = list((REPO_ROOT / "data" / "phase22_dataset" / "images" / "val").glob("*.jpg"))[0]

img = Image.open(test_img_path).convert("RGB")
print(f"Profiling using test image: {test_img_path.name} ({img.size})")

# 1. Profile Detection Service (Production B2)
ext_cfg = model_config.get("exterior_detection", {})
det_service = DetectionService(mock_mode=False, config=ext_cfg)

# Warmup 2 passes
for _ in range(2):
    det_service.run(img)
if torch.cuda.is_available():
    torch.cuda.synchronize()

det_latencies = []
vram_allocs = []

print("\nRunning 10 consecutive inference passes with Production Model B2...")
for i in range(10):
    t0 = time.perf_counter()
    res = det_service.run(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    timing = res.get("timing", {})
    tm = res.get("timing_metrics", {})
    inf_ms = tm.get("per_model_inference_ms") or timing.get("inference_time_ms") or getattr(res.get("response"), "inference_time_ms", 0.0)
    pre_ms = tm.get("preprocess_time_ms") or timing.get("preprocess_time_ms", 0.0)
    post_ms = (tm.get("rendering_time_ms", 0.0) + tm.get("saving_time_ms", 0.0)) or timing.get("postprocess_time_ms", 0.0)
    
    alloc_mb = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    res_mb = torch.cuda.memory_reserved(0) / (1024**2) if torch.cuda.is_available() else 0.0
    
    det_latencies.append({
        "pass": i + 1,
        "total_ms": round(total_ms, 2),
        "inference_ms": round(inf_ms, 2),
        "preprocess_ms": round(pre_ms, 2),
        "postprocess_ms": round(post_ms, 2),
        "vram_allocated_mb": round(alloc_mb, 2),
        "vram_reserved_mb": round(res_mb, 2),
    })

# Check memory stability
mem_start = det_latencies[0]["vram_allocated_mb"]
mem_end = det_latencies[-1]["vram_allocated_mb"]
mem_delta = mem_end - mem_start
print(f"VRAM Allocated: Pass 1={mem_start} MB -> Pass 10={mem_end} MB (Delta: {mem_delta:.2f} MB)")
assert abs(mem_delta) < 10.0, f"Possible VRAM leak detected: delta={mem_delta} MB"

# 2. Profile Scene3D Depth Service
s3d_cfg = model_config.get("scene3d", {})
s3d_service = Scene3DService(mock_mode=False, config=s3d_cfg)

# Warmup 1 pass
s3d_service.run(img)
if torch.cuda.is_available():
    torch.cuda.synchronize()

s3d_latencies = []
print("\nRunning 5 consecutive passes with Scene3D Depth Service...")
for i in range(5):
    t0 = time.perf_counter()
    res = s3d_service.run(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    inf_ms = res.get("timing", {}).get("inference_time_ms", 0.0)
    alloc_mb = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    s3d_latencies.append({
        "pass": i + 1,
        "total_ms": round(total_ms, 2),
        "inference_ms": round(inf_ms, 2),
        "vram_allocated_mb": round(alloc_mb, 2),
    })

profile_summary = {
    "test_image": test_img_path.name,
    "image_resolution": list(img.size),
    "detection_b2": {
        "num_passes": len(det_latencies),
        "median_total_ms": round(float(np.median([d["total_ms"] for d in det_latencies])), 2),
        "p95_total_ms": round(float(np.percentile([d["total_ms"] for d in det_latencies], 95)), 2),
        "median_inference_ms": round(float(np.median([d["inference_ms"] for d in det_latencies])), 2),
        "median_preprocess_ms": round(float(np.median([d["preprocess_ms"] for d in det_latencies])), 2),
        "median_postprocess_ms": round(float(np.median([d["postprocess_ms"] for d in det_latencies])), 2),
        "memory_leak_delta_mb": round(mem_delta, 2),
        "per_pass_details": det_latencies,
    },
    "scene3d_depth": {
        "num_passes": len(s3d_latencies),
        "median_total_ms": round(float(np.median([s["total_ms"] for s in s3d_latencies])), 2),
        "p95_total_ms": round(float(np.percentile([s["total_ms"] for s in s3d_latencies], 95)), 2),
        "per_pass_details": s3d_latencies,
    }
}

out_file = REPO_ROOT / "reports" / "operational_reliability_profile.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(profile_summary, f, indent=2)

print(f"\n[PASS] Operational reliability profiling completed successfully! Saved to: {out_file}")
print(f"Detection B2: Median Total={profile_summary['detection_b2']['median_total_ms']}ms (Inference: {profile_summary['detection_b2']['median_inference_ms']}ms)")
print(f"Scene3D Depth: Median Total={profile_summary['scene3d_depth']['median_total_ms']}ms")

