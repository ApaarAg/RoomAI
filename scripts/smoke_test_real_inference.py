"""Real-inference smoke benchmark for production RoomAI models on CUDA.

Runs each production model on a single held-out image and reports:
  - cold load time, warm inference time, device, detected objects,
  - peak VRAM delta (CUDA only).

Guarded so it is skipped cleanly when checkpoints/data are unavailable.
Run with the tf-gpu interpreter::

    python -m scripts.smoke_test_real_inference
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

INDOOR_IMG_DIR = REPO_ROOT / "data" / "deepfurniture" / "images" / "test"
EXTERIOR_IMG_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "val"
PARKING_IMG_DIR = REPO_ROOT / "data" / "parking_canonical" / "images" / "test"


def _load_first(directory: Path) -> Image.Image:
    first = sorted(directory.glob("*.jpg"))[0]
    return Image.open(first).convert("RGB")


def _cuda_alloc_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def benchmark_detection(name: str, cfg: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
    from services.detection_service import DetectionService

    t0 = time.perf_counter()
    svc = DetectionService(mock_mode=False, config=cfg)
    load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    before = _cuda_alloc_mb()
    _cuda_sync()
    ti = time.perf_counter()
    result = svc.run(image, confidence_threshold=cfg.get("confidence_threshold", 0.30), save_annotated=False)
    warm_ms = round((time.perf_counter() - ti) * 1000.0, 2)
    _cuda_sync()
    after = _cuda_alloc_mb()

    resp = result["response"]
    return {
        "model": name,
        "checkpoint": resp.checkpoint if resp else None,
        "device": resp.device if resp else None,
        "is_simulated": resp.is_simulated if resp else None,
        "load_ms": load_ms,
        "warm_inference_ms": warm_ms,
        "vram_before_mb": before,
        "vram_after_mb": after,
        "vram_delta_mb": round(after - before, 2),
        "objects": resp.total_objects if resp else 0,
        "timing_ms": resp.inference_time_ms if resp else 0.0,
    }


def benchmark_segmentation(cfg: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
    from services.segmentation_service import SegmentationService

    t0 = time.perf_counter()
    svc = SegmentationService(mock_mode=False, config=cfg)
    load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    before = _cuda_alloc_mb()
    _cuda_sync()
    ti = time.perf_counter()
    result = svc.run(image)
    warm_ms = round((time.perf_counter() - ti) * 1000.0, 2)
    _cuda_sync()
    after = _cuda_alloc_mb()

    resp = result["response"]
    return {
        "model": "YOLO11n-seg (pretrained)",
        "device": getattr(resp, "device", None) if resp else None,
        "is_simulated": resp.is_simulated if resp else None,
        "load_ms": load_ms,
        "warm_inference_ms": warm_ms,
        "vram_before_mb": before,
        "vram_after_mb": after,
        "vram_delta_mb": round(after - before, 2),
        "masks": len(resp.masks) if resp else 0,
    }


def benchmark_depth(cfg: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
    from services.scene3d_service import Scene3DService

    t0 = time.perf_counter()
    svc = Scene3DService(mock_mode=False, config=cfg)
    load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    before = _cuda_alloc_mb()
    _cuda_sync()
    ti = time.perf_counter()
    result = svc.run(image)
    warm_ms = round((time.perf_counter() - ti) * 1000.0, 2)
    _cuda_sync()
    after = _cuda_alloc_mb()

    resp = result["response"]
    return {
        "model": "Depth Anything V2 Small",
        "device": resp.device if resp else None,
        "is_simulated": resp.is_simulated if resp else None,
        "load_ms": load_ms,
        "warm_inference_ms": warm_ms,
        "vram_before_mb": before,
        "vram_after_mb": after,
        "vram_delta_mb": round(after - before, 2),
        "peak_memory_mb": resp.peak_memory_mb if resp else 0.0,
        "depth_min": resp.depth_min if resp else 0.0,
        "depth_max": resp.depth_max if resp else 0.0,
        "relative_depth_only": resp.relative_depth_only if resp else True,
    }


def main() -> None:
    import yaml

    with open(REPO_ROOT / "config" / "model_config.yaml", "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    results: Dict[str, Any] = {}

    try:
        import torch

        results["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            results["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        results["cuda_available"] = False

    # Indoor A3
    if INDOOR_IMG_DIR.exists():
        img = _load_first(INDOOR_IMG_DIR)
        cfg = dict(model_config.get("detection", {}))
        results["indoor_a3"] = benchmark_detection("Indoor A3 (800px)", cfg, img)
    else:
        results["indoor_a3"] = "skipped (no indoor test images)"

    # Exterior B2
    if EXTERIOR_IMG_DIR.exists():
        img = _load_first(EXTERIOR_IMG_DIR)
        cfg = dict(model_config.get("exterior_detection", {}))
        results["exterior_b2"] = benchmark_detection("Exterior B2", cfg, img)
    else:
        results["exterior_b2"] = "skipped (no exterior val images)"

    # Parking C2
    if PARKING_IMG_DIR.exists():
        img = _load_first(PARKING_IMG_DIR)
        cfg = dict(model_config.get("parking_detection", {}))
        results["parking_c2"] = benchmark_detection("Parking C2", cfg, img)
    else:
        results["parking_c2"] = "skipped (no parking test images)"

    # Segmentation
    if INDOOR_IMG_DIR.exists():
        img = _load_first(INDOOR_IMG_DIR)
        results["segmentation"] = benchmark_segmentation(model_config.get("segmentation", {}), img)

    # Depth
    if INDOOR_IMG_DIR.exists():
        img = _load_first(INDOOR_IMG_DIR)
        results["depth_anything_v2_small"] = benchmark_depth(model_config.get("scene3d", {}), img)

    print(json.dumps(results, indent=2, default=str))
    out = REPO_ROOT / "reports" / "real_inference_smoke_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()