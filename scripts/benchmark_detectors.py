"""Benchmark script comparing YOLO11n and YOLO11s on RTX 4060 GPU."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from ultralytics import YOLO

# Set seeds
torch.manual_seed(42)

VAL_DIR = Path(r"c:\Users\apaar\Desktop\RoomAI\val2017")
ANN_FILE = Path(r"c:\Users\apaar\fiftyone\coco-2017\raw\instances_val2017.json")

# Find representative room images containing furniture classes
FURNITURE_CAT_IDS = {62: "chair", 63: "couch", 65: "bed", 67: "dining table", 64: "potted plant", 72: "tv"}

def find_representative_room_images(num_samples: int = 50) -> List[Path]:
    """Finds COCO validation images containing multiple furniture items."""
    with open(ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    img_furniture_counts = {}

    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid in FURNITURE_CAT_IDS:
            iid = ann["image_id"]
            img_furniture_counts[iid] = img_furniture_counts.get(iid, 0) + 1

    # Sort images by most furniture items
    sorted_imgs = sorted(img_furniture_counts.items(), key=lambda x: x[1], reverse=True)
    selected = []
    for iid, _ in sorted_imgs:
        fname = img_id_to_file[iid]
        p = VAL_DIR / fname
        if p.exists():
            selected.append(p)
            if len(selected) >= num_samples:
                break

    return selected

def benchmark_model(model_name: str, image_paths: List[Path], imgsz: int = 640, half: bool = True) -> Dict[str, Any]:
    """Benchmarks a YOLO model for latency, peak VRAM, throughput, and detections."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Clean GPU memory before test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**2)
    else:
        mem_before = 0.0

    # 1. Model Loading Time
    t0 = time.perf_counter()
    model = YOLO(f"{model_name}.pt")
    load_time = time.perf_counter() - t0

    # Model size on disk
    ckpt_path = Path(f"{model_name}.pt")
    ckpt_size_mb = ckpt_path.stat().st_size / (1024**2) if ckpt_path.exists() else 0.0

    # 2. Warm-up inference
    warmup_img = str(image_paths[0])
    t_w0 = time.perf_counter()
    _ = model.predict(source=warmup_img, imgsz=imgsz, device=device, half=half, verbose=False)
    warmup_latency_ms = (time.perf_counter() - t_w0) * 1000

    # 3. Steady-State Latency Measurement across representative images
    latencies = []
    total_detections = 0
    furniture_detections = 0

    for p in image_paths:
        t_start = time.perf_counter()
        results = model.predict(source=str(p), imgsz=imgsz, device=device, half=half, verbose=False)
        lat = (time.perf_counter() - t_start) * 1000
        latencies.append(lat)

        for r in results:
            total_detections += len(r.boxes)
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0].item())]
                if cls_name in ["chair", "couch", "bed", "dining table", "potted plant", "tv"]:
                    furniture_detections += 1

    # Measure memory
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_vram_mb = 0.0

    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    throughput_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "model_name": model_name,
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "load_time_sec": round(load_time, 3),
        "warmup_ms": round(warmup_latency_ms, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(throughput_fps, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "images_tested": len(image_paths),
        "total_detections": total_detections,
        "furniture_detections": furniture_detections,
        "imgsz": imgsz,
        "precision": "FP16" if half else "FP32",
    }

if __name__ == "__main__":
    print("Finding representative room images from val2017...")
    sample_images = find_representative_room_images(num_samples=50)
    print(f"Selected {len(sample_images)} representative furniture-dense room images.")

    models_to_test = ["yolo11n", "yolo11s"]
    results = []

    print("\nStarting Benchmarking on NVIDIA RTX 4060 Laptop GPU...\n" + "="*60)
    for m in models_to_test:
        print(f"\nEvaluating {m} (imgsz=640, FP16)...")
        res = benchmark_model(m, sample_images, imgsz=640, half=True)
        results.append(res)
        print(f"  Load Time: {res['load_time_sec']}s | Checkpoint: {res['checkpoint_size_mb']} MB")
        print(f"  Warmup: {res['warmup_ms']} ms | Steady-state: {res['mean_latency_ms']} ms (FPS: {res['throughput_fps']})")
        print(f"  Peak VRAM: {res['peak_vram_mb']} MB")
        print(f"  Furniture Detections: {res['furniture_detections']} across {res['images_tested']} room images")

    # Save benchmark results
    out_file = Path("artifacts/detector_benchmark_results.json")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")
