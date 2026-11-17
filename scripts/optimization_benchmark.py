"""Measures optimization trade-offs on RTX 4060: FP16 vs FP32 and Image Resolutions."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

TEST_IMG = r"c:\Users\apaar\Desktop\RoomAI\artifacts\sample_room.jpg"


def measure_config(model_path: str, imgsz: int, half: bool, num_runs: int = 30):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(model_path)

    # Warmup
    for _ in range(5):
        _ = model.predict(source=TEST_IMG, imgsz=imgsz, device=device, half=half, verbose=False)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(source=TEST_IMG, imgsz=imgsz, device=device, half=half, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    return {
        "model": Path(model_path).name,
        "imgsz": imgsz,
        "precision": "FP16" if half else "FP32",
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1),
        "peak_vram_mb": round(peak_vram, 1),
    }


def main():
    print("Running hardware optimization benchmarks on RTX 4060 GPU...")
    experiments = [
        # 1. Precision comparison on YOLO11s (imgsz=640)
        {"model_path": "yolo11s.pt", "imgsz": 640, "half": False},  # FP32
        {"model_path": "yolo11s.pt", "imgsz": 640, "half": True},   # FP16
        # 2. Resolution comparison on YOLO11s (FP16)
        {"model_path": "yolo11s.pt", "imgsz": 512, "half": True},   # Lower res
        {"model_path": "yolo11s.pt", "imgsz": 800, "half": True},   # Higher res
        # 3. Model comparison (YOLO11n FP16 imgsz=640)
        {"model_path": "yolo11n.pt", "imgsz": 640, "half": True},
    ]

    results = []
    for exp in experiments:
        tag = f"{exp['model_path']} | imgsz={exp['imgsz']} | {'FP16' if exp['half'] else 'FP32'}"
        print(f"Testing {tag}...")
        res = measure_config(**exp)
        results.append(res)
        print(f"  -> Latency: {res['mean_latency_ms']} ms | FPS: {res['throughput_fps']} | Peak VRAM: {res['peak_vram_mb']} MB")

    out_file = Path("artifacts/optimization_benchmarks.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved optimization benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
