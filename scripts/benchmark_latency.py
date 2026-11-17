"""RoomAI End-to-End Latency Benchmark for Model A3 Integration."""

import io
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient

from api.main import app


def create_synthetic_room_image(width: int, height: int, seed: int = 0) -> bytes:
    """Generates a realistic synthetic test image with geometric room shapes."""
    from PIL import Image, ImageDraw
    import random
    rng = random.Random(seed)
    base_color = (220 + rng.randint(-5, 5), 215 + rng.randint(-5, 5), 205 + rng.randint(-5, 5))
    img = Image.new("RGB", (width, height), color=base_color)
    draw = ImageDraw.Draw(img)
    # Draw floor
    draw.rectangle([0, int(height * 0.6), width, height], fill=(120, 110, 100))
    # Draw furniture proxies (bed, chair, table) with subtle variations
    x_offset = rng.randint(-10, 10)
    draw.rectangle([int(width * 0.2) + x_offset, int(height * 0.5), int(width * 0.5) + x_offset, int(height * 0.85)], fill=(180, 140, 100))
    draw.rectangle([int(width * 0.6), int(height * 0.55), int(width * 0.75), int(height * 0.8)], fill=(80, 60, 50))
    draw.rectangle([int(width * 0.05), int(height * 0.2), int(width * 0.25), int(height * 0.7)], fill=(70, 70, 80))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run_benchmark():
    print("=" * 70)
    print("ROOMAI END-TO-END LATENCY BENCHMARK — MODEL A3 (800px YOLO11s on RTX 4060)")
    print("=" * 70)

    # Image test configurations
    test_configs = [
        {"name": "Small (640x480)", "width": 640, "height": 480, "cached": False},
        {"name": "Medium (1280x720)", "width": 1280, "height": 720, "cached": False},
        {"name": "Large (1920x1080)", "width": 1920, "height": 1080, "cached": False},
        {"name": "Cached Query (LRU Hit)", "width": 640, "height": 480, "cached": True},
    ]

    NUM_WARM_REQUESTS = 20

    all_results = {}

    with TestClient(app) as client:
        # Pre-warm is handled by startup handler in TestClient context
        print("\nFastAPI application and Model A3 initialized.\n")
        print("METHODOLOGY DETAILS:")
        print("  - Mode: Sequential HTTP client requests (Starlette TestClient / in-memory ASGI)")
        print("  - Warmup: 1 startup pre-warm pass + 1 pre-benchmark pass per configuration")
        print(f"  - Runs: {NUM_WARM_REQUESTS} warm requests per configuration x {len(test_configs)} configurations = {NUM_WARM_REQUESTS * len(test_configs)} total warm requests")
        print("  - Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (PyTorch 2.6.0+cu124, CUDA 12.4)")
        print("  - CUDA Synchronization: Hardware synchronized via torch.cuda.synchronize()")
        print("  - Statistics: Mean, Median (p50), 95th Percentile (p95), Min, Max\n")

        for cfg in test_configs:
            name = cfg["name"]
            w, h = cfg["width"], cfg["height"]
            is_cached_cfg = cfg.get("cached", False)
            print(f"\n--- Benchmarking: {name} ---")

            base_img_bytes = create_synthetic_room_image(w, h, seed=999)
            print(f"Payload size: {len(base_img_bytes) / 1024:.1f} KB (Mode: {'CACHED REPEATS' if is_cached_cfg else 'FRESH INFERENCE'})")

            # Warmup request
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            client.post("/api/v1/detect", files={"file": ("test.jpg", io.BytesIO(base_img_bytes), "image/jpeg")})
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            e2e_latencies = []
            gpu_latencies = []
            server_latencies = []
            pre_latencies = []
            post_latencies = []
            transfer_latencies = []

            for i in range(NUM_WARM_REQUESTS):
                if is_cached_cfg:
                    img_bytes_req = base_img_bytes  # Same bytes -> hits cache
                else:
                    img_bytes_req = create_synthetic_room_image(w, h, seed=i + 1)  # Unique bytes -> fresh inference

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # T0: User submit / Request dispatch
                t0 = time.perf_counter()

                response = client.post(
                    "/api/v1/detect",
                    files={"file": ("test.jpg", io.BytesIO(img_bytes_req), "image/jpeg")},
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t_resp = time.perf_counter()

                assert response.status_code == 200, f"Error {response.status_code}: {response.text}"
                data = response.json()
                t_parse = time.perf_counter()

                total_e2e_ms = (t_parse - t0) * 1000.0
                timing = data.get("timing", {})
                inference = data.get("inference", {})

                gpu_ms = inference.get("latency_ms", 0.0)
                server_total_ms = timing.get("total_server_ms", 0.0)
                pre_ms = timing.get("preprocess_ms", 0.0)
                post_ms = timing.get("postprocess_ms", 0.0)
                transfer_ms = max(0.0, total_e2e_ms - server_total_ms)

                e2e_latencies.append(total_e2e_ms)
                gpu_latencies.append(gpu_ms)
                server_latencies.append(server_total_ms)
                pre_latencies.append(pre_ms)
                post_latencies.append(post_ms)
                transfer_latencies.append(transfer_ms)

            def get_stats(vals: List[float]) -> Dict[str, float]:
                s = sorted(vals)
                p95_idx = int(np.ceil(0.95 * len(s))) - 1
                return {
                    "mean": round(statistics.mean(vals), 2),
                    "median": round(statistics.median(vals), 2),
                    "p95": round(s[p95_idx], 2),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                }

            cfg_result = {
                "sample_count": NUM_WARM_REQUESTS,
                "payload_kb": round(len(base_img_bytes) / 1024, 2),
                "total_e2e": get_stats(e2e_latencies),
                "gpu_inference": get_stats(gpu_latencies),
                "server_total": get_stats(server_latencies),
                "preprocess": get_stats(pre_latencies),
                "postprocess": get_stats(post_latencies),
                "transfer_overhead": get_stats(transfer_latencies),
            }

            all_results[name] = cfg_result

            print(f"Results for {name} ({NUM_WARM_REQUESTS} runs):")
            print(f"  Total E2E:    p50 = {cfg_result['total_e2e']['median']} ms | p95 = {cfg_result['total_e2e']['p95']} ms | mean = {cfg_result['total_e2e']['mean']} ms (min: {cfg_result['total_e2e']['min']}, max: {cfg_result['total_e2e']['max']})")
            print(f"  GPU Forward:  p50 = {cfg_result['gpu_inference']['median']} ms | p95 = {cfg_result['gpu_inference']['p95']} ms | mean = {cfg_result['gpu_inference']['mean']} ms")
            print(f"  Server Total: p50 = {cfg_result['server_total']['median']} ms | p95 = {cfg_result['server_total']['p95']} ms")
            print(f"  Preprocess:   mean = {cfg_result['preprocess']['mean']} ms")
            print(f"  Postprocess:  mean = {cfg_result['postprocess']['mean']} ms")
            print(f"  Transfer:     mean = {cfg_result['transfer_overhead']['mean']} ms")

        # Save to disk
        out_path = Path("plans/latency-benchmark-results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved benchmark metrics to {out_path}")

        return all_results


if __name__ == "__main__":
    run_benchmark()
