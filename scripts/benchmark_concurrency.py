"""RoomAI Concurrency & GPU Lock Benchmark for Model A3 on RTX 4060."""

import asyncio
import io
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx
import numpy as np
from PIL import Image, ImageDraw
import torch

from api.main import app
from api.routers.detect import GLOBAL_MODEL_A3_ENGINE


def get_nvidia_smi_memory_mb() -> float:
    """Queries nvidia-smi for total device memory used in MiB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(out.split("\n")[0].strip())
    except Exception:
        return 0.0


def make_unique_room_image(seed: int, width: int = 640, height: int = 480) -> bytes:
    """Generates unique synthetic room images to bypass LRU cache."""
    rng = random.Random(seed)
    base_color = (200 + rng.randint(-15, 15), 195 + rng.randint(-15, 15), 185 + rng.randint(-15, 15))
    img = Image.new("RGB", (width, height), color=base_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(height * 0.6), width, height], fill=(110, 100, 90))
    x_off = rng.randint(-20, 20)
    draw.rectangle([int(width * 0.2) + x_off, int(height * 0.5), int(width * 0.5) + x_off, int(height * 0.85)], fill=(160, 130, 90))
    draw.rectangle([int(width * 0.6), int(height * 0.55), int(width * 0.75), int(height * 0.8)], fill=(75, 55, 45))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


async def run_concurrency_level(
    client: httpx.AsyncClient, concurrency: int, num_rounds: int = 5
) -> Dict[str, Any]:
    print(f"\nEvaluating Concurrency = {concurrency} across {num_rounds} batches...")

    all_req_latencies = []
    all_queue_waits = []
    all_gpu_times = []
    total_requests = concurrency * num_rounds
    failed_requests = 0

    # Ensure LRU cache is clean and seeds are unique across concurrency levels
    GLOBAL_MODEL_A3_ENGINE.clear_cache()

    # Pre-generate unique payloads so image construction overhead on CPU does not pollute batch duration
    all_batches_payloads = [
        [
            make_unique_room_image(seed=(concurrency * 10000 + r * 100 + i + 1))
            for i in range(concurrency)
        ]
        for r in range(num_rounds)
    ]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_allocated_start = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_reserved_start = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        mem_allocated_start = 0.0
        mem_reserved_start = 0.0

    t_batch_start_total = time.perf_counter()

    for r in range(num_rounds):
        payloads = all_batches_payloads[r]

        async def send_single(payload: bytes, req_idx: int):
            t_req_start = time.perf_counter()
            files = {"file": (f"room_c{concurrency}_r{r}_{req_idx}.jpg", io.BytesIO(payload), "image/jpeg")}
            data = {"confidence_threshold": "0.30"}
            resp = await client.post("/api/v1/detect", files=files, data=data, timeout=30.0)
            t_req_end = time.perf_counter()
            return resp, (t_req_end - t_req_start) * 1000.0

        tasks = [send_single(p, i) for i, p in enumerate(payloads)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                failed_requests += 1
                continue
            resp, req_ms = res
            if resp.status_code == 200:
                body = resp.json()
                all_req_latencies.append(req_ms)
                timing = body.get("timing", {})
                all_queue_waits.append(timing.get("queue_wait_ms", 0.0))
                all_gpu_times.append(timing.get("inference_ms", 0.0))
            else:
                failed_requests += 1

    total_batch_duration = time.perf_counter() - t_batch_start_total
    throughput_rps = round(len(all_req_latencies) / total_batch_duration, 2)

    if torch.cuda.is_available():
        mem_allocated_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        mem_reserved_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        peak_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        peak_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
    else:
        mem_allocated_mb = 0.0
        mem_reserved_mb = 0.0
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    nvidia_smi_total_used_mb = get_nvidia_smi_memory_mb()

    def calc_stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
        s = sorted(vals)
        p95_idx = int(np.ceil(0.95 * len(s))) - 1
        return {
            "mean": round(statistics.mean(vals), 2),
            "median": round(statistics.median(vals), 2),
            "p95": round(s[p95_idx], 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(all_req_latencies),
        "failed_requests": failed_requests,
        "throughput_rps": throughput_rps,
        "vram_metrics": {
            "torch_cuda_memory_allocated_mb": mem_allocated_mb,
            "torch_cuda_memory_reserved_mb": mem_reserved_mb,
            "torch_cuda_max_memory_allocated_mb": peak_allocated_mb,
            "torch_cuda_max_memory_reserved_mb": peak_reserved_mb,
            "nvidia_smi_device_used_mb": nvidia_smi_total_used_mb,
            "interpretation": (
                "Peak allocated represents active PyTorch tensors; reserved represents blocks allocated "
                "from CUDA driver by caching allocator; nvidia-smi measures total OS-level device memory."
            ),
        },
        "latency_stats": calc_stats(all_req_latencies),
        "queue_wait_stats": calc_stats(all_queue_waits),
        "gpu_inference_stats": calc_stats(all_gpu_times),
    }


async def main():
    print("=" * 75)
    print("ROOMAI CONCURRENCY & VRAM AUDIT: 1, 2, 4, 8 WORKERS (RTX 4060 8GB)")
    print("=" * 75)

    # Pre-warm Model A3 engine
    GLOBAL_MODEL_A3_ENGINE.load_and_warmup()

    concurrency_levels = [1, 2, 4, 8]
    summary_results = {}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Client-level pre-warmup: send 3 warmup requests to ensure route initialization / JIT complete
        print("Executing client HTTP pipeline pre-warmup...")
        for w in range(3):
            warm_bytes = make_unique_room_image(seed=999000 + w)
            files = {"file": ("warmup.jpg", io.BytesIO(warm_bytes), "image/jpeg")}
            await client.post("/api/v1/detect", files=files, data={"confidence_threshold": "0.30"})
        print("Pre-warmup complete.")

        for c in concurrency_levels:
            metrics = await run_concurrency_level(client, concurrency=c, num_rounds=5)
            summary_results[f"concurrency_{c}"] = metrics

            print(f"Results for Concurrency = {c}:")
            print(f"  Success:     {metrics['successful_requests']}/{metrics['total_requests']} (Failures: {metrics['failed_requests']})")
            print(f"  Throughput:  {metrics['throughput_rps']} req/s")
            print(f"  E2E Latency: p50 = {metrics['latency_stats']['median']} ms | p95 = {metrics['latency_stats']['p95']} ms | mean = {metrics['latency_stats']['mean']} ms")
            print(f"  Queue Wait:  p50 = {metrics['queue_wait_stats']['median']} ms | p95 = {metrics['queue_wait_stats']['p95']} ms | mean = {metrics['queue_wait_stats']['mean']} ms")
            print(f"  GPU Time:    mean = {metrics['gpu_inference_stats']['mean']} ms (min = {metrics['gpu_inference_stats']['min']} ms, max = {metrics['gpu_inference_stats']['max']} ms)")
            v = metrics["vram_metrics"]
            print(f"  VRAM Breakdown:")
            print(f"    - torch.cuda.memory_allocated():     {v['torch_cuda_memory_allocated_mb']} MB")
            print(f"    - torch.cuda.max_memory_allocated(): {v['torch_cuda_max_memory_allocated_mb']} MB (peak active tensors)")
            print(f"    - torch.cuda.memory_reserved():      {v['torch_cuda_memory_reserved_mb']} MB (PyTorch reserved blocks)")
            print(f"    - torch.cuda.max_memory_reserved():  {v['torch_cuda_max_memory_reserved_mb']} MB")
            print(f"    - nvidia-smi total device used:     {v['nvidia_smi_device_used_mb']} MB")

    out_file = Path("plans/concurrency-benchmark-results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nWrote full concurrency benchmark metrics to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
