"""End-to-End Inference Benchmarking for Fine-Tuned DeepFurniture YOLO11n."""

import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "deepfurniture_yolo11n_finetune_best.pt"
TEST_IMG_DIR = REPO_ROOT / "data" / "deepfurniture" / "images" / "test"
REPORT_PATH = REPO_ROOT / "runs" / "benchmarks" / "inference_benchmark_report.json"


def benchmark_inference():
    print("=" * 65)
    print(" TASK C: DETAILED END-TO-END INFERENCE BENCHMARK")
    print("=" * 65)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Target Device:     {device_str} ({gpu_name})")
    print(f"Checkpoint:        {CHECKPOINT_PATH}")

    # 1. Measure Cold Start Model Loading Time
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    t_load_start = time.perf_counter()
    model = YOLO(str(CHECKPOINT_PATH))
    # Trigger model preparation on GPU
    _ = model.to(device_str)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_time_ms = (time.perf_counter() - t_load_start) * 1000.0
    print(f"Model Cold-Load Time: {model_load_time_ms:.2f} ms")

    # Collect test images
    test_images = sorted(list(TEST_IMG_DIR.glob("*.jpg")))[:50]
    print(f"Benchmarking across {len(test_images)} test images (Batch Size = 1, ImgSz = 640)")

    # 2. Warm-Up Phase
    t_warmup_start = time.perf_counter()
    warmup_count = 10
    for img_path in test_images[:warmup_count]:
        img = Image.open(img_path).convert("RGB")
        _ = model.predict(source=img, imgsz=640, conf=0.35, device=device_str, half=False, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_time_ms = (time.perf_counter() - t_warmup_start) * 1000.0
    print(f"Warm-Up Time ({warmup_count} iterations): {warmup_time_ms:.2f} ms ({warmup_time_ms / warmup_count:.2f} ms/iter)")

    # 3. Component-Wise Timings across 50 Real Room Photos
    prep_times = []
    infer_times = []
    post_times = []
    total_times = []

    # Reset peak memory for steady-state inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    for img_path in test_images:
        pil_img = Image.open(img_path).convert("RGB")

        # Total application end-to-end timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_e2e_0 = time.perf_counter()

        # Run predict with return of results
        results = model.predict(
            source=pil_img,
            imgsz=640,
            conf=0.35,
            iou=0.45,
            device=device_str,
            half=False,
            verbose=False,
        )[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_e2e_1 = time.perf_counter()

        total_ms = (t_e2e_1 - t_e2e_0) * 1000.0

        # Extract Ultralytics internal phase breakdown (in milliseconds)
        speed = results.speed  # dict with 'preprocess', 'inference', 'postprocess', 'loss'
        prep_ms = speed.get("preprocess", 0.0)
        inf_ms = speed.get("inference", 0.0)
        post_ms = speed.get("postprocess", 0.0)

        prep_times.append(prep_ms)
        infer_times.append(inf_ms)
        post_times.append(post_ms)
        total_times.append(total_ms)

    peak_gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)

    total_times_arr = np.array(total_times)
    mean_e2e = float(np.mean(total_times_arr))
    median_e2e = float(np.median(total_times_arr))
    p95_e2e = float(np.percentile(total_times_arr, 95))
    min_e2e = float(np.min(total_times_arr))
    max_e2e = float(np.max(total_times_arr))
    fps = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0

    mean_prep = float(np.mean(prep_times))
    mean_infer = float(np.mean(infer_times))
    mean_post = float(np.mean(post_times))

    print("\n--- Latency Breakdown (Mean per Image) ---")
    print(f"  1. Preprocessing:      {mean_prep:>6.2f} ms")
    print(f"  2. Model Inference:    {mean_infer:>6.2f} ms")
    print(f"  3. Postprocessing/NMS: {mean_post:>6.2f} ms")
    print(f"  -----------------------------------")
    print(f"  End-to-End Total Mean: {mean_e2e:>6.2f} ms ({fps:.1f} FPS)")
    print(f"  End-to-End Median:     {median_e2e:>6.2f} ms")
    print(f"  End-to-End P95:        {p95_e2e:>6.2f} ms")
    print(f"  Min / Max:             {min_e2e:.2f} ms / {max_e2e:.2f} ms")
    print(f"  Peak GPU VRAM:         {peak_gpu_mem_mb:.2f} MB (~{peak_gpu_mem_mb / 1024:.2f} GB)")

    benchmark_data = {
        "hardware": {
            "device": device_str,
            "gpu_name": gpu_name,
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
        },
        "config": {
            "checkpoint": str(CHECKPOINT_PATH),
            "batch_size": 1,
            "imgsz": 640,
            "precision": "FP32",
            "samples_evaluated": len(test_images),
        },
        "loading_and_warmup": {
            "model_load_time_ms": round(model_load_time_ms, 2),
            "warmup_time_ms": round(warmup_time_ms, 2),
            "warmup_iterations": warmup_count,
        },
        "component_latencies_ms": {
            "preprocessing_mean": round(mean_prep, 2),
            "inference_mean": round(mean_infer, 2),
            "postprocessing_mean": round(mean_post, 2),
        },
        "end_to_end_latencies_ms": {
            "mean": round(mean_e2e, 2),
            "median": round(median_e2e, 2),
            "p95": round(p95_e2e, 2),
            "min": round(min_e2e, 2),
            "max": round(max_e2e, 2),
            "throughput_fps": round(fps, 1),
        },
        "memory": {
            "peak_gpu_vram_mb": peak_gpu_mem_mb,
            "peak_gpu_vram_gb": round(peak_gpu_mem_mb / 1024, 2),
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nSaved benchmark report to: {REPORT_PATH}")


if __name__ == "__main__":
    benchmark_inference()
