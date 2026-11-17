import os
import sys
import time
from pathlib import Path

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
test_imgs = sorted(list((REPO_ROOT / "data" / "exterior_v2_curated" / "images" / "test").glob("*.jpg")))[:50]

print(f"Loaded {len(test_imgs)} test images for latency audit.")

ckpts = {
    "B2": REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt",
    "B3": REPO_ROOT / "checkpoints" / "exterior_detector_b3_yolo11s_rfs_best.pt",
    "B4": REPO_ROOT / "checkpoints" / "exterior_detector_b4_yolo11s_curated_best.pt",
}

for name, ckpt_path in ckpts.items():
    print(f"\n==========================================")
    print(f"Benchmarking Model {name}: {ckpt_path.name}")
    model = YOLO(str(ckpt_path))
    
    # Warm-up 10 passes
    for p in test_imgs[:10]:
        _ = model.predict(source=str(p), imgsz=640, device=0, verbose=False)
    torch.cuda.synchronize()
    
    latencies = []
    for idx, p in enumerate(test_imgs):
        t0 = time.perf_counter()
        _ = model.predict(source=str(p), imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0
        latencies.append(lat)
        if lat > 60.0:
            print(f"  [OUTLIER] Sample {idx:02d} ({p.name}): {lat:.2f} ms")
            
    lats = np.array(latencies)
    print(f"Stats for {name}:")
    print(f"  Min:    {np.min(lats):.2f} ms")
    print(f"  Mean:   {np.mean(lats):.2f} ms")
    print(f"  Median: {np.median(lats):.2f} ms")
    print(f"  P90:    {np.percentile(lats, 90):.2f} ms")
    print(f"  P95:    {np.percentile(lats, 95):.2f} ms")
    print(f"  P99:    {np.percentile(lats, 99):.2f} ms")
    print(f"  Max:    {np.max(lats):.2f} ms")
    print(f"  Top 5 slowest: {sorted(latencies, reverse=True)[:5]}")
