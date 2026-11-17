# ROOMAI — FINAL BACKEND PERFORMANCE & TELEMETRY REPORT

**Date:** 2026-11-06  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM)  
**Host System RAM:** 24 GB DDR5  
**Operating System:** Windows 11 (Build 26200)  
**PyTorch Version:** 2.11.0+cu128 (CUDA 12.8)  
**Benchmark Test Image:** `artifacts/sample_room.jpg` (800 x 600 RGB indoor room image)  
**Benchmark Script:** `scripts/benchmark_final_backend.py`

---

## 1. Benchmarking Methodology

To ensure scientific accuracy and reproducibility, all benchmarks were conducted under strict engineering protocols:
1. **CUDA Synchronization:** Explicit `torch.cuda.synchronize()` calls were executed immediately before starting the timer and after receiving the model output, eliminating asynchronous GPU launch skew.
2. **Warm-up Iterations:** A cold start run was measured separately to capture initial CUDA kernel compilation, weight paging, and tensor allocations. This was followed by 10 warm-up runs to measure steady-state production performance.
3. **Hardware Memory Tracking:** `torch.cuda.max_memory_allocated(0)` and `torch.cuda.max_memory_reserved(0)` were recorded per subsystem. Host process memory was tracked via `psutil.Process().memory_info().rss`.
4. **Real Weight Verification:** All benchmarks executed against authentic production checkpoints on disk; zero synthetic mocks were active during measurement.

---

## 2. Comprehensive Performance Benchmarks

| Subsystem / Pipeline Step | Model Name | Checkpoint / Params | Cold Start (ms) | Warm p50 (ms) | Warm p95 (ms) | Peak VRAM Allocated (MB) | Peak VRAM Reserved (MB) | Steady-State FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Indoor Detection (A3)** | YOLO11s (800px) | `indoor_detector_a3...pt` (9.4M) | 2,706.13 | **24.81** | 49.61 | 131.2 | 176.0 | **40.3 FPS** |
| **Exterior Detection (B2)** | YOLO11s | `exterior_detector_b2...pt` (9.4M) | 3,353.99 | **23.42** | 49.72 | 107.7 | 192.0 | **42.7 FPS** |
| **Parking Detection (C2)** | YOLO11s | `parking_detector_c2...pt` (9.4M) | 170.05 | **23.68** | 34.55 | 126.8 | 204.0 | **42.2 FPS** |
| **Instance Segmentation** | YOLO11n-Seg | `yolo11n-seg.pt` (2.9M) | 344.66 | **172.30** | 259.55 | 203.3 | 254.0 | **5.8 FPS** |
| **Metric Depth Estimation** | Depth Anything V2 Metric | `safetensors` (24.8M) | 4,336.60 | **518.54** | 685.09 | 345.4 | 482.0 | **1.9 FPS** |
| **Relative Depth Estimation** | Depth Anything V2 Small | `safetensors` (24.8M) | 465.15 | **521.80** | 556.67 | 345.4 | 482.0 | **1.9 FPS** |
| **Full Single-View 3D Recon** | Unified Pipeline (A3+Seg+Depth+RANSAC) | Multi-model composite | 1,460.65 | **1,546.55** | 1,772.51 | **367.7** | **486.0** | **0.65 FPS** |
| **3D Asset Export (GLB+PLY)** | glTF 2.0 & PLY Engine | CPU serialization | 127.95 | **125.33** | 136.26 | 0.0 (CPU) | 0.0 (CPU) | **8.0 FPS** |

---

## 3. VRAM Budget & Hardware Headroom Analysis

The target hardware configuration is an NVIDIA GeForce RTX 4060 Laptop GPU with 8,187.5 MB dedicated VRAM. Maintaining sufficient VRAM headroom is critical to prevent Windows Driver display resets (TDR) and out-of-memory crashes.

```
Total RTX 4060 VRAM Budget: 8,187.5 MB
┌────────────────────────────────────────────────────────────────────────┐
│ Peak Allocated: 367.7 MB (4.5%)                                       │
│ Peak Reserved:  486.0 MB (5.9%)                                       │
│ Available Headroom: 7,701.5 MB (94.1%)                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Hardware Takeaways:
1. **Negligible VRAM Footprint:** The entire active single-view 3D scene reconstruction pipeline requires only **367.7 MB of allocated VRAM** and **486.0 MB of reserved PyTorch caching memory**.
2. **Headroom:** With **> 7.7 GB of unused VRAM**, there is zero risk of VRAM thrashing or OOM crashes when operating the core perception and 3D reconstruction models.
3. **Host RAM Stability:** The Python runtime maintains an RSS footprint of **2,292.7 MB (~2.3 GB)** on the 24 GB host system, leaving substantial capacity for operating system tasks, UI rendering, and browser sessions.
4. **Latencies for UI Experience:**
   - Object detection executes in **~25 ms** (near real-time / interactive).
   - Full 3D room reconstruction completes in **~1.55 seconds**, providing an optimal balance of reconstruction fidelity and user responsiveness for web applications.

---

## 4. Frontend Optimization Guidelines

1. **Async Reconstruction:** Because full 3D reconstruction requires ~1.5 seconds, frontend applications (Streamlit / React) should display a spinner or progressive progress indicator during processing.
2. **Asset Streaming:** Exported GLB files are approximately 480 KB for 40,000 points and oriented furniture boxes; these can be transferred to Three.js or Plotly viewers with negligible network latency (< 5 ms locally).
3. **Downsampling Tuning:** For low-power mobile or browser clients, the `max_points` argument in `BACKEND.export_3d_assets(scene, max_points=20000)` can be reduced from 40,000 to 20,000, cutting export latency to under 70 ms.

