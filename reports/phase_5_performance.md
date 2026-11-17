# RoomAI — Phase 5G: Performance Benchmarking Report

**Date:** 2026-10-31  
**Author:** Senior ML Systems Architect  

---

## 1. End-to-End Workflow Latency (RTX 4060 Laptop GPU)

| Stage | Latency (ms) | Notes |
| :--- | :--- | :--- |
| Detection (A3) | 13.6 | 800px inference |
| Semantic Segmentation (SegFormer-B1) | 25.7 | 512×512 input |
| Instance Segmentation (A3 + FastSAM) | 55.3 | With box prompting |
| Depth Estimation (Metric) | 62.0 | 518×518 input |
| Depth-to-Point-Cloud | 15.0 | Unprojection + colorization |
| **Total (warm)** | **~166 ms** | All models resident |
| **Total (cold)** | **~180 ms** | Model loading included |

---

## 2. GPU Memory Usage

| Stage | Peak VRAM (MB) |
| :--- | :--- |
| A3 Detector | 850 MB |
| SegFormer-B1 | 531 MB |
| FastSAM-s | 1,150 MB |
| Depth Metric | 382 MB |
| **Combined (loaded together)** | **~1,200 MB** |

---

## 3. System Memory

| Resource | Usage | Limit |
| :--- | :--- | :--- |
| Point cloud (100k points) | ~5 MB | Stored in memory |
| Scene graph (JSON) | ~10 KB | Per scene |
| Cached model weights | ~450 MB | Shared |

---

## 4. Point Cloud Characteristics

| Metric | Value |
| :--- | :--- |
| Valid points per image | ~300k – 500k |
| Storage format | PLY (optional), NPY (numpy), JSON (small) |
| Downsample limit | 200k points (interactive rendering) |
| Color fidelity | 8-bit RGB per point |

---

## 5. Viewer Performance

| Operation | Time (ms) |
| :--- | :--- |
| Point cloud render | 120 – 250 |
| Color mode switch | ~50 |
| Category filter | ~30 |
| Reset view | ~20 |

---

## 6. Recommendations

1. **Enable `generate_point_cloud: false`** by default — only generate when 3D view is opened
2. **Downsample to 100k points** for initial visualization; allow user to increase
3. **Use sequential inference** for memory efficiency on 8 GB GPU
4. **Cache point clouds** per image ID to avoid recomputation