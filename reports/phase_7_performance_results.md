# RoomAI — Phase 7: Performance & Reliability Benchmark Report

**Date:** November 5, 2026  
**Auditor / Benchmarker:** Senior ML Systems Architect & Performance Profiler  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), Intel i7-13700H, 24 GB RAM, Windows 11  
**Benchmark Suite:** 20-Image Diverse Indoor Validation Set (`archive (2)/ADEChallengeData2016/images/validation`)  
**Status:** Complete & Verified  

---

## 1. Executive Performance Summary

| Metric | Measured Value | Operational Budget / Target | Status |
| :--- | :--- | :--- | :--- |
| **Model Load Latency (Lazy)** | **9,338.98 ms** | $\le 15,000$ ms | **PASS** |
| **First (Cold) Inference Latency** | **3,096.69 ms** | $\le 5,000$ ms | **PASS** |
| **Warm Inference Average Latency** | **1,295.47 ms** | $\le 2,500$ ms | **PASS** |
| **Fastest Reconstruction** | **576.2 ms** (`ADE_val_00000018`) | N/A | **PASS** |
| **Peak GPU VRAM Allocated** | **339.9 MB** | $\le 4,000$ MB (Budget: 8,187.5 MB) | **PASS (4.1% of VRAM)** |
| **Peak GPU VRAM Reserved** | **456.0 MB** | $\le 6,000$ MB | **PASS** |
| **Host System RAM Consumption** | **~2.8 GB** | $\le 16.0$ GB (Available: 24 GB) | **PASS** |
| **Point Cloud Synthesis Latency** | **45 – 95 ms** | $\le 200$ ms | **PASS** |
| **GLB / PLY Export Latency** | **18 – 42 ms** | $\le 100$ ms | **PASS** |
| **Average 3D Points per Scene** | **216,947 points** | $\ge 50,000$ points | **PASS** |

---

## 2. 20-Image Per-Scene Benchmark Breakdown

| Scene Index & File | Resolution | Total Latency | Peak VRAM | Objects | Metric 3D Points | Room Dimensions ($W \times L \times H$) | Spatial Relations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `01` `ADE_val_00000001.jpg` | $683 \times 512$ | 3,096.7 ms | 317.9 MB | 1 | 342,702 | $6.3\text{m} \times 8.5\text{m} \times 3.9\text{m}$ | 0 |
| `02` `ADE_val_00000002.jpg` | $500 \times 364$ | 1,260.0 ms | 254.2 MB | 0 | 178,359 | $8.3\text{m} \times 11.1\text{m} \times 4.6\text{m}$ | 0 |
| `03` `ADE_val_00000003.jpg` | $400 \times 300$ | 1,091.3 ms | 257.4 MB | 3 | 117,599 | $5.3\text{m} \times 16.3\text{m} \times 2.9\text{m}$ | 10 |
| `04` `ADE_val_00000004.jpg` | $683 \times 512$ | 1,594.9 ms | 318.8 MB | 3 | 342,701 | $10.8\text{m} \times 16.1\text{m} \times 4.7\text{m}$ | 8 |
| `05` `ADE_val_00000005.jpg` | $400 \times 300$ | 961.0 ms | 258.3 MB | 6 | 117,600 | $4.2\text{m} \times 17.6\text{m} \times 2.2\text{m}$ | 32 |
| `06` `ADE_val_00000006.jpg` | $683 \times 512$ | 1,582.2 ms | 317.9 MB | 0 | 342,701 | $4.1\text{m} \times 2.5\text{m} \times 2.7\text{m}$ | 0 |
| `07` `ADE_val_00000007.jpg` | $500 \times 342$ | 1,373.9 ms | 249.1 MB | 2 | 167,580 | $2.1\text{m} \times 14.9\text{m} \times 2.2\text{m}$ | 2 |
| `08` `ADE_val_00000008.jpg` | $375 \times 500$ | 1,430.9 ms | 256.8 MB | 1 | 183,748 | $5.8\text{m} \times 15.8\text{m} \times 5.8\text{m}$ | 0 |
| `09` `ADE_val_00000009.jpg` | $768 \times 512$ | 1,820.7 ms | 339.9 MB | 10 | 385,351 | $7.9\text{m} \times 14.0\text{m} \times 2.5\text{m}$ | 99 |
| `10` `ADE_val_00000010.jpg` | $500 \times 345$ | 1,259.4 ms | 251.7 MB | 3 | 169,050 | $4.8\text{m} \times 8.2\text{m} \times 2.7\text{m}$ | 8 |
| `11` `ADE_val_00000011.jpg` | $478 \times 320$ | 1,141.5 ms | 249.6 MB | 5 | 149,900 | $2.0\text{m} \times 3.5\text{m} \times 2.2\text{m}$ | 23 |
| `12` `ADE_val_00000012.jpg` | $568 \times 400$ | 1,508.8 ms | 252.5 MB | 5 | 222,656 | $3.8\text{m} \times 9.8\text{m} \times 2.2\text{m}$ | 25 |
| `13` `ADE_val_00000013.jpg` | $430 \times 430$ | 1,366.0 ms | 282.9 MB | 2 | 181,201 | $2.0\text{m} \times 2.8\text{m} \times 2.2\text{m}$ | 4 |
| `14` `ADE_val_00000014.jpg` | $683 \times 512$ | 1,435.0 ms | 318.5 MB | 4 | 342,702 | $4.7\text{m} \times 14.4\text{m} \times 4.3\text{m}$ | 8 |
| `15` `ADE_val_00000015.jpg` | $512 \times 682$ | 1,847.8 ms | 317.9 MB | 3 | 342,200 | $2.7\text{m} \times 15.9\text{m} \times 3.9\text{m}$ | 8 |
| `16` `ADE_val_00000016.jpg` | $480 \times 640$ | 1,557.4 ms | 319.6 MB | 6 | 301,056 | $2.2\text{m} \times 11.8\text{m} \times 4.4\text{m}$ | 30 |
| `17` `ADE_val_00000017.jpg` | $204 \times 380$ | 694.4 ms | 234.7 MB | 0 | 75,969 | $2.0\text{m} \times 10.9\text{m} \times 2.7\text{m}$ | 0 |
| `18` `ADE_val_00000018.jpg` | $320 \times 240$ | 576.2 ms | 256.5 MB | 0 | 75,264 | $4.3\text{m} \times 15.0\text{m} \times 2.2\text{m}$ | 0 |
| `19` `ADE_val_00000019.jpg` | $600 \times 402$ | 1,400.1 ms | 249.6 MB | 5 | 236,376 | $2.0\text{m} \times 3.4\text{m} \times 2.2\text{m}$ | 31 |
| `20` `ADE_val_00000020.jpg` | $256 \times 256$ | 712.4 ms | 282.5 MB | 1 | 64,225 | $3.4\text{m} \times 14.4\text{m} \times 2.2\text{m}$ | 1 |

---

## 3. Reliability & Hardware Headroom Assessment

1. **Peak VRAM Headroom:** Peak allocation of $339.9\text{ MB}$ leaves $> 7,800\text{ MB}$ ($95.8\%$) of GPU memory unencumbered. Zero Out-Of-Memory (OOM) faults were observed across all 20 runs.
2. **Deterministic Stability:** 20 out of 20 scenes ($100\%$) completed reconstruction, scene graph generation, and 3D export without a single unhandled exception or NaN/Inf coordinate defect.
3. **Execution Latency:** Average warm reconstruction of $1,295.5\text{ ms}$ enables responsive interactive usage in the Streamlit application.

