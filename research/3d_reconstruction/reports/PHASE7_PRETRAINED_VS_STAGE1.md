# ROOMAI — PHASE 7: PRETRAINED BASELINE VS. STAGE 1 HEAD-TO-HEAD EVALUATION

**Author:** Independent Senior 3D Computer Vision Evaluation Scientist  
**Date:** November 8, 2026  
**Status:** RIGOROUS EVALUATION COMPLETE  
**Benchmark Artifact:** `research/3d_reconstruction/experiments/phase7_ground_truth_benchmark.json`  
**Dataset:** Validation Split (402 standardized frames: 100 Hypersim CAD, 302 ScanNet IR Structured Light)  
**Quarantine:** Test Split (484 frames across 2 scenes) remained strictly frozen and untouched.

---

## 1. EXECUTIVE SUMMARY

An exhaustive head-to-head empirical evaluation was conducted comparing the **Original Pretrained WorldMirror-1.0 Baseline** (`production_model.ckpt`) against the **Fine-Tuned Stage 1 Checkpoint** (`best_stage1.ckpt`) across all 402 standardized validation frames under identical inference conditions (BF16, single view, native resolution $518 \times 518$, NVIDIA RTX 4060 Laptop GPU).

### Primary Finding:
The Stage 1 fine-tuned checkpoint **failed completely to generalize to real physical sensor data**. While displaying minor fluctuations on synthetic CAD path-traced views (Hypersim), its reconstruction accuracy on real physical structured-light scans (ScanNet) experienced a catastrophic collapse:
- **Depth AbsRel error exploded by +914.0%** (0.5456 $\to$ 5.5317).
- **Depth RMSE error surged by +611.6%** (1.528 m $\to$ 10.871 m).
- **Metric Point Cloud Chamfer Distance surged by +652.6%** (1.439 m $\to$ 10.828 m).
- **Overall Normal MAE degraded by +11.2%** (62.66° $\to$ 69.68°).

Under RoomAI’s mandatory domain guardrail ("*a candidate cannot win by improving on synthetic CAD while regressing on physical sensor data*"), **Stage 1 is categorically rejected**. The **Original Pretrained Foundation Baseline remains demonstrably superior**.

---

## 2. COMPREHENSIVE VALIDATION BENCHMARK RESULTS

### 2.1 Overall Validation Split (402 Frames)

| Evaluation Metric | Pretrained Baseline (`production_model.ckpt`) | Stage 1 Fine-Tuned (`best_stage1.ckpt`) | Absolute Delta | Relative Change | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Depth AbsRel** | **0.5953** | 4.3341 | +3.7388 | +628.1% (Regressed) | **Pretrained** |
| **Depth RMSE (m)** | **2.8929 m** | 9.7075 m | +6.8146 m | +235.6% (Regressed) | **Pretrained** |
| **Depth Median AE (m)** | **1.9076 m** | 8.4682 m | +6.5606 m | +343.9% (Regressed) | **Pretrained** |
| **Depth $\delta < 1.25$ Acc** | **0.0584** | 0.0458 | -0.0126 | -21.6% (Regressed) | **Pretrained** |
| **Surface Normal MAE** | **62.66°** | 69.68° | +7.02° | +11.2% (Regressed) | **Pretrained** |
| **Normal Median AE** | **64.63°** | 72.45° | +7.82° | +12.1% (Regressed) | **Pretrained** |
| **Normal Acc $< 30^\circ$** | **0.1495** | 0.1410 | -0.0085 | -5.7% (Regressed) | **Pretrained** |
| **Shape Chamfer (Sim3)** | **1.1704 m** | 1.3625 m | +0.1921 m | +16.4% (Regressed) | **Pretrained** |
| **Shape F-score @ 50mm** | **0.0078** | 0.0007 | -0.0071 | -91.0% (Regressed) | **Pretrained** |
| **Shape F-score @ 100mm** | **0.0440** | 0.0051 | -0.0389 | -88.4% (Regressed) | **Pretrained** |
| **Sim(3) Alignment Scale** | 4.0736 | **0.8863** | -3.1873 | Shifted scale | Stage 1 (Scale only) |
| **Metric Chamfer (Raw)** | **2.5130 m** | 9.3379 m | +6.8249 m | +271.6% (Regressed) | **Pretrained** |
| **Metric F-score @ 50mm** | **0.0126** | 0.0031 | -0.0095 | -75.4% (Regressed) | **Pretrained** |
| **Metric F-score @ 100mm** | **0.0250** | 0.0064 | -0.0186 | -74.4% (Regressed) | **Pretrained** |
| **Median Latency (ms)** | **279.1 ms** | 292.7 ms | +13.6 ms | +4.9% (Slower) | **Pretrained** |
| **Peak VRAM Allocated** | 3393.7 MB | **3392.7 MB** | -1.0 MB | Parity | Parity |

---

## 3. DOMAIN BREAKDOWN: SYNTHETIC VS. REAL SENSORS

### 3.1 Synthetic CAD Domain — Hypersim (`ai_016_001` - 100 Frames)
In the clean synthetic CAD domain where ray paths and textures are mathematically perfect:
- **Depth AbsRel:** Improved marginally from 0.7452 $\to$ **0.7172** (-3.76%).
- **Depth RMSE:** Decreased from 7.016 m $\to$ **6.195 m** (-11.71%).
- **Metric Chamfer:** Reduced from 5.757 m $\to$ **4.836 m** (-16.00%).
- **Shape Chamfer (Sim3):** Regressed slightly from **3.240 m** $\to$ 3.314 m (+2.28%).

### 3.2 Real Sensor Domain — ScanNet (`scene0030_00/01/02` - 302 Frames)
In the physical world with handheld motion blur, infrared structured-light dropouts, ambient lighting, and non-Lambertian surfaces:
- **Depth AbsRel:** Catastrophic regression from **0.5456** $\to$ **5.5317** (**+914.0% error increase**).
- **Depth RMSE:** Catastrophic regression from **1.528 m** $\to$ **10.871 m** (**+611.6% error increase**).
- **Metric Chamfer:** Catastrophic regression from **1.439 m** $\to$ **10.828 m** (**+652.6% error increase**).
- **Shape Chamfer (Sim3):** Degraded from **0.485 m** $\to$ **0.716 m** (+47.6% error increase).

---

## 4. ROOT-CAUSE FORENSIC ANALYSIS: WHY STAGE 1 FAILED

The empirical divergence between synthetic Hypersim and physical ScanNet points directly to the fundamental flaw identified in Phase 5:

1. **Pseudo-Label Contamination:**
   Stage 1 was trained using pseudo-ground-truth generated by Depth Anything V2 Indoor Small (`depth_anything_v2_vits.pth`). Depth Anything V2 is an affine-invariant monocular model whose depth predictions represent relative visual disparity rather than physical metric meters.
2. **Metric Head Warp:**
   During Stage 1 backpropagation, the loss function forced WorldMirror's DPT depth head to warp its scale and shift parameters to mimic Depth Anything V2's relative gradients.
3. **Disastrous Transfer to Physical IR Structured Light:**
   While the relative gradients slightly mimicked Hypersim's large open-hall CAD geometry (where rooms extend 15–25 meters), when applied to typical real-world residential rooms (ScanNet scene0030, where depth is 0.5–5.0 meters), the model predicted wildly inflated depth values ($> 15\text{ meters}$ for objects sitting 1.5 meters from the lens). This caused the 10-fold explosion in absolute relative error.

---

## 5. FORMAL PROMOTION GATE DECISION

Section 11 of the RoomAI Specification mandates:
1. Candidate model must demonstrate lower validation loss/error across all primary metrics.
2. Candidate model must not regress on real physical sensor data.
3. Candidate model must maintain or improve surface normal fidelity.

**Outcome:**
Stage 1 failed all three gates.
- Primary error worsened across 14 out of 15 metrics.
- Physical sensor error worsened by an order of magnitude (+914%).
- Surface normals degraded from 62.66° to 69.68°.

**Definitive Verdict:** `STAGE 1 DOES NOT GENERALIZE` — The original Pretrained Foundation Model (`production_model.ckpt`) is retained as RoomAI's active production candidate.

