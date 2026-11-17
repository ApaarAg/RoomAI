# RoomAI — Phase 3: Canonical Pretrained Baseline Certification Report

**Author:** Principal CV Research Scientist / Foundation Model Optimization Engineer  
**Date:** 2026-11-10 (IST 11:18)  
**Status:** COMPLETE & FORMALLY CERTIFIED  
**Evaluated Checkpoint:** `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt`  
**Checkpoint SHA-256:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`  
**Parameter Count:** 1,223,250,748 parameters (WorldMirror-1.0)  
**Execution Environment:** Windows 11, CUDA 12.8, PyTorch 2.11, RTX 4060 Laptop (8GB VRAM)  
**Structured Output:** `research/3d_reconstruction/experiments/phase9/pretrained_baseline_evaluation.json`

---

## 1. Executive Summary

The untouched official WorldMirror-1.0 foundation model (`production_model.ckpt`) was evaluated strictly on the certified held-out validation sets across both Track A (Hypersim - Commercial) and Track B (ScanNet - Non-commercial Research):
- **Zero Failures:** 175/175 frames successfully evaluated (Failure Rate: **0.0%**).
- **Peak VRAM:** **3,379.8 MB** (well within the 8 GB GPU hardware budget).
- **Average Frame Latency:** **462.3 ms** per multi-task inference pass.

This evaluation confirms the canonical baseline benchmarks and serves as the immutable reference point for all subsequent fine-tuning experiments.

---

## 2. Certified Canonical Baseline Metrics

| Metric | Track A (Hypersim Held-Out: `ai_016_001`) | Track B (ScanNet Held-Out: `scene0030`) |
| :--- | :---: | :---: |
| **Total Frames Evaluated** | 100 / 100 | 75 / 75 |
| **Prediction Failure Rate** | **0.0%** | **0.0%** |
| **Raw Metric Depth AbsRel** | 0.7452 ± 0.1257 | 0.5556 ± 0.0559 |
| **Raw Metric Depth RMSE** | 7.0159 m | 1.5285 m |
| **Raw Metric Depth $\delta < 1.25$** | 0.0388 | 0.2140 |
| **Scaled Depth AbsRel** | **0.5469 ± 0.3177** | **0.2730 ± 0.1004** |
| **Scaled Depth RMSE** | 4.6727 m | 0.6618 m |
| **Scaled Depth $\delta < 1.25$** | **0.2926** | **0.5959** |
| **Point Map Raw MAE** | 5.7572 m | 1.3412 m |
| **Point Map Sim(3) MAE** | **3.2401 m** | **0.4633 m** |
| **Surface Normal Mean AE** | **61.53°** | **62.94°** |
| **Surface Normal Median AE** | 67.60° | 63.85° |
| **Normals $< 30^\circ$ Accuracy** | 26.4% | 24.1% |
| **Inference Latency** | 462.3 ms | 462.3 ms |
| **Peak VRAM** | 3,379.8 MB | 3,379.8 MB |

---

## 3. Strict Fine-Tuning Acceptance Criteria

In compliance with Part 11 of the Master Prompt:
1. **Primary Geometric Gate:** Any fine-tuned Stage 1 candidate must improve at least two primary geometric metrics by **$\ge 3\%$ relative** to the baseline:
   - Hypersim Scaled AbsRel threshold: $\le 0.5469 \times 0.97 = \mathbf{0.5305}$
   - Hypersim Scaled Delta1 threshold: $\ge 0.2926 \times 1.03 = \mathbf{0.3014}$
   - Point Cloud Sim(3) MAE threshold: $\le 3.2401 \times 0.97 = \mathbf{3.1429\text{ m}}$
2. **Generalization & Catastrophic Forgetting Gate:**
   - ScanNet (Track B) Scaled AbsRel must not regress by more than $5\%$ ($\le 0.2867$).
3. **Hardware Gate:**
   - Peak VRAM must remain $\le 7,500\text{ MB}$.
   - No NaNs or Infs during validation inference.

If any candidate fails these criteria, the original pretrained model `production_model.ckpt` will be maintained as the production winner.

