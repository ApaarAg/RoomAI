# ROOMAI MASTER TRAINING — PHASE 3: CANONICAL BASELINE BENCHMARK REPORT

**Author:** Principal 3D Vision Research Scientist & Evaluation Lead  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Phase:** Phase 03 (Untouched Pretrained WorldMirror-1.0 Baseline)  
**Status:** CANONICAL REFERENCE CERTIFIED  
**Evaluated Checkpoint:** `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt`  
**Checkpoint SHA-256:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`  
**JSON Results Archive:** [`results/master_training/baseline/baseline_evaluation.json`](file:///F:/RoomAI/results/master_training/baseline/baseline_evaluation.json)  

---

## 1. EXECUTIVE SUMMARY & BENCHMARK CERTIFICATION

The original, untouched pretrained WorldMirror-1.0 model (`production_model.ckpt`) was evaluated across all held-out validation sequences under the Canonical RoomAI Validation Protocol v1.0. This evaluation serves as the immutable reference against which every candidate fine-tuned checkpoint will be judged.

### System & Hardware Profile
- **Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM)
- **Input Resolution:** $518 \times 518$ (Bicubic interpolation, normalized $[0, 1]$)
- **Peak VRAM Allocated:** **3,379.75 MB** (41.3% of 8 GB VRAM capacity)
- **Mean Inference Latency:** **376.8 ms / frame** (2.65 FPS)
- **Prediction Failure Rate:** **0.00%** (175 / 175 frames successfully predicted with finite values)

---

## 2. CANONICAL BENCHMARK METRICS SUMMARY TABLE

| Metric Category | Specific Metric | Track A: Hypersim Held-Out (`ai_016_001`, 100 Frames) | Track B: ScanNet Held-Out (3 Scenes, 75 Frames) | Primary / Secondary Gate |
| :--- | :--- | :---: | :---: | :---: |
| **Depth Estimation** | **Scaled AbsRel** | **0.5469 ± 0.318** (Median: 0.4655) | **0.2730 ± 0.100** (Median: 0.2770) | **PRIMARY GATE (Goal: ≥ 3% rel. reduction)** |
| | **Scaled RMSE** | **4.6727 m** (Median: 4.6813 m) | **0.6618 m** (Median: 0.6381 m) | Secondary |
| | **Scaled $\delta < 1.25$ Acc** | **29.26%** (Median: 24.58%) | **59.59%** (Median: 57.24%) | Secondary |
| | **Scaled $\delta < 1.25^2$ Acc**| **53.07%** (Median: 52.43%) | **80.68%** (Median: 81.95%) | Secondary |
| | Raw AbsRel | 0.7452 ± 0.126 | 0.5556 ± 0.056 | Informational |
| | Raw RMSE | 7.0159 m | 1.5285 m | Informational |
| **Point Geometry** | **Sim(3) Umeyama L1 MAE** | **3.2401 m** (Median: 3.3076 m) | **0.4633 m** (Median: 0.4655 m) | **PRIMARY GATE (Goal: ≥ 3% rel. reduction)** |
| | Sim(3) RMSE | 4.0093 m | 0.5349 m | Secondary |
| | Raw Point L1 MAE | 5.7572 m | 1.4612 m | Informational |
| **Surface Normals** | **Mean Angular Error** | **61.53°** (Median: 59.54°) | **62.94°** (Median: 62.58°) | Secondary (Goal: No > 1% rel. regression) |
| | Median Angular Error | 67.60° | 63.41° | Secondary |
| | % Normals < 11.25° | 7.28% | 1.43% | Secondary |
| | % Normals < 22.50° | 20.25% | 6.19% | Secondary |
| | % Normals < 30.00° | 26.45% | 11.62% | Secondary |
| **Runtime Profile** | Mean Latency | 377.8 ms | 375.9 ms | SLA Gate (≤ 450 ms) |
| | Peak GPU VRAM | 3,379.8 MB | 3,379.8 MB | Hardware Gate (< 7,000 MB) |
| | Failure Rate | 0.00% | 0.00% | Critical Reliability Gate (0.0%) |

---

## 3. PER-SCENE BREAKDOWN (TRACK B: SCANNET)

| Scene Identifier | Evaluated Frames | Scaled AbsRel | Scaled RMSE | Scaled $\delta < 1.25$ | Point Sim(3) MAE | Normal MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `scene0030_00` | 25 frames | **0.2707 ± 0.098** | 0.6481 m | 60.12% | 0.4512 m | 62.15° |
| `scene0030_01` | 25 frames | **0.2860 ± 0.115** | 0.6912 m | 57.48% | 0.4820 m | 63.82° |
| `scene0030_02` | 25 frames | **0.2622 ± 0.084** | 0.6462 m | 61.18% | 0.4568 m | 62.84° |
| **Track B Stratified Aggregate** | **75 frames** | **0.2730 ± 0.100** | **0.6618 m** | **59.59%** | **0.4633 m** | **62.94°** |

---

## 4. PREDEFINED SUCCESS GATES FOR CANDIDATE MODELS

To achieve promotion over this canonical baseline, a candidate fine-tuned checkpoint must satisfy all criteria defined in Phase 7:
1. **Primary Geometric Improvement:**
   - Track A (Hypersim): Scaled AbsRel $\le 0.5469 \times (1 - 0.03) = \mathbf{0.5305}$ AND/OR Point Sim(3) MAE $\le 3.2401 \times (1 - 0.03) = \mathbf{3.1429\text{ m}}$.
   - Track B (ScanNet): Scaled AbsRel $\le 0.2730 \times (1 - 0.03) = \mathbf{0.2648}$ AND/OR Point Sim(3) MAE $\le 0.4633 \times (1 - 0.03) = \mathbf{0.4494\text{ m}}$.
2. **Regression Constraint:**
   - No other critical metric may regress by more than $1.0\%$ relative.
3. **Cross-Scene Consistency:**
   - Demonstrated improvement on at least 2 distinct validation scenes.
4. **Reliability & Latency:**
   - 0.0% failure rate; latency $\le 450\text{ ms}$; VRAM $\le 7,000\text{ MB}$.

