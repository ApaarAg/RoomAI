# ROOMAI MASTER TRAINING — PHASE 5: EXPERIMENT RESULTS & COMPARATIVE AUDIT

**Author:** Principal 3D Vision Research Scientist & Foundation Model Fine-Tuning Lead  
**Date:** 2026-11-09  
**Project:** RoomAI (`F:\RoomAI`)  
**Phase:** Phase 05 (Controlled Multi-Dataset Experiment Program)  
**Status:** COMPLETE & INDEPENDENTLY EVALUATED  

---

## 1. EXECUTIVE SUMMARY & EXPERIMENT PROGRAM OVERVIEW

Under the Autonomous Experiment Controller ([`master_experiment_controller.py`](file:///F:/RoomAI/research/3d_reconstruction/scripts/master_experiment_controller.py)), a structured series of multi-dataset indoor fine-tuning experiments was executed on the RTX 4060 GPU. Training utilized the unified 204-frame multi-dataset training split across 6 physical scenes (3 Hypersim scenes + 3 ScanNet sequences), completely eliminating the previous single-dataset sampling constraint.

### Certified Experiment Registry Summary (`experiment_registry.csv`):
| Experiment ID | Adaptation Mode | Dataset Scope | LR (Heads / Backbone) | Epochs / Steps | Checkpoint SHA-256 | Track A AbsRel ($\Delta$) | Track B AbsRel ($\Delta$) | Gate Passed |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Untouched Pretrained** | Baseline | Pretrained Weights | — | — | `dbc9c9b0...` | **0.5469** (Ref) | **0.2730** (Ref) | **WINNER** |
| **`exp_01_multidataset_head_only`** | Head-Only | Multi-Dataset (204) | $5 \times 10^{-5}$ / $0.0$ | 2 / 408 | `da0cc6b3...` | 0.5737 (+4.91%) | 0.2818 (+3.24%) | **FAIL** |
| **`exp_02_multidataset_conservative`**| Head-Only | Multi-Dataset (204) | $2 \times 10^{-5}$ / $0.0$ | 2 / 408 | `b81289bc...` | 0.5934 (+8.50%) | 0.2825 (+3.49%) | **FAIL** |
| **`exp_03_multidataset_partial_backbone`**| Partial ViT | Multi-Dataset (204) | $3 \times 10^{-5}$ / $5 \times 10^{-7}$ | 2 / 408 | `30560852...` | 0.5829 (+6.60%) | 0.2884 (+5.67%) | **FAIL** |

---

## 2. DETAILED METRIC COMPARISON TABLE ACROSS ALL EXPERIMENTS

Evaluated under the Canonical RoomAI Validation Protocol v1.0 across held-out Track A (Hypersim `ai_016_001`, 100 frames) and Track B (ScanNet `scene0030`, 302 frames):

### 2.1 Track A: Synthetic Indoor Held-Out (Hypersim `ai_016_001`, 100 Frames)
| Metric Dimension | Pretrained Baseline | Exp 01 (Head-Only, 5e-5) | Exp 02 (Head-Only, 2e-5) | Exp 03 (Partial Backbone) | Best Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Scaled AbsRel** | **0.5469 ± 0.318** | 0.5737 (+4.91%) | 0.5934 (+8.50%) | 0.5829 (+6.60%) | **Pretrained Baseline** |
| **Scaled RMSE (m)** | **4.6727 m** | 4.7471 m (+1.59%) | 4.7676 m (+2.03%) | 4.7635 m (+1.94%) | **Pretrained Baseline** |
| **Scaled $\delta < 1.25$ Acc** | **29.26%** | 28.35% (-0.91%) | 27.60% (-1.66%) | 27.94% (-1.32%) | **Pretrained Baseline** |
| **Point Cloud Sim(3) MAE** | **3.2401 m** | 4.2759 m (+31.97%) | 4.3028 m (+32.80%) | 4.2704 m (+31.80%)| **Pretrained Baseline** |
| **Surface Normal MAE** | **61.53°** | 70.91° (+15.24%) | 72.28° (+17.47%) | 73.27° (+19.08%) | **Pretrained Baseline** |

### 2.2 Track B: Real-World Indoor Held-Out (ScanNet `scene0030`, 302 Frames)
| Metric Dimension | Pretrained Baseline | Exp 01 (Head-Only, 5e-5) | Exp 02 (Head-Only, 2e-5) | Exp 03 (Partial Backbone) | Best Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Scaled AbsRel** | **0.2730 ± 0.100** | 0.2818 (+3.24%) | 0.2825 (+3.49%) | 0.2884 (+5.67%) | **Pretrained Baseline** |
| **Scaled RMSE (m)** | **0.6618 m** | 0.7042 m (+6.40%) | 0.6977 m (+5.42%) | 0.7154 m (+8.09%) | **Pretrained Baseline** |
| **Scaled $\delta < 1.25$ Acc** | **59.59%** | 54.07% (-5.52%) | 55.38% (-4.21%) | 53.76% (-5.83%) | **Pretrained Baseline** |
| **Point Cloud Sim(3) MAE** | **0.4633 m** | 0.9707 m (+109.5%) | 1.0243 m (+121.1%) | 0.9959 m (+115.0%)| **Pretrained Baseline** |
| **Surface Normal MAE** | 62.94° | **50.25° (-20.16%)** | **50.87° (-19.18%)** | **51.50° (-18.17%)** | Exp 01 |

---

## 3. DEEP SCIENTIFIC DIAGNOSIS & FAILURE ANALYSIS

### 3.1 The Representation Distortion Phenomenon
1. **Training Loss vs Validation Reality:**
   - In all three experiments, training loss converged smoothly (from initial ~0.55 down to ~0.27 on ScanNet and ~0.45 on Hypersim).
   - However, held-out validation error consistently increased across all models (+3.2% to +8.5% in AbsRel; +31% to +121% in 3D Point MAE).
2. **Mechanism of Regression:**
   - WorldMirror-1.0 is a 1.22-billion-parameter foundation model pretrained on massive Web-scale multiview datasets.
   - The available training split consists of 204 frames across only 6 scenes.
   - Optimizing 314M parameters (prediction heads) on only 6 camera trajectories induces immediate representation collapse: the prediction heads overfit to the specific camera heights, field of views, and depth distributions of those 6 training scenes, degrading their zero-shot physical depth triangulation on unseen geometries.
3. **The Point Cloud Disruption:**
   - Point Cloud Sim(3) error doubled on ScanNet (from 0.46m to 0.97m+).
   - This occurs because the 3D point prediction head (`pts_head`) is tightly coupled to the internal vision transformer features. Adjusting the head weights without hundreds of thousands of diverse training pairs breaks the fine coordinate calibration established during large-scale foundation pretraining.
4. **Surface Normal Trade-off:**
   - Surface normals on ScanNet improved from 62.94° down to 50.25° (a 20% improvement).
   - However, under Phase 7 Non-Negotiable Rule #1 (*"Avoid more than 1% relative regression on any other predefined critical metric"*), a 20% gain in normals cannot justify a 109% regression in point geometry and a 3.2% regression in metric depth.

---

## 4. PHASE 5 CONCLUSION
All three fine-tuned models demonstrably degrade 3D geometric reconstruction relative to the untouched pretrained model.
In strict accordance with the scientific method and Non-Negotiable Rule #9 (*"Never report a checkpoint as production-ready merely because it loads"*), **all three fine-tuned checkpoints are REJECTED for production promotion**.

The untouched pretrained checkpoint `production_model.ckpt` (`dbc9c9b0...`) remains the undisputed reference winner.

