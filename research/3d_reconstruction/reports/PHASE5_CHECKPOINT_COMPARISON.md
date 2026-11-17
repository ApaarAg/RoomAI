# ROOMAI — PHASE 5: HEAD-TO-HEAD CHECKPOINT COMPARISON

**Document ID:** `research/3d_reconstruction/reports/PHASE5_CHECKPOINT_COMPARISON.md`  
**Date:** 2026-11-08  
**Auditor:** Senior 3D Computer Vision Research Scientist & Independent Model Evaluation Lead  
**Scope:** Rigorous empirical evaluation comparing Pretrained Baseline, Stage 1 Fine-Tuned, and Stage 2 3DGS Fine-Tuned checkpoints under identical test conditions.

---

## 1. Evaluation Protocol

To ensure absolute scientific validity, all three models were evaluated under **100% identical test conditions**:
* **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Windows 11.
* **Input Resolution:** $518 \times 518$ RGB, standardized $[0, 1]$ floating-point tensor.
* **Test Scenes Evaluated:**
  1. `Bright_Room` Frame 0 (Validation set)
  2. `Bright_Room` Frame 1 (Validation set)
  3. `Office` Frame 0 (Validation set)
  4. `Sample_Room` Frame 0 (Held-out benchmark test set)
* **Metric Formulation:** Exact identical AbsRel, RMSE, Point-Map L2, Normal Angular Error, Latency, and VRAM memory profiling.

---

## 2. Checkpoint Provenance & Inventory

| Checkpoint Name | Local File Path | Parameters | File Size | SHA-256 | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Pretrained Baseline** | `checkpoints/production_candidate/production_model.ckpt` | 1,223,250,748 | 2,707.66 MB | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` | Base WorldMirror-1.0 architecture initialization |
| **Stage 1 Fine-Tuned** | `checkpoints/stage1/best_stage1.ckpt` | 1,223,250,748 | 2,707.64 MB | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` | 10-step geometric fine-tuning on real indoor room dataset |
| **Stage 2 3DGS Fine-Tuned**| `checkpoints/stage2/stage2_real_verified.ckpt` | 1,256,222,250 | 2,522.40 MB | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` | 5-step 3DGS radiance and opacity regularization training |

---

## 3. Aggregate Performance Comparison

| Metric | Pretrained Baseline | Stage 1 Fine-Tuned | Stage 2 Fine-Tuned | Stage 1 vs Baseline Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Point-Map Accuracy (m)** | **4.602** | **2.546** | **2.546** | **-44.7% (Substantial Improvement)** |
| **Mean Depth AbsRel** | **0.7562** | **1.0040** | **1.0040** | +32.8% (Skewed by `Office` outlier) |
| **Mean Depth RMSE (m)** | **3.550** | **4.120** | **4.120** | +16.1% (Skewed by `Office` outlier) |
| **Mean Normal MAE (deg)** | **60.71°** | **63.45°** | **63.45°** | +4.5% (Parity) |
| **Median Inference Latency (ms)**| **263.8** | **248.9** | **482.5** | Stage 1: -5.6%, Stage 2: +82.9% |
| **Peak VRAM Allocated (MB)** | **3,392.0** | **3,389.9** | **3,471.3** | Parity (~3.4 GB) |

---

## 4. Per-Scene Detailed Breakdown

### 4.1 Point-Cloud Accuracy (Euclidean Distance in Meters) — Lower is Better

| Scene | Split | Pretrained Baseline | Stage 1 Fine-Tuned | Stage 2 Fine-Tuned | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Bright_Room` Frame 0 | Val | 5.469 m | **3.303 m** | **3.303 m** | **-39.6% error** |
| `Bright_Room` Frame 1 | Val | 5.843 m | **2.681 m** | **2.681 m** | **-54.1% error** |
| `Office` Frame 0 | Val | 3.877 m | **2.892 m** | **2.892 m** | **-25.4% error** |
| `Sample_Room` Frame 0 | Test | 3.220 m | **1.308 m** | **1.308 m** | **-59.4% error** |
| **Average Across Scenes** | — | **4.602 m** | **2.546 m** | **2.546 m** | **-44.7% error** |

> **Key Finding:** Point-cloud accuracy improved substantially across **100% of evaluated scenes**. Fine-tuning on indoor room scenes successfully aligned WorldMirror's internal point map coordinate system closer to room camera dimensions.

---

### 4.2 Metric Depth AbsRel Error — Lower is Better

| Scene | Split | Pretrained Baseline | Stage 1 Fine-Tuned | Stage 2 Fine-Tuned | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Bright_Room` Frame 0 | Val | 0.7905 | **0.6967** | **0.6967** | **Improved (-11.9%)** |
| `Bright_Room` Frame 1 | Val | 0.8094 | **0.3091** | **0.3091** | **Massive Improvement (-61.8%)** |
| `Sample_Room` Frame 0 | Test | 0.6828 | **0.1999** | **0.1999** | **Exceptional Improvement (-70.7%)** |
| `Office` Frame 0 | Val | **0.7420** | 2.8103 | 2.8103 | Outlier scale divergence (3.63x scale) |

> **Key Finding:** On 3 out of 4 scenes (`Bright_Room` f0, `Bright_Room` f1, and `Sample_Room`), Stage 1 achieved major improvements in depth estimation accuracy, cutting relative depth error by up to 70.7%. However, on the single `Office` scene, the fine-tuned model overpredicted depth scale, illustrating that pseudo-label supervision without multi-scene scale anchoring is susceptible to scale drift on specific room layouts.

---

### 4.3 Metric Depth RMSE (Meters) — Lower is Better

| Scene | Split | Pretrained Baseline | Stage 1 Fine-Tuned | Stage 2 Fine-Tuned |
| :--- | :---: | :---: | :---: | :---: |
| `Bright_Room` Frame 0 | Val | 4.552 m | **3.179 m** | **3.179 m** |
| `Bright_Room` Frame 1 | Val | 4.815 m | **2.056 m** | **2.056 m** |
| `Sample_Room` Frame 0 | Test | 2.059 m | **0.627 m** | **0.627 m** |
| `Office` Frame 0 | Val | **2.773 m** | 10.616 m | 10.616 m |

---

## 5. Scientific Validation of Stage 1 and Stage 2

### 5.1 Stage 1 Status
1. **Did Stage 1 improve agreement with Depth Anything V2?**  
   **YES.** In 3 out of 4 scenes, depth error dropped significantly, and point-cloud distance was reduced from 4.60m to 2.55m across all scenes.
2. **Does this prove independent ground-truth 3D reconstruction quality?**  
   **NO.** The target labels were Depth Anything V2 pseudo-labels. Stage 1 demonstrates **high-efficiency pseudo-label distillation**, not physical ground truth validation.
3. **Guardrail Decision on `best_stage1.ckpt`:**  
   Because `Office` regressed in raw unscaled AbsRel by more than 5%, strict automated guardrails reject unconditional promotion of `best_stage1.ckpt` into production until median scale alignment or true metric training is applied.

### 5.2 Stage 2 Status
1. Stage 2 verified the mathematical and computational mechanics of 3D Gaussian Splatting (`enable_gs=True`), opacity regularization, and rasterization on local GPU hardware.
2. However, because the camera poses used during Stage 2 training were synthetic heuristic rotations ($R_y(0.05 \cdot v)$) rather than calibrated multi-view camera matrices, **Stage 2 cannot be claimed to have learned generalizable novel-view synthesis**.
3. It must remain classified as **EXPERIMENTAL 3DGS PROOF-OF-CONCEPT**.

---

## 6. Checkpoint Comparison Verdict

* **Pretrained Baseline:** Stable general-purpose baseline, but exhibits higher initial Euclidean offset to indoor camera frames.
* **Stage 1 Fine-Tuned:** Substantially improved indoor room point alignment (-44.7% point error) and sharp depth prediction on 3/4 scenes, but subject to unanchored scale drift on cubicle scenes (`Office`).
* **Stage 2 Fine-Tuned:** Preserves Stage 1 geometry, adds 3DGS splat heads, but doubles latency (482ms vs 249ms) with uncalibrated novel-view poses.

**Winner Selection:** For RoomAI production release, **Pretrained Baseline remains the primary production checkpoint**, while **Stage 1 is preserved as the distilled indoor geometric candidate**.

