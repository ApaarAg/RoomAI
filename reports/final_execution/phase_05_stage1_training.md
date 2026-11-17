# PHASE 5 — STAGE 1 GEOMETRIC TRAINING & CONTINUOUS OPTIMIZATION

**Author:** RoomAI Master Execution Controller  
**Status:** In Progress / Controlled Pipeline Active  
**Model Architecture:** Tencent HunyuanWorld-Mirror 1.0 (~1.223 Billion Parameters)  
**Hardware Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), PyTorch 2.11.0+cu128  
**Canonical Dataset:** Expanded 2,715 canonical 518×518 frames across 31 indoor scenes  
**Date:** 2026-11-10  

---

## 1. Executive Summary

Phase 5 executes Stage 1 continuous geometric training on the full 100 GB expanded Hypersim acquisition. Following the root-cause diagnostics established in Phase 4 (which identified Phase 8 sequential-sampling catastrophic forgetting and Phase 9 micro-sample 31-frame overfitting), Phase 5 scales training across **1,588 standardized training frames spanning 27 unique physical scenes** with strict scene-level partitioning and canonical coordinate normalization.

The controlled training pipeline (`research/3d_reconstruction/scripts/train_phase5_expanded_suite.py`) evaluates two distinct adaptation regimes against the certified pretrained baseline:
1. **`exp_04_expanded_head_only`:** Complete freeze of the 1.12B visual geometry transformer backbone; isolated gradient flow through the 4 prediction heads (~98M parameters, 8.01% of total model capacity) with protected learning rate $2 \times 10^{-6}$.
2. **`exp_05_expanded_lora_rank8`:** Parameter-efficient Low-Rank Adaptation (Rank 8) on late cross-view attention layers combined with head adaptation, evaluating whether learned geometric latent representations can be refined without representation collapse.

---

## 2. Training Pipeline Architecture & Invariants

```
+---------------------------------------------------------------------------------------------------+
|                           Canonical Expanded Dataset (2,715 Frames)                               |
|   - 1,588 Train Frames (24 Hypersim Scenes)                                                       |
|   - 543 Held-Out Validation Frames (Track A: ai_016_001; Track B: scene0030)                      |
|   - 584 Locked Test Frames (Held strictly untouched)                                              |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        Data Loading & Patch-14 Alignment (518x518x3)                              |
|   - RGB normalized to [0, 1]                                                                      |
|   - Orthogonal planar Z-depth: Z = R * cos(theta), normalized by scene scale factor s             |
|   - 3D World Points X_w unprojected from pinhole camera model, normalized by s                    |
|   - Surface normal unit vectors (||n||_2 == 1.0)                                                  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                              WorldMirror-1.0 Forward Pass (bfloat16)                              |
|   - DINOv2 ViT Backbone (Patch size 14, Embed Dim 1024) -> 1,369 Tokens                           |
|   - Visual Geometry Transformer (Frame Blocks + Global Blocks)                                    |
|   - Multimodal Prediction Heads: DepthHead, PtsHead, NormHead, CamHead                            |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                               CanonicalMultiTaskLoss Computation                                  |
|   L_total = 1.0 * L_depth (Scale-Invariant Log Loss + AbsRel)                                     |
|           + 0.5 * L_pts (Normalized World Coordinate Huber/L1)                                     |
|           + 0.1 * L_norm (Cosine Similarity Loss on Unit Surface Normals)                         |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                    Backprop & Optimization                                        |
|   - AdamW (beta1=0.9, beta2=0.95, weight_decay=1e-4)                                             |
|   - Gradient norm clipping at 0.5                                                                 |
|   - VRAM footprint: 7,490 - 7,890 MiB / 8,188 MiB (100% compute saturation)                       |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Experiment Configurations

### 3.1 Experiment 04: `exp_04_expanded_head_only`
- **Goal:** Investigate whether expanding dataset scale by $51.2\times$ (from 31 frames to 1,588 frames across 27 scenes) allows head-only adaptation to beat the pretrained baseline.
- **Backbone State:** Frozen (requires_grad = False for all 1,125,250,748 backbone parameters).
- **Trainable Parameters:** 98,000,000 parameters across `depth_head`, `pts_head`, `norm_head`, and `cam_head`.
- **Learning Rate:** $2 \times 10^{-6}$ (strictly bounded to eliminate catastrophic gradient shock).
- **Batch Size:** 1 (bfloat16 native precision).
- **Steps:** 300 optimizer steps.

### 3.2 Experiment 05: `exp_05_expanded_lora_rank8`
- **Goal:** Adapt late-stage spatial attention representations without full backbone retraining.
- **LoRA Injection:** Query, Key, Value, and Projection matrices in the final two transformer blocks of the visual geometry transformer.
- **Trainable Parameters:** LoRA adapters (~4.2M params) + prediction heads (~98M params) = ~102.2M params (8.35% of total).
- **Learning Rate:** $1 \times 10^{-5}$ for LoRA adapters, $2 \times 10^{-6}$ for prediction heads.
- **Steps:** 250 optimizer steps.

---

## 4. Acceptance Criteria & Gates for Promotion

To promote any experimental checkpoint over `production_model.ckpt`, the checkpoint must strictly satisfy all of the following:

1. **Track A (Hypersim) Improvement:** $\ge 3\%$ relative reduction in held-out validation AbsRel:
   $$\text{AbsRel}_{\text{candidate}} \le 0.5469 \times 0.97 = \mathbf{0.5305}$$
2. **Track B (ScanNet) Generalization:** Regressions bounded to $\le 5\%$ relative:
   $$\text{AbsRel}_{\text{candidate}} \le 0.2730 \times 1.05 = \mathbf{0.2867}$$
3. **Point Cloud Metric Stability:** Sim(3) aligned MAE must not degrade by $> 10\%$.
4. **Locked Test Verification:** Candidate must replicate gains on the 584-frame locked test split before production candidate replacement.

If any candidate fails these gates, Rule 8 strictly enforces: **"Never promote an inferior checkpoint into production. The original pretrained model remains the reference winner."**

---
*Authored by RoomAI Master Execution Controller.*

