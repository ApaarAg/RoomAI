# ROOMAI MASTER TRAINING — PHASE 6: DOMAIN GENERALIZATION & CROSS-DATASET ANALYSIS

**Author:** Principal 3D Vision Research Scientist & Model Evaluation Lead  
**Date:** 2026-11-09  
**Project:** RoomAI (`F:\RoomAI`)  
**Phase:** Phase 06 (Cross-Dataset Generalization & Scene-Level Performance)  
**Status:** COMPLETE & EMPIRICALLY AUDITED  

---

## 1. CROSS-DATASET GENERALIZATION EVALUATION

To verify whether geometric adaptations transfer across physical environments, performance was audited across two completely distinct visual distributions:
1. **Track A (Hypersim):** Synthetic, physically based path-traced rendering with clean specular surfaces, complex indoor architectural geometry, high dynamic range, and exact CAD depth.
2. **Track B (ScanNet):** Real-world hand-held commodity RGB-D sensor observations with motion blur, sensor noise, dropped depth pixels, and physical clutter.

---

## 2. CROSS-DATASET DOMAIN PERFORMANCE MATRIX

| Model Checkpoint | Track A: Hypersim AbsRel | Track A: Point Sim(3) MAE | Track B: ScanNet AbsRel | Track B: Point Sim(3) MAE | Generalization Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Untouched Pretrained** | **0.5469** | **3.2401 m** | **0.2730** | **0.4633 m** | **Robust Foundation Baseline** |
| `exp_01_multidataset_head_only` | 0.5737 (+4.9%) | 4.2759 m (+32.0%) | 0.2818 (+3.2%) | 0.9707 m (+109.5%) | Degraded in both domains |
| `exp_02_multidataset_conservative` | 0.5934 (+8.5%) | 4.3028 m (+32.8%) | 0.2825 (+3.5%) | 1.0243 m (+121.1%) | Degraded in both domains |
| `exp_03_multidataset_partial_backbone`| 0.5829 (+6.6%) | 4.2704 m (+31.8%) | 0.2884 (+5.7%) | 0.9959 m (+115.0%) | Degraded in both domains |

---

## 3. SCENE-LEVEL HELD-OUT PERFORMANCE AUDIT

A breakdown of performance across each individual held-out validation scene illustrates that the regression is systematic, not an artifact of a single anomalous scene:

### 3.1 Scene `ai_016_001` (Track A Held-Out, 100 Frames)
- Pretrained Baseline AbsRel: **0.5469**
- Exp 01 AbsRel: **0.5737** (Degraded)
- Exp 02 AbsRel: **0.5934** (Degraded)
- Exp 03 AbsRel: **0.5829** (Degraded)
- *Finding:* In open-plan corridors and large rooms, fine-tuned models over-predict near distances, compressing the dynamic depth range.

### 3.2 Scene `scene0030_00` (Track B Held-Out, 124 Frames)
- Pretrained Baseline AbsRel: **0.2707**
- Exp 01 AbsRel: **0.2798** (Degraded)
- Exp 02 AbsRel: **0.2811** (Degraded)
- Exp 03 AbsRel: **0.2854** (Degraded)

### 3.3 Scene `scene0030_01` (Track B Held-Out, 82 Frames)
- Pretrained Baseline AbsRel: **0.2860**
- Exp 01 AbsRel: **0.2941** (Degraded)
- Exp 02 AbsRel: **0.2952** (Degraded)
- Exp 03 AbsRel: **0.3012** (Degraded)

### 3.4 Scene `scene0030_02` (Track B Held-Out, 96 Frames)
- Pretrained Baseline AbsRel: **0.2622**
- Exp 01 AbsRel: **0.2715** (Degraded)
- Exp 02 AbsRel: **0.2712** (Degraded)
- Exp 03 AbsRel: **0.2786** (Degraded)

---

## 4. SCIENTIFIC ROOT CAUSE OF GENERALIZATION COLLAPSE

1. **Pretrained Parameter Stiffness:**
   A 1.22B parameter model pretrained with self-supervised and multiview consistency objectives encodes rich geometric priors in high-dimensional manifolds. Updating 314M parameters with AdamW on 204 training frames applies updates that lie primarily in the subspace of the training scenes' specific camera geometry and lighting, moving the weights off the optimal foundation manifold.
2. **The Point Head Distortion:**
   The point prediction head is an MLP projection that decodes spatial 3D rays. While the normal head benefits somewhat from local planar regularization (improving normal MAE on ScanNet from 62.9° to 50.2°), the point head's metric scale calibration is disrupted, doubling Umeyama alignment error.
3. **Minimum Scale Threshold for Foundation Adaptation:**
   In foundation vision models (such as DINOv2, Depth Anything, and WorldMirror), successful fine-tuning without catastrophic forgetting typically requires:
   - At least 50 to 100 diverse physical scenes.
   - At least 5,000 to 10,000 frames.
   - Feature distillation or parameter-efficient fine-tuning (e.g. LoRA with rank $r \le 16$, or strict feature-regularization loss $\mathcal{L}_{\text{reg}} = \|\mathbf{f}_{\text{tuned}} - \mathbf{f}_{\text{frozen}}\|_2$).
   - Full-head gradient unfreezing on 204 frames causes immediate representation drift.

---

## 5. CONCLUSION & DECISION FOR PRODUCTION
The cross-dataset generalization audit confirms that the original pretrained checkpoint `production_model.ckpt` maintains strictly superior generalization across both synthetic and real-world indoor domains. None of the fine-tuned variants qualify for production promotion.

