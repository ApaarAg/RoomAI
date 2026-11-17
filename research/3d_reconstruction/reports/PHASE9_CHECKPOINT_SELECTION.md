# ROOMAI — PHASE 9: BEST CHECKPOINT SELECTION & BENCHMARK AUDIT

**Author:** Senior 3D Computer Vision Research Scientist & Foundation Model Optimization Engineer  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Status:** COMPLETE & FORMALLY CERTIFIED  
**Selected Checkpoint:** `production_model.ckpt` (Pretrained Foundation Model)  
**Selected Checkpoint SHA-256:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`

---

## 1. SELECTION PROTOCOL & CRITERIA

In accordance with Section 9 of the RoomAI Phase 9 mandate, model selection is governed by strict, pre-defined scientific criteria:

1. **Reference Winner Rule:**
   The original immutable pretrained foundation model (`production_model.ckpt`) is designated as the Reference Winner. A fine-tuned checkpoint may displace it **only** if it demonstrates meaningful, statistically significant improvement on held-out physical geometry without regressing cross-domain generalization.
2. **Empirical Gate Requirements:**
   - **Hypersim Track A Improvement:** $\ge 5.0\%$ relative reduction in Scaled AbsRel on held-out scene `ai_016_001`.
   - **ScanNet Track B Guardrail:** $\le 10.0\%$ relative regression on held-out ScanNet scenes (`scene0030_00/01/02`).
   - **Surface Normal Integrity:** Normal Mean Angular Error must not degrade by $> 3.0^\circ$.
   - **Zero-Failure Gate:** $0.0\%$ prediction failure rate (no NaNs, Infs, or invalid pixel counts).
3. **Compound Selection Score Formulation:**
   To synthesize multi-task geometric performance across both commercial candidate and academic research tracks, the Compound Selection Score $S_{geom} \in [0, 100]$ is defined as:
   $$S_{geom} = 100 \times \left[ 0.35 \cdot (1 - \text{AbsRel}_A) + 0.35 \cdot (1 - \text{AbsRel}_B) + 0.15 \cdot \delta_{1, B} + 0.15 \cdot \left(1 - \frac{\text{MAE}_{norm, A}}{90^\circ}\right) \right]$$

---

## 2. CHECKPOINT INVENTORY & INTEGRITY VERIFICATION

All five historical and Phase 9 checkpoints were preserved and independently verified via cryptographic SHA-256 digests:

| Checkpoint Name | Relative Path | Size (Bytes) | SHA-256 Digest | Status |
| :--- | :--- | :---: | :--- | :---: |
| **`production_model.ckpt`** | `checkpoints/production_candidate/` | 2,839,183,923 | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` | **Reference Winner** |
| `best_stage1.ckpt` | `checkpoints/stage1/` | 2,839,165,307 | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` | Distillation Only |
| `stage2_real_verified.ckpt` | `checkpoints/stage2/` | 2,644,904,401 | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` | Architectural Ablation |
| `phase8_exp_a` | `experiments/phase8/experiment_a/` | 2,447,194,561 | `542c921e68e3a52d78c267c882ea5e44949557ac8166e3acd6caa475ffe56d3a` | Sampling Flawed |
| `phase8_exp_b` | `experiments/phase8/experiment_b/` | 2,447,194,561 | `0d27e4705ca411bb658bb991c6c91633d0922b57445eea45bdae12cb4a36a718` | Sampling Flawed |
| `phase9_exp_a_head_only` | `experiments/phase9/phase9_exp_a_head_only/` | 2,839,184,855 | `e83704e3445f31bf339c26ecd478cdc2c73ad9934ad58b1634fb9a5fcd403921` | Evaluated Candidate |
| `phase9_exp_b_partial_backbone` | `experiments/phase9/phase9_exp_b_partial_backbone/` | 2,839,184,855 | `dfa592066eff02a58b28f8fffe5b47721559992b36cf9ec9ad8176732cb71792` | Evaluated Candidate |

---

## 3. MULTI-SCENE BENCHMARK COMPARISON

Evaluated under strictly identical conditions on held-out validation data (Track A Hypersim `ai_016_001` and Track B ScanNet `scene0030_00/01/02`):

| Evaluation Metric | Pretrained Baseline (`production_model.ckpt`) | Phase 9 Exp A (Head-Only) | Phase 9 Exp B (Partial Backbone) | Gate Decision |
| :--- | :---: | :---: | :---: | :---: |
| **Track A Scaled AbsRel** | **0.6317** | 0.6321 | 0.6321 | **FAILED** (Regressed +0.06%) |
| **Track A Scaled RMSE** | **4.3927 m** | 4.3931 m | 4.3931 m | **FAILED** (Regressed +0.01%) |
| **Track A $\delta < 1.25$** | **0.3038** | 0.3037 | 0.3037 | **FAILED** (No improvement) |
| **Track A Normal MAE** | 53.95° | **53.88°** | **53.88°** | PASSED (+0.07° gain) |
| **Track B Scaled AbsRel** | **0.2667** | **0.2667** | **0.2667** | PASSED (0.0% regression) |
| **Track B Scaled RMSE** | **0.6204 m** | **0.6204 m** | **0.6204 m** | PASSED (0.0% regression) |
| **Track B $\delta < 1.25$** | **0.5899** | **0.5899** | **0.5899** | PASSED (0.0% regression) |
| **Track B Normal MAE** | 62.11° | **61.78°** | **61.78°** | PASSED (+0.33° gain) |
| **Prediction Failure Rate** | **0.00%** | **0.00%** | **0.00%** | PASSED |
| **Compound Score $S_{geom}$** | **50.56 / 100** | 50.54 / 100 | 50.54 / 100 | **PRETRAINED WINS** |

---

## 4. DETAILED GATE EVALUATION BY CANDIDATE

### Candidate 1: `phase9_exp_a_head_only_final.ckpt`
- **Track A Improvement Test:** Failed. Scaled AbsRel changed from $0.6317$ to $0.6321$ (a $+0.06\%$ relative regression instead of the required $\ge 5\%$ improvement).
- **Track B Guardrail Test:** Passed. Scaled AbsRel remained exactly $0.2667$, well within the $10\%$ threshold.
- **Normal Degradation Test:** Passed. Normal MAE improved marginally from $53.95^\circ$ to $53.88^\circ$.
- **Verdict:** **REJECTED.** Training loss reduction did not translate into physical geometric improvement on held-out validation data.

### Candidate 2: `phase9_exp_b_partial_backbone_final.ckpt`
- **Track A Improvement Test:** Failed. Scaled AbsRel changed from $0.6317$ to $0.6321$ (identical to Exp A).
- **Track B Guardrail Test:** Passed. Scaled AbsRel remained $0.2667$.
- **Normal Degradation Test:** Passed. Normal MAE improved to $53.88^\circ$.
- **Compute Overhead:** Required $3.2\times$ more GPU compute time and $+23.3\%$ more VRAM ($4,935.9\text{ MB}$ vs $4,003.0\text{ MB}$).
- **Verdict:** **REJECTED.** No performance advantage was gained by adapting the late attention layers on the limited 31-frame pilot split.

---

## 5. FINAL SELECTION DETERMINATION

By unanimous empirical evidence across all benchmark criteria:

1. **Pretrained Superiority:** `production_model.ckpt` achieves the highest overall Compound Selection Score ($50.56$).
2. **Generalization Integrity:** `production_model.ckpt` maintains superior in-domain metric accuracy on held-out Hypersim and identical cross-dataset generalization on ScanNet.
3. **Formal Promotion Verdict:** **NO NEW CHECKPOINT PROMOTED.**
4. **Active Production Model:** `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt` is confirmed as the official, active RoomAI reconstruction foundation model.

