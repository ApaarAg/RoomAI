# Phase 8: Checkpoint Comparison and Selection Report

**Author:** Senior 3D Computer Vision Research Scientist & Model Evaluation Lead  
**Repository:** `F:\RoomAI`  
**Evaluation Standard:** Held-Out Validation Split Only (402 frames: 100 Hypersim, 302 ScanNet)  
**Untouched Test Set:** Strictly Quarantined (484 frames: 100 Hypersim, 384 ScanNet)  
**Date:** 2026-11-08  

---

## 1. Preserved Checkpoint Inventory & Provenance

All historical and Phase 8 checkpoints are recorded with exact cryptographic SHA-256 hashes and file sizes:

| Checkpoint Name | Origin / Phase | Path | Size (MB) | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`production_model.ckpt`** | Pretrained Foundation | `checkpoints/production_candidate/` | $2,707.66$ | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` | **IMMUTABLE REFERENCE** |
| **`best_stage1.ckpt`** | Phase 4/5 (Unnormalized) | `checkpoints/stage1/` | $2,707.64$ | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` | **REJECTED (Domain Regression)** |
| **`stage2_real_verified.ckpt`** | Phase 4/5 (3DGS) | `checkpoints/stage2/` | $2,522.40$ | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` | **PRESERVED (Multi-view only)** |
| **`experiment_a_hypersim_final.ckpt`** | Phase 8 (Canonical Exp A) | `experiments/phase8_corrected/experiment_a_hypersim/` | $2,707.66$ | `542c921e68e3a52d78c267c882ea5e44949557ac8166e3acd6caa475ffe56d3a` | **VALIDATED EXPERIMENT** |
| **`experiment_b_multiview_final.ckpt`** | Phase 8 (Canonical Exp B) | `experiments/phase8_corrected/experiment_b_multiview/` | $2,707.66$ | `0d27e4705ca411bb658bb991c6c91633d0922b57445eea45bdae12cb4a36a718` | **VALIDATED EXPERIMENT** |

---

## 2. Guardrail Audits & Decision Criteria

To ensure that only genuinely superior models are deployed, every candidate was subjected to 4 mandatory decision gates:

| Decision Gate | Evaluation Criteria | Old Stage 1 | Phase 8 Exp A | Phase 8 Exp B | Pretrained Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gate 1: Metric Finiteness** | $100\%$ finite evaluation metrics (0 NaN, 0 Inf) | PASSED | PASSED | PASSED | **PASSED** |
| **Gate 2: Target Normalization** | Canonical scene normalization ($s = \operatorname{nanmedian}(\|\mathbf{X}_w\|_2)$) | FAILED | PASSED | PASSED | **PASSED** |
| **Gate 3: Domain Regression** | ScanNet Scaled AbsRel must not regress $>10\%$ | **FAILED ($+48.5\%$)** | **PASSED ($0.0\%$)** | **PASSED ($0.0\%$)** | **REFERENCE ($0.2667$)** |
| **Gate 4: Validation Superiority** | Must demonstrate statistically significant gain over baseline | FAILED | FAILED ($+0.0003$) | FAILED ($+0.0003$) | **WINNER (Unbeaten)** |

---

## 3. Quantitative Validation Comparison

Performance evaluated on identical held-out validation scenes (`ai_016_001` for synthetic Hypersim, `scene0030` for sensor ScanNet):

| Metric | Pretrained Reference (`production_model.ckpt`) | Old Stage 1 (`best_stage1.ckpt`) | Phase 8 Exp A (`experiment_a`) | Phase 8 Exp B (`experiment_b`) | Best Performer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hypersim Val AbsRel** | **$0.6317$** | $0.6311$ | $0.6320$ | $0.6320$ | **Pretrained** |
| **Hypersim Val RMSE** | **$4.3927\text{ m}$** | $4.4501\text{ m}$ | $4.3930\text{ m}$ | $4.3930\text{ m}$ | **Pretrained** |
| **ScanNet Val AbsRel** | **$0.2667$** | $0.3691$ (Severe Regression) | $0.2666$ | $0.2666$ | **Pretrained / Parity** |
| **ScanNet Val RMSE** | **$0.6204\text{ m}$** | $0.8654\text{ m}$ | $0.6203\text{ m}$ | $0.6203\text{ m}$ | **Pretrained / Parity** |
| **ScanNet $\delta < 1.25$** | **$0.5899$** | $0.4812$ | $0.5899$ | $0.5899$ | **Pretrained / Parity** |

---

## 4. Final Checkpoint Selection Verdict

1. **Rejection of Fine-Tuned Candidates:** While Phase 8 successfully eliminated the defects of Stage 1 (zero domain regression, gradient stability, canonical scaling), fine-tuning only the prediction heads on 31 to 204 pilot frames did **not** produce a statistically measurable improvement over the original pretrained weights.
2. **Promotion Decision:** In accordance with project policy ("Your objective is to produce a genuinely improved reconstruction model, not a lower training loss. If fine-tuned models do not demonstrably beat the original pretrained baseline on the validation split, the original pretrained baseline remains the winning model"):
   **The winning production checkpoint remains `production_model.ckpt`.**
3. **No Overwrites:** Neither `production_model.ckpt` nor any historical checkpoint has been modified or overwritten.

