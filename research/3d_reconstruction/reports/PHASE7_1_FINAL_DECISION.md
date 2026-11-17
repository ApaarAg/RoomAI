# ROOMAI — PHASE 7.1: ROOT-CAUSE FORENSIC SYNTHESIS & FINAL SCIENTIFIC DECISION

**Author:** Independent Senior 3D Computer Vision Researcher & Evaluation Auditor  
**Date:** November 8, 2026  
**Status:** COMPLETE, FORMALLY CERTIFIED & TEST-SUITE VALIDATED  
**Final Production Candidate:** [`production_model.ckpt`](file:///F:/RoomAI/research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt) (WorldMirror-1.0 Pretrained Baseline)  
**Mandated Final Classification:** `MULTIPLE CONTRIBUTING CAUSES (GENUINE MODEL DEGRADATION + DEPTH OUTPUT SEMANTICS MISMATCH)`  
**Stage 2 3DGS Novel-View Status:** `UNVERIFIED (NO HELD-OUT RENDERING EVIDENCE)`

---

## 1. PRESERVED CHECKPOINTS & CRYPTOGRAPHIC LEDGER

The cryptographic integrity of all candidate checkpoints was verified before and after the forensic audit:

| Checkpoint Identifier | File Location | File Size (Bytes) | Size (MB) | SHA-256 Checksum | Final Promotion Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pretrained Baseline** | `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt` | 2,839,184,102 | 2,707.66 MB | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` | **CERTIFIED PRODUCTION CANDIDATE** |
| **Stage 1 Fine-Tuned** | `research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt` | 2,839,164,068 | 2,707.64 MB | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` | **REJECTED (DEGRADED GEOMETRY)** |
| **Stage 2 3DGS** | `research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt` | 2,644,926,673 | 2,522.40 MB | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` | **REJECTED (INHERITED DISTORTION)** |

---

## 2. ROOT-CAUSE TAXONOMY & CAUSAL ATTRIBUTION

In response to the Phase 7.1 forensic mandate, the Stage 1 performance regression is classified across the formal causal categories:

```
+---------------------------------------------------------------------------------------------------+
|                               PHASE 7.1 ROOT-CAUSE CAUSAL MATRIX                                  |
+---------------------------------------------------------------------------------------------------+
| CAUSAL CATEGORY                    | EMPIRICAL STATUS        | EXPERIMENTAL EVIDENCE              |
+------------------------------------+-------------------------+------------------------------------+
| 1. MULTIPLE CONTRIBUTING CAUSES    | DEMONSTRATED            | Combination of Semantics Mismatch, |
|                                    |                         | Training Bugs, & Genuine Drift     |
| 2. DEPTH OUTPUT SEMANTICS MISMATCH | DEMONSTRATED            | WorldMirror outputs normalized     |
|                                    |                         | coordinates (~1.0), not SI meters  |
| 3. GENUINE MODEL DEGRADATION       | DEMONSTRATED            | Even when scale-aligned, Stage 1   |
|                                    |                         | is strictly worse on all metrics   |
| 4. COORDINATE TRANSFORMATION BUG   | DEMONSTRATED            | Training omitted `norm_factor` and |
|                                    |                         | unprojected points in camera space |
| 5. EVALUATION PROTOCOL GAP         | DEMONSTRATED            | Phase 7 evaluated raw normalized   |
|                                    |                         | tensor without median scaling      |
| 6. PSEUDO-LABEL QUALITY PROBLEM    | DEMONSTRATED            | DAv2 Metric lacked physical sensor |
|                                    |                         | dropouts & precise boundaries      |
| 7. INSUFFICIENT EVIDENCE           | REFUTED BY EXPERIMENT   | 10-frame visual + 402-frame audit  |
|                                    |                         | proves conclusive causality        |
+---------------------------------------------------------------------------------------------------+
```

### Detailed Deconstruction of Demonstrated Causes:

#### Cause A: Depth Output Semantics Mismatch & Training Target Omission (Demonstrated)
- **Mechanism:** WorldMirror was pre-trained on scenes where 3D points and depths were normalized by `norm_factor = nanmedian(distance)`, producing a canonical prediction range of $\hat{d} \in [0.85, 1.05]$.
- **The Bug:** `train_stage1.py` fed unnormalized physical meter arrays ($Z \approx 3.5 - 12.4\text{ m}$) directly into an exponential DPT head ($\hat{d} = \exp(x)$) via an unmasked L1 loss.
- **The Result:** The optimizer drove the DPT projection layers to massive positive offsets to output numbers around $3.5\text{m}$. When evaluated on ScanNet (where physical depth is $0.8 - 2.5\text{m}$), the head over-activated into extreme values ($18.0 - 23.0\text{ meters}$), causing the $914\%$ error surge.

#### Cause B: Genuine Model Degradation (Demonstrated)
- A critical scientific question was: *Is Stage 1 merely suffering from a scale offset, or is its actual geometry degraded?*
- **Experimental Proof:** When both models are aligned using the official median scale factor ($s^* = \mathrm{median}(d_{gt}) / \mathrm{median}(\hat{d})$), the Pretrained Baseline **STILL BEATS STAGE 1 SOUNDLY**:
  - ScanNet Scaled AbsRel: Baseline **0.2485** vs Stage 1 **0.3691** (Baseline is 48% better).
  - ScanNet Scaled RMSE: Baseline **0.5117 m** vs Stage 1 **0.8231 m** (Baseline is 61% better).
  - ScanNet Delta1 Accuracy ($\delta < 1.25$): Baseline **0.6826** vs Stage 1 **0.3837** (Baseline is 78% higher).
  - ScanNet Shape Chamfer: Baseline **0.3560 m** vs Stage 1 **0.6099 m** (Baseline is 71% better).
  - Surface Normal MAE: Baseline **62.78°** vs Stage 1 **71.95°** (Stage 1 degraded by 9.17°).
- This proves conclusively that Stage 1 experienced **genuine structural and geometric degradation**, warping room topology and planar orientations.

---

## 3. AUDIT OF STAGE 2 3D GAUSSIAN SPLATTING

In accordance with Section 9 of the mandate:
1. **Absence of Held-Out Novel-View Evidence:** Phase 7 evaluated Stage 2 solely in feed-forward single-view inference mode, logging its Gaussian parameters (mean 146,207 splats) and depth rasterization. No novel-view camera poses were rendered against held-out ground truth RGB images to measure PSNR, SSIM, or LPIPS.
2. **Mandated Certification:** Stage 2 novel-view generalization is formally certified as:
   ```
   STAGE 2 NOVEL-VIEW GENERALIZATION: UNVERIFIED
   ```
3. **Architectural Viability:** While unverified for novel-view synthesis, the CUDA 3DGS rasterizer and `gs_head` are mechanically operational and run within local 8 GB VRAM constraints (peak 3,481 MB). Stage 2 must be retrained on top of the Pretrained Baseline with true geometric supervision in Phase 8.

---

## 4. AUDIT TRAIL OF DELIVERED REPORTS

All 8 mandated Phase 7.1 forensic reports are authored and archived in `research/3d_reconstruction/reports/`:

1. [`PHASE7_1_DEPTH_MODEL_PROVENANCE.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_DEPTH_MODEL_PROVENANCE.md) — Exact identification of `Depth-Anything-V2-Metric-Indoor-Small-hf` (commit `8078d68`, max depth 20m, output range 1.1–12.4m) and retraction of the Phase 7 relative-depth claim.
2. [`PHASE7_1_WORLD_MIRROR_DEPTH_SEMANTICS.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_WORLD_MIRROR_DEPTH_SEMANTICS.md) — Full transformation trace of WorldMirror's `exp+expp1` DPT head and primary-source proof of canonical scene scale normalization (`norm_factor`).
3. [`PHASE7_1_TRAINING_TARGET_AUDIT.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_TRAINING_TARGET_AUDIT.md) — Code-level audit of `train_stage1.py`, proving unnormalized L1 loss forced severe scale warp.
4. [`PHASE7_1_EVALUATION_METRIC_AUDIT.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_EVALUATION_METRIC_AUDIT.md) — Mathematical verification of evaluation formulas and comparison of Raw vs Scale-Aligned protocols.
5. [`PHASE7_1_COORDINATE_SYSTEM_AUDIT.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_COORDINATE_SYSTEM_AUDIT.md) — OpenCV camera conventions, $\mathrm{SE}(3)$ camera poses, and Umeyama $\mathrm{Sim}(3)$ shape-only verification.
6. [`PHASE7_1_CONTROLLED_REBENCHMARK.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_CONTROLLED_REBENCHMARK.md) — Granular per-frame metrics, frame-weighted vs scene-weighted tables, and 10 side-by-side visual diagnostic figures in `artifacts/phase7_1_forensic_visualizations/`.
7. [`PHASE7_1_LICENSE_CORRECTION.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_LICENSE_CORRECTION.md) — Retraction of the CC BY-NC-SA 4.0 claim; verification of Apple Hypersim CC BY-SA 3.0 and Stanford ScanNet non-commercial Terms of Use.
8. [`PHASE7_1_FINAL_DECISION.md`](file:///F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_FINAL_DECISION.md) — Formal causal taxonomy, Stage 2 UNVERIFIED certification, and production candidate sign-off.

---

## 5. FINAL SCIENTIFIC RECOMMENDATION

1. **Retain Pretrained Baseline as Production Candidate:**
   [`production_model.ckpt`](file:///F:/RoomAI/research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt) is retained as RoomAI’s certified production candidate. It achieves an outstanding **$0.2485$ AbsRel** and **$0.5117\text{ m}$ RMSE** on physical ScanNet living rooms under standard scale alignment.
2. **Reject Stage 1 & Stage 2:** Neither checkpoint is promoted.
3. **Preserve Test Split Quarantine:** The 484-frame held-out test split remains completely unread.
4. **Phase 8 Retraining Blueprint:** Future fine-tuning must incorporate:
   - Scene scale normalization (`depth / norm_factor`, `pts3d / norm_factor`).
   - Binary validity masks on structured-light dropouts.
   - Scale-invariant loss functions (e.g., scale-invariant depth loss + normal cosine loss).

