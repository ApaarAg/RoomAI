# ROOMAI — PHASE 5: FINAL SCIENTIFIC DECISION & EVALUATION SIGN-OFF

**Document ID:** `research/3d_reconstruction/reports/PHASE5_FINAL_DECISION.md`  
**Date:** 2026-11-08  
**Auditor:** Senior 3D Computer Vision Research Scientist & Independent Model Evaluation Lead  
**Scope:** Final sign-off, checkpoint selection, status scorecard, and definitive scientific conclusion for Phase 5.

---

## 1. Overall Scientific Conclusion

# **VERDICT: PSEUDO-LABEL DISTILLATION ONLY**

> **Formal Statement:**  
> Training mechanics, backpropagation, optimizer updates, and 3D Gaussian Splatting execution have been empirically validated on local GPU hardware.  
> However, **independent physical geometric improvement has NOT been demonstrated**.  
> The supervision signals were model-predicted pseudo-labels (Depth Anything V2) and synthetic camera trajectories ($R_y(0.05\cdot v)$), not physical ground truth.  
> While Stage 1 reduced discrepancy against Depth Anything V2 by 44.7%, it also introduced a severe 3.63x scale divergence on specific room layouts (`Office`), and Stage 2 novel-view rendering remains unanchored by calibrated multi-view camera poses.

---

## 2. Definitive Status Scorecard

| Evaluation Gate | Verified Status | Empirical Evidence & Findings |
| :--- | :---: | :--- |
| **1. Dataset Provenance Verified** | **VERIFIED** | Audited all 11 scenes (39 views). 100% of depth maps are Depth Anything V2 pseudo-labels; all normals are finite-difference derivatives; all poses are synthetic heuristics. |
| **2. Real Geometric Ground Truth Verified** | **FALSE / NOT VERIFIED** | Zero LiDAR, zero structured light, and zero CAD ground-truth exist in the current local dataset. |
| **3. Camera-Pose Provenance Verified** | **VERIFIED (SYNTHETIC)** | Verified that camera extrinsics were generated via an assumed circular heuristic ($R_y(0.05\cdot v), t_x = 0.1\cdot v$), not COLMAP or calibrated hardware. |
| **4. Genuine Multi-View Sequences Verified** | **PARTIALLY VERIFIED** | 5 scenes have genuine video overlap (>190 ORB matches); 2 scenes are wide-baseline/disjoint; 2 scenes are single photographs. All lack calibrated poses. |
| **5. Validation Metrics Verified** | **VERIFIED** | Audited poor validation metrics (2.959m point acc, 1.272 AbsRel). Proven to stem from the `Office` 3.63x scale outlier and coordinate normalization offset. |
| **6. Pretrained vs Stage 1 Comparison Completed** | **VERIFIED** | Stage 1 improved point accuracy from 4.602m to 2.546m (-44.7% error against pseudo-GT) and improved 3/4 scenes in depth, but regressed on `Office` (10.6m RMSE). |
| **7. Pretrained vs Stage 2 Comparison Completed** | **VERIFIED** | Stage 2 preserves Stage 1 geometry and generates finite 3DGS splats, but doubles latency (482.5ms vs 248.9ms) without calibrated multi-view pose supervision. |
| **8. Held-Out Generalization Demonstrated** | **FALSE / NOT DEMONSTRATED** | The single held-out test view (`Sample_Room`) evaluated pseudo-label agreement, which cannot substantiate generalizable 3D reconstruction. |
| **9. Best Checkpoint Selected** | **VERIFIED** | Retained Pretrained Baseline as the official production release; preserved Stage 1 as an experimental distilled candidate. |

---

## 3. Checkpoint Selection & Production Gate

### 3.1 Decision on Production Candidate
* **Stage 1 Fine-Tuned Checkpoint (`best_stage1.ckpt`):**
  * **PROS:** Sharp room geometry, reduced Euclidean point offset (-44.7%), fast inference (248.9 ms).
  * **CONS:** Unanchored scale drift on cubicles (`Office` 3.63x scale error); trained exclusively on pseudo-labels.
  * **DECISION:** **REJECTED FOR PRODUCTION PROMOTION**. Preserved in `research/3d_reconstruction/checkpoints/stage1/` as an experimental checkpoint.
* **Stage 2 3DGS Checkpoint (`stage2_real_verified.ckpt`):**
  * **PROS:** Generates valid 3D Gaussian Splats with regularized opacities.
  * **CONS:** 2x inference latency penalty (482.5 ms); unanchored novel-view radiance due to synthetic camera poses.
  * **DECISION:** **REJECTED FOR PRODUCTION PROMOTION**. Retained as research proof-of-concept.
* **Official Pretrained WorldMirror Baseline (`production_model.ckpt`):**
  * **PROS:** Stable general-purpose baseline, well-calibrated across broad indoor categories, zero fine-tuning scale collapse.
  * **DECISION:** **RETAINED AS PRIMARY PRODUCTION BASELINE**.

---

## 4. Checkpoint Selection Manifest

```json
{
  "selection_timestamp": "2026-11-08T03:25:00Z",
  "selection_verdict": "RETAIN_PRETRAINED_BASELINE",
  "production_checkpoint": {
    "path": "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt",
    "sha256": "dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5",
    "parameters": 1223250748,
    "size_bytes": 2839184102,
    "status": "APPROVED_PRODUCTION_BASELINE"
  },
  "research_candidates": {
    "stage1_distilled": {
      "path": "research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt",
      "sha256": "fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2",
      "status": "EXPERIMENTAL_PSEUDO_DISTILLED"
    },
    "stage2_3dgs": {
      "path": "research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt",
      "sha256": "e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece",
      "status": "EXPERIMENTAL_PROOF_OF_CONCEPT"
    }
  },
  "guardrail_reasons": [
    "Stage 1 regressed on Office test scene depth scale by 3.63x",
    "Current dataset uses Depth Anything V2 pseudo-labels rather than physical ground truth",
    "Camera poses are synthetic heuristics, preventing valid novel-view 3DGS verification"
  ]
}
```

---

## 5. Immediate Next Actions

1. Preserve all existing checkpoints in their designated research directories without modification.
2. Execute the Pilot Ground-Truth Acquisition Plan ([`PHASE5_DATASET_ACQUISITION_PLAN.md`](file:///f:/RoomAI/research/3d_reconstruction/reports/PHASE5_DATASET_ACQUISITION_PLAN.md)) to download 10 validated ScanNet and Hypersim scenes with true LiDAR / CAD depth and COLMAP poses.
3. Integrate Umeyama Sim(3) alignment and median depth scaling into evaluation scripts to eliminate artificial coordinate offset errors.
4. Re-evaluate WorldMirror on the true ground-truth benchmark before considering future production model promotion.

