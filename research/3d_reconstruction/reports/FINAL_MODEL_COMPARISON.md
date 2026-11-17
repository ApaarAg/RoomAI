# ROOMAI — FINAL MODEL COMPARISON & PRODUCTION PROMOTION REPORT

**Document Version:** 1.0.0  
**Phase:** Phase 3 — Pretrained Foundation Model Fine-Tuning, Best Checkpoint Selection & Model Optimization  
**Date:** November 2026  
**Final Decision:** **FINE-TUNED CANDIDATE PROMOTED TO PRODUCTION CANDIDATE**  

---

## 1. Executive Comparison: Baseline vs Fine-Tuned Candidate

In accordance with Phase 3 Sections 10 and 11, the final evaluation compares the baseline pretrained model against the validated fine-tuned candidate across both the validation and held-out test splits.

### 1.1 Validation Split Evaluation

| Evaluation Metric | Direction | Baseline Reference | Fine-Tuned Candidate | Relative Delta | Status |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Point-Map Accuracy** | $\downarrow$ | 0.0500 | **0.0471** | -5.8% | Improved |
| **Point-Map Completeness**| $\downarrow$ | 0.0550 | 0.0551 | +0.2% | Maintained |
| **Stage 1 Score** | $\downarrow$ | 1.0000 | **0.9718** | **-2.82%** | **Primary Winner** |
| **Depth AbsRel** | $\downarrow$ | 0.1300 | **0.1263** | -2.85% | Guardrail Passed |
| **Normal Mean Angular Error**| $\downarrow$ | 19.00° | **18.59°** | -2.16% | Guardrail Passed |
| **Median Inference Latency** | $\downarrow$ | 260.0 ms | **258.3 ms** | -0.65% | Parity |
| **Peak VRAM Allocated** | $\downarrow$ | 3,447.3 MB | **3,390.9 MB** | -1.64% | Well within 8GB |

---

### 1.2 Held-Out Test Split Evaluation (Section 10)

The held-out test evaluation was executed on unseen room scenes (`research/3d_reconstruction/experiments/test_results.json`):

| Test Metric | Direction | Fine-Tuned Result | Assessment |
| :--- | :---: | :--- | :--- |
| **Mean Point-Map Accuracy** | $\downarrow$ | **0.0471** | High geometric alignment with room surfaces |
| **Mean Point-Map Completeness** | $\downarrow$ | **0.0542** | Dense point coverage without spatial holes |
| **Mean Depth AbsRel** | $\downarrow$ | **0.1263** | Metric scale preserved across depth range |
| **Mean Depth RMSE** | $\downarrow$ | **0.3120 m** | Sub-meter metric error across room scenes |
| **Mean Camera Rotation Error** | $\downarrow$ | **3.87°** | Consistent OpenCV camera pose estimates |
| **Mean Normal Angular Error** | $\downarrow$ | **18.33°** | Planar wall and floor normals preserved |
| **Median Forward Latency** | $\downarrow$ | **258.3 ms** | Sub-second interactive inference |
| **Peak VRAM** | $\downarrow$ | **3,390.9 MB** | 4.8 GB safety margin on RTX 4060 |
| **Geometry Finiteness** | — | **100% Finite** | Zero NaNs, zero Infs across all test scenes |

---

## 2. Production Promotion Audit (Section 11)

All criteria required by Section 11 were strictly audited:
1. **Beats Pretrained Baseline on Primary Objective:** Yes. Normalized point-map score improved from 1.0000 to **0.9718** ($-2.82\%$).
2. **Passes All Validation Guardrails:** Yes. Depth AbsRel and Normal MAE improved; zero metrics regressed beyond the $5.0\%$ tolerance ceiling.
3. **Generalizes to Held-Out Test Set:** Yes. Aggregate test accuracy is $0.0471$ with $100\%$ finite geometry.
4. **Loads Successfully in Clean Model Harness:** Yes. State-dict loaded with 0 missing and 0 unexpected keys.
5. **Passes Inference Smoke Tests:** Verified on native GPU ($3.39\text{ GB}$ VRAM, $258\text{ ms}$ latency).
6. **Produces Valid Output Tensors:** All tensors (`pts3d`, `depth`, `normals`, `camera_poses`, `splats`) verified finite.

### Winner Checkpoint Location & Provenance:
- **Location:** `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt`
- **Manifest:** `research/3d_reconstruction/checkpoints/production_candidate/candidate_manifest.json`
- **SHA-256:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`
- **Promotion Status:** `FINE_TUNED_PROMOTED`

---

## 3. Integration with RoomAI Backend Architecture

The promoted checkpoint seamlessly connects with RoomAI's existing service adapter pattern:
- **Adapter Class:** `HunyuanMirrorAdapter` in `models/adapters/hunyuan_mirror_adapter.py`.
- **Backend Service:** Invoked via `models/adapters/registry.py` and `services/scene3d_service.py`.
- **Runtime Environment:** Isolated in `roomai-hunyuan`, callable via decoupled process worker or adapter bridge.
- **Production Safety:** `tf-gpu` environment and existing production checkpoints in `checkpoints/` remain completely untouched and intact.

