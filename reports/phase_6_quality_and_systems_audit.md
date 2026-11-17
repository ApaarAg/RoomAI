# RoomAI — Phase 6: Model Quality, Weaknesses & System Stability Audit

**Date:** 2026-11-05  
**Auditor:** Principal Computer Vision Engineer, QA Architect & Systems Optimization Specialist  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, 24 GB RAM, Windows 11)

---

## 1. Executive Summary

In accordance with user directives, Phase 5.9 reports have been permanently preserved as research artifacts. No generative 3D models (TripoSR, SF3D, Shap-E, etc.) are being integrated. 

The project priority has shifted to **optimizing, hardening, stabilizing, and preparing RoomAI for local deployment**.

This audit inspects every existing model, measures its operational quality, diagnoses root-cause defects, catalogs test suite regressions, and establishes the technical blueprint for the Phase 6 stabilization sprint.

---

## 2. Comprehensive Model Inventory & Empirical Quality Audit

### 2.1 Model A3 — Indoor Furniture & Household Objects Detector
* **Role:** Production Primary for Indoor Scene Understanding (Group A).
* **Architecture:** Ultralytics YOLO11s trained at high resolution ($800 \times 800\text{ px}$).
* **Active Checkpoint:** `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` (18.3 MB, 9.43M parameters).
* **Taxonomy (12 Classes):** `cabinet#shelf`, `table`, `chair#stool`, `lamp`, `door`, `bed`, `sofa`, `plant`, `decoration`, `curtain`, `home-appliance`, `window`.
* **Empirical Quality:**
  * **Test mAP50:** **0.7374** on DeepFurniture/HomeObjects canonical test set.
  * **Forward-Pass Latency (GPU):** $13.60\text{ ms} - 16.85\text{ ms}$ (warm inference).
  * **Static Parameter VRAM:** $36.3\text{ MB}$.
* **Measurable Weaknesses:**
  1. *Armchair / Sofa Class Conflict:* Overlapping predictions where high-backed armchairs trigger both `chair#stool` and `sofa` boxes with near-identical coordinates ($IoU > 0.80$).
  2. *Slender Leg Fragmentation:* Thin wooden or metal legs of dining chairs lose contrast against patterned floors, requiring higher resolution at the cost of letterbox latency.
  3. *Inference Overhead at 800px:* Letterboxing and NMS overhead increases total service latency from $\sim 15\text{ ms}$ to $\sim 45\text{ ms}$ on larger images.

### 2.2 Model B2 — Exterior & Building Components Detector
* **Role:** Production Default for Exterior Architecture (Group B).
* **Architecture:** Ultralytics YOLO11s ($640 \times 640\text{ px}$).
* **Active Checkpoint:** `checkpoints/exterior_detector_b2_yolo11s_best.pt` (19.2 MB, 9.43M parameters).
* **Taxonomy (10 Classes):** `building`, `facade`, `window`, `door`, `roof_cornice`, `balcony`, `porch`, `pillar_column`, `shutter_blind`, `molding_deco`.
* **Empirical Quality:**
  * **Test mAP50:** **0.4554** (Window mAP50: 0.703).
  * **Forward-Pass Latency (GPU):** $24.3\text{ ms} - 28.5\text{ ms}$.
  * **Static Parameter VRAM:** $36.0\text{ MB}$.
* **Measurable Weaknesses:**
  1. *Rare Class Underperformance:* `porch` has low recall due to limited training samples ($N=24$) in the canonical training split.
  2. *Harsh Shadow Sensitivity:* Outdoor facades with strong direct sunlight next to dark shadows show false negatives on facade moldings.

### 2.3 Model C2 — Parking & Outdoor Property Detector
* **Role:** Production Primary for Parking & Outdoor Property (Group C).
* **Architecture:** Ultralytics YOLO11s ($640 \times 640\text{ px}$).
* **Active Checkpoint:** `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` (19.2 MB, 9.43M parameters).
* **Taxonomy (8 Classes):** `car_vehicle`, `motorcycle`, `bicycle`, `outdoor_seating`, `outdoor_table`, `plant_bush`, `tree`, `outdoor_fixture`.
* **Empirical Quality:**
  * **Test mAP50:** **0.4491**; Ensemble mAP50: 0.3920.
  * **Forward-Pass Latency (GPU):** $24.9\text{ ms} - 26.4\text{ ms}$.
* **Measurable Weaknesses:**
  1. *Vegetation Confusion:* Borderline overlap between dense `plant_bush` and low-hanging `tree` foliage.

### 2.4 Instance Segmentation — FastSAM-s & YOLO11n-seg
* **Role:** Instance-level polygon mask extraction.
* **Active Checkpoints:** `FastSAM-s.pt` (22.7 MB, 11.79M params) & `checkpoints/yolo11n-seg.pt` (5.9 MB, 2.89M params).
* **Empirical Quality:**
  * FastSAM Prompt Latency: $\sim 55\text{ ms}$ per object on GPU.
* **Measurable Weaknesses:**
  1. *Batch Argmax Mask Leakage:* When prompted with all boxes simultaneously, FastSAM's global `argmax` causes masks to select mismatched global objects (23.06% containment ratio).
  2. *Per-Box Overhead:* Calling FastSAM sequentially per-box achieves near 100% containment but scales linearly ($N \times 55\text{ ms}$).
  3. *Unimplemented Stub in `models/segmentation_model.py`:* `CustomRoomSegmenter` raises `ModelNotImplementedError`, preventing local fallback to `yolo11n-seg.pt`.

### 2.5 Semantic Segmentation — SegFormer-B1
* **Role:** Architectural surface segmentation (walls, floors, ceilings).
* **Architecture:** SegFormer-B1 (`nvidia/segformer-b1-finetuned-ade-512-512`, 13.72M parameters).
* **Empirical Quality:**
  * Target-10 mIoU: **0.6638**; All-88 mIoU: **0.3017** on quarantined Golden Test Set.
  * Inference Latency: $25.7\text{ ms}$ on GPU.
* **Measurable Weaknesses:**
  1. *Pretraining Overlap:* Model was pretrained on all ADE20K images; residential fine-tuning in Phase E pilot demonstrated slight negative transfer (-0.68%) at 1% data regime.
  2. *Surface-to-3D Gap:* 2D semantic masks are not yet extruded into continuous planar surface equations in the 3D viewer.

### 2.6 Depth Estimation — Depth Anything V2 Small
* **Role:** Dense monocular depth and 3D point cloud unprojection.
* **Architecture:** DINOv2 ViT-Small backbone + DPT Head (24.79M parameters).
* **Active Checkpoint:** `checkpoints/depth_anything_v2_small/` (94.6 MB).
* **Empirical Quality:**
  * Relative inverse depth range: $[0.0, 1.0]$.
  * Inference Latency: $62.0\text{ ms}$ warm GPU / $159.8\text{ ms}$ cold.
  * Static Parameter VRAM: $94.6\text{ MB}$.
* **Measurable Weaknesses:**
  1. *Relative Depth Only:* Non-metric; distances are relative inverse values, preventing true physical dimension measurements.
  2. *Visual Heatmap Colormap Limitations:* Viewer only provided a fixed Inferno map without user-selectable visualization controls.

---

## 3. Measurable Discrepancies and Root-Cause Diagnostics

Executing the test suite revealed **31 failing tests** out of 146 total tests. The root causes were systematically diagnosed:

### Defect 1: Configuration Desynchronization (`config/model_config.yaml`)
- **Severity:** High (System Configuration Failure)
- **Root Cause:** `config/model_config.yaml` was overwritten with an unconfigured default template from git commit `8f4bf66` containing generic `yolo11s.pt` (COCO 80-class) and missing `exterior_detection` and `parking_detection` blocks.
- **Impact:** Failed 14 tests across `test_detection_deployment_readiness`, `test_per_class_thresholds`, `test_specialist_ensemble`, and `test_ui_integration`.
- **Remediation:** Restore the verified hardened production configuration from `config/model_config.yaml.backup_hardening_step1` while preserving Phase 5.8.1 `scene3d.enabled: true` and `generate_point_cloud: true`. Add explicit candidate stubs (`exterior_detection_b6_candidate: enabled: false`).

### Defect 2: Missing `timing_metrics` in `services/detection_service.py`
- **Severity:** High (Service Contract Regression)
- **Root Cause:** `services/detection_service.py` was reverted to a bare version lacking `timing_metrics`, per-class thresholding, and session-scoped artifact preservation, while `services/detection_service.py.backup_phase11` preserved the exact complete implementation.
- **Impact:** Throws `KeyError: 'timing_metrics'` across `test_deployment_hardening.py`, `test_detection_deployment_readiness.py`, and `test_ui_integration.py`.
- **Remediation:** Merge the complete hardened `DetectionService` and `resolve_detector` from `backup_phase11` into `services/detection_service.py`.

### Defect 3: Unimplemented `CustomRoomSegmenter` & Missing `HEX_PALETTE`
- **Severity:** High (Functional Defect)
- **Root Cause:** `models/segmentation_model.py` retains student stub raising `ModelNotImplementedError`, and fails to export `HEX_PALETTE`.
- **Impact:** `test_custom_segmenter_with_pretrained_checkpoint`, `test_segmentation_workflow`, and `test_segmentation_palette_alignment` fail.
- **Remediation:** Implement `CustomRoomSegmenter` using `yolo11n-seg.pt` (with graceful fallback to `MockRoomSegmenter` if weights missing) and export aligned `HEX_PALETTE`.

### Defect 4: Model Cache Instantiation in `models/detection_model.py`
- **Severity:** Medium (Performance & Cache Integrity)
- **Root Cause:** `load_model()` in `models/detection_model.py` was calling `self.model = YOLO(self.checkpoint_path)` instead of `get_cached_yolo(self.checkpoint_path)`. Additionally, default constructor checkpoint without config was hardcoded to `yolo11s.pt` rather than `deepfurniture_yolo11n_finetune_best.pt`.
- **Impact:** `test_cached_model_loading` and `test_checkpoint_exists_and_class_mapping` fail.
- **Remediation:** Route `load_model()` through `get_cached_yolo()` and set default constructor path to `checkpoints/deepfurniture_yolo11n_finetune_best.pt`.

### Defect 5: Stale State Invalidation in `ui/upload_view.py` (DEF-1 & DEF-2)
- **Severity:** High (Data Integrity)
- **Root Cause:** `ui/upload_view.py` on `master` branch omitted `classification_result`, `vlm_result`, and `design_result` from the invalidation loop, and lacked the atomic Clear button handler.
- **Impact:** `test_state_invalidation.py` fails 2 tests.
- **Remediation:** Update `ui/upload_view.py` with the complete canonical 9-key invalidation list on image switch and clear button.

### Defect 6: Decompression Bomb Error Message Catch in `utils/image_validation.py`
- **Severity:** Low (Exception Handling Specificity)
- **Root Cause:** `validate_image_file` caught all exceptions in verify block and returned `"Image file appears corrupted or unreadable."` without retaining the `"decompression"` keyword.
- **Impact:** `test_decompression_bomb_rejection` fails.
- **Remediation:** Catch `Image.DecompressionBombError` explicitly and return descriptive message containing `"decompression"`.

---

## 4. Phase 6 Optimization and Stabilization Roadmap

| Milestone | Objective | Files Impacted |
|---|---|---|
| **Phase 6.1: Core Service & Config Stabilization** | Fix config desynchronization, restore `timing_metrics`, implement `CustomRoomSegmenter`, and fix caching. | `config/model_config.yaml`, `services/detection_service.py`, `models/detection_model.py`, `models/segmentation_model.py`, `utils/image_validation.py` |
| **Phase 6.2: UI & State Hardening** | Restore DEF-1/DEF-2 state invalidation in `ui/upload_view.py`, enhance relative depth visualization controls. | `ui/upload_view.py`, `ui/analysis_view.py`, `ui/three_d_viewer.py` |
| **Phase 6.3: Inference Latency & VRAM Optimization** | Measure warm GPU latency, enforce memory cleanup hooks, profile PyTorch allocator. | `utils/gpu_concurrency.py`, `services/scene3d_service.py` |
| **Phase 6.4: Full Regression Testing** | Execute complete 146-test test suite; achieve 100% pass rate. | `tests/` |
| **Phase 6.5: Local Deployment Readiness** | Verify headless Streamlit healthcheck on port 8501, document local launch protocol for RTX 4060. | `README.md`, `reports/phase_6_deployment_readiness.md` |

