# RoomAI — Phase 6.2: Mock Mode Default and Production Model Selection Fix

**Date:** November 5, 2026  
**Status:** Complete & Independently Validated  
**Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), Windows 11, Conda `tf-gpu` (Python 3.10.20, PyTorch 2.11.0+cu128)  
**Test Suite Status:** 155 passed / 0 failed in 38.21s  

---

## 1. Problem Diagnosis & Root Causes

Inspection of the Streamlit application startup state revealed three root causes for the reported behavior:

1. **Mock Mode Defaulted to True:**
   - In `config/app.yaml`, `mock_mode` was set to `true`.
   - In `app.py`, `app_config.get("mock_mode", True)` defaulted to `True`, automatically activating Mock Mode and displaying the simulated output warning upon launch.
   - `st.session_state` was not actively guarding the user's explicit preference across session reruns.

2. **Hardcoded Generic Detectors in UI Selectbox:**
   - In `ui/analysis_view.py` (Tab 1: Object Detection), the detector architecture options were hardcoded as `ckpt_options = ["yolo11s.pt", "yolo11n.pt"]`.
   - The production configuration specifies `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt`. Because this path was absent from `ckpt_options`, the selectbox fell back to generic options, displaying `yolo11n.pt` / `yolo11s.pt` and substituting generic weights instead of Model A3.

3. **VRAM Metric Misinterpretation:**
   - PyTorch allocates CUDA memory dynamically on demand when tensor operations or model weights are loaded into VRAM.
   - Prior to running inference, PyTorch reported `< 0.05 GB` allocated, which was formatted as `0 GB`, causing an impression of GPU failure or inaction.

---

## 2. Remediations & Technical Implementations

### 2.1 Real Inference Enabled as Default
- Updated [`config/app.yaml`](file:///F:/RoomAI/config/app.yaml): Set `mock_mode: false`.
- Updated [`app.py`](file:///F:/RoomAI/app.py):
  - Initialized `st.session_state["mock_mode"] = bool(config_mock)` (defaulting to `False`).
  - Added persistent sidebar checkbox binding with explicit session-state persistence.
  - When mock mode is off (default), displays green banner: `🚀 Real Inference Active: Running trained models on GPU.`
  - When mock mode is explicitly checked by the user, displays warning banner: `⚠️ Mock Mode Active: Outputs are simulated.`
  - Ensured no silent fallback occurs if real model weights fail to load; an explicit `st.error` is displayed and recorded in `st.session_state["last_model_error"]`.

### 2.2 Correct Production Model Selection
- Updated [`ui/analysis_view.py`](file:///F:/RoomAI/ui/analysis_view.py):
  - Defined canonical production dictionary mappings:
    - `INDOOR_DETECTOR_CHOICES`: Model A3 (800px YOLO11s) ordered first as active production default (`checkpoints/indoor_detector_a3_yolo11s_800px_best.pt`), with A2, A1, and DeepFurniture baseline as secondary options.
    - `EXTERIOR_DETECTOR_CHOICES`: Model B2 (YOLO11s) ordered first as active production default (`checkpoints/exterior_detector_b2_yolo11s_best.pt`).
    - `PARKING_DETECTOR_CHOICES`: Model C2 (YOLO11s) ordered first as active production default (`checkpoints/parking_detector_c2_yolo11s_pilot_best.pt`).
  - Added a domain selector (`Task Domain`) allowing seamless switching between Indoor (A3), Exterior (B2), and Parking (C2) without hardcoded generic checkpoints.
  - Ensured `imgsz=800` is dynamically set when Model A3 is active.

### 2.3 Real Inference Verification
- Verified that running with Mock Mode disabled:
  - Loads the production checkpoint (`indoor_detector_a3_yolo11s_800px_best.pt`).
  - Uses `cuda:0` acceleration on the RTX 4060 GPU.
  - Returns real detections with valid bounding boxes and confidence scores.
  - Emits `is_simulated=False` in the `DetectionResponse` schema.
  - Displays real latency (ms), device (`CUDA:0`), model architecture, and checkpoint name in the UI banner.

### 2.4 Detailed Hardware & VRAM Status
- Updated [`utils/device.py`](file:///F:/RoomAI/utils/device.py):
  - Enhanced `get_device_info()` to report `allocated_vram_mb`, `allocated_vram_gb`, `peak_vram_mb`, `peak_vram_gb`, `cuda_version`, and `pytorch_version`.
- Updated [`ui/components.py`](file:///F:/RoomAI/ui/components.py):
  - Display both MB and GB for current and peak allocated VRAM.
  - Added contextual note explaining that PyTorch allocates memory dynamically on demand, clarifying that 0 GB initial idle allocation is expected behavior.
  - Added surface notification for `last_model_error` if any model fails loading.

---

## 3. Verification & Regression Testing

A dedicated test suite was implemented in [`tests/test_mock_mode_and_production_selection.py`](file:///F:/RoomAI/tests/test_mock_mode_and_production_selection.py) covering all requirements:

| Test Name | Verified Contract | Result |
|---|---|---|
| `test_mock_mode_defaults_to_false` | `mock_mode` is False in `config/app.yaml` and defaults to False | **PASSED** |
| `test_explicit_mock_mode_activation` | Explicit mock toggle produces `is_simulated=True` across all services | **PASSED** |
| `test_explicit_real_mode_activation` | Real mode executes A3 on CUDA/CPU with `is_simulated=False` | **PASSED** |
| `test_correct_a3_checkpoint_selection` | Model A3 800px is configured in yaml, registry, and UI index 0 | **PASSED** |
| `test_b2_and_c2_production_selection` | Models B2 and C2 are selected as primary production checkpoints | **PASSED** |
| `test_missing_checkpoint_handling_no_silent_fallback` | Missing weights raise explicit error, never falling back to mock | **PASSED** |
| `test_is_simulated_metadata_integrity` | Strict parity between execution mode and `is_simulated` flag | **PASSED** |
| `test_session_state_mock_mode_persistence` | Session state preserves user toggle across application reruns | **PASSED** |
| `test_hardware_status_reporting` | GPU info reports MB and GB without treating idle 0 GB as failure | **PASSED** |

### Full Test Suite Run:
```
============================ test session starts ============================
platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
rootdir: F:\RoomAI, configfile: pytest.ini
testpaths: tests
collected 155 items

tests\test_deepfurniture_detector.py .......                          [  4%]
tests\test_deployment_hardening.py .............                      [ 12%]
tests\test_depth_estimation.py ........                               [ 18%]
tests\test_detection_deployment_readiness.py ..............           [ 27%]
tests\test_ensemble_service.py ........                               [ 32%]
tests\test_exterior_ensemble.py ........                              [ 37%]
tests\test_image_validation.py ......                                 [ 41%]
tests\test_masked_loss.py .....                                       [ 44%]
tests\test_mock_mode_and_production_selection.py .........            [ 50%]
tests\test_model_interfaces.py .........                              [ 56%]
tests\test_model_registry.py ....                                     [ 58%]
tests\test_parking_ensemble.py .....                                  [ 61%]
tests\test_per_class_thresholds.py .......                            [ 66%]
tests\test_phase19_multidataset.py ..........                         [ 72%]
tests\test_schemas.py .......                                         [ 77%]
tests\test_services.py .......                                        [ 81%]
tests\test_specialist_ensemble.py ..........                          [ 88%]
tests\test_state_invalidation.py ...                                  [ 90%]
tests\test_timing_utils.py ...                                        [ 92%]
tests\test_ui_integration.py ............                             [100%]

========================= 155 passed in 38.21s ==========================
```

### Streamlit Startup Health Check:
```
Starting Streamlit on port 8503...
[PASS] Streamlit started successfully in 3.04s! Health response: 'ok'
Terminating Streamlit process...
[PASS] Streamlit process stopped cleanly.
=== Streamlit Application Startup Verified Successfully! ===
```

---

## 4. Preservation Invariant Compliance

- **No Retraining or Modifying Checkpoints:** Production checkpoints (`indoor_detector_a3_yolo11s_800px_best.pt`, `exterior_detector_b2_yolo11s_best.pt`, `parking_detector_c2_yolo11s_pilot_best.pt`, `yolo11n-seg.pt`, `depth_anything_v2_small/`) remain strictly bit-for-bit preserved.
- **Class Vocabularies Preserved:** The 12 indoor classes, 10 exterior classes, and 8 parking classes remain unaltered.
- **No Unsolicited Architectural Complexity:** All fixes are direct, robust, and contained to configuration, device utilities, UI component rendering, and regression tests.

