# RoomAI — Backend Test Results & Verification Audit

**Date:** October 29, 2026  
**Environment:** Windows 11, Conda `tf-gpu`, Python 3.10.20, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM)  
**Execution Command:** `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pytest -q`  
**Overall Status:** **117 PASSED / 0 FAILED / 0 SKIPPED (100% PASS RATE)**  
**Total Runtime:** 74.68 seconds (verified clean execution)  

---

## 1. Test Suite Summary by Module

| Test Module | Total Tests | Passed | Failed | Skipped | Test Focus / Capabilities Covered |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `tests/test_deepfurniture_detector.py` | 7 | 7 | 0 | 0 | DeepFurniture YOLO11n checkpoint loading, response schemas, confidence filtering, mock mode isolation, cached model loading, missing checkpoint clean failure, detection service integration. |
| `tests/test_depth_estimation.py` | 8 | 8 | 0 | 0 | Mock Scene3D model, custom Scene3D unconfigured exception handling, missing checkpoint exception handling, invalid input validation, real inference shape and relative depth range, CPU fallback behavior, service integration, depth heatmap visualization generation. |
| `tests/test_detection_deployment_readiness.py` | 13 | 13 | 0 | 0 | Mock mode isolation, zero silent mock substitution in real mode, missing checkpoint exception handling, 12-class indoor taxonomy mapping, confidence threshold filtering, timing metrics integrity, cross-class conflict resolution, overlapping object preservation, 800px resolution scaling, ModelRegistry primary A3 and rollback A2 verification. |
| `tests/test_ensemble_service.py` | 9 | 9 | 0 | 0 | Weighted Box Fusion (WBF), consensus bonus, single-model confidence penalization, cross-class IoU conflict filtering, empty detection handling, mock mode ensemble fallback. |
| `tests/test_exterior_ensemble.py` | 13 | 13 | 0 | 0 | 10-class exterior taxonomy consistency, exterior ensemble WBF fusion, cross-class bounding box suppression, threshold compliance, latency limits. |
| `tests/test_masked_loss.py` | 5 | 5 | 0 | 0 | Class-conditional binary cross-entropy loss masking, mathematical identity on fully annotated images (`max abs diff = 0.0`), zero loss and zero backward gradient on unannotated classes (`mask=0.0`), finite loss and gradient flow. |
| `tests/test_model_registry.py` | 13 | 13 | 0 | 0 | Dynamic VRAM routing, LRU model eviction under memory pressure, task-specific model registration, primary model resolution, fallback model switching, metadata tracking. |
| `tests/test_parking_ensemble.py` | 9 | 9 | 0 | 0 | 8-class outdoor parking taxonomy, Parking C2 primary model resolution, lightweight C1 fallback, ensemble fusion mechanics. |
| `tests/test_per_class_thresholds.py` | 13 | 13 | 0 | 0 | Per-class operating thresholds for exterior classes, fallback default thresholds, threshold dictionary validation, preservation of high-precision rare classes. |
| `tests/test_phase19_multidataset.py` | 16 | 16 | 0 | 0 | Multi-dataset harmonization, 64-bit DCT perceptual hash (pHash) near-duplicate screening ($d \ge 5$), exact MD5 deduplication, bounding box coordinate normalization and bounds validation ($0 < x, y, w, h \le 1.0$). |
| `tests/test_specialist_ensemble.py` | 8 | 8 | 0 | 0 | Verification of Model B2 as active production default, opt-in candidate status for B5 and B6 (`enabled: false`), xyxy box IoU calculation, specialist ensemble mock mode, specialist fusion with general detector. |
| `tests/test_timing_utils.py` | 3 | 3 | 0 | 0 | Precision latency benchmarking, CPU/GPU timing synchronization, execution timer context manager. |
| **TOTAL** | **117** | **117** | **0** | **0** | **Full Suite Pass** |

---

## 2. Application Smoke Test Results

**Script:** `scripts/smoke_test_application.py`  
**Execution Status:** **PASS** (Exit Code: 0)  

### Verified Workflow Checks
1. **Configuration Loading:** Both `config/app.yaml` and `config/model_config.yaml` load and parse cleanly with zero schema warnings.
2. **Production Default Verification:** Verified `exterior_detection` active default is `checkpoints/exterior_detector_b2_yolo11s_best.pt` with `status: "production_ready"`.
3. **Candidate Model Quarantine:** Verified experimental candidates B5 and B6 are registered with `enabled: false` and `status: "experimental_candidate"`.
4. **ModelRegistry Specifications:** Verified `exterior_b2_finetuned` is `is_primary=True`; B5, B6, and C1 are non-primary.
5. **Real Inference Execution:**
   - Production Model B2 executed cleanly on GPU (`cuda:0`).
   - Candidate Model B6 executed cleanly in 259.86 ms.
   - Candidate Model B5 executed cleanly in 702.99 ms.
6. **Mock Mode Isolation:** Executed mock detection service; returned 5 simulated detections with `is_simulated: True`.
7. **Explicit Failure Integrity:** Pointed detection service to `checkpoints/non_existent_model.pt`; verified it raises `FileNotFoundError` explicitly and **never silently substitutes mock output**.
8. **VRAM Headroom:** GPU memory allocated during smoke test was 140.8 MB (reserved 212.0 MB), well within the 4,500 MB budget.

---

## 3. Streamlit Application Startup Verification

**Script:** `scripts/test_streamlit_startup.py`  
**Target:** `app.py`  
**Execution Status:** **PASS** (Exit Code: 0)  

### Verified Lifecycle Checks
- Started Streamlit server headless on dedicated port 8503.
- Polled internal health endpoint `http://localhost:8503/_stcore/health`.
- Received HTTP 200 OK with response body `"ok"` in 9.31 seconds.
- Terminated process cleanly with zero orphaned subprocesses.

