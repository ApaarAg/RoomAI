# RoomAI — Production Readiness Audit: Baseline Report

**Audit Date:** October 29, 2026  
**Lead Engineer / Auditor:** Senior ML Engineer, Computer Vision Engineer, QA Architect & MLOps Lead  
**Audit Scope:** Production Readiness, Architecture Trace, Latency/VRAM Telemetry, Automated Testing, Security & Deployment  
**Repository Path:** `C:\Users\apaar\Desktop\RoomAI`  
**Git Working Tree Status:** **DIRTY / UNCOMMITTED CHANGES PRESENT**  

---

## 1. Git & Environment Baseline

### 1.1 Git Working Tree Status
* **Branch:** `master`
* **Head Commit:** `8f4bf662eb73518866f516710b68a99c6b03f37a`
* **Tracked Modified Files (15 files):**
  - `README.md`
  - `app.py`
  - `config/app.yaml`
  - `config/model_config.yaml`
  - `models/detection_model.py`
  - `models/scene3d_model.py`
  - `models/segmentation_model.py`
  - `schemas/responses.py`
  - `services/detection_service.py`
  - `services/image_service.py`
  - `services/scene3d_service.py`
  - `ui/analysis_view.py`
  - `ui/components.py`
  - `ui/upload_view.py`
  - `utils/image_validation.py`
* **Untracked Files:** ~139 untracked files/directories (historical reports, datasets in `data/`, evaluation artifacts in `artifacts/`, migration scripts in `scripts/`, test suite additions in `tests/`).
* **Preservation Status:** All pre-existing modified files and untracked artifacts are preserved. No destructive Git commands have been or will be executed.

### 1.2 Python Runtime & Hardware Environment
* **Python Executable:** `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`
* **Python Version:** 3.10.20 (64-bit AMD64)
* **PyTorch Version:** 2.11.0+cu128
* **CUDA Available:** True (`cuda:0`)
* **GPU Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8,192 MB VRAM)
* **Core Libraries:**
  - `ultralytics`: 8.4.66
  - `streamlit`: 1.58.0
  - `pillow`: 12.3.0
  - `pydantic`: 2.13.5
  - `transformers`: AutoImageProcessor & DepthAnything loaded cleanly in `tf-gpu`
  - `torchvision`: Compatible with PyTorch cu128

---

## 2. Checkpoint & Model Inventory

| Task Domain | Configured Checkpoint Path | Architecture | On-Disk Status | File Size | Primary / Default Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Indoor Detection** | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | YOLO11s (800px) | **EXISTS** | 19,173,786 B | **Primary Recommended** |
| **Indoor Rollback** | `checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt` | YOLO11s (640px) | **EXISTS** | 19,156,442 B | Rollback Candidate |
| **Indoor Pilot** | `checkpoints/indoor_detector_a1_yolo11n_pilot_best.pt` | YOLO11n (640px) | **EXISTS** | 5,446,490 B | Experimental Pilot |
| **Indoor Baseline** | `checkpoints/deepfurniture_yolo11n_finetune_best.pt` | YOLO11n (640px) | **EXISTS** | 5,460,186 B | Baseline Rollback |
| **Exterior Detection** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | YOLO11s (640px) | **EXISTS** | 19,163,034 B | **Production Default** |
| **Exterior Candidate**| `checkpoints/exterior_detector_b5_transfer_best.pt` | YOLO11s (640px) | **EXISTS** | 19,167,770 B | Quarantined (`enabled: false`) |
| **Exterior Candidate**| `checkpoints/exterior_detector_b6_expanded_best.pt` | YOLO11s (640px) | **EXISTS** | 19,167,194 B | Quarantined (`enabled: false`) |
| **Parking Detection** | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | YOLO11s (640px) | **EXISTS** | 19,157,978 B | **Primary Recommended** |
| **Parking Fallback** | `checkpoints/parking_detector_c1_yolo11n_pilot_best.pt` | YOLO11n (640px) | **EXISTS** | 5,446,554 B | Fallback Candidate |
| **Segmentation** | `checkpoints/yolo11n-seg.pt` | YOLO11n-Seg | **EXISTS** | 6,182,636 B | Pretrained MS COCO |
| **Relative Depth** | `checkpoints/depth_anything_v2_small` | ViT-S + DPT Head | **EXISTS** | Directory (~24.8M params) | Primary Geometry |

*Stubbed / Mocked Modules:*
- Scene Classification: `checkpoints/classification_best.pt` (Not present on disk; mock fallback)
- Vision-Language: `checkpoints/vlm_best.pt` (Not present on disk; mock fallback)
- Generative AI Staging: `checkpoints/generator_best.pt` (Not present on disk; mock fallback)

---

## 3. Application Entrypoint & Launch Command

* **Main Entrypoint:** [`app.py`](file:///C:/Users/apaar/Desktop/RoomAI/app.py)
* **Standard Launch Command:**  
  ```powershell
  conda activate tf-gpu
  streamlit run app.py
  ```
* **Documented Headless Health Check Command:**
  ```powershell
  streamlit run app.py --server.port=8503 --server.headless=true
  # HTTP probe: http://localhost:8503/_stcore/health -> 200 OK "ok"
  ```

---

## 4. Test Suite Baseline & Discovery Investigation

### 4.1 Test Count & Status in `tests/`
* **Test Directory:** `tests/`
* **Total Discovered Tests in `tests/`:** **129 tests across 16 test modules** (including 3 new regression tests in `tests/test_ui_integration.py`).
* **Execution Status:** **129 PASSED / 0 FAILED in 40.90 seconds (100% pass rate)**
* **Verified Command:** `$env:USE_TF="0"; C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pytest -q`

### 4.2 Pytest Rootdir Discovery Discrepancy (Finding P2-01) — RESOLVED
* **Symptom:** Executing bare `python -m pytest -q` without `tests/` path argument resulted in an error:
  `ERROR collecting scripts/test_streamlit_startup.py` (`TimeoutExpired: 25s timeout`).
* **Root Cause:**
  1. No `pytest.ini` was present in the repository root defining `testpaths = tests`.
  2. Standalone verification scripts in `scripts/` (e.g., `scripts/test_streamlit_startup.py`, `scripts/test_load_depth_anything.py`, `scripts/test_midas_load.py`) follow the `test_*.py` naming pattern.
  3. These scripts execute executable logic at the module level rather than guarding execution behind `if __name__ == "__main__":`.
* **Remediation Completed:**
  1. Created [`pytest.ini`](file:///C:/Users/apaar/Desktop/RoomAI/pytest.ini) with `testpaths = tests`.
  2. Guarded standalone test scripts with `if __name__ == "__main__":` entrypoint functions.

### 4.3 TensorFlow / NumPy 2.x Environment Dependency Discovery
* **Discovery:** Conda environment `tf-gpu` contains legacy TensorFlow 2.x alongside `numpy==2.2.6`. Importing `transformers.AutoImageProcessor` triggers TensorFlow's C-extension (`_pywrap_bfloat16.TF_bfloat16_type()`), crashing with `TypeError` because TF was compiled for NumPy 1.x.
* **Verified Resolution:** RoomAI is a pure PyTorch application. Setting `$env:USE_TF="0"` causes Transformers to bypass TensorFlow entirely, enabling 100% of tests (129/129) and depth models to execute on GPU without issues.

---

## 5. Frontend Integration Evidence Inspection

All 6 artifacts in [`artifacts/frontend_integration/`](file:///C:/Users/apaar/Desktop/RoomAI/artifacts/frontend_integration/) were inspected:
1. `ui_indoor_annotated_a3.png`: $1600 \times 1200$, 1.89 MB. Valid RGB PIL image with crisp bounding boxes (`table`, `lamp`, `cabinet#shelf`, `curtain`, `decoration`).
2. `ui_exterior_annotated_b2.png`: $768 \times 1024$, 567 KB. Valid RGB PIL image with 70 detected exterior elements (`window`, `door`, `facade`, `molding_deco`).
3. `ui_parking_annotated_c2.png`: $500 \times 375$, 242 KB. Valid RGB image with bounding box on `outdoor_seating`.
4. `ui_segmentation_overlay.png`: $1600 \times 1200$, 1.90 MB. Color overlay for `dining table` and `chair`.
5. `ui_relative_depth_heatmap.png`: $1600 \times 1200$, 175 KB. Inferno-colormapped relative inverse depth map.
6. `frontend_integration_evidence.json`: 1,087 bytes. Structured JSON agreeing with the artifacts.

*Conclusion:* Artifacts are genuine, readable, and properly generated from real GPU inference on real datasets.

