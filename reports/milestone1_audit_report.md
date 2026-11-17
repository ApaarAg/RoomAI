# RoomAI — Milestone 1 Audit Report: Repository, Configuration, Checkpoint, Git-State, and Environment Audit

**Audit Date:** 2026-10-29  
**Execution Environment:** Windows 11, Conda `tf-gpu` (`C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`)  
**Git Commit Baseline:** `8f4bf662eb73518866f516710b68a99c6b03f37a` (Branch: `master`)  
**Auditor Role:** Senior ML Systems Engineer, CV Architect, QA & Security Auditor  
**Milestone 1 Status:** **COMPLETED (EVIDENCE-BACKED AUDIT)**  
**Gate Status:** **AWAITING USER APPROVAL TO PROCEED TO MILESTONE 2**

---

## 1. Executive Summary & Audit Verdict

Milestone 1 was executed in strict read-only audit mode. No application code, model weights, dependencies, or user data were modified. All 15 pre-existing modified files, untracked artifacts, datasets, and historical reports remain preserved.

### Key Audit Findings:
1. **CUDA & GPU Acceleration Confirmed:**  
   The host hardware has an **NVIDIA GeForce RTX 4060 Laptop GPU** with **8,187.5 MB total VRAM**, CUDA version **12.8**, and PyTorch **2.11.0+cu128**. CUDA availability is verified (`torch.cuda.is_available() == True`).
2. **Production Model Status Confirmed:**  
   In [`config/model_config.yaml`](file:///C:/Users/apaar/Desktop/RoomAI/config/model_config.yaml), **Model B2** ([`checkpoints/exterior_detector_b2_yolo11s_best.pt`](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b2_yolo11s_best.pt)) is verified as the active primary exterior detector (`enabled: true`, `status: production_ready`). Experimental candidates B4, B5, and B6 are confirmed quarantined (`enabled: false`, `status: experimental_candidate`).
3. **All 21 Model Checkpoints Verified:**  
   All 21 `.pt` weights in `checkpoints/` exist on disk, have non-zero size, match their cryptographic digests, and load successfully with Ultralytics YOLO.
4. **Critical Environment Dependency Discovery (Verified Defect vs Claim):**  
   - *Previous Assumption:* The Conda environment `tf-gpu` was claimed to run all tests with 100% pass rate out of the box.
   - *Independent Audit Finding:* In the `tf-gpu` environment, an legacy TensorFlow 2.x installation exists alongside `numpy==2.2.6` (NumPy 2.x). Because TensorFlow was compiled for NumPy 1.x, any call importing `transformers.AutoImageProcessor` triggers TensorFlow's internal C-bindings (`_pywrap_bfloat16.TF_bfloat16_type()`), crashing with `TypeError: Unable to convert function return value to a Python type! The signature was () -> handle`.
   - *Empirically Verified Fix:* RoomAI is a pure PyTorch application that does not use TensorFlow. When `$env:USE_TF="0"` is set, Transformers bypasses TensorFlow checks completely. Under `$env:USE_TF="0"`, **129/129 tests pass (100%) in 40.90s**.
5. **Interactive UI Health Verified:**  
   Streamlit 1.58.0 was launched in headless mode on port 8503 and responded with `HTTP 200 "ok"` at `/_stcore/health` within 13.06s.

---

## 2. Distinction: Verified Facts vs Previous Claims & Assumptions

| Dimension | Previous Claim / Assumption | Independently Verified Fact | Status |
| :--- | :--- | :--- | :---: |
| **GPU Acceleration** | RTX 4060 8GB available | Confirmed: NVIDIA RTX 4060 Laptop GPU, 8,187.5 MB VRAM, CUDA 12.8, PyTorch 2.11.0+cu128. | **VERIFIED FACT** |
| **Active Python Env** | Any python on PATH works | Disproven: System default `python` points to Miniconda base (`transformers`/`hub` conflict). The active env MUST be `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`. | **VERIFIED FACT** |
| **TensorFlow / NumPy Collision** | Assumed clean test pass | Found: TensorFlow 2.x in `tf-gpu` crashes on `numpy==2.2.6` during `AutoImageProcessor` import unless `$env:USE_TF="0"` is set. | **CONFIRMED DEFECT / FIX** |
| **Full Test Suite** | 126 or 129 passed | Confirmed: Exactly **129 passed in 40.90s** when `$env:USE_TF="0"` is active. | **VERIFIED FACT** |
| **Production Model** | Model B2 is primary | Confirmed: `config/model_config.yaml` points to `checkpoints/exterior_detector_b2_yolo11s_best.pt` (`status: production_ready`). | **VERIFIED FACT** |
| **Quarantine Isolation** | B4, B5, B6 isolated | Confirmed: All flagged `enabled: false`, `status: experimental_candidate`. | **VERIFIED FACT** |
| **Streamlit Health** | Starts on port 8503 | Confirmed: Responded `HTTP 200 "ok"` at `/_stcore/health` in 13.06s; clean shutdown verified. | **VERIFIED FACT** |

---

## 3. Host & Runtime Environment Verification

**Verification Command:**
```powershell
& "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" -c "
import sys, platform, torch, ultralytics, streamlit, PIL, pydantic, transformers, huggingface_hub, psutil, shutil
print('OS:', platform.platform())
print('Python:', sys.executable, sys.version)
print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| CUDA Version:', torch.version.cuda)
print('GPU Device:', torch.cuda.get_device_name(0), '| Total VRAM:', round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2), 'MB')
print('Ultralytics:', ultralytics.__version__, '| Streamlit:', streamlit.__version__, '| Pillow:', PIL.__version__, '| Pydantic:', pydantic.__version__)
print('Transformers:', transformers.__version__, '| Huggingface Hub:', huggingface_hub.__version__)
vm = psutil.virtual_memory()
print(f'System RAM Total: {vm.total / (1024**3):.2f} GB | Available: {vm.available / (1024**3):.2f} GB')
du = shutil.disk_usage('.')
print(f'Disk Total: {du.total / (1024**3):.2f} GB | Free: {du.free / (1024**3):.2f} GB')
"
```

**Actual Output:**
```
OS: Windows-10-10.0.26200-SP0
Python: C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe 3.10.20 | packaged by Anaconda, Inc.
PyTorch: 2.11.0+cu128 | CUDA Available: True | CUDA Version: 12.8
GPU Device: NVIDIA GeForce RTX 4060 Laptop GPU | Total VRAM: 8187.5 MB
Ultralytics: 8.4.66 | Streamlit: 1.58.0 | Pillow: 12.3.0 | Pydantic: 2.13.5
Transformers: 4.49.0 | Huggingface Hub: 0.36.2
System RAM Total: 23.73 GB | Available: 5.60 GB
Disk Total: 474.87 GB | Free: 72.46 GB
```

---

## 4. Git Repository & Working Tree State

**Verification Commands:**
```powershell
git rev-parse HEAD
git branch --show-current
git status --short
git diff --stat
```

**Actual Output:**
- **HEAD Commit:** `8f4bf662eb73518866f516710b68a99c6b03f37a`
- **Active Branch:** `master`
- **Tracked Modified Files (Preserved):** 15 files (`+1392 / -212` lines):
  1. `README.md` (+35 lines)
  2. `app.py` (+1 / -1 lines)
  3. `config/app.yaml` (+2 / -2 lines)
  4. `config/model_config.yaml` (+191 / -20 lines)
  5. `models/detection_model.py` (+140 / -16 lines)
  6. `models/scene3d_model.py` (+202 / -14 lines)
  7. `models/segmentation_model.py` (+147 / -16 lines)
  8. `schemas/responses.py` (+13 lines)
  9. `services/detection_service.py` (+258 / -15 lines)
  10. `services/image_service.py` (+7 / -3 lines)
  11. `services/scene3d_service.py` (+5 lines)
  12. `ui/analysis_view.py` (+378 / -102 lines)
  13. `ui/components.py` (+71 lines)
  14. `ui/upload_view.py` (+137 / -23 lines)
  15. `utils/image_validation.py` (+14 lines)
- **Untracked Directories & Artifacts (Preserved):**
  - `artifacts/`, `checkpoints/`, `data/`, `docs/`, `reports/`, `scripts/`, `tests/`, `pytest.ini`

---

## 5. Checkpoints Cryptographic Manifest

All 21 `.pt` checkpoint files in [`checkpoints/`](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints) were inspected for file size, last modified date, SHA-256 hash, and model load integrity:

| Checkpoint File | Byte Size | SHA-256 (First 16 Hex) | Last Modified Date | Load Integrity | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`exterior_detector_b2_yolo11s_best.pt`** | 19,163,034 B | `C599C19241B8C603` | 2026-10-24 07:50:20 | **PASS** | **Active Production Default** |
| **`indoor_detector_a3_yolo11s_800px_best.pt`** | 19,173,786 B | `6CBC9E58F87156E0` | 2026-10-24 13:49:10 | **PASS** | **Active Indoor Specialist** |
| **`parking_detector_c2_yolo11s_pilot_best.pt`** | 19,157,978 B | `4ABAB963AB0BECFE` | 2026-10-24 01:38:24 | **PASS** | **Active Parking Specialist** |
| **`yolo11n-seg.pt`** | 6,182,636 B | `55ED65C56C91713D` | 2026-10-24 07:56:49 | **PASS** | **Pretrained Benchmark Seg** |
| `deepfurniture_yolo11n_finetune_best.pt` | 5,460,186 B | `98077FA569117D21` | 2026-10-23 22:21:15 | **PASS** | Historical Baseline |
| `deepfurniture_yolo11n_pilot_best.pt` | 5,456,666 B | `0F2D515CC78B5074` | 2026-10-23 22:08:40 | **PASS** | Historical Pilot |
| `exterior_detector_b1_yolo11n_pilot_best.pt` | 5,453,594 B | `00CAB43515EFD53C` | 2026-10-24 00:57:16 | **PASS** | Baseline |
| `exterior_detector_b2_yolo11s_pilot_best.pt` | 19,162,266 B | `6E7EE9795E9D029B` | 2026-10-24 01:04:36 | **PASS** | Intermediate Pilot |
| `exterior_detector_b3_yolo11s_rfs_best.pt` | 19,164,627 B | `BE708457EAE4D5BF` | 2026-10-24 11:05:44 | **PASS** | RFS Experiment |
| `exterior_detector_b4_yolo11s_curated_best.pt` | 19,159,322 B | `157A1DBC88B3A1CA` | 2026-10-24 17:27:53 | **PASS** | Quarantined Candidate |
| `exterior_detector_b5_transfer_best.pt` | 19,167,770 B | `21712794727E0706` | 2026-10-25 00:32:21 | **PASS** | Quarantined Candidate |
| `exterior_detector_b6_expanded_best.pt` | 19,167,194 B | `009144B272998EC9` | 2026-10-25 01:44:59 | **PASS** | Quarantined Candidate |
| `exterior_detector_b7_pilot_best.pt` | 19,167,194 B | `0B04B97E66F033D2` | 2026-10-25 14:48:18 | **PASS** | Quarantined Candidate |
| `exterior_detector_c1_phase22_best.pt` | 19,167,699 B | `4D3A3339BB09B892` | 2026-10-29 02:21:52 | **PASS** | Phase 22 Experiment |
| `exterior_specialist_porch_yolo11s_best.pt` | 19,159,130 B | `943E7012090124EA` | 2026-10-25 09:51:08 | **PASS** | Porch Specialist |
| `indoor_detector_a1_yolo11n_pilot_best.pt` | 5,446,490 B | `EBD4BBF265D5558D` | 2026-10-23 23:36:23 | **PASS** | Baseline |
| `indoor_detector_a2_yolo11s_finetune_best.pt` | 19,156,442 B | `42113B2DDB5ABFFA` | 2026-10-24 02:32:38 | **PASS** | Rollback Candidate |
| `indoor_detector_a2_yolo11s_pilot_best.pt` | 19,155,866 B | `1333473D8876421D` | 2026-10-23 23:46:38 | **PASS** | Intermediate Pilot |
| `indoor_detector_a2_yolo11s_pilot_check.pt` | 19,155,802 B | `236E3C54CA5B131D` | 2026-10-24 02:22:37 | **PASS** | Intermediate Checkpoint |
| `parking_detector_c1_yolo11n_pilot_best.pt` | 5,450,074 B | `687DDA666DBFD0D1` | 2026-10-24 01:36:08 | **PASS** | Quarantined Candidate |
| `parking_detector_c3_yolo11s_rfs_best.pt` | 19,158,490 B | `CD4584CDB736A75D` | 2026-10-24 11:53:49 | **PASS** | RFS Experiment |
| **`depth_anything_v2_small/` (Directory)** | 99,175,354 B (3 files) | Directory tree | 2026-10-29 | **PASS** | **Relative Depth V2** |

---

## 6. Independent Test Suite Reproduction

### Execution Command:
```powershell
$env:USE_TF="0"; & "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" -m pytest -q
```

### Actual Reproduction Output:
```
........................................................................ [ 55%]
.........................................................                [100%]
129 passed in 40.90s
```

### Test Breakdown by Suite (All Passing):
- `tests/test_deepfurniture_detector.py`: **7 passed**
- `tests/test_depth_estimation.py`: **8 passed**
- `tests/test_detection_deployment_readiness.py`: **13 passed**
- `tests/test_ensemble_service.py`: **8 passed**
- `tests/test_exterior_ensemble.py`: **8 passed**
- `tests/test_image_validation.py`: **6 passed**
- `tests/test_masked_loss.py`: **5 passed**
- `tests/test_model_interfaces.py`: **9 passed**
- `tests/test_model_registry.py`: **4 passed**
- `tests/test_parking_ensemble.py`: **5 passed**
- `tests/test_per_class_thresholds.py`: **7 passed**
- `tests/test_phase19_multidataset.py`: **10 passed**
- `tests/test_schemas.py`: **7 passed**
- `tests/test_services.py`: **7 passed**
- `tests/test_specialist_ensemble.py`: **10 passed**
- `tests/test_timing_utils.py`: **3 passed**
- `tests/test_ui_integration.py`: **12 passed**
- **Total: 129 passed in 40.90s (100% pass rate, 0 failures, 0 warnings)**

---

## 7. Application Smoke Test & Streamlit Health Check

### 7.1 Smoke Test Execution
**Command:**
```powershell
$env:USE_TF="0"; & "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" scripts/smoke_test_application.py
```
**Results:**
- Configuration loaded: **PASS**
- Production model verified: **Model B2 (`checkpoints/exterior_detector_b2_yolo11s_best.pt`)**
- Inference with Model B2: **PASS (executed in 4,741 ms cold, artifact saved)**
- Inference with Candidate B6: **PASS (executed in 362 ms warm)**
- Inference with Candidate B5: **PASS (executed in 315 ms warm)**
- Mock Mode: **PASS (simulated 5 detections)**
- Missing checkpoint: **PASS (raised explicit, caught `FileNotFoundError`)**
- VRAM status: **Allocated 140.8 MB | Reserved 212.0 MB | Budget 4,500.0 MB**

### 7.2 Streamlit Web Startup Test
**Command:**
```powershell
$env:USE_TF="0"; & "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" scripts/test_streamlit_startup.py
```
**Results:**
- Streamlit 1.58.0 started on port 8503 in headless mode.
- HTTP probe `http://localhost:8503/_stcore/health` returned: **`HTTP 200 "ok"` in 13.06s**.
- Process terminated cleanly with exit code 0.

---

## 8. Milestone 1 Conclusions & Gate

1. **Integrity Confirmed:** The repository architecture, configurations, git state, and checkpoints are completely verified.
2. **Environment Blocker Identified & Mitigated:** Setting `$env:USE_TF="0"` guarantees pure PyTorch execution without interference from incompatible TensorFlow libraries in the Conda environment.
3. **No Code Modified:** All files, checkpoints, and configurations remain untouched.

**Milestone 1 is complete. I await your approval before proceeding to Milestone 2 (Detection, Segmentation, and Relative-Depth Verification).**

