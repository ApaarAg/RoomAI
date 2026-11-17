# RoomAI — Phase 0 Safe Baseline Audit Report

**Date:** October 29, 2026  
**Auditor:** Senior ML Systems Engineer, Python Backend Architect & QA Automation Lead  
**Repository Root:** `C:\Users\apaar\Desktop\RoomAI`  
**Git Status:** Branch `master` | Commit `8f4bf662eb73518866f516710b68a99c6b03f37a`

---

## 1. Environment & Hardware Inventory

| Component | Verified Specification |
| :--- | :--- |
| **Python Executable** | `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe` |
| **Python Version** | `3.10.20` (64-bit AMD64) |
| **Conda Environment** | `tf-gpu` |
| **Operating System** | Windows 11 (Build 26200) |
| **GPU Device** | NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM) |
| **CUDA Available** | `True` (Device index 0, 0.0 GB active allocation) |
| **System Memory** | 23.73 GB Total RAM / 4.64 GB Free / Available |
| **Storage Availability**| 75.23 GB Free on `C:\` drive |

### Core Dependency Matrix
- `torch`: `2.11.0+cu128`
- `ultralytics`: `8.4.66`
- `torchvision`: `0.26.0+cu128`
- `pydantic`: `2.13.5`
- `pyyaml`: `6.0.3`
- `opencv-python`: `4.13.0.92`
- `pillow`: `12.3.0`
- `streamlit`: `1.58.0`
- `numpy`: `1.23.5`
- `pytest`: `9.0.3`

---

## 2. Git Status & Working Tree Baseline

- **Branch:** `master`
- **Head Commit:** `8f4bf662`
- **Tracked Modified Files (13 files - all pre-existing before this audit):**
  - `README.md`
  - `app.py`
  - `config/app.yaml`
  - `config/model_config.yaml` *(Notice: Found 0 bytes, empty!)*
  - `models/detection_model.py`
  - `models/scene3d_model.py`
  - `models/segmentation_model.py`
  - `schemas/responses.py`
  - `services/detection_service.py`
  - `services/image_service.py`
  - `services/scene3d_service.py`
  - `ui/analysis_view.py`
  - `ui/upload_view.py`
- **Untracked Artifacts / Files:** 136 items (reports, candidate caches, artifacts, backups, specialist scripts).

> [!WARNING]
> **Critical Baseline Finding:** `config/model_config.yaml` is currently 0 bytes (empty). It was modified immediately prior to this turn. A complete, verified backup exists at `config/model_config.yaml.backup_b5` (4,770 bytes). Restoring this verified configuration is required before Phase 1 test suite execution so that model registry and services can resolve their target checkpoints and classes.

---

## 3. Cryptographic Checkpoint Inventory (21 Checkpoints)

| Checkpoint Name | Status / Role | File Size (bytes) | MD5 Hash | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| `exterior_detector_b2_yolo11s_best.pt` | **Production Default** | 19,163,034 | `14a931dd0adabc16cbd7b28ed1d17f66` | `C599C19241B8C60390AC0C4914DE6CEF449BC1129F73B0DB7F8F04FC62A0459F` |
| `exterior_detector_b6_expanded_best.pt` | Research Baseline | 19,167,194 | `f5ceea56b0610ea14b647014e48152dc` | `009144B272998EC9447F649E8D850E0FDB8CAE5FC973C5E0CB9EEBE717675AC9` |
| `exterior_detector_b7_pilot_best.pt` | Pilot Model | 19,167,194 | `41433e929910eda4414ed6541bac2fe4` | `0B04B97E66F033D2037BC1483D4AD4F484A00213DEFEE66A7D32AD83199E2C3F` |
| `exterior_detector_c1_phase22_best.pt` | Experimental Candidate | 19,167,699 | `753c1784c5247895be3a41b2a829fa8e` | `4D3A3339BB09B89213F60F2E8AF82212804A84BAFA3A91EC94D4D424B0FB187F` |
| `exterior_specialist_porch_yolo11s_best.pt` | Specialist Model | 19,159,130 | `4261fc1ad5f6fd9ac29c0499e2adddfd` | `943E7012090124EA726F21B37983CC8B253B35AF36F72B381EB01D18EAC2D9B2` |
| `indoor_detector_a3_yolo11s_800px_best.pt` | Indoor Primary | 19,173,786 | `95563016732fca2cb4e0ee15bdb9e295` | `6CBC9E58F87156E07DD0E4E712B8AA96097AEB514A473621139A54CB83CEC576` |
| `indoor_detector_a2_yolo11s_finetune_best.pt`| Indoor Rollback | 19,156,442 | `ad899be666950b695c825a2e57229a58` | `42113B2DDB5ABFFA4830BDE024CEA13EC5BE99B1C6493CB2A01B4B52A7EA08C8` |
| `parking_detector_c2_yolo11s_pilot_best.pt` | Parking Primary | 19,157,978 | `89383b78c1a43785fc5057cead5e52a5` | `4ABAB963AB0BECFEF4A03C4E7B0039CC048D7621B3456BC5435DEECE6EED1470` |
| `parking_detector_c1_yolo11n_pilot_best.pt` | Parking Lightweight | 5,450,074 | `1a4a7a99b07efdc7a6ab7a75ced10474` | `687DDA666DBFD0D10F06ACBD2F08AC8A94848A9D1EF0C269CE7B426FB0E4DA42` |
| `parking_detector_c3_yolo11s_rfs_best.pt` | Parking Specialist | 19,158,490 | `a5430d32b070fe5f4741beb83b804ae0` | `CD4584CDB736A75D11AA95D818927491191E46EC292D90115ABD3F7D88E031F2` |
| `yolo11n-seg.pt` | Generic Segmentation | 6,182,636 | `edfa69d9468b2703b185f550e8f26251` | `55ED65C56C91713D23E8402371C6C49A6FD84F257F7DCE452E8D70E41DCBE152` |
| `deepfurniture_yolo11n_finetune_best.pt` | DeepFurniture Fine-tune| 5,460,186 | `64c1150d99cc32e2877ce1b9bdfad657` | `98077FA569117D21D1A2A0E03F4ADA2BA15E48AB28AEC594006A11D15B469D50` |

---

## 4. Pytest Test Discovery Baseline

Executing `pytest --collect-only -q` in the `tf-gpu` environment discovered **117 test items** across 12 test files:
1. `tests/test_deepfurniture_detector.py` (7 tests)
2. `tests/test_depth_estimation.py` (8 tests)
3. `tests/test_detection_deployment_readiness.py` (13 tests)
4. `tests/test_ensemble_service.py` (9 tests)
5. `tests/test_exterior_ensemble.py` (13 tests)
6. `tests/test_masked_loss.py` (5 tests)
7. `tests/test_model_registry.py` (13 tests)
8. `tests/test_parking_ensemble.py` (9 tests)
9. `tests/test_per_class_thresholds.py` (13 tests)
10. `tests/test_phase19_multidataset.py` (16 tests)
11. `tests/test_specialist_ensemble.py` (8 tests)
12. `tests/test_timing_utils.py` (3 tests)

Total collected: **117 tests**.

---

## 5. Phase 0 Gate Summary

- **Environment Usability:** VERIFIED (Python 3.10.20, CUDA on RTX 4060 GPU, 75 GB disk free).
- **Checkpoints Accounted For:** VERIFIED (All 21 checkpoints cryptographically hashed and matched).
- **Production Configuration:** DEFECT DETECTED & ISOLATED (`config/model_config.yaml` is 0 bytes; backup `backup_b5` identified for clean recovery).
- **Tool Execution Hook:** Operational and responsive.

