# RoomAI Exterior Detector B5 — Pre-Training & System Audit Report

**Audit Date:** 2026-10-24  
**Auditor:** Senior Computer Vision Engineer & Object Detection Researcher  
**Workspace:** `C:\Users\apaar\Desktop\RoomAI`  
**Phase:** Phase 1 — Audit the Existing System  

---

## 1. Executive Summary

This audit establishes the pre-training foundation for training **Model B5**, the next-generation exterior detector for RoomAI. B5 aims to improve detection recall and precision across all 10 canonical architectural classes via transfer learning with an expanded, curated dataset.

All non-negotiable safeguards have been verified:
1. **Existing Checkpoints Preserved Byte-for-Byte:** B2 (production), B3 (rollback), and B4 (experimental) hashes match their expected values exactly.
2. **Production Integrity Maintained:** `config/model_config.yaml` remains actively configured with Model B2 (`exterior_detector_b2_yolo11s_best.pt`) in `production_ready` status.
3. **Split Protection:** Validation and test splits are strictly cordoned off from training. No test data will be used for threshold tuning or model selection.

---

## 2. Hardware, Environment, & Compute Budget

| Parameter | Measured Specification | Evaluation & Suitability |
|---|---|---|
| **Operating System** | Windows 11 Home (64-bit AMD64) | Fully compatible with CUDA and PyTorch |
| **Python Environment** | Conda `tf-gpu` (Python 3.10.20) | Active virtual environment with all ML dependencies |
| **PyTorch & CUDA** | PyTorch 2.11.0+cu128 (CUDA 12.8) | Hardware accelerated with CUDA stream synchronization |
| **GPU Hardware** | NVIDIA GeForce RTX 4060 Laptop GPU | 8.00 GB Dedicated VRAM |
| **System Memory** | 23.73 GB Total System RAM (5.06 GB Available) | Sufficient for multithreaded data loading |
| **Disk Storage** | 474.87 GB Total Drive, **32.04 GB Free** | Ample for targeted dataset expansion (~2–4 GB) |

### Memory & VRAM Guidelines for B5 Training
- To operate safely within the 8.00 GB VRAM envelope on RTX 4060:
  - Architecture: YOLO11s (`yolo11s.pt`) or YOLO11m with conservative batch sizing.
  - Image size: $640\times640$ resolution.
  - Batch size: $4$ to $8$ (with gradient accumulation if required).
  - Mixed Precision: AMP (`half=True`) enabled and verified stable.

---

## 3. Checkpoint Lineage & Integrity Verification

Every existing checkpoint was independently hashed using MD5:

| Checkpoint File | Role | Expected MD5 | Measured MD5 | Status | File Size |
|---|---|---|---|---|---:|
| `exterior_detector_b2_yolo11s_best.pt` | Active Production Exterior Detector | `14a931dd0adabc16cbd7b28ed1d17f66` | `14a931dd0adabc16cbd7b28ed1d17f66` | **VERIFIED** | 19,163,034 bytes |
| `exterior_detector_b3_yolo11s_rfs_best.pt` | Rollback Exterior Detector | `2c44291fa642aba3893db001520deba0` | `2c44291fa642aba3893db001520deba0` | **VERIFIED** | 19,164,627 bytes |
| `exterior_detector_b4_yolo11s_curated_best.pt` | Experimental Exterior Detector | `d2710fba760713e68ea306e029ab651c` | `d2710fba760713e68ea306e029ab651c` | **VERIFIED** | 19,159,322 bytes |

- **Production Configuration:** `config/model_config.yaml` lines 76–95 point directly to B2.
- **Rollback Path:** If B5 fails any verification gate, B2 remains production default and B3 remains rollback default.

---

## 4. Canonical 10-Class Taxonomy & Existing Datasets

### 4.1 Class Taxonomy
RoomAI Group B exterior detection operates on a strict 10-class canonical taxonomy:

```
Class 0: building        Class 5: balcony
Class 1: facade          Class 6: porch
Class 2: window          Class 7: pillar_column
Class 3: door            Class 8: shutter_blind
Class 4: roof_cornice    Class 9: molding_deco
```

### 4.2 Existing Exterior Datasets
The repository currently houses two generations of exterior datasets:

1. **`data/exterior_canonical/` (Phase 3 Baseline):**
   - 726 total images: 458 train, 134 val, 134 test.
   - Sources: CMP Facade (606 images) + Open Images V7 canonical exterior subset (120 images).
2. **`data/exterior_v2_curated/` (Phase 9–11 Experimental):**
   - 876 total images: 568 train, 154 val, 154 test.
   - Sources: 726 canonical images + 150 Open Images curated porch scenes (110 train, 20 val, 20 test).

### 4.3 Detailed Class Distribution in `exterior_v2_curated`

| Class ID | Class Name | Train Instances (568 imgs) | Val Instances (154 imgs) | Test Instances (154 imgs) | Total Instances (876 imgs) | Imbalance Ratio (vs Porch) |
|---|---|---:|---:|---:|---:|---:|
| 0 | **building** | 204 | 40 | 40 | 284 | 1.5 : 1 |
| 1 | **facade** | 2,411 | 495 | 579 | 3,485 | 18.8 : 1 |
| 2 | **window** | 12,677 | 3,845 | 4,090 | 20,612 | 111.4 : 1 |
| 3 | **door** | 1,379 | 372 | 432 | 2,183 | 11.8 : 1 |
| 4 | **roof_cornice** | 2,797 | 445 | 666 | 3,908 | 21.1 : 1 |
| 5 | **balcony** | 1,091 | 345 | 393 | 1,829 | 9.9 : 1 |
| 6 | **porch** | **134** | **24** | **28** | **186** | **1.0 : 1 (Severely Rare)** |
| 7 | **pillar_column** | 2,270 | 494 | 896 | 3,660 | 19.8 : 1 |
| 8 | **shutter_blind** | 2,487 | 901 | 664 | 4,052 | 21.9 : 1 |
| 9 | **molding_deco** | 7,070 | 1,543 | 2,038 | 10,651 | 57.6 : 1 |
| **Total** | — | **32,520** | **8,504** | **9,826** | **50,850** | — |

---

## 5. Provenance Audit & Cross-Split Contamination Risks

### 5.1 Historical Finding from Phase 10 Audit
During the Phase 9 data curation, 82 images originally in Open Images V7 official test split and 28 from validation were assigned to RoomAI training data (`data/exterior_v2_curated/images/train`).
- **Impact:** While internal RoomAI splits (`exterior_v2_curated` train vs val vs test) have zero MD5 or perceptual hash overlap, RoomAI models cannot be evaluated against the public Open Images test benchmark without training contamination.

### 5.2 Strict Rules for B5 Dataset Expansion
To ensure complete scientific integrity for B5:
1. **Source Separation:** Any new Open Images images added to B5 training MUST be drawn strictly from the official **Open Images V7 `train` split**. Under no circumstances will Open Images validation or test images be used for B5 training.
2. **Evaluation Split Isolation:** The existing RoomAI evaluation splits (`exterior_v2_curated/val` and `exterior_v2_curated/test`) must remain preserved and untouched.
3. **Deduplication:** Every candidate image must be screened against existing canonical hashes (MD5 and 64-bit DCT pHash with Hamming distance $< 5$) to prevent near-duplicate leakage.

---

## 6. Pre-Training Gap Analysis

| Class | Current B4 Validation F1 | Root Cause of Errors | B5 Strategy |
|---|---:|---|---|
| **porch** | 0.3922 (calibrated) / 0.3077 (uniform) | Severe training data scarcity (only 134 train instances) | Download hundreds of new porch instances from Open Images V7 train split. |
| **facade** | 0.3876 | Visual ambiguity between full building and individual facade surfaces | Add diverse exterior scene envelopes with distinct building and facade boundaries. |
| **pillar_column** | 0.3145 | Vertical aspect ratios confused with door frames and window mullions | Enhance training variety with freestanding and structural columns. |
| **door** | 0.4714 | Small doors obscured by porches or shadows | Increase exterior residential door examples. |
| **building** | 0.5185 | Wide scale variance (skyscrapers to single-family bungalows) | Expand building envelopes across different architectural styles. |

---

## 7. Audit Conclusion & Next Steps

The system environment, hardware, storage, and baseline checkpoints are verified and ready.
- **Proceed to Phase 2:** Curate a larger public dataset from Open Images V7 Train split and complementary public architectural sources.
- **Save Backup:** `config/model_config.yaml` backed up to `config/model_config.yaml.backup_b5`.
