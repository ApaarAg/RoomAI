# RoomAI — Phase 14 Stage A: Model B5 Pre-Retraining Audit & Weak-Class Diagnosis

**Date:** October 25, 2026  
**Auditor:** Senior Computer Vision Engineer & ML Research Mentor  
**Target Model:** Model B5 (`exterior_detector_b5_transfer_best.pt`, MD5: `e12b4181fee244a3569dfa782ba3c3c3`)  
**Scope:** Dataset Integrity, Evaluation Protocol Consistency, Checkpoint Lineage, and Weak-Class Error Diagnosis  
**Structured Audit Results:** [exterior_b5_audit_results.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/exterior_b5_audit_results.json)  

---

## 1. Executive Summary

Prior to launching dataset expansion and training for candidate **Model B6**, a comprehensive Stage A audit of Model B5 was executed.

### Key Audit Findings:
1. **Checkpoint Integrity:** All baseline checkpoints (B2, B3, B4, B5) were verified byte-for-byte against their recorded MD5 hashes.
2. **Dataset Composition & Split Separation:** The B5 dataset consists of 1,211 total images (Train: 903, Val: 154, Test: 154) comprising 52,280 bounding boxes. Cross-split MD5 collision checks proved **strict zero leakage** across train, validation, and test sets.
3. **Reproduced Validation Metrics:** Validation metrics reproduced on the 154 validation images:
   - **mAP50:** `0.4869`
   - **mAP50-95:** `0.2954`
   - **Macro-F1 (conf=0.25):** `0.4960`
   - **Micro-F1 (conf=0.25):** `0.6371`
4. **Evaluation Code Bug Fixed:** During the audit, an index accumulation defect in the per-class confusion loop was discovered and resolved (`matched_gt_indices` scope was corrected to per-class), ensuring mathematical consistency across all metric calculations.
5. **Target Weak Classes Identified:** Deep failure analysis isolated the specific failure modes for `porch`, `pillar_column`, `facade`, and `shutter_blind`.

---

## 2. Checkpoint Lineage & Integrity Verification

| Checkpoint Name | Checkpoint Path | Expected MD5 | Actual MD5 | File Size (Bytes) | Status |
|---|---|---|---|:---:|:---:|
| **B2 (Production)** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | `14a931dd0adabc16cbd7b28ed1d17f66` | `14a931dd0adabc16cbd7b28ed1d17f66` | 19,163,034 | **VERIFIED INTACT** |
| **B3 (Rollback)** | `checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt` | `2c44291fa642aba3893db001520deba0` | `2c44291fa642aba3893db001520deba0` | 19,164,627 | **VERIFIED INTACT** |
| **B4 (Experimental)** | `checkpoints/exterior_detector_b4_yolo11s_curated_best.pt` | `d2710fba760713e68ea306e029ab651c` | `d2710fba760713e68ea306e029ab651c` | 19,159,322 | **VERIFIED INTACT** |
| **B5 (Transfer Base)** | `checkpoints/exterior_detector_b5_transfer_best.pt` | `e12b4181fee244a3569dfa782ba3c3c3` | `e12b4181fee244a3569dfa782ba3c3c3` | 19,167,770 | **VERIFIED INTACT** |

### Pretrained Lineage & Checkpoint Metadata:
- **Base Architecture:** YOLO11s (`yolo11s.pt` COCO pretrained)
- **Trained Image Size:** $640 \times 640$ pixels
- **Batch Size:** 8
- **Optimizer:** AdamW (`auto` mode in Ultralytics)
- **Initial Learning Rate:** $0.01$ (scaled by optimizer)

---

## 3. Dataset Composition & Split Representation

The B5 dataset combines the CMP Facade dataset with 335 carefully curated Open Images V7 scenes.

| Canonical Class | Train Boxes | Train Images | Val Boxes | Val Images | Test Boxes | Test Images | Total Instances | Imbalance Ratio (vs Min) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `building` | 491 | 275 | 40 | 29 | 40 | 26 | 571 | $4.3\times$ |
| `facade` | 2,411 | 378 | 495 | 113 | 579 | 116 | 3,485 | $26.0\times$ |
| `window` | 13,525 | 635 | 3,845 | 135 | 4,090 | 138 | 21,460 | $160.1\times$ |
| `door` | 1,607 | 483 | 372 | 92 | 432 | 100 | 2,411 | $18.0\times$ |
| `roof_cornice` | 2,797 | 277 | 445 | 70 | 666 | 73 | 3,908 | $29.2\times$ |
| `balcony` | 1,091 | 216 | 345 | 74 | 393 | 76 | 1,829 | $13.6\times$ |
| `porch` | **134** | **111** | **24** | **22** | **28** | **24** | **186** | **$1.0\times$ (MIN)** |
| `pillar_column` | 2,270 | 237 | 494 | 60 | 896 | 62 | 3,660 | $27.3\times$ |
| `shutter_blind` | 2,554 | 184 | 901 | 67 | 664 | 60 | 4,119 | $30.7\times$ |
| `molding_deco` | 7,070 | 372 | 1,543 | 103 | 2,038 | 107 | 10,651 | $79.5\times$ |
| **All Classes** | **33,950** | **903** | **8,504** | **154** | **9,826** | **154** | **52,280** | — |

**Leakage Audit:** Pairwise MD5 collision analysis confirmed **zero overlap** between any split ($0$ train-val collisions, $0$ train-test collisions, $0$ val-test collisions).

---

## 4. Reproduced Baseline Validation Metrics (Model B5)

Evaluated on 154 validation images ($640$px resolution, conf threshold $0.25$, IoU threshold $0.50$):

| Class | GT Count | TP | FP | FN | Precision | Recall | F1 Score | AP50 | AP50-95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `building` | 40 | 18 | 19 | 22 | 0.4865 | 0.4500 | 0.4675 | 0.4582 | 0.3040 |
| `facade` | 495 | 150 | 184 | 345 | 0.4491 | 0.3030 | 0.3619 | 0.3314 | 0.2120 |
| `window` | 3,845 | 2,962 | 1,086 | 883 | 0.7317 | 0.7704 | 0.7505 | 0.7891 | 0.5410 |
| `door` | 372 | 191 | 220 | 181 | 0.4647 | 0.5134 | 0.4879 | 0.4626 | 0.2510 |
| `roof_cornice` | 445 | 315 | 243 | 130 | 0.5645 | 0.7079 | 0.6281 | 0.6753 | 0.3640 |
| `balcony` | 345 | 229 | 130 | 116 | 0.6379 | 0.6638 | 0.6506 | 0.6565 | 0.4060 |
| `porch` | 24 | 1 | 4 | 23 | 0.2000 | 0.0417 | 0.0690 | 0.0719 | 0.0346 |
| `pillar_column` | 494 | 173 | 375 | 321 | 0.3157 | 0.3502 | 0.3321 | 0.2662 | 0.1330 |
| `shutter_blind` | 901 | 496 | 143 | 405 | 0.7762 | 0.5505 | 0.6442 | 0.6360 | 0.4310 |
| `molding_deco` | 1,543 | 891 | 700 | 652 | 0.5600 | 0.5774 | 0.5686 | 0.5216 | 0.2830 |
| **All Classes** | **8,504** | **5,426** | **3,104** | **3,078** | **0.6361** | **0.6381** | **0.4960 (Macro)** | **0.4869** | **0.2954** |

---

## 5. Diagnostic Error Analysis for 4 Target Weak Classes

### A. `porch` (F1 = 0.0690 on val, 0.1667 on test)
- **Data Volume:** Only 134 boxes in 111 training images ($0.39\%$ of all training annotations).
- **False Negatives:** 23 out of 24 validation porches are completely missed (Recall: $4.17\%$).
- **Root Cause:** Severe acute sample sparsity. The detector lacks sufficient visual variations of real residential porches, steps, overhangs, and porticos across diverse lighting conditions.
- **Action for B6:** Curate additional verified porch and portico scenes from Open Images V7.

### B. `pillar_column` (F1 = 0.3321 on val, 0.3199 on test)
- **False Positives:** 375 FPs. Major confusion sources:
  - 238 on background (vertical building edges, drainpipes, downspouts, door jambs)
  - 40 on `molding_deco` (vertical pilasters and ornamental wall reliefs)
  - 38 on `roof_cornice`
- **False Negatives:** 321 FNs (27 misclassified as cornice, 20 as molding).
- **Root Cause:** High visual similarity between flat decorative pilasters (labeled as molding in CMP) and structural columns. The model struggles with columns that have low relief or blend into the wall texture.
- **Action for B6:** Curate freestanding columns and colonnades with prominent capital/base features and clear 3D separation from the facade.

### C. `facade` (F1 = 0.3619 on val, 0.3313 on test)
- **False Negatives:** 345 FNs out of 495 ground truth instances (Recall: $30.30\%$).
- **False Positives:** 184 FPs (107 on background, 15 on molding, 12 on window).
- **Root Cause:** Hierarchy and scale ambiguity. `facade` is a macro-bounding box enclosing windows, doors, and cornices. In complex multi-story structures or buildings with recessed wings, the boundary between `building` (the entire structural envelope) and `facade` (individual exterior walls) is ambiguous.
- **Action for B6:** Ingest high-resolution images with clearly bounded single-plane facades and portico-framed building faces.

### D. `shutter_blind` (F1 = 0.6442 on val, 0.3061 on test)
- **Distribution Shift:** Performs moderately well on val ($0.6442$) but regressed sharply on the locked test set ($0.3061$).
- **False Negatives:** 405 on val (64 misclassified directly as `window`).
- **False Positives:** 143 on val (49 overlapping with `window`).
- **Root Cause:** Visual co-occurrence. Exterior shutters are mounted directly adjacent to windows; dark or slatted shutters are frequently grouped into the window bounding box or vice versa.
- **Action for B6:** Add targeted training examples of distinct louvered, paneled, and colored shutters flanking windows to train sharper spatial discrimination between the window frame and the shutter panel.

---

## 6. Action Plan for Model B6 Data Expansion

1. **Source Candidate Datasets:**
   - Query Open Images V7 validation/train pools for:
     - `/m/04brg2` (`Porch`)
     - `/m/02dgv` (`Door`) / `/m/03jm5` (`House`) / Architectural elements
     - `/m/031n1` (`Window blind`) / exterior shutter subsets
     - Verified classical architectural columns and porticos
2. **Strict Quality Gates:**
   - Enforce MD5 and 64-bit DCT perceptual hash ($d < 5$) checks against all existing train, val, and locked test images.
   - Reject any candidates overlapping with validation or locked test sets.
3. **Candidate Model Training (Model B6):**
   - Initialize from B5 weights (`exterior_detector_b5_transfer_best.pt`) to retain learned transfer features while fine-tuning on the expanded weak-class dataset.
   - Maintain locked validation split for early stopping and model selection.
   - Leave the locked test set untouched.
