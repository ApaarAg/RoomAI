# RoomAI Phase 3 — Exterior Detector Data & Error Analysis

**Author:** Senior Object Detection Research Engineer  
**Date:** October 2026  
**Target:** Group B Exterior & Building Architecture Detector  
**Baseline Model:** Model B2 (`checkpoints/exterior_detector_b2_yolo11s_best.pt`)  
**Baseline Checkpoint MD5:** `14a931dd0adabc16cbd7b28ed1d17f66` (19,163,034 bytes)  
**Dataset:** Canonical 10-Class Exterior Dataset (`data/exterior_canonical/exterior_canonical.yaml`)

---

## 1. Executive Summary

This report establishes the baseline evaluation, data distribution audit, and failure mode analysis for the RoomAI Group B Exterior Detector. The baseline checkpoint (`exterior_detector_b2_yolo11s_best.pt`) achieves **0.3961 mAP50** and **0.2427 mAP50–95** on the 134-image held-out test split, with precision at **0.4991**, recall at **0.3990**, and F1 at **0.4435**.

While high-frequency, well-demarcated classes such as `window` (mAP50 0.8029) and `roof_cornice` (mAP50 0.6735) perform reliably, several key architectural classes suffer from acute recall deficits (`shutter_blind` R=0.1943, `pillar_column` R=0.2467, `facade` R=0.2401) and catastrophic failure on rare classes (`porch` mAP50 0.0000, R=0.0000).

Our deep data audit reveals that these failures stem from three distinct structural causes:
1. **Severe Data Starvation & Label Artifacts (`porch`):** Only 6 total bounding boxes exist across all 49,890 annotations in the 726-image dataset (imbalance ratio 3,332.5:1). Critically, the training split contains **only 1 single image with 1 porch annotation**. In the test split, one of the two porch instances is an annotation artifact covering 100% of the image (`[0.5, 0.5, 1.0, 1.0]`).
2. **Extreme Geometric Aspect Ratios & Tiny Scale (`pillar_column`, `molding_deco`):** Pillars have a median area of only **0.148%** of the image (slender vertical strips), frequently downsampled below detection thresholds at 640×640.
3. **Hierarchical Class Confusion & Occlusion (`building` vs. `facade`, `shutter_blind` vs. `window`):** Annotations inconsistently label full buildings versus facade planes, while shutters are visually co-located with and overshadowed by high-contrast window frames.

---

## 2. Baseline Reproduction & Verification

The baseline model was verified and re-evaluated strictly using the frozen checkpoint without modifications:

| Metric | Validation Split (134 images) | Held-Out Test Split (134 images) | Delta (Test vs Val) |
| :--- | :---: | :---: | :---: |
| **mAP50** | **0.4484** | **0.3961** | -0.0523 |
| **mAP50–95** | **0.2689** | **0.2427** | -0.0262 |
| **Precision** | **0.5473** | **0.4991** | -0.0482 |
| **Recall** | **0.4268** | **0.3990** | -0.0278 |
| **F1 Score** | **0.4796** | **0.4435** | -0.0361 |

### Per-Class Test Performance (Baseline B2)

| Class ID | Class Name | Test Instances | Precision | Recall | F1 Score | mAP50 | mAP50–95 | Diagnosis |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `building` | 24 | 0.5045 | 0.3750 | 0.4302 | 0.4280 | 0.2914 | Recall limited; confusion with `facade` |
| **1** | `facade` | 579 | 0.4608 | 0.2401 | 0.3157 | 0.2703 | 0.1749 | High FN rate; bounding box overlap |
| **2** | `window` | 4,017 | 0.7592 | 0.7872 | 0.7729 | 0.8029 | 0.5707 | Strong performance; dominant class |
| **3** | `door` | 428 | 0.5455 | 0.4346 | 0.4838 | 0.4134 | 0.2584 | Moderate; shadow & glass confusion |
| **4** | `roof_cornice` | 666 | 0.6222 | 0.6862 | 0.6526 | 0.6735 | 0.3711 | Good horizontal feature detection |
| **5** | `balcony` | 393 | 0.5096 | 0.5929 | 0.5481 | 0.5301 | 0.3059 | Solid localization |
| **6** | `porch` | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **Data-starved (0 detections)** |
| **7** | `pillar_column` | 896 | 0.3847 | 0.2467 | 0.3006 | 0.1906 | 0.0929 | Severe recall loss; sub-pixel width |
| **8** | `shutter_blind` | 664 | 0.6565 | 0.1943 | 0.2998 | 0.2299 | 0.1483 | Precision intact; massive FN misses |
| **9** | `molding_deco` | 2,038 | 0.5480 | 0.4331 | 0.4838 | 0.4217 | 0.2130 | Clutter & small boundary ambiguities |
| **ALL** | **Overall** | **9,707** | **0.4991** | **0.3990** | **0.4435** | **0.3961** | **0.2427** | Baseline reference |

---

## 3. Dataset Distribution & Geometry Audit

The canonical exterior dataset contains 726 total images (458 train, 134 validation, 134 test) and 49,890 bounding boxes across 10 classes.

### Per-Class Instance & Image Distribution Across Splits

| Class Name | Train Boxes (Images) | Val Boxes (Images) | Test Boxes (Images) | Total Boxes | Median Area (% img) | Median Aspect Ratio (W/H) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `building` | 101 (63) | 22 (17) | 24 (18) | 147 | 50.88% | 1.00 |
| `facade` | 2,411 (378) | 495 (113) | 579 (113) | 3,485 | 0.51% | 0.67 |
| `window` | 12,224 (405) | 3,754 (121) | 4,017 (120) | 19,995 | 0.32% | 0.63 |
| `door` | 1,362 (318) | 367 (87) | 428 (95) | 2,157 | 0.79% | 0.78 |
| `roof_cornice` | 2,797 (277) | 445 (70) | 666 (83) | 3,908 | 0.39% | 2.47 |
| `balcony` | 1,091 (216) | 345 (74) | 393 (70) | 1,829 | 0.89% | 2.21 |
| `porch` | **1 (1)** | **3 (2)** | **2 (2)** | **6** | 6.96% | 1.64 |
| `pillar_column` | 2,270 (237) | 494 (60) | 896 (88) | 3,660 | 0.15% | 0.93 |
| `shutter_blind` | 2,487 (145) | 901 (67) | 664 (34) | 4,052 | 0.25% | 0.40 |
| `molding_deco` | 7,070 (372) | 1,543 (103) | 2,038 (109) | 10,651 | 0.29% | 2.58 |
| **Total** | **31,814 (458)** | **8,369 (134)** | **9,707 (134)** | **49,890** | — | — |

---

## 4. In-Depth Root Cause Analysis of Low-Performing Classes

### 4.1. `porch` — The 6-Instance Starvation Problem

The deep audit conclusively resolves the question of the 6 reported porch instances:
- **Total Instances:** Exactly **6 bounding boxes in 5 distinct images** across the entire repository.
- **Split Breakdown:**
  - **Train:** **1 image** (`oi_ext_train_0012.txt`), **1 box**: `[0.8000, 0.7021, 0.2750, 0.0833]`, normalized area = 2.29%, aspect ratio 3.30.
  - **Val:** **2 images** (`oi_ext_val_0084.txt`: 1 box, area 18.36%; `oi_ext_val_0085.txt`: 2 boxes, areas 6.13% and 2.64%) = 3 boxes.
  - **Test:** **2 images** (`oi_ext_test_0106.txt`: 1 box; `oi_ext_test_0113.txt`: 1 box) = 2 boxes.
- **Root Cause & Label Quality Findings:**
  1. **Data Starvation:** A deep CNN/YOLO model cannot learn a multi-view visual concept of "porch" from 1 single training image containing 1 small strip box.
  2. **Severe Annotation Artifact in Test Set:** In `oi_ext_test_0106.txt`, the porch annotation is `0.5 0.5 1.0 1.0` (100% full-image bounding box). In reality, the image depicts a house with a modest front entryway, but the Open Images V7 labeler applied the box to the entire photograph. Any valid, localized prediction would have near-zero IoU against a 1.0×1.0 ground truth box.
  3. **Visual Verification:** Visualizations generated in `artifacts/exterior_error_analysis/porch_gt_vs_pred_*.jpg` confirm that the baseline detector outputs zero candidate boxes for porch because its confidence never exceeds the background suppression threshold.
  4. **Engineering Recommendation:** **Porch cannot be fixed by model augmentation alone.** Synthesizing or oversampling 1 image 15 times can prevent gradient vanishing for the porch head, but true generalization requires ingesting 150–300 verified exterior porch images.

### 4.2. `pillar_column` — Tiny Spatial Footprint (Median Area 0.148%)

- **Observations:** Test precision is 0.3847, but recall is only 0.2467 (mAP50 = 0.1906).
- **Geometric Analysis:**
  - `pillar_column` has the smallest median area of all classes: **0.148%** of the image area.
  - At 640×640 input resolution, an area of 0.148% corresponds to a box of approximately 12×50 pixels. Slender vertical columns often have a width of only 4–8 pixels.
  - In feature pyramid networks (P3/P4/P5 strides 8, 16, 32), an 8-pixel wide column is smaller than a single stride-16 grid cell, causing significant feature attenuation and IoU mismatch during NMS.
- **Intervention:** Small-object feature retention or higher input resolution (imgsz 768 or 800) would benefit pillars, but within fixed 640×640 constraints, tuning anchor matching or NMS IoU is necessary.

### 4.3. `shutter_blind` — High Precision (0.6565), Extremely Low Recall (0.1943)

- **Observations:** When predicted, shutters are accurate (65.65% precision), but 80.57% of ground truth shutters are missed completely.
- **Root Cause:**
  - Shutters flank windows. In the CMP Facade dataset, windows are overwhelmingly annotated (12,224 train boxes). The visual receptive field of the detector is dominated by window frames and glass reflections.
  - Shutters are often in deep shadow under eaves or partially open, causing their confidence scores to hover between 0.12 and 0.22, just below the standard 0.25 confidence threshold.
- **Intervention:** Repeat Factor Sampling (RFS) to increase frequency of shutter-bearing facades and validation threshold tuning to lower the deployment confidence threshold without harming precision.

### 4.4. `facade` vs. `building` — Hierarchical Bounding Box Conflict

- **Observations:** `facade` mAP50 is only 0.2703 (recall 0.2401); `building` recall is 0.3750.
- **Root Cause:**
  - In CMP Facade, the entire wall surface is labeled `facade`. In Open Images, the identical structure is labeled `building`.
  - In images where both classes are present, the ground truth contains two virtually identical, heavily overlapping bounding boxes (one for building, one for facade).
  - During NMS, whichever prediction has slightly higher confidence suppresses the other, immediately generating a false negative on the co-located class.

---

## 5. Visual Error Analysis Galleries

Visual inspection artifacts have been rendered and saved to `artifacts/exterior_error_analysis/`:

1. `porch_gt_vs_pred_train_oi_ext_train_0012.jpg`: Ground truth porch strip box on suburban home vs. baseline prediction (no porch detected; predicted `building` and `roof_cornice`).
2. `porch_gt_vs_pred_val_oi_ext_val_0084.jpg`: Front porch patio ground truth vs. baseline predictions (doors and windows detected; porch missed).
3. `porch_gt_vs_pred_val_oi_ext_val_0085.jpg`: Two small porch boxes vs. baseline predictions.
4. `porch_gt_vs_pred_test_oi_ext_test_0106.jpg`: Pathological 1.0×1.0 full-image porch ground truth box demonstrating Open Images label noise.
5. `porch_gt_vs_pred_test_oi_ext_test_0113.jpg`: Test image porch ground truth vs. baseline predictions.

---

## 6. Summary of Controlled Interventions for Phase 3

Based on this audit, the controlled experiment must address class balance without corrupting the test split or fabricating ungrounded synthetic samples:

1. **Implement Repeat Factor Sampling (RFS):** Formulate a deterministic manifest where images containing underrepresented and low-recall classes (`porch` 15x, `building` 2x, `shutter_blind` 2x) are sampled more frequently during training.
2. **Preserve Untouched Validation & Test Splits:** Training occurs exclusively on the RFS train manifest (`data/exterior_canonical/train_rfs.txt`); validation and held-out test splits remain 100% canonical, identical, and unaugmented.
3. **Validation Threshold Sweep:** Determine deployment operating points strictly on validation data to recapture missed shutter and column detections while controlling false positives.
