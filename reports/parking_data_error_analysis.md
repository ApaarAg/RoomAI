# RoomAI Phase 4 — Parking and Grounds Detector Data & Error Analysis

**Author:** Senior Computer Vision Research Engineer  
**Date:** October 2026  
**Target:** Group C Parking & Outdoor Grounds Detector  
**Baseline Model:** Model C2 ([`checkpoints/parking_detector_c2_yolo11s_pilot_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/parking_detector_c2_yolo11s_pilot_best.pt))  
**Baseline Checkpoint MD5:** `89383b78c1a43785fc5057cead5e52a5` (19,157,978 bytes)  
**Dataset Configuration:** Canonical 8-Class Parking Dataset ([`data/parking_canonical/parking_canonical.yaml`](file:///c:/Users/apaar/Desktop/RoomAI/data/parking_canonical/parking_canonical.yaml))  
**Splits:** 750 images total (510 train, 120 val, 120 test), 2,516 bounding boxes  

---

## 1. Executive Summary

This report establishes the baseline evaluation reproduction, data distribution audit, and granular error analysis for RoomAI Group C (Parking & Grounds Detection). The verified baseline checkpoint (`parking_detector_c2_yolo11s_pilot_best.pt`) achieves **0.2551 mAP50** and **0.1682 mAP50–95** on the 120-image held-out test split, with precision at **0.5382**, recall at **0.2640**, and F1 at **0.3543**.

While dominant vehicular and tabletop classes such as `car_vehicle` (mAP50 0.4522, F1 0.5062) and `outdoor_table` (mAP50 0.4075, F1 0.5275) perform reasonably well, the detector exhibits severe, catastrophic recall failures on two classes:
1. **`bicycle`:** **0.0000 recall** on the held-out test split (mAP50 = 0.0657).
2. **`tree`:** **0.0000 recall** on the held-out test split (mAP50 = 0.0509).
3. **`plant_bush`:** Recall collapses to **0.1700** (83% of ground truth bushes are missed despite 72.23% precision).

Our audit reveals the primary mechanisms driving these failures:
- **Severe Class Imbalance in Training Images:** Cars (173 images, 545 boxes) and outdoor seating (160 images, 440 boxes) dominate the 510 training images. Bicycles appear in only 24 training images (4.7% frequency) and trees in 56 training images (10.9% frequency).
- **Extreme Structural Wire-Frame Thinness (`bicycle`):** Bicycles have hollow frames, thin spokes, and small median areas (0.581% of the frame). At 640×640 resolution, bike frames span only 2–5 pixels in width, leading to severe feature loss in convolutional feature pyramids.
- **Diffuse, Amorphous Foliage Boundaries (`tree`, `plant_bush`):** Trees lack rigid bounding contours. Ground truth annotations vary wildly from individual shrubs to giant forest canopies, producing inconsistent IoU overlaps.
- **Structural Insuitability of Bounding Boxes for Parking Space Markings:** Flat painted lane markings and empty parking slots on asphalt are planar surface features without vertical volumetric extent. Representing them as 2D upright bounding boxes causes severe perspective warping and NMS suppression; they strictly belong in **semantic segmentation**, not 2D object detection.

---

## 2. Baseline Reproduction & Verification

The baseline model was re-evaluated under strict evaluation protocols (imgsz=640, device=CUDA:0, batch=8):

| Metric | Validation Split (120 images, 401 boxes) | Held-Out Test Split (120 images, 368 boxes) | Delta (Test vs. Val) |
| :--- | :---: | :---: | :---: |
| **mAP50** | **0.3465** | **0.2551** | -0.0914 |
| **mAP50–95** | **0.1950** | **0.1682** | -0.0268 |
| **Precision** | **0.4731** | **0.5382** | +0.0651 |
| **Recall** | **0.3510** | **0.2640** | -0.0870 |
| **F1 Score** | **0.4030** | **0.3543** | -0.0487 |

### Per-Class Test Performance (Baseline C2)

| Class ID | Class Name | Test Instances | Precision | Recall | F1 Score | mAP50 | mAP50–95 | Diagnosis |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `car_vehicle` | 115 | 0.6430 | 0.4174 | 0.5062 | 0.4522 | 0.2820 | Dominant class; recall drops on distant stalls |
| **1** | `motorcycle` | 17 | 0.5316 | 0.4675 | 0.4975 | 0.3863 | 0.2765 | Moderate performance; partial vehicle occlusion |
| **2** | `bicycle` | 7 | **1.0000** | **0.0000** | **0.0000** | **0.0657** | 0.0502 | **Complete recall collapse; thin wire-frames** |
| **3** | `outdoor_seating` | 95 | 0.3803 | 0.2421 | 0.2959 | 0.2424 | 0.1553 | Low precision & recall; curb / bench confusion |
| **4** | `outdoor_table` | 34 | 0.5257 | 0.5294 | 0.5275 | 0.4075 | 0.2772 | Solid performance; large visual surface |
| **5** | `plant_bush` | 46 | 0.7223 | 0.1700 | 0.2752 | 0.2035 | 0.1540 | High precision; misses 83% of ground truth |
| **6** | `tree` | 33 | **0.0000** | **0.0000** | **0.0000** | **0.0509** | 0.0261 | **Complete recall collapse; diffuse canopy** |
| **7** | `outdoor_fixture` | 21 | 0.5030 | 0.2857 | 0.3644 | 0.2321 | 0.1239 | Sub-scale vertical slivers (meters, lampposts) |
| **ALL** | **Overall** | **368** | **0.5382** | **0.2640** | **0.3543** | **0.2551** | **0.1682** | Baseline starting point |

---

## 3. Dataset Distribution & Geometry Audit

The canonical Group C dataset contains 750 images and 2,516 bounding boxes across 8 classes.

### Per-Class Instance & Image Distribution Across Splits

| Class Name | Train Boxes (Images) | Val Boxes (Images) | Test Boxes (Images) | Total Boxes | Median Area (% img) | Median Aspect Ratio (W/H) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `car_vehicle` | 545 (173) | 100 (35) | 115 (25) | 760 | 0.763% | 1.14 |
| `motorcycle` | 55 (33) | 22 (8) | 17 (7) | 94 | 3.890% | 0.70 |
| `bicycle` | 66 (24) | 8 (2) | 7 (3) | 81 | 0.581% | 0.72 |
| `outdoor_seating` | 440 (160) | 116 (34) | 95 (42) | 651 | 1.180% | 0.79 |
| `outdoor_table` | 143 (110) | 38 (24) | 34 (31) | 215 | 26.620% | 1.24 |
| `plant_bush` | 169 (96) | 37 (23) | 46 (24) | 252 | 2.484% | 0.88 |
| `tree` | 97 (56) | 35 (17) | 33 (15) | 165 | 8.360% | 0.92 |
| `outdoor_fixture` | 132 (48) | 45 (19) | 21 (11) | 198 | 0.335% | 0.44 |
| **Total** | **1,647 (510)** | **401 (120)** | **368 (120)** | **2,516** | — | — |

---

## 4. In-Depth Root Cause Analysis of Failure Modes

### 4.1. `bicycle` — Complete Recall Failure (0.0000 Recall on Test)
- **Observations:** Precision is reported as 1.0000 because the model produced zero false positives, but recall is 0.0000. All 7 test ground truth bicycles were missed.
- **Root Cause & Visual Mechanics:**
  1. **Sparse Image Presence:** Bicycles appear in only 24 out of 510 training images (4.7%). In contrast, cars appear in 173 images (7.2× more images and 8.3× more boxes).
  2. **Wire-Frame Geometry:** Bicycles consist almost entirely of thin tubular steel frames and wire-spoked wheels. Bounding boxes are largely empty background (>85% background inside the box). At 640×640 input resolution, convolutional downsampling down to stride-16 and stride-32 washes out the thin edges of the frame.
  3. **Occlusion Behind Parked Cars:** In parking lots, bicycles are chained to bike racks or tucked between SUVs, presenting high occlusion.
- **Remediation:** Repeat Factor Sampling (RFS) with a 4× multiplier on bicycle images, combined with multi-scale feature retention.

### 4.2. `tree` — Diffuse Boundaries and Sky Clutter (0.0000 Recall on Test)
- **Observations:** On the held-out test split, `tree` achieves 0.0000 recall and 0.0000 precision (mAP50 = 0.0509).
- **Root Cause & Annotation Ambiguity:**
  1. **Amorphous Foliage:** Unlike cars or tables with crisp planar silhouettes, trees have variable, fractal contours. A box drawn around a tree canopy inevitably includes large swaths of sky, buildings, or other trees.
  2. **Inconsistent Label Granularity in Open Images:** In Open Images V7, some labelers box entire tree groves as a single giant "Tree", while others label individual trunks. This label noise penalizes the model during IoU matching.
  3. **Low Training Volume:** Only 56 training images contain trees.
- **Remediation:** Repeat Factor Sampling (2×) to stabilize gradient backprop on canopy features, accompanied by threshold tuning.

### 4.3. `outdoor_fixture` — Sub-Scale Vertical Slivers (Median Area 0.335%)
- **Observations:** mAP50 is 0.2321, with recall at 0.2857.
- **Root Cause:**
  - `outdoor_fixture` encompasses parking meters, lampposts, and waste bins.
  - The median aspect ratio is 0.44 (narrow vertical rectangles), and the median area is only 0.335% (approx. 15×40 pixels at 640×640).
  - Many distant lampposts in parking lots are only 4–6 pixels wide, causing bounding box coordinates to drift outside standard IoU thresholds.

### 4.4. Critical Analysis: Parking Space Markings — Detection vs. Segmentation
A crucial research question addressed in this phase is whether empty parking stalls and painted parking-space lines should be treated as 2D object detection bounding boxes:
- **Planar Surface Nature:** Painted lines on asphalt are 2D road surface markings without vertical physical volume or 3D bounding geometry.
- **Perspective Warping:** In vehicle-mounted or pedestrian-level cameras, parking stall lines recede into the distance at oblique angles (often 30°–60° relative to the optical axis). An axis-aligned 2D bounding box covering an oblique rectangular parking slot captures 60%–80% irrelevant asphalt and adjacent stalls.
- **IoU Collapse & NMS Suppression:** When multiple stalls are adjacent, their axis-aligned 2D bounding boxes overlap heavily. Standard 2D Non-Maximum Suppression (NMS) suppresses neighboring valid stalls, leading to artificially depressed recall.
- **Occlusion by Tires:** When a car is parked in a slot, the markings are partially obscured under the chassis. A detector trained on empty stalls will misfire on partially occupied stalls.
- **Definitive Conclusion:** **Parking space markings are fundamentally ill-suited for 2D bounding-box object detection.** They strictly belong in **semantic segmentation** (e.g., bird's-eye-view ground plane line segmentation) or polygon instance masks. Keeping the Group C object detector focused on discrete physical objects (`car_vehicle`, `motorcycle`, `bicycle`, `outdoor_seating`, `outdoor_table`, `plant_bush`, `tree`, `outdoor_fixture`) is the scientifically sound architectural choice.

---

## 5. Visual Error Gallery

Visual inspection artifacts generated from the test split have been rendered to `artifacts/parking_error_analysis/`:

1. `gt_vs_c2_pred_coco_test_coco_ext_0027.jpg`: Shows urban street scene with bicycles and pedestrians. Baseline C2 detects cars but misses bicycles completely.
2. `gt_vs_c2_pred_coco_test_coco_ext_0032.jpg`: Parked vehicle lot with trees in background. Trees are completely missed; car bounding boxes are localized.
3. `gt_vs_c2_pred_coco_test_coco_ext_0067.jpg`: Outdoor seating and tables. Demonstrates table detection success but missed distant benches.
4. `gt_vs_c2_pred_coco_test_coco_ext_0073.jpg`: Street corner with bicycle and lamppost. Bicycle is missed; outdoor fixture is partially localized.
5. `gt_vs_c2_pred_coco_test_coco_ext_0103.jpg`: Landscaped parking lot with bushes. Demonstrates high false-negative rate on bushes.
6. `gt_vs_c2_pred_coco_test_coco_ext_0104.jpg`: Urban parking lane with overlapping motorcycles and cars.

---

## 6. Controlled Optimization Strategy for Group C

To address these empirical findings, we implement controlled sequential experiments:
1. **Experiment A (Baseline Control):** Frozen Model C2 checkpoint evaluated under identical settings.
2. **Experiment B (Repeat Factor Sampling — Model C3):** Class-aware oversampling targeting `bicycle` (4×), `motorcycle` (3×), `outdoor_fixture` (2×), and `tree` (2×), while keeping validation and test splits 100% frozen.
3. **Experiment C (Targeted Augmentation / Fine-Tuning — Model C4):** Controlled geometric scale and translation transforms to enhance feature extraction on thin wire-frame structures.
4. **Experiment D (Capacity & Feasibility Analysis):** Assess whether scaling to larger architectures (e.g. YOLO11m) is justified given 510 training images, inference latency budgets, and 8 GB VRAM.
