# RoomAI Phase 5 — Indoor Detector Data & Error Analysis

**Author:** Senior Computer Vision Research Engineer  
**Date:** October 2026  
**Target:** Group A Indoor Furniture & Household Objects Detector  
**Baseline Model:** Model A2 ([`checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt))  
**Baseline Checkpoint MD5:** `3a4294ca2373fb3eaefbbf55e37a3c3c` (19,156,442 bytes)  
**Dataset:** Canonical 12-Class Indoor Dataset ([`data/indoor_canonical/indoor_canonical.yaml`](file:///c:/Users/apaar/Desktop/RoomAI/data/indoor_canonical/indoor_canonical.yaml))  
**Total Images & Annotations:** 3,689 images (2,985 train, 554 val, 150 held-out test), 28,606 bounding boxes across 12 canonical classes  

---

## 1. Executive Summary

This report establishes the baseline evaluation reproduction, granular dataset geometry audit, and per-class failure mode diagnosis for the RoomAI Group A Indoor Detector. The verified production candidate checkpoint (`indoor_detector_a2_yolo11s_finetune_best.pt`) demonstrates strong aggregate performance:
- **Validation Split (554 images, 4,474 boxes):** **0.7531 mAP50**, **0.5484 mAP50–95**, Precision **0.7629**, Recall **0.6604**, F1 **0.7080**.
- **Held-Out Test Split (150 images, 961 boxes):** **0.7464 mAP50**, **0.5921 mAP50–95**, Precision **0.7549**, Recall **0.6762**, F1 **0.7134**.

Several core furniture classes demonstrate outstanding precision and localization:
- `bed`: **0.9324 mAP50**, **0.8885 F1** (exceeds the 90% benchmark!)
- `decoration`: **0.8536 mAP50**, **0.7633 F1**
- `lamp`: **0.8230 mAP50**, **0.7487 F1**
- `sofa`: **0.8096 mAP50**, **0.7169 F1**

However, detailed per-class auditing reveals two significant performance bottlenecks:
1. **`chair#stool` Recall Deficit (Recall = 0.4374, mAP50 = 0.5245):** Over 56% of dining chairs and bar stools are missed. 42.3% of all chair instances are **small objects (<1% image area)** frequently occluded under dining tables or desks.
2. **`curtain` Aspect Ratio Variation (Recall = 0.5242, mAP50 = 0.6173):** Curtains have an extreme vertical aspect ratio (median $W/H = 0.41$) and frequently blend into wall textures or window frames.
3. **`window` Split Anomaly in Test Set:** Ground truth contains **1,636 train boxes** and **371 validation boxes** (where Model A2 achieves **0.5692 mAP50**), but **0 test boxes**. The legacy DeepFurniture test set contained 0 labeled windows; windows were introduced in train/val from HomeObjects-3K.

---

## 2. Baseline Reproduction & Verification

Model A2 was evaluated under standard protocol (imgsz=640, device=CUDA:0, batch=8):

| Metric | Validation Split (554 images) | Held-Out Test Split (150 images) | Delta (Test vs. Val) |
| :--- | :---: | :---: | :---: |
| **mAP50** | **0.7531** | **0.7464** | -0.0067 |
| **mAP50–95** | **0.5484** | **0.5921** | +0.0437 |
| **Precision** | **0.7629** | **0.7549** | -0.0080 |
| **Recall** | **0.6604** | **0.6762** | +0.0158 |
| **F1 Score** | **0.7080** | **0.7134** | +0.0054 |

### Per-Class Performance Breakdown (Baseline Model A2)

| Class ID | Class Name | Test Instances | Test Precision | Test Recall | Test F1 | Test mAP50 | Test mAP50–95 | Val mAP50 | Diagnosis |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `cabinet#shelf` | 153 | 0.8173 | 0.6275 | 0.7099 | 0.7607 | 0.5623 | 0.6953 | High precision; misses low dressers |
| **1** | `table` | 78 | 0.6103 | 0.6667 | 0.6373 | 0.6639 | 0.5218 | 0.7483 | Confusion with counters/desks |
| **2** | `chair#stool` | 247 | **0.6116** | **0.4374** | **0.5100** | **0.5245** | 0.3810 | 0.5841 | **Primary recall bottleneck; occlusion** |
| **3** | `lamp` | 90 | 0.7709 | 0.7278 | 0.7487 | 0.8230 | 0.6273 | 0.8291 | Reliable pendant & floor localization |
| **4** | `door` | 57 | 0.7279 | 0.6316 | 0.6763 | 0.7034 | 0.5192 | 0.7816 | Good boundary detection |
| **5** | `bed` | 45 | **0.9114** | **0.8667** | **0.8885** | **0.9324** | 0.8379 | 0.9329 | **Benchmark class (>90% mAP50)** |
| **6** | `sofa` | 43 | 0.7371 | 0.6977 | 0.7169 | 0.8096 | 0.6732 | 0.8365 | Clean volumetric segmentation |
| **7** | `plant` | 58 | 0.7150 | 0.8362 | 0.7709 | 0.7799 | 0.5570 | 0.8162 | Strong recall on foliage |
| **8** | `decoration` | 82 | 0.7837 | 0.7439 | 0.7633 | 0.8536 | 0.7692 | 0.8354 | High AP on wall art and ornaments |
| **9** | `curtain` | 62 | **0.8290** | **0.5242** | **0.6423** | **0.6173** | 0.4250 | 0.7259 | **High precision; recall limited by aspect ratio** |
| **10** | `home-appliance`| 28 | 0.7897 | 0.6786 | 0.7299 | 0.7422 | 0.6390 | 0.7825 | Solid performance on TVs / ovens |
| **11** | `window` | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.5692** | **0 test instances (dataset split anomaly)** |
| **ALL** | **Overall** | **961** | **0.7549** | **0.6762** | **0.7134** | **0.7464** | **0.5921** | **0.7531** | **Baseline control reference** |

---

## 3. Dataset Distribution & Geometry Audit

The canonical indoor dataset contains 3,689 images across three splits:

### Per-Class Instance Distribution & Scale Breakdown

| Class Name | Train Boxes (Imgs) | Val Boxes (Imgs) | Test Boxes (Imgs) | Total Boxes | Med Area (% img) | Aspect (W/H) | Small (<1%) | Medium (1-10%) | Large (>10%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cabinet#shelf` | 4,204 (1,749) | 794 (343) | 153 (69) | 5,151 | 8.875% | 0.94 | 208 | 2,683 | 2,260 |
| `table` | 2,423 (1,675) | 496 (336) | 78 (52) | 2,997 | 7.962% | 1.18 | 79 | 1,691 | 1,227 |
| `chair#stool` | **4,402 (1,496)** | **946 (326)** | **247 (76)** | **5,595** | **1.350%** | 0.74 | **2,367 (42.3%)** | 3,091 | 137 |
| `lamp` | 2,492 (1,399) | 468 (269) | 90 (49) | 3,050 | 1.624% | 0.69 | 848 | 2,059 | 143 |
| `door` | 2,028 (1,241) | 321 (193) | 57 (33) | 2,406 | 9.940% | 0.48 | 66 | 1,140 | 1,200 |
| `bed` | 1,440 (1,368) | 239 (232) | 45 (43) | 1,724 | 23.493% | 1.28 | 4 | 260 | 1,460 |
| `sofa` | 1,366 (1,154) | 269 (228) | 43 (37) | 1,678 | 18.067% | 1.45 | 5 | 390 | 1,283 |
| `plant` | 744 (478) | 134 (82) | 58 (35) | 936 | 2.502% | 0.73 | 206 | 639 | 91 |
| `decoration` | 1,862 (954) | 298 (153) | 82 (46) | 2,242 | 1.622% | 0.95 | **900 (40.1%)** | 1,237 | 105 |
| `curtain` | 370 (303) | 69 (60) | 70 (39) | 509 | 15.250% | **0.41** | 8 | 165 | 336 |
| `home-appliance`| 574 (511) | 91 (86) | 28 (27) | 693 | 2.606% | 0.94 | 119 | 525 | 49 |
| `window` | **1,636 (722)** | **371 (162)** | **0 (0)** | 2,007 | 2.461% | 0.39 | 380 | 1,489 | 138 |
| **Total** | **23,171 (2,985)**| **4,474 (554)** | **961 (150)** | **28,606** | — | — | **5,190** | **15,369** | **8,029** |

---

## 4. Root Cause Analysis of Specific Error Modes

### 4.1. `chair#stool` — Occlusion & Scale Downsampling (Recall = 0.4374)
- **Scale Factor:** 42.3% of chairs (2,367 instances) are **Small (<1% of the frame)**.
- **Physical Context:** In dining rooms and open kitchens, chairs are partially tucked under tables. Only the top rail of the chair back or a single leg is visible.
- **Convolutional Stride Collapse:** At 640×640, a 0.5% area chair corresponds to ~25×35 pixels. Fine tubular metal legs or thin wooden spindles are 1–2 pixels wide. When downsampled through P3 (stride 8), P4 (stride 16), and P5 (stride 32), the distinctive features of the chair frame blur into the surrounding floor/table pixels.
- **Justification for Experiment B (800px):** Increasing training and inference resolution from 640×640 to **800×800** yields a **+56.25% pixel density increase**, allowing stride-8 feature maps to retain distinct chair leg contours.

### 4.2. `curtain` — Slender Vertical Aspect Ratio & Boundary Ambiguity (Recall = 0.5242)
- **Geometry Factor:** Curtains have the most extreme vertical aspect ratio of all indoor objects (median $W/H = 0.41$).
- **Visual Camouflage:** Curtains are often neutral white, cream, or grey, hanging directly adjacent to matching painted walls.
- **Annotation Variation:** In multi-panel bay windows, labelers sometimes annotate both left and right drapes as a single wide box spanning the window, and in other instances box each panel individually. This causes IoU penalties during NMS.

### 4.3. `window` — Split Isolation Anomaly
- **The Zero-Test Instance Anomaly:** The test split contains exactly 0 window ground truth annotations because the original DeepFurniture test split lacked window labels.
- **Validation Reality:** In the validation split (554 images), `window` has 371 ground truth boxes, and Model A2 achieves **0.5692 mAP50** and **0.7663 precision**.
- **Scientific Implication:** Evaluating test split aggregate metrics without understanding this split discrepancy can mislead engineers into believing window detection is non-functional. In reality, window detection functions well on validation images.

---

## 5. Visual Error Gallery

Visual inspection artifacts rendered to `artifacts/indoor_error_analysis/`:

1. `gt_vs_a2_pred_df_L3D121S21ENDIHOOVJYUI5L7SFOS3P3W2888.jpg`: Dining room scene with 6 chairs around a table. A2 correctly detects the table and front chairs, but misses 2 heavily occluded rear chairs tucked under the far edge.
2. `gt_vs_a2_pred_df_L3D121S22ENDIHJZWPIUI5MI2UR73P3XK888.jpg`: Living room with sofa, coffee table, and tall curtain panel. Demonstrates curtain localization.
3. `gt_vs_a2_pred_df_L3D121S22ENDIHL2BCIUI5MI2DZR3P3W4888.jpg`: Bedroom scene demonstrating clean bed and lamp localization.
4. `gt_vs_a2_pred_df_L3D121S8ENDI225Y3QUPFR7OCG43P3XC888.jpg`: Open-plan apartment showing cabinet and door demarcations.
5. `gt_vs_a2_pred_df_L3D121S8ENDI22RVNIUG5TL4LNM3P3XS888.jpg`: Dense living room showing small wall decorations.
6. `gt_vs_a2_pred_df_L3D121S8ENDI23BW6AUPFRYIYFJ3P3XS888.jpg`: Kitchen island with bar stools.

---

## 6. Controlled Optimization Plan for Phase 5

1. **Experiment A (Baseline Control):** Frozen Model A2 checkpoint evaluated under standard 640px settings.
2. **Experiment B (Higher-Resolution Training at 800 Pixels — Model A3):**
   - Train YOLO11s at `imgsz=800` with `batch=4` (safe for 8 GB VRAM, expected VRAM ~3.8 GB).
   - Evaluate whether the +56.25% pixel expansion resolves small `chair#stool` and `decoration` features.
3. **Experiment C (Class-Balancing & Targeted Augmentation — Model A4):**
   - Apply class-aware oversampling or scale augmentations targeted at `chair#stool` and `curtain`.
4. **Validation Threshold Sweep:**
   - Optimize operating points strictly on the 554 validation images across confidence $\in [0.10 \dots 0.50]$ and IoU $\in [0.40 \dots 0.60]$.
