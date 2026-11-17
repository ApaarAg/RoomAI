# RoomAI Phase 19 — Multi-Dataset Expansion and Training Strategy Audit Report

**Auditor:** Senior Computer-Vision Engineer & ML Systems Architect  
**Repository:** `C:/Users/apaar/Desktop/RoomAI`  
**Execution Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM), CUDA 12.x, PyTorch 2.5.1+cu124  
**Integrity Guarantee:** Zero models retrained in Phase 19. Checkpoints `Model B2`, `Model B6`, and `Porch Specialist` remain byte-for-byte immutable. The Phase 17 test set is strictly quarantined.

---

## Executive Summary

Phase 19 provides an architectural audit, data harmonization framework, and staged training plan to transition RoomAI from a single-source classical facade detector into a multi-dataset, competition-ready exterior understanding system.

### Key Audit Findings:
1. **Critical Hardware Bottleneck:** System C: drive has **27.18 GB of free disk space**. Massive indiscriminate dataset downloads (e.g., full Open Images V7 at >500 GB, Mapillary Vistas at 190 GB) are physically impossible and strictly prohibited. Ingestion must rely on **curated targeted subsets (< 4.5 GB total storage)**.
2. **Taxonomic Compatibility Gap:**
   - **COCO 80-Class Detection:** Contains **zero** architectural exterior classes (`building`, `facade`, `window`, `door`, `porch`, `column`, `balcony`, `cornice`, `shutter`, and `molding` do not exist in COCO-80). It is **unsupported** for object detection training.
   - **ADE20K (MIT CSAIL):** Exhibits the **highest taxonomic alignment** with RoomAI in the literature, explicitly annotating object-part hierarchies for `porch`, `column/pillar`, `balcony`, `cornice`, `shutter`, `facade`, and `door`.
   - **Open Images V7:** High-value for rare classes (`porch`, `shutter_blind`), but requires **class-conditional loss masking** to prevent severe false-positive penalties from unannotated windows and doors.
3. **Facade Geometry Paradigm Shift:** Bounding boxes are mathematically and geometrically misspecified for `facade` surfaces (Phase 17 showed 45.6% missed facades and mean IoU of 0.41 due to non-convex, stepped, L-shaped walls). Facade detection must transition to **semantic polygon/mask segmentation (YOLO11s-seg)**.
4. **Actionable Recommendation:** Execute **Pilot B-1 (300 curated images: 150 ADE20K outdoor + 150 Open Images entrance scenes)** with class-aware loss masking and temperature-scaled sampling before committing to a multi-thousand-image retraining run.

---

## Stage 1: Existing Repository Audit

### 1.1 Current Data Sources and Split Allocations

The RoomAI repository currently maintains three iterations of exterior datasets in `data/`:
1. `data/exterior_canonical/`: The original baseline dataset.
2. `data/exterior_b5_unified/`: The Phase 14 transfer-learning dataset.
3. `data/exterior_b6_expanded/`: The active Phase 15–17 expanded dataset.

Audit of image and label allocations across current splits:

| Dataset Directory | Split | Image Count | Label Count | Total Annotations | Dominant Classes | Sparsest Classes |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `exterior_b6_expanded` | **train** | **1,302** | 1,302 | 35,515 | window (14,176), molding (7,070) | porch (509), building (777) |
| `exterior_b6_expanded` | **val** | **154** | 154 | 8,504 | window (3,845), molding (1,543) | porch (24), building (40) |
| `exterior_b6_expanded` | **test** | **154** | 154 | 9,826 | window (4,090), molding (2,038) | porch (28), building (40) |
| `exterior_b5_unified` | train | 903 | 903 | 33,950 | window (13,525), molding (7,070) | porch (134), building (491) |
| `exterior_canonical` | train | 458 | 458 | 31,814 | window (12,224), molding (7,070) | porch (1), building (101) |

### 1.2 Data Provenance Breakdown (Current Training Set of 1,302 Images)
The 1,302 images in `exterior_b6_expanded/images/train/` originate from only two primary sources:
- **CMP Facade Database:** 378 images (28,500 annotations). Rectified European urban facades with dense windows and moldings, but virtually zero porches (1 instance).
- **Open Images V7 Canonical (Phase 8–10):** 80 images (targeted residential buildings and porches).
- **Open Images V7 B5 Ingestion (Phase 14):** 445 images (targeted porch and door scenes).
- **Open Images V7 B6 Ingestion (Phase 14 Stage B):** 399 images (targeted porch and window blind scenes).

### 1.3 Class Distributions by Split (`exterior_b6_expanded`)

```
====================================================================================================
Canonical Class ID & Name        Train Boxes (%)          Val Boxes (%)           Test Boxes (%)
====================================================================================================
0: building                         777 (2.19%)              40 (0.47%)              40 (0.41%)
1: facade                         2,411 (6.79%)             495 (5.82%)             579 (5.89%)
2: window                        14,176 (39.92%)          3,845 (45.21%)          4,090 (41.62%)
3: door                           1,633 (4.60%)             372 (4.37%)             432 (4.40%)
4: roof_cornice                   2,797 (7.88%)             445 (5.23%)             666 (6.78%)
5: balcony                        1,091 (3.07%)             345 (4.06%)             393 (4.00%)
6: porch                            509 (1.43%)              24 (0.28%)              28 (0.28%)
7: pillar_column                  2,270 (6.39%)             494 (5.81%)             896 (9.12%)
8: shutter_blind                  2,781 (7.83%)             901 (10.60%)            664 (6.76%)
9: molding_deco                   7,070 (19.91%)          1,543 (18.14%)          2,038 (20.74%)
====================================================================================================
TOTAL                            35,515 (100.0%)          8,504 (100.0%)          9,826 (100.0%)
====================================================================================================
```

### 1.4 Existing Safeguards and Overlap Screening
The codebase currently implements:
- **MD5 Content Verification:** Evaluates full file byte hashes to prevent exact duplicate images across train, val, and test splits.
- **64-bit DCT Perceptual Hashing (pHash):** Evaluates Hamming distance across all pairs. A threshold of Hamming distance $\ge 5$ is enforced against validation and locked test images to prevent near-duplicate scene leakage.
- **Immutable Checkpoints:**
  - `exterior_detector_b2_yolo11s_best.pt` (`14a931dd0adabc16cbd7b28ed1d17f66`) [Production Default]
  - `exterior_detector_b6_expanded_best.pt` (`f5ceea56b0610ea14b647014e48152dc`) [Candidate Baseline]
  - `exterior_specialist_porch_yolo11s_best.pt` (`4261fc1ad5f6fd9ac29c0499e2adddfd`) [Specialist Profile]

### 1.5 System Hardware and Execution Constraints
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB GDDR6 VRAM.
  - Safe VRAM operating budget: **$\le 6.5\text{ GB}$**. Peak VRAM during single-model batch-16 training at 800px with PyTorch AMP fp16 is $\sim 4.8\text{ GB}$.
  - Concurrent GPU training is **strictly prohibited**.
- **System RAM:** 23.73 GB total, **5.39 GB available**. Background data loading workers must be capped at `workers=4` to prevent system memory swapping.
- **System Disk (C:):** Total 474.87 GB, **Free: 27.18 GB**.
  - Maximum allowable storage for new dataset expansions: **$\le 4.5\text{ GB}$**.

### 1.6 Identified Gaps in Current Multi-Dataset Pipeline
1. **Absence of Class-Conditional Loss Masking:** Current YOLO training treats any unannotated pixel in Open Images as true background negative. Because Open Images only annotates queried classes per image, unannotated windows and moldings generate false negative penalties during backpropagation.
2. **Bounding Box Geometry Limitation for Facades:** Bounding boxes over non-convex facades cause extreme spatial ambiguity (Phase 17 showed 243 background false positives and 45.6% missed facades).
3. **Severe Perspective Bias:** CMP Facade is 100% rectified orthogonal European architecture. Real-world consumer and competition images are street-level, perspective-distorted suburban homes.

---

## Stage 2: Audit of Candidate Datasets

Every candidate dataset was audited for license terms, access protocols, annotation formats, class overlap, sample volume, and disk requirements:

| Candidate Dataset | Official Source & Access | License & Competition Terms | Annotation Types | Relevant RoomAI Classes | Usable Samples | Storage Required | Expected Value & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Open Images V7** | Google Open Images / AWS S3 (Direct HTTP/S3) | CC BY 4.0 (annotations) / CC BY 2.0 (images). Fully eligible for commercial/competition use. | 2D BBoxes, Instance Segmentation Masks, Visual Relations. | `porch`, `door`, `window`, `shutter_blind`, `building`. | ~4,200 porch scenes; ~8,500 blind scenes; ~45,000 door scenes. | Full: >500 GB (Prohibited).<br>Targeted subset: **1.2 GB / 1,000 imgs**. | **High:** Diverse suburban houses and porches.<br>**Limitation:** Sparse partial annotations; no labels for `facade`, `roof_cornice`, `balcony`, `pillar_column`. |
| **Mapillary Vistas (v1.2 & v2.0)** | Mapillary / Meta (Portal auth token required) | Mapillary Research License (Non-commercial research). Commercial license required for production. | Pixel semantic segmentation (65/124 classes), polygon instances. | `building`, `facade` (wall), `window` (v2), `door` (v2). | ~4,500 street scenes with prominent facades. | Full: 190 GB (Prohibited).<br>Targeted crops: **1.8 GB / 500 imgs**. | **Moderate:** Excellent street-view urban context.<br>**Limitation:** Heavy tree/car occlusions; distant oblique angles; commercial license restriction. |
| **ADE20K** | MIT CSAIL (Direct HTTP tarball) | CC BY-NC-SA 4.0 (Non-Commercial ShareAlike). Eligible for research & academic benchmarks. | Dense polygons, semantic masks, **hierarchical object-part annotations**. | `building`, `facade`, `window`, `door`, `roof_cornice`, `balcony`, `porch`, `pillar_column`, `shutter_blind`, `molding_deco`. | ~4,100 outdoor architectural scenes (2,100 columns, 480 porches, 1,200 balconies). | Full: 3.8 GB comp. / **6.5 GB uncomp.** (Fits disk constraint).<br>Targeted: **850 MB**. | **Highest:** The ONLY dataset explicitly annotating porch, column, balcony, cornice, and facade parts on natural imagery.<br>**Limitation:** Requires scene filtering to exclude indoor rooms. |
| **COCO (2017 Detection)** | cocodataset.org (Direct HTTP) | CC BY 4.0 (annotations) / Flickr (images). Fully eligible. | 80 object categories (BBoxes, polygons). | **NONE.** (0 of 10 canonical classes exist in COCO-80). | **0 usable architectural detection samples.** | Full: 19 GB (Not recommended). | **Unsupported:** COCO-80 does not annotate buildings, windows, doors, or architectural trim. Ingesting COCO detection labels is rejected. |
| **CMP Facade (Existing)** | CTU Prague (Radim Tylecek, 2013) | Academic research / free with citation. Integrated locally. | 12-class pixel segmentation maps converted to bounding boxes. | 10 canonical classes (dense windows, moldings, cornices, pillars). | 606 images (378 train, 114 val, 114 test). | **145 MB** (Already local). | **Baseline:** Dense annotations for classical trim.<br>**Limitation:** Rectified European facades; near-zero porches; bounding-box facade mismatch. |

---

## Stage 3: Label Harmonization Design

### 3.1 Explicit Mapping Strategy
RoomAI enforces a strict 10-class canonical taxonomy. All incoming datasets are harmonized via `reports/phase19_dataset_mapping.csv` (47 defined rules).

Key mapping definitions:
- **`building` (ID 0):** Enclosing bounding box of the entire structure. Mapped from Open Images `/m/0cgh4` (Building), `/m/03jm5` (House), `/m/021sj1` (Office building), ADE20K `building, edifice`, `house`, and Mapillary `construction--structure--building`.
- **`facade` (ID 1):** Exterior wall surface envelope. Mapped from CMP `facade`, ADE20K `facade`, `wall (exterior)`. Freestanding perimeter walls in Mapillary are flagged as **ambiguous** and quarantined.
- **`window` (ID 2):** Architectural glazed window aperture. Mapped from CMP `window`, Open Images `/m/0d4v4`, ADE20K `windowpane, window`, and Mapillary v2.0 `object--window`.
- **`door` (ID 3):** Ground-level and balcony entry portals. Mapped from CMP `door`, CMP `shop`, Open Images `/m/02dgv`, ADE20K `door`, and Mapillary v2.0 `object--door`. Vehicle doors are filtered via scene context.
- **`roof_cornice` (ID 4):** Roofline crowning trim and eaves. Mapped from CMP `cornice`, ADE20K `cornice`, and ADE20K `roof` (clipped to lower fascia).
- **`balcony` (ID 5):** Exterior projecting platform with railing. Mapped from CMP `balcony` and ADE20K `balcony`. Open Images `/m/02p3w7` is flagged as **ambiguous** due to crowd-source sparsity.
- **`porch` (ID 6):** Covered exterior entrance structure / veranda. Mapped from Open Images `/m/04m6gz` and ADE20K `porch`.
- **`pillar_column` (ID 7):** Vertical structural or decorative pillar / column. Mapped from CMP `pillar` and ADE20K `column, pillar`. Mapillary `object--support--pole` (utility pole) and Open Images `/m/02522` (text/newspaper column) are **unsupported and rejected**.
- **`shutter_blind` (ID 8):** Louvered exterior shutters and window blinds. Mapped from CMP `blind`, Open Images `/m/031b6r`, and ADE20K `shutter, blind`.
- **`molding_deco` (ID 9):** Exterior profiled trim bands, window sills, and decorative reliefs. Mapped from CMP `sill`, `deco`, `molding`, and ADE20K `molding`.

### 3.2 Handling Partially Annotated Images (Open Images & ADE20K)
Treating an unannotated window or door in an Open Images frame as negative background severely damages detector precision. RoomAI defines two mandatory training safeguards:
1. **Class-Conditional Loss Masking:** Modify the training loss computation so that classification loss is backpropagated only for classes explicitly verified in the image's source annotation record. Unannotated classes are masked out ($w_{class} = 0$).
2. **Offline High-Precision Pseudo-Labeling (Alternative):** Run Model B2 at a strict threshold ($\ge 0.70$ confidence) over new Open Images frames to auto-label obvious missing windows and doors before ingestion, preventing false-negative gradient signals.

### 3.3 Facade Geometry: Segmentation Masks vs. Bounding Boxes
Phase 17 empirical evidence demonstrated that bounding boxes for `facade` fail:
- Facades are non-convex, multi-planar, L-shaped surfaces with recessed portals and gables.
- Fitting an axis-aligned bounding box over a stepped facade forces the box to enclose large expanses of sky, grass, and neighboring houses, causing high false-positive rates (243 FPs in Phase 17).
- **Engineering Verdict:** For `facade`, **semantic polygon segmentation (YOLO11s-seg or Mask2Former)** is required. Bounding boxes must be retained exclusively for discrete object instances (`window`, `door`, `balcony`, `porch`, `pillar_column`), while the facade surface is predicted as a pixel mask.

---

## Stage 4: Staged Training Plan

All experiments are fully defined in `reports/phase19_training_plan.json`. Zero training runs were initiated in this phase.

```
+---------------------------------------------------------------------------------------------------+
|                                  STAGED TRAINING PIPELINE                                         |
+---------------------------------------------------------------------------------------------------+
|  Experiment A: Frozen Model B6 Baseline (Control Benchmark)                                       |
|  - Checkpoint: checkpoints/exterior_detector_b6_expanded_best.pt                                  |
|  - Role: Immutable baseline on new holdout evaluation set.                                        |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  Pilot B-1: Multi-Source Harmonized Detection Pilot (Small-Scale Verification)                    |
|  - Size: 300 Curated Images (150 ADE20K Outdoor + 150 Open Images Porch/Entrance)                 |
|  - Hyperparameters: 800px, Batch 16, AMP fp16, 30 Epochs, Patience 15, AdamW                      |
|  - Sampling: Temperature-scaled inverse-frequency sampling (T=0.5; Porch 3x, Column 2x)           |
|  - Storage / Memory: 420 MB disk, 5.2 GB peak VRAM, ~18 minutes training time                     |
|  - Gate: Must achieve Val Porch F1 >= 0.20 and Macro-F1 >= 0.50 before full run                  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  Experiment B: Full Multi-Source Detection Detector (Candidate B7)                                |
|  - Target: 2,200 Clean Images (378 CMP + 924 Existing OI + 600 ADE20K + 300 Entrance Expansion)   |
|  - Hyperparameters: 800px, Batch 16, AMP fp16, 80 Epochs, Patience 20, AdamW                      |
|  - Objective: Lift Macro-F1 above 0.50 and Porch Recall above 0.40 on holdout benchmark.          |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|  Experiment C: Dedicated Facade Mask Segmentation Baseline                                        |
|  - Architecture: YOLO11s-seg (Pretrained on COCO-seg)                                             |
|  - Target Masks: facade, roof, and entrance openings                                              |
|  - Data: CMP Facade Polygons (378) + ADE20K Exterior Wall Polygons (400)                          |
|  - Objective: Eliminate non-convex bounding box error for facade surfaces.                        |
+---------------------------------------------------------------------------------------------------+
```

---

## Stage 5: Data Integrity & Evaluation Protocol

### 5.1 Automated Quality Gates
Implemented and verified in `utils/multidataset_harmonization.py` and `tests/test_phase19_multidataset.py`:
1. **MD5 Collision Screening:** Automated verification that no training image shares an exact MD5 hash with any validation or test image.
2. **64-bit DCT pHash Near-Duplicate Screening:** Rejects any candidate image having a perceptual Hamming distance $< 5$ against protected validation and test images.
3. **Coordinate Bounds Verification:** Bounding boxes must satisfy $x_c, y_c, w, h \in [0.0, 1.0]$, $w \ge 0.003$, $h \ge 0.003$, and $x_{min} \ge 0, x_{max} \le 1, y_{min} \ge 0, y_{max} \le 1$.
4. **Header and Decode Verification:** OpenCV and PIL decode verification to catch corrupt headers, truncated streams, or bad EXIF tags.
5. **Provenance Logging:** Every image records its source dataset, original ID, license, and ingestion timestamp.

### 5.2 Independent Holdout Evaluation Set (Holdout Set 2)
To prevent overfitting and repeated tuning on the Phase 17 test set:
- **Size:** 200 high-resolution outdoor scenes.
- **Composition:** 100 held-out ADE20K exterior scenes + 100 held-out Open Images verified entrance scenes.
- **Isolation Policy:** Strictly quarantined. Zero hyperparameter tuning, model selection, or threshold calibration permitted against Holdout Set 2.

### 5.3 Standardized Evaluation Metrics
All future model comparisons must report:
- Overall: mAP50, mAP50–95, Macro-F1, Micro-F1.
- Per-class: Precision, Recall, F1, AP50, TP, FP, FN.
- Latency & Resources: Model forward pass median/p90/p95, full-pipeline median/p90/p95 (with `torch.cuda.synchronize()`), peak allocated VRAM, peak reserved VRAM.
- Uncertainty: 1,000-resample scene-level bootstrap 95% confidence intervals.

---

## Test Suite Verification

Executed full repository test suite including newly implemented Phase 19 harmonization tests:
```bash
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pytest tests/ -q
........................................................................ [ 64%]
........................................                                 [100%]
112 passed in 29.91s
```
- **New Tests:** 10/10 tests in `tests/test_phase19_multidataset.py` passing (covering canonical taxonomy, mapping coverage, coordinate validation, pHash deduplication, mock split leakage, and artifact verification).
- **Existing Tests:** 102/102 tests passing without regressions.

---

## Concrete Recommendation for First Pilot Experiment

Before committing to a multi-thousand-image training pipeline, RoomAI should execute **Pilot B-1**:
1. **Pilot Scope:** 300 curated images (150 ADE20K outdoor building scenes + 150 Open Images residential entrance scenes).
2. **Storage Footprint:** ~420 MB (safe for the 27.18 GB disk limit).
3. **Training Parameters:** Fine-tune Model B6 at 800px, batch size 16, AMP fp16, 30 epochs with early stopping patience 15.
4. **Validation Gate:** The pilot candidate must demonstrate a Porch F1 $\ge 0.20$ and Macro-F1 $\ge 0.50$ on validation data while maintaining zero degradation on window and door precision before proceeding to full-scale training.

---

## Deliverables Summary

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **Multi-Dataset Audit Report** | [phase19_multidataset_audit.md](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase19_multidataset_audit.md) | Comprehensive engineering audit report covering Stages 1–5. |
| **Master Mapping CSV** | [phase19_dataset_mapping.csv](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase19_dataset_mapping.csv) | 47 machine-readable mapping rules with exact, defensible, ambiguous, and unsupported statuses. |
| **Training Plan JSON** | [phase19_training_plan.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase19_training_plan.json) | Complete experimental schema, hyperparameter configs, pilot parameters, hardware budgets, and evaluation metrics. |
| **Harmonization Utilities** | [multidataset_harmonization.py](file:///C:/Users/apaar/Desktop/RoomAI/utils/multidataset_harmonization.py) | Reusable module providing taxonomy, mapping registry, YOLO box validation, pHash, and split isolation checks. |
| **Automated Tests** | [test_phase19_multidataset.py](file:///C:/Users/apaar/Desktop/RoomAI/tests/test_phase19_multidataset.py) | 10 unit and integration tests verifying mappings, coordinate bounds, split isolation, and artifact structures. |
