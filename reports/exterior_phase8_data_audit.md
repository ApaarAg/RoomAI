# RoomAI Phase 8 — Exterior Detector Data Quality & Error-Reduction Audit

**Author:** Senior Computer Vision Engineer & Dataset Auditor  
**Date:** October 2026  
**Status:** Audit Completed & Experiment Plan Formulated  
**Baseline Model:** Model B2 ([`checkpoints/exterior_detector_b2_yolo11s_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b2_yolo11s_best.pt))  
**Candidate Model:** Model B3 ([`checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt))  
**Dataset Under Audit:** Canonical 10-Class Exterior Dataset ([`data/exterior_canonical/exterior_canonical.yaml`](file:///c:/Users/apaar/Desktop/RoomAI/data/exterior_canonical/exterior_canonical.yaml))  
**Hardware Profile:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), 24 GB System RAM, Windows 11, PyTorch 2.11.0+cu128  

---

## 1. Executive Summary

This phase conducts an exhaustive data-quality audit and diagnostic comparison for Group B exterior/building detection in RoomAI. In accordance with the prompt guidelines, **no model training was executed** and **no checkpoints were modified or overwritten**.

### Core Audit Findings
1. **Severe Rare-Class Data Starvation (`porch`):**  
   Across all 49,890 bounding boxes in the 726-image dataset, there exist **only 6 total porch annotations in 5 distinct images** (an extreme imbalance ratio of **3,332.5 : 1** against `window`). Critically, the training split contains **exactly 1 image with 1 single bounding box**.
2. **Confirmed Ground-Truth Error in Test Split (`oi_ext_test_0106.jpg`):**  
   In held-out test image `oi_ext_test_0106.jpg`, the `porch` class is labeled as `0.5 0.5 1.0 1.0` (a **100% full-frame box**). Because standard evaluation requires $\text{IoU} \ge 0.50$, any properly localized prediction on this building's actual entrance achieves $\text{IoU} \le 0.15$ against the full-frame ground truth, rendering a true positive **mathematically impossible** for any valid object detector.
3. **Severe Spatial Downsampling on Slender Features (`pillar_column`, `molding_deco`):**  
   19,420 annotations (**38.9% of the dataset**) are small objects ($\text{area} < 0.25\%$), and 1,960 boxes have a width or height $< 8$ pixels at 640px input resolution. Over 52% of `pillar_column` instances have sub-8px widths, attenuating feature maps in P3/P4/P5 strides and causing massive false-negative recall loss ($R = 0.2467$ in B2, $R = 0.2076$ in B3).
4. **Hierarchical Boundary Ambiguity (`building` vs. `facade`):**  
   CMP Facade annotations segment discrete 2D facade planes, whereas Open Images annotates 3D building envelopes. This semantic discrepancy induces overlapping predictions and false-positive penalties.
5. **Empirical Lesson from Model B3 (RFS):**  
   Model B3 applied Repeat Factor Sampling (15× on porch, 2× on building). While B3 improved overall mAP50 by **+2.32%** (from 0.3961 to 0.4193) and precision by **+12.99%** (from 49.91% to 62.90%), `porch` remained at **0.0000 recall and 0.0000 mAP50**. Oversampling 1 single image 15 times cannot teach a neural network to generalize to novel building styles or overcome an erroneous full-frame test label.

---

## 2. Checkpoint Preservation & Cryptographic Integrity

Before initiating data analysis, all Group B model checkpoints were inspected and cryptographically verified:

| Checkpoint Name | File Path | File Size (Bytes) | MD5 Checksum | Role & Status |
| :--- | :--- | :---: | :---: | :--- |
| **Model B2 (Standard)** | [`checkpoints/exterior_detector_b2_yolo11s_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b2_yolo11s_best.pt) | 19,163,034 | `14a931dd0adabc16cbd7b28ed1d17f66` | **Verified Preserved.** Production baseline. |
| **Model B3 (RFS Candidate)** | [`checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt) | 19,164,627 | `2c44291fa642aba3893db001520deba0` | **Verified Preserved.** Experimental candidate. |
| **Model B1 (Pilot)** | [`checkpoints/exterior_detector_b1_yolo11n_pilot_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b1_yolo11n_pilot_best.pt) | 5,453,594 | `c61d28da725a0b05dc45f0f7f19fcab3` | **Verified Preserved.** Pilot nano checkpoint. |
| **Model B2 (Pilot)** | [`checkpoints/exterior_detector_b2_yolo11s_pilot_best.pt`](file:///c:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b2_yolo11s_pilot_best.pt) | 19,162,266 | `c54b6de00c7dfb853edee3602382f548` | **Verified Preserved.** Pre-training checkpoint. |

*No checkpoints were modified, overwritten, or promoted. The production configuration remains locked on Model B2.*

---

## 3. Dataset Distribution & Geometry Audit

The canonical exterior dataset ([`data/exterior_canonical/`](file:///c:/Users/apaar/Desktop/RoomAI/data/exterior_canonical/)) comprises **726 total images** and **49,890 bounding boxes** across 10 canonical classes, sourced from:
- **CMP Facade Database:** 606 images (378 train, 114 val, 114 test)
- **Open Images V7 Exterior Subset:** 120 images (80 train, 20 val, 20 test)

### 3.1. Class Distribution Across Splits

| Class ID | Class Name | Train Boxes (Images) | Val Boxes (Images) | Test Boxes (Images) | Total Boxes (Images) | Train Prevalence | Class Imbalance Ratio (vs. Window) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `building` | 101 (63) | 22 (17) | 24 (18) | 147 (98) | 0.32% | 136.0 : 1 |
| **1** | `facade` | 2,411 (378) | 495 (113) | 579 (113) | 3,485 (604) | 7.58% | 5.7 : 1 |
| **2** | `window` | 12,224 (405) | 3,754 (121) | 4,017 (120) | 19,995 (646) | 38.42% | **1.0 : 1 (Dominant)** |
| **3** | `door` | 1,362 (318) | 367 (87) | 428 (95) | 2,157 (500) | 4.28% | 9.3 : 1 |
| **4** | `roof_cornice` | 2,797 (277) | 445 (70) | 666 (83) | 3,908 (430) | 8.79% | 5.1 : 1 |
| **5** | `balcony` | 1,091 (216) | 345 (74) | 393 (70) | 1,829 (360) | 3.43% | 10.9 : 1 |
| **6** | `porch` | **1 (1)** | **3 (2)** | **2 (2)** | **6 (5)** | **0.003%** | **3,332.5 : 1 (Acute)** |
| **7** | `pillar_column` | 2,270 (237) | 494 (60) | 896 (88) | 3,660 (385) | 7.14% | 5.5 : 1 |
| **8** | `shutter_blind` | 2,487 (145) | 901 (67) | 664 (34) | 4,052 (246) | 7.82% | 4.9 : 1 |
| **9** | `molding_deco` | 7,070 (372) | 1,543 (103) | 2,038 (109) | 10,651 (584) | 22.22% | 1.9 : 1 |
| **ALL** | **Total** | **31,814 (458)** | **8,369 (134)** | **9,707 (134)** | **49,890 (726)** | 100.0% | — |

### 3.2. Geometric Anomaly & Bounding Box Quality Audit

An exhaustive scan across all 49,890 lines in `data/exterior_canonical/labels/` revealed:
- **Corrupt / Invalid Boxes:** **0** (no negative coordinates, no non-numeric tokens, no invalid class IDs).
- **Out of Bounds Boxes:** **0** (all bounding coordinates lie within $[0.0, 1.0]$).
- **Empty Label Files:** **0** (every image has an associated non-empty annotation file).
- **Tiny Bounding Boxes ($\text{area} < 0.0025$):** **19,420 boxes (38.9%)**. Under 640×640 input resolution, these boxes occupy fewer than $32 \times 32$ pixels, causing high vulnerability to spatial pooling losses.
- **Thin Bounding Boxes ($\text{width} < 8\text{px}$ or $\text{height} < 8\text{px}$ at 640px):** **1,960 boxes (3.9%)**. Concentrated in `pillar_column` (slender vertical columns) and `molding_deco` (horizontal window sills).
- **Suspicious Full-Frame Boxes ($\text{area} \ge 85\%$):** **346 boxes**.
  - `facade`: 295 boxes (representing the entire building face in CMP photos).
  - `building`: 49 boxes (representing the entire house in Open Images).
  - `door`: 1 box ([`oi_ext_val_0098.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/data/exterior_canonical/images/val/oi_ext_val_0098.jpg), area 98.97% — extreme close-up of a front door).
  - `porch`: 1 box ([`oi_ext_test_0106.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/data/exterior_canonical/images/test/oi_ext_test_0106.jpg), area 100.0% — **erroneous ground-truth label**).

---

## 4. Deep Investigation of the `porch` Class

Every single `porch` annotation in the repository was extracted, localized, and visually rendered into individual crops and full-image verification sheets in [`artifacts/exterior_phase8/porch_audit/`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_audit/):

### 4.1. Complete Porch Annotation Inventory

| # | Split | Image Filename | Normalized YOLO Box `[xc, yc, w, h]` | Normalized Area (%) | Aspect Ratio (W/H) | Image Dimensions | Visual Verification Artifact |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Train** | `oi_ext_train_0012.jpg` | `[0.8000, 0.7021, 0.2750, 0.0833]` | 2.29% | 3.30 | 1024 × 768 | [`artifacts/exterior_phase8/porch_audit/porch_01_train_oi_ext_train_0012.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_audit/porch_01_train_oi_ext_train_0012.jpg) |
| **2** | **Val** | `oi_ext_val_0084.jpg` | `[0.4367, 0.8949, 0.8733, 0.2102]` | 18.36% | 4.16 | 1024 × 682 | [`artifacts/exterior_phase8/porch_audit/porch_02_val_oi_ext_val_0084.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_audit/porch_02_val_oi_ext_val_0084.jpg) |
| **3** | **Val** | `oi_ext_val_0085.jpg` | `[0.2810, 0.7091, 0.3260, 0.1881]` | 6.13% | 1.73 | 1024 × 683 | [`artifacts/exterior_phase8/porch_audit/porch_03_val_oi_ext_val_0085.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_audit/porch_03_val_oi_ext_val_0085.jpg) |
| **4** | **Val** | `oi_ext_val_0085.jpg` | `[0.8901, 0.5675, 0.2021, 0.1305]` | 2.64% | 1.55 | 1024 × 683 | [`artifacts/exterior_phase8/porch_audit/porch_04_val_oi_ext_val_0085.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_audit/porch_04_val_oi_ext_val_0085.jpg) |
| **5** | **Test** | `oi_ext_test_0106.jpg` | `[0.5000, 0.5000, 1.0000, 1.0000]` | **100.00%** | **1.00** | 683 × 1024 | [`artifacts/exterior_phase8/porch_audit/porch_05_test_oi_ext_test_0106.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_audit/porch_05_test_oi_ext_test_0106.jpg) |
| **6** | **Test** | `oi_ext_test_0113.jpg` | `[0.5375, 0.4240, 0.3313, 0.2354]` | 7.80% | 1.41 | 1024 × 768 | [`artifacts/exterior_phase8/porch_audit/porch_06_test_oi_ext_test_0113.jpg`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_audit/porch_06_test_oi_ext_test_0113.jpg) |

![Complete Porch Audit Contact Sheet](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/porch_complete_audit_gallery.jpg)

### 4.2. Porch Error Analysis & Key Conclusions
1. **The Test Full-Frame Error is Fatal for Metrics:**  
   In `oi_ext_test_0106.jpg`, Open Images annotators placed a $1.0 \times 1.0$ box over the entire image. The true porch is a front entryway canopy spanning $\approx [0.48, 0.48, 0.28, 0.22]$ (area $\approx 6.2\%$). Any model that accurately predicts this localized porch receives an $\text{IoU} \approx 0.062$ against the $1.0 \times 1.0$ ground truth, failing the $\text{IoU} \ge 0.50$ threshold. Thus, **50% of the test split for porch is mathematically unwinnable**.
2. **Extreme Cross-Image Semantic Inconsistency:**  
   The single training instance (`oi_ext_train_0012.jpg`) is a tiny side entrance overhang (2.29% area). The validation instances are wide suburban decks (18.36% area). The model is given zero training signal on standard covered porticos or colonial columns.
3. **Targeted Acquisition Requirement:**  
   To achieve reliable, deployable detection ($>60\%$ recall), the model requires **150–200 diverse, accurately annotated porch images** (120 train, 40 val, 40 test).

---

## 5. Duplicate Images & Scene-Prefix Leakage Audit

We audited exact content hashes (MD5) and perceptual hashes (dHash) across all 726 images:
- **Exact Cross-Split File Duplicates:** **0**. All train, val, and test image files are distinct byte-streams.
- **Open Images Cross-Split Correlation:** **0 pairs** with perceptual Hamming distance $\le 12$. The 120 Open Images photographs depict independent geographic scenes.
- **CMP Facade Near-Duplicates:**  
  At Hamming distance $\le 8$, exactly **2 cross-split pairs** were identified:
  - `cmp_train_0015.jpg` $\leftrightarrow$ `cmp_test_0014.jpg` (Hamming distance 8)
  - `cmp_train_0244.jpg` $\leftrightarrow$ `cmp_test_0046.jpg` (Hamming distance 8)
  
  Visual inspection ([`artifacts/exterior_phase8/cmp_pairs/`](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/cmp_pairs/)) shows these are different buildings in Prague with similar neoclassical window grid geometry, not duplicate photographs of the same scene.
- **Split Preservation:**  
  The dataset strictly preserved the original CTU Prague benchmark splits (`train` $\rightarrow$ `train`, `eval` $\rightarrow$ `val`, `test` $\rightarrow$ `test`). No silent split manipulation occurred.

---

## 6. Fair Comparison: Model B2 vs. Model B3

Both models were evaluated on the held-out test split using Ultralytics standard 101-point COCO interpolation across confidence sweeps $[0.001, 1.0]$ and IoU thresholds $[0.50:0.05:0.95]$:

![B2 vs B3 Per-Class Performance Comparison](file:///c:/Users/apaar/Desktop/RoomAI/artifacts/exterior_phase8/b2_vs_b3_per_class_comparison.png)

### 6.1. Overall Performance Comparison

| Metric | Baseline B2 (Standard) | Candidate B3 (RFS) | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Validation mAP@50** | 0.4484 | **0.5090** | +0.0606 | +13.51% |
| **Validation mAP@50–95** | 0.2689 | **0.3173** | +0.0484 | +18.00% |
| **Held-Out Test mAP@50** | 0.3961 | **0.4193** | +0.0232 | +5.86% |
| **Held-Out Test mAP@50–95** | 0.2427 | **0.2572** | +0.0145 | +5.97% |
| **Held-Out Test Precision** | 0.4991 | **0.6290** | **+0.1299** | **+26.03%** |
| **Held-Out Test Recall** | 0.3990 | **0.3993** | +0.0003 | +0.08% |
| **Held-Out Test F1 Score** | 0.4435 | **0.4885** | **+0.0450** | **+10.15%** |

### 6.2. Per-Class Test Breakdown (B2 vs. B3)

| Class ID | Class Name | B2 Test mAP50 | B3 Test mAP50 | Delta mAP50 | B2 Recall | B3 Recall | Delta Recall | B2 Precision | B3 Precision | Diagnosis & Impact of RFS |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `building` | 0.4280 | **0.4736** | **+0.0456** | 0.3750 | **0.4583** | **+0.0833** | 0.5045 | 0.3738 | **Clear win:** 2× RFS boosted building recall by +8.33% |
| **1** | `facade` | 0.2703 | **0.2770** | +0.0067 | 0.2401 | **0.2623** | +0.0222 | 0.4608 | 0.4340 | Minor recall gain; boundary overlap remains |
| **2** | `window` | **0.8029** | 0.8004 | -0.0025 | **0.7872** | 0.7707 | -0.0165 | 0.7592 | **0.8037** | Stable dominant class; precision up +4.45% |
| **3** | `door` | 0.4134 | **0.4480** | **+0.0346** | **0.4346** | 0.4065 | -0.0281 | 0.5455 | **0.6056** | Precision improved significantly (+6.01%) |
| **4** | `roof_cornice` | 0.6735 | **0.6751** | +0.0016 | **0.6862** | 0.6426 | -0.0436 | 0.6222 | **0.7170** | Precision improved (+9.48%), slight recall loss |
| **5** | `balcony` | 0.5301 | **0.5934** | **+0.0633** | 0.5929 | **0.6056** | +0.0127 | 0.5096 | **0.6073** | **Strong gain:** +6.33% mAP50, +9.77% precision |
| **6** | `porch` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000* | **Failure:** 15× oversampling 1 image produced 0 detections |
| **7** | `pillar_column` | 0.1906 | **0.2213** | **+0.0307** | **0.2467** | 0.2076 | -0.0391 | 0.3847 | **0.4558** | Precision up (+7.11%), but recall dropped (-3.91%) |
| **8** | `shutter_blind` | 0.2299 | **0.2518** | **+0.0219** | 0.1943 | **0.2100** | +0.0157 | 0.6565 | **0.6957** | Modest gain from 2× RFS |
| **9** | `molding_deco` | 0.4217 | **0.4520** | **+0.0303** | **0.4331** | 0.4293 | -0.0038 | 0.5480 | **0.5843** | Precision improved (+3.63%) |

*Note: In Ultralytics `DetMetrics`, when 0 predictions are made for a class, precision evaluates to 1.0 by convention in the absence of false positives; with 0 recall, F1 remains 0.0000.*

### 6.3. False Positives vs. False Negatives Analysis
- **False Positives Greatly Reduced in B3:** Precision jumped from **49.91% to 62.90%** (+12.99%). RFS stabilized background suppression on balconies, roof cornices, doors, and window blinds.
- **False Negatives Remain Unresolved on Rare Classes:** Overall recall remained flat (39.90% vs. 39.93%). While `building` recall improved (+8.33%), `pillar_column` recall suffered (-3.91%), and `porch` remained completely undetected.

---

## 7. Actionable Experiment Plan

We propose **two controlled experiments**, with Experiment B4 designated as the **Primary Recommendation**.

> [!IMPORTANT]
> **Test Split Disclosure:** The held-out test split (134 images) has been utilized for baseline B2 and candidate B3 evaluations in Phases 3 and 8. To prevent data leakage and benchmark contamination, **all threshold tuning and hyperparameter decisions must be performed exclusively on the validation split**. Final test-set evaluation must be reported as a reused diagnostic split.

### Primary Recommendation: Experiment B4 — Data-Centric Quality Correction & Rare-Class Curation

**Rationale:** Algorithmic techniques (RFS, weighting) cannot synthesize visual variance that does not exist in the training data. Fixing the known test error and injecting 150–200 diverse porch images directly resolves the root cause.

1. **Step 1 — Correct Test Split Annotation Artifact:**
   - Correct the bounding box in `data/exterior_canonical/labels/test/oi_ext_test_0106.txt` from `[0.5, 0.5, 1.0, 1.0]` to the true localized portico canopy: `[0.4850, 0.4780, 0.2840, 0.2160]`.
   - Update `split_manifest.json` and document the exact coordinate revision in the audit ledger.
2. **Step 2 — Targeted Data Acquisition:**
   - Ingest 160 curated, verified exterior images containing residential porches, colonnades, and porticos from open-source architectural sources (Open Images V7, Places365).
   - Partition deterministically: **100 train, 30 val, 30 test**.
   - Annotate using the canonical 10-class taxonomy with strict boundaries.
3. **Step 3 — Re-Train YOLO11s (Experiment B4):**
   - Resolution: 640×640, epochs: 20, batch: 4, optimizer: SGD, initial lr: 0.01.
4. **Success Criteria:**
   - `porch` test recall $> 0.50$ (currently $0.00$).
   - `porch` test mAP50 $> 0.40$ (currently $0.00$).
   - Overall test mAP50 $> 0.4500$ (B2: $0.3961$, B3: $0.4193$).
   - No regression on `window` ($> 0.79$) or `roof_cornice` ($> 0.65$).

### Secondary Recommendation: Experiment B5 — High-Resolution (800×800) Slender Feature Optimization

**Rationale:** Over 52% of `pillar_column` instances have widths $< 8$ pixels at 640px. Model A3 (Indoor) demonstrated that scaling resolution from 640px to 800px provided a **+9.93% relative gain in mAP50–95** and dramatically improved recall on slender furniture legs.

1. **Interventions:**
   - Scale input resolution to 800×800 pixels.
   - Adjust anchor-free loss weight to penalize false negatives on slender objects (`pillar_column`, `molding_deco`).
2. **Success Criteria:**
   - `pillar_column` test recall $> 0.35$ (B2: $0.2467$, B3: $0.2076$).
   - `pillar_column` test mAP50 $> 0.26$ (B2: $0.1906$).
   - Pipeline latency $< 40$ ms on RTX 4060 GPU.

---

## 8. Final Synthesis

### The Three Most Important Bottlenecks
1. **Extreme Data Starvation & Full-Frame Annotation Artifact on `porch`:**  
   Having only 1 training box and an erroneous 100% full-frame test box guarantees zero measured performance on the porch class regardless of model architecture.
2. **Feature Downsampling on Slender Architectural Features (`pillar_column`, `molding_deco`):**  
   Sub-8-pixel widths at 640px resolution cause extreme false-negative rates on columns and trim.
3. **Semantic Hierarchy Conflict (`building` vs. `facade`):**  
   Inconsistent co-labeling of whole buildings versus facade planes penalizes precision through false cross-class competitions.

### Recommended Next Experiment
**Experiment B4 (Data-Centric Quality Correction & Rare-Class Curation).** Data curation must precede architecture tuning.

### Expected Runtime on Target Hardware (RTX 4060 Laptop GPU, 8 GB VRAM)
- **Dataset Size:** $\approx 600$ training images (458 current + 100 new porch images + 42 balance images).
- **Training Duration:** $\approx 28$ seconds/epoch $\times$ 20 epochs = **$\approx 9.3$ minutes**.
- **Peak VRAM:** $\approx 2,450$ MiB (leaving $>5.5$ GB VRAM headroom).

### Evidence Needed to Replace Model B2 in Production
To justify promoting a new exterior candidate to replace production baseline B2:
1. `porch` test recall must reach $\ge 0.50$ with mAP50 $\ge 0.40$.
2. Overall held-out test mAP50 must achieve $\ge 0.4500$ (representing a $+0.054$ improvement over B2).
3. Test F1 score must exceed $\ge 0.5000$ (B2: $0.4435$).
4. Real inference latency must remain $\le 35$ ms per image on the RTX 4060 GPU.
5. All regression tests in `tests/test_exterior_ensemble.py` and `tests/test_detection_deployment_readiness.py` must pass unconditionally.
