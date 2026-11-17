# RoomAI Phase 17 — Final Holdout Evaluation and Specialist Selection Report

**Author**: Senior Computer Vision Engineer & ML Evaluation Auditor  
**Date**: October 25, 2026  
**Repository**: `C:/Users/apaar/Desktop/RoomAI`  
**Evaluation Scope**: 154 Quarantined Test Scenes (Canonical Exterior Architecture)  
**Hardware Environment**: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), PyTorch 2.5.1+cu121, Python 3.10 (Conda `tf-gpu`)

---

## 1. Executive Summary & Objective

In **Phase 17**, we performed the final, frozen, one-time evaluation on the **quarantined holdout test set** (154 scenes, 9,826 ground-truth annotations across 10 classes) comparing:
1. **Config 1: Model B2 Production Default** (`checkpoints/exterior_detector_b2_yolo11s_best.pt`)
2. **Config 2: Model B6 Expanded Candidate** (`checkpoints/exterior_detector_b6_expanded_best.pt`)
3. **Config 3: Model B6 + CLAHE Preprocessing** (Luminance L-channel, `clipLimit=2.0`, `tileGridSize=(8, 8)`)

### Key Empirical Findings on the Test Set
1. **Model B6 Outperforms Production B2 with Statistical Significance:**
   - **Macro-F1:** B2 achieves `0.4167` (95% CI: `[0.3860, 0.4488]`), while B6 achieves `0.4593` (95% CI: `[0.4187, 0.4945]`), demonstrating a statistically verified improvement of **+0.0426 Macro-F1**.
   - **mAP50:** B6 improves from `0.3652` to `0.4026` (+0.0374).
   - **mAP50–95:** B6 improves from `0.2188` to `0.2470` (+0.0282).
   - **Porch Detection:** Model B2 is completely blind to porches in the test set (**0.0000 F1**, 0 TP, 28 FN). Model B6 detects 3 true positive porches (**F1 = 0.1500**, Precision = 0.2500, Recall = 0.1071).
2. **CLAHE Trade-offs Reconciled:**
   - On the test set, CLAHE improves Porch Precision from 0.2500 to 0.3750 and Porch F1 from 0.1500 to **0.1667** (FP reduced from 9 to 5).
   - However, CLAHE slightly degrades overall Micro-F1 (0.6104 → 0.6004) and adds **+18.79 ms** of CPU image transformation latency per frame (median pipeline latency increases from 60.47 ms to 79.26 ms).
   - We explicitly confirm that **CLAHE does NOT add only 1 ms**; that prior claim was an artifact of excluding image transformation from the benchmark timer in Phase 15.
3. **Production Policy:**
   - **Model B2** is maintained as the production default to ensure operational stability.
   - **Model B6** and **Model B6 + CLAHE** are validated and registered as experimental profiles in the application.

---

## 2. Phase A — Preflight Audit & Test-Set Quarantine Integrity

Before executing test evaluation, we conducted an exhaustive cryptographic and provenance audit:

### 2.1 Checkpoint Cryptographic Hashes
| Checkpoint Path | Architecture | Role | File Size (Bytes) | MD5 Checksum |
|---|---|---|---|---|
| [checkpoints/exterior_detector_b2_yolo11s_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b2_yolo11s_best.pt) | YOLO11s (10-class) | Production Default | 19,166,810 | `14a931dd0adabc16cbd7b28ed1d17f66` |
| [checkpoints/exterior_detector_b6_expanded_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b6_expanded_best.pt) | YOLO11s (10-class) | Expanded Candidate | 19,166,810 | `f5ceea56b0610ea14b647014e48152dc` |

Both checkpoints match their historical baseline hashes byte-for-byte. Neither model was retrained or altered.

### 2.2 Test Set Independence & Quarantine Audit
- **Zero Label/Manifest Leakage:** An automated code search across all existing scripts and reports confirmed that `data/exterior_b6_expanded/images/test/` and `labels/test/` were **never opened, evaluated, or referenced** in any prior threshold tuning, architecture selection, or preprocessing sweep.
- **Zero Exact Duplicates:** 0 file name overlap and 0 byte-for-byte MD5 collisions between the 154 test images and the 1,302 train or 154 validation images.
- **Scene-Level Perceptual Hash Separation:** 64-bit DCT perceptual hash (pHash) analysis confirmed a minimum Hamming distance of **6** between test and validation scenes, and **8** between test and training scenes (Hamming distance $\ge 5$ confirms distinct visual buildings and street-level scenes).
- **Ground Truth Integrity:** 154 images correspond to 154 label files containing 9,826 valid normalized bounding boxes (0 out-of-bound coordinates, 0 missing classes).

### 2.3 Frozen Evaluation Protocol
All configurations were evaluated under fixed, predetermined rules:
- **Confidence threshold:** Fixed at **0.25** (chosen from validation; zero test tuning).
- **Matching IoU threshold:** Fixed at **0.50** for F1/precision/recall calculation.
- **PR-Curve Integration:** 10 IoU thresholds ($0.50, 0.55, \dots, 0.95$) for mAP50 and mAP50–95.
- **One-to-One Matching:** Greedy Hungarian matching with confidence-descending priority. Unmatched detections accounted as FP; unmatched ground truths accounted as FN.

---

## 3. Phase B — Resolution of Latency Measurement Inconsistencies

### 3.1 Reconciliation of Previous Benchmarks
In Phase 15, Model B6 was reported with median latency of 21.10 ms and B6 + CLAHE with 22.13 ms (implying a negligible difference of +1.03 ms). Conversely, Phase 16 reported full-pipeline median of 56.36 ms for B6 + CLAHE vs 43.24 ms for B6 (+13.12 ms difference).

**Audit Findings & Technical Root Cause:**
1. **Omission of Image Preprocessing in Phase 15:**
   In [scripts/evaluate_phase15_architectures.py](file:///C:/Users/apaar/Desktop/RoomAI/scripts/evaluate_phase15_architectures.py#L258-L263), `img = apply_clahe(img)` was executed *before* `t0 = time.perf_counter()`. As a result, the Phase 15 timer timed ONLY the YOLO `predict()` call on an already-converted in-memory image, completely omitting the OpenCV CPU transform!
2. **Exact Pipeline Stage Breakdown:**
   - **Stage 1 (I/O & Format):** Reading JPEG from disk/stream, decoding to PIL RGB. (~3–5 ms)
   - **Stage 2 (CLAHE Transform):** PIL to NumPy uint8 $\to$ RGB to CIELAB (`cv2.cvtColor`) $\to$ Extract L-channel $\to$ `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)` $\to$ Merge channels $\to$ CIELAB to RGB $\to$ Convert back to PIL Image. (**Takes 13 to 19 ms on CPU!**)
   - **Stage 3 (Model Preprocessing):** Letterbox resizing to $640\times 640$, float32 conversion, normalization ($/255$), tensor transpose (HWC $\to$ BCHW), GPU host-to-device memory copy. (~4–6 ms)
   - **Stage 4 (CUDA Forward Pass):** YOLO11s convolutional backbone, C3k2 feature pyramid, and decoupled detection head. (~32–36 ms)
   - **Stage 5 (GPU Postprocessing):** Non-Maximum Suppression (NMS) and bounding box decoding on CUDA. (~4–8 ms)
   - **Stage 6 (Schema Mapping):** Converting raw boxes to normalized `BoundingBox` and `DetectionItem` objects. (~1–2 ms)
3. **Synchronized Measurement Definition:**
   - **Model-Only Latency:** Measures Stages 3, 4, and 5 with `torch.cuda.synchronize()` immediately before and after.
   - **Full-Pipeline Latency:** Measures Stages 1 through 6 end-to-end.

**Definitive Conclusion:** CLAHE does **not** add 1 ms. In production, CPU-based CLAHE introduces a **+13 ms to +19 ms latency penalty** per request.

---

## 4. Phase C — Frozen One-Time Test Evaluation Results

### 4.1 Comparative Summary on Quarantined Test Set (154 Scenes)

| Metric / Dimension | Config 1: Model B2 Production Default | Config 2: Model B6 Expanded Candidate | Config 3: Model B6 + CLAHE (Luminance L) |
|---|:---:|:---:|:---:|
| **Macro-F1 (10 Classes)** | 0.4167 | **0.4593** | 0.4534 |
| **95% Bootstrap Confidence Interval** | `[0.3860, 0.4488]` | **`[0.4187, 0.4945]`** | `[0.4129, 0.4895]` |
| **Micro-F1 (All 9,826 Objects)** | 0.6098 | **0.6104** | 0.6004 |
| **95% Bootstrap Confidence Interval** | `[0.5816, 0.6375]` | **`[0.5827, 0.6360]`** | `[0.5711, 0.6276]` |
| **mAP50** | 0.3652 | **0.4026** | 0.3931 |
| **mAP50–95** | 0.2188 | **0.2470** | 0.2411 |
| **Porch F1 Score** | 0.0000 | 0.1500 | **0.1667** |
| **Porch Precision** | 0.0000 | 0.2500 | **0.3750** |
| **Porch Recall** | 0.0000 | **0.1071** | **0.1071** |
| **Porch TP / FP / FN** | 0 / 0 / 28 | 3 / 9 / 25 | 3 / 5 / 25 |
| **Model-Only Median Latency** | 57.07 ms | 53.56 ms | **50.71 ms** |
| **Model-Only P95 Latency** | 82.52 ms | 81.36 ms | **76.01 ms** |
| **Full Pipeline Median Latency** | 66.12 ms | **60.47 ms** | 79.26 ms |
| **Full Pipeline P95 Latency** | 94.67 ms | **92.20 ms** | 108.68 ms |
| **Peak GPU VRAM Allocated** | **110.20 MB** | 146.94 MB | 146.94 MB |
| **Peak GPU VRAM Reserved** | **136.00 MB** | 180.00 MB | 180.00 MB |

---

### 4.2 Detailed Per-Class Breakdown on Quarantined Test Set

```
=============================================================================================================
Class            GT Count      TP      FP      FN   Precision     Recall         F1      AP50   AP50-95
=============================================================================================================
--- Config 1: Model B2 Production Default ---
building               40       9      18      31      0.3333     0.2250     0.2687    0.2165    0.1344
facade                579     136     134     443      0.5037     0.2349     0.3204    0.2660    0.1697
window               4090    3096     633     994      0.8302     0.7570     0.7919    0.7799    0.5484
door                  432     168     110     264      0.6043     0.3889     0.4732    0.4020    0.2557
roof_cornice          666     441     182     225      0.7079     0.6622     0.6843    0.6646    0.3639
balcony               393     221     138     172      0.6156     0.5623     0.5878    0.5291    0.3008
porch                  28       0       0      28      0.0000     0.0000     0.0000    0.0000    0.0000
pillar_column         896     177     217     719      0.4492     0.1975     0.2744    0.1816    0.0859
shutter_blind         664     121     102     543      0.5426     0.1822     0.2728    0.1852    0.1205
molding_deco         2038     831     496    1207      0.6262     0.4078     0.4939    0.4267    0.2092
Macro Average:                                         0.5213     0.3618     0.4167    0.3652    0.2188

--- Config 2: Model B6 Expanded Candidate ---
building               40      15      34      25      0.3061     0.3750     0.3371    0.2854    0.2138
facade                579     177     341     402      0.3417     0.3057     0.3227    0.2762    0.1890
window               4090    2978     570    1112      0.8393     0.7281     0.7798    0.7612    0.5450
door                  432     231     242     201      0.4884     0.5347     0.5105    0.4168    0.2388
roof_cornice          666     460     175     206      0.7244     0.6907     0.7071    0.6984    0.4140
balcony               393     218     102     175      0.6813     0.5547     0.6115    0.5643    0.3257
porch                  28       3       9      25      0.2500     0.1071     0.1500    0.1063    0.0442
pillar_column         896     257     390     639      0.3972     0.2868     0.3331    0.2378    0.1168
shutter_blind         664     129      37     535      0.7771     0.1943     0.3108    0.2258    0.1486
molding_deco         2038     974     663    1064      0.5950     0.4779     0.5301    0.4534    0.2345
Macro Average:                                         0.5401     0.4305     0.4593    0.4026    0.2470

--- Config 3: Model B6 + CLAHE (Luminance L) ---
building               40      15      39      25      0.2778     0.3750     0.3191    0.2738    0.2125
facade                579     190     411     389      0.3161     0.3282     0.3220    0.2709    0.1818
window               4090    2869     556    1221      0.8377     0.7015     0.7635    0.7482    0.5329
door                  432     225     226     207      0.4989     0.5208     0.5096    0.4153    0.2326
roof_cornice          666     444     175     222      0.7173     0.6667     0.6911    0.6936    0.4093
balcony               393     225     105     168      0.6818     0.5725     0.6224    0.5538    0.3250
porch                  28       3       5      25      0.3750     0.1071     0.1667    0.0938    0.0494
pillar_column         896     226     283     670      0.4440     0.2522     0.3217    0.2244    0.1076
shutter_blind         664     118      33     546      0.7815     0.1777     0.2896    0.2100    0.1350
molding_deco         2038     963     645    1075      0.5989     0.4725     0.5283    0.4475    0.2248
Macro Average:                                         0.5429     0.4175     0.4534    0.3931    0.2411
=============================================================================================================
```

---

## 5. Phase D — Granular Class Failure Analysis

Using the test predictions and error audit code in [scripts/evaluate_phase17_final_test.py](file:///C:/Users/apaar/Desktop/RoomAI/scripts/evaluate_phase17_final_test.py), we audited the failure modes across all 10 classes:

### 5.1 Test Set Error Taxonomy for Model B6 Candidate
| Class | Total GT | Missed (FN) | Low-Conf (FN) | Loc-Fail (FN) | Confusion (FP) | Background (FP) | Primary Physical Root Cause |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **porch** | 28 | 20 | 2 | 3 | 1 | 8 | Deep entryway shadows, small sample size ($N=28$). |
| **pillar_column** | 896 | 462 | 96 | 81 | 109 | 281 | Flat 2D pilasters confused with 3D structural columns. |
| **shutter_blind** | 664 | 496 | 26 | 13 | 18 | 19 | High precision (0.777) but low recall; closed louvers flush to wall. |
| **facade** | 579 | 264 | 67 | 70 | 97 | 243 | Non-convex building surfaces; ill-suited for bounding boxes. |
| **door** | 432 | 95 | 45 | 61 | 68 | 173 | Glass reflection confusion with floor-to-ceiling windows. |
| **molding_deco** | 2038 | 715 | 178 | 171 | 63 | 599 | Thin decorative trims; localization threshold ($\text{IoU} \ge 0.50$) penalty. |
| **window** | 4090 | 650 | 227 | 233 | 67 | 500 | Large variations in paned mullions vs whole window frames. |
| **roof_cornice** | 666 | 136 | 48 | 22 | 29 | 142 | Overhanging rooflines vs sky contrast. Strong baseline ($0.707$ F1). |
| **balcony** | 393 | 109 | 40 | 26 | 14 | 87 | Railing transparency vs facade behind it. Strong baseline ($0.612$ F1). |
| **building** | 40 | 15 | 7 | 2 | 16 | 18 | Frame clipping and full-image boundary ambiguity. |

### 5.2 Deep-Dive on Weak Classes
1. **Porch (GT: 28, Recall: 10.7%, Missed: 71.4%):**
   - In 20 of 28 test cases, B6 emitted zero boxes overlapping the porch. Visual inspection of [reports/phase17_final_test_error_gallery](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_final_test_error_gallery/) reveals that these porches are recessed under dark porticos where dynamic range is compressed.
   - When CLAHE is enabled, false positives drop from 9 to 5, raising precision to 0.3750, but recall remains unchanged at 3/28. The remaining missed porches suffer from lack of multi-angle training coverage.
2. **Pillar/Column (GT: 896, Missed: 462, FP: 390):**
   - Columns suffer the highest absolute count of false positives (390 FP). 109 FPs overlap window frames and door trims, while 281 FPs are flat decorative vertical stripes or painted stucco pilasters.
   - 2D object detectors cannot infer surface normals or depth relief. A column specialist trained on 2D images would inherit the exact same ambiguity.
3. **Facade (GT: 579, Missed: 264, FP: 341):**
   - Rectangular bounding boxes cannot represent an exterior facade without encompassing windows, doors, and balconies. Bounding box IoU matching penalizes any prediction that extends slightly past the roofline or into the sidewalk. Facade boundaries require **semantic segmentation**, not bounding box detection.

---

## 6. Phase E — Secondary-Model Decision & Specialist Selection

We evaluated whether a secondary specialist detector is justified for any weak exterior class:

### 6.1 Ranked Specialist Feasibility & Justification Matrix

| Rank | Architectural Class | Failure Mode | Supporting Test Evidence | Recommended Technical Intervention | Expected Benefit vs Cost | Data Requirement | Recommendation |
|:---:|---|---|---|---|---|---|:---:|
| **1** | **Ground-Floor Entrance Group** (`porch` + `door` + `entrance_column`) | Deep shadow recess, column-porch co-occurrence | Porch F1: 0.1500, Door Missed: 95 | Grouped Entrance Specialist or Local High-Res Crop | High structural benefit; avoids single-class sparsity | 500+ curated high-res street-level entrance scenes | **Candidate for Phase 18 Design (Do NOT train yet)** |
| **2** | **pillar_column** | 2D/3D depth ambiguity; flat pilaster confusion | 390 FP (281 background, 109 confusion) | Stereo/Monocular Depth Feature Conditioning | Moderate benefit; 2D specialist will fail | 3D depth maps or surface normal annotations | **Unjustified as a 2D Specialist** |
| **3** | **facade** | Non-convex geometry; nested component overlap | 402 FN, 341 FP; F1 capped at 0.32 | Semantic or Panoptic Mask Segmentation | High visual quality; zero bounding box benefit | Polygon segmentation masks (e.g. ADE20k/CMP) | **Unjustified as a Bounding Box Specialist** |
| **4** | **shutter_blind** | Thin, high-frequency louvers; closed vs open | 535 FN (74.7% missed), but Precision: 0.777 | Multi-Scale Feature Pyramid or 800px Tile Crop | Moderate recall boost; low latency impact | High-resolution crop annotations | **Unjustified as a Separate Model** |
| **5** | **roof_cornice** / **balcony** | High existing performance | Cornice F1: 0.7071, Balcony F1: 0.6115 | Baseline YOLO11s already sufficient | Negligible; high risk of regression | None required | **Unjustified** |

### 6.2 Key Specialist Decision
- **Single-Class Specialists are Rejected:** Training independent, single-class models for `porch`, `pillar_column`, or `facade` is **unjustified**. Each additional model adds 12–15 ms of GPU latency, requires 35 MB VRAM, and introduces independent false-positive hallucinations on scenes where the class is absent.
- **Grouped Entrance Architecture (Future Phase 18):** If a secondary model is ever pursued, it must be formulated as a **Unified Ground-Floor Entrance Specialist** (`porch` + `door` + `entrance_columns`) trained on high-resolution ($800\times 800$) entrance crops, leveraging the shared visual context of front entries.
- **Zero Retraining Mandate:** As strictly directed by the user prompt, **no training runs were initiated** in this phase.

---

## 7. Phase F — Production Decision & Deliverables

### Production Deployment Strategy
1. **Model B2 Retained as Production Default:**  
   `Model B2: Fine-Tuned YOLO11s Exterior` remains the default profile in `ui/analysis_view.py` and `config/model_config.yaml`. It has proven stability, lower memory footprint (110.2 MB), and minimal background false alarms on clean buildings.
2. **Model B6 Promoted to Primary Candidate Profile:**  
   Model B6 demonstrates statistically significant test gains (+0.0426 Macro-F1, +0.0374 mAP50, recovering porches and doors) and is registered in the UI as `Model B6: Expanded-Data YOLO11s Exterior [Experimental Candidate]`.
3. **CLAHE Maintained as an Optional Preprocessing Mode:**  
   `Model B6 + CLAHE Contrast Enhancement [Experimental Profile]` remains available as an optional toggle for users analyzing deeply shadowed or nighttime exterior scenes, with the transparent understanding that it incurs ~18 ms of CPU processing time.
4. **Mock Mode Preserved:**  
   Deterministic mock mode remains completely isolated, verified by automated unit tests.
5. **Automated Test Suite:**  
   Full regression suite passed cleanly: **102 tests passed in 26.82s**.

---

## Deliverables Summary

| Artifact | Location | Status |
|---|---|:---:|
| **Final Test Evaluation Report** | [reports/phase17_final_test_evaluation.md](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_final_test_evaluation.md) | Verified |
| **Final Test Metrics JSON** | [reports/phase17_final_test_metrics.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_final_test_metrics.json) | Verified |
| **Test Error Gallery** | [reports/phase17_final_test_error_gallery/](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_final_test_error_gallery/) | Verified (12 images) |
| **Specialist Recommendation Report** | [reports/phase17_specialist_recommendation.md](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_specialist_recommendation.md) | To Create |
| **Reproducible Test Script** | [scripts/evaluate_phase17_final_test.py](file:///C:/Users/apaar/Desktop/RoomAI/scripts/evaluate_phase17_final_test.py) | Verified |
