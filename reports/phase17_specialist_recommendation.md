# RoomAI Phase 17 — Specialist Selection & Architectural Recommendation

**Author**: Senior Computer Vision Engineer, Object Detection Researcher & ML Systems Mentor  
**Date**: October 25, 2026  
**Repository**: `C:/Users/apaar/Desktop/RoomAI`  
**Basis**: Empirical Findings from 154 Quarantined Test Scenes (9,826 Annotations) and Controlled Phase 15/16/17 Experiments

---

## 1. Executive Summary

Based on final evaluation across the quarantined test set and detailed error taxonomy analysis, this report provides the definitive architectural recommendation regarding secondary specialist models in RoomAI.

### Core Recommendation: Data Curation & Segmentation First; No Specialist Retraining
1. **Single-Class Specialists are Rejected:** The empirical evidence from Phase 15 and Phase 17 demonstrates that training standalone 2D bounding-box specialist detectors for individual weak classes (`porch`, `pillar_column`, or `facade`) is fundamentally flawed. Standalone single-class models lack holistic architectural context, produce excessive background hallucinations, and impose a prohibitive +50% to +60% latency penalty.
2. **Top Architectural Intervention Classes:**
   - **Ground-Floor Entrances (`porch` + `door`):** Suffer from severe illumination/shadow clipping and vertical context truncation.
   - **Structural Columns (`pillar_column`):** Suffer from 2D depth ambiguity (cannot distinguish flat pilasters from freestanding 3D columns).
   - **Wall Planes (`facade`):** Suffer from rectangular bounding-box geometric incompatibility.
3. **Strategic Decision:** **Do NOT train a new specialist model at this time.**
   - For `facade`: Transition from bounding boxes to **semantic segmentation**.
   - For `pillar_column`: Incorporate **monocular depth estimation** or reject pilaster boxes via surface normal filters.
   - For `porch`: Formulate a **Unified Ground-Floor Entrance Group** (`porch` + `entrance_door` + `entrance_columns`) trained on $800\times 800$ resolution crops ONLY after collecting 500+ dedicated, multi-angle entrance scenes.

---

## 2. Top Classes Requiring Intervention & Root Cause Analysis

Based on the 9,826 test ground truth annotations and the error taxonomy from `reports/phase17_final_test_metrics.json`:

```
Class            Test GT   Missed FN   Low-Conf   Loc-Fail   Confusion FP   Background FP    Test F1
--------------------------------------------------------------------------------------------------
porch                 28          20          2          3              1               8     0.1500
pillar_column        896         462         96         81            109             281     0.3331
shutter_blind        664         496         26         13             18              19     0.3108
facade               579         264         67         70             97             243     0.3227
door                 432          95         45         61             68             173     0.5105
```

### 2.1 Porch (Test F1: 0.1500 | Recall: 10.7% | 71.4% Missed)
- **Visual Evidence:** Inspection of `b6_cand_oi_curated_porch_test_...` images shows that front porches are typically covered by deep roofs or pediments that cast heavy shadows over the entryway. At $640\times 640$ full-building resolution, the entryway occupies less than $5\%$ of image pixels, compressing subtle portico floor and column features into low-contrast blobs.
- **Root Cause:** Dual failure of **spatial resolution** (features too small on whole-building scale) and **dynamic range** (shadow underexposure). Furthermore, single-class models hallucinate porches on flat driveways because they lack full vertical building context.

### 2.2 Pillar / Column (Test F1: 0.3331 | 462 Missed | 390 FP)
- **Visual Evidence:** 281 background false positives occur on decorative vertical wall grooves, stucco stripes, and corner pilasters. 109 confusion false positives occur on window mullions and door jambs.
- **Root Cause:** **Fundamental physical ambiguity of 2D bounding boxes.** A 2D RGB patch of a painted flat pilaster on a wall looks identical to a round structural column. Without 3D surface depth relief, 2D appearance alone cannot resolve whether the structure is freestanding or flush with the wall.

### 2.3 Facade (Test F1: 0.3227 | 264 Missed | 341 FP)
- **Visual Evidence:** Facades span the entire building, yet contain large negative cutouts for windows, doors, balconies, and retail signs. A bounding box must either encompass the whole building (overlapping all components and penalizing IoU against cropped wall sections) or partition the wall into arbitrary rectangular tiles.
- **Root Cause:** **Geometric representation mismatch.** Bounding boxes are mathematically incapable of representing non-convex, planar architectural surfaces with holes.

### 2.4 Shutter / Blind (Test F1: 0.3108 | 496 Missed | Precision: 0.7771)
- **Visual Evidence:** When exterior shutters are folded flush or window blinds are retracted, high-frequency slats are lost at 640px scaling. When detected, predictions are highly accurate (Precision is 0.7771 with only 37 FPs).
- **Root Cause:** **Pixel sampling limit at 640px resolution.** High-frequency slat textures require higher spatial sampling or targeted window-level crops.

---

## 3. Evaluation of Candidate Technical Interventions

We audited six alternative technical strategies across the candidate classes:

| Intervention Strategy | Applicable Classes | Expected Metric Lift | Latency Impact | GPU VRAM Impact | Technical Feasibility & Verdict |
|---|---|:---:|:---:|:---:|---|
| **A. Standalone Single-Class Specialist (YOLO11s)** | `porch` or `pillar_column` | Low (+0.01 to +0.02) | **Severe (+15 to +20 ms per model)** | **+35 MB per model** | **REJECTED:** Incurs high latency and hallucinations; tested in Phase 15 and failed. |
| **B. Unified Entrance Group Specialist (`porch`+`door`+`col`)** | `porch`, `door`, `pillar_column` | Moderate (+0.04 to +0.07) | Moderate (+14 ms) | +35 MB (single shared model) | **RECOMMENDED FOR FUTURE:** Leverages shared entrance context; avoids single-class sparsity. |
| **C. Semantic / Panoptic Mask Segmentation** | `facade`, `roof_cornice`, `balcony` | Very High (+0.15 on facade) | Low (replaces bbox branch) | Negligible (+5 MB) | **STRONGLY RECOMMENDED:** Resolves non-convex wall boundary failure. |
| **D. Monocular Depth-Conditioned Filtering** | `pillar_column`, `balcony` | Moderate (+0.05 Precision) | Low (+5 ms via Depth Anything) | Shared depth buffer | **FEASIBLE:** Suppresses flat 2D pilaster false positives using relative depth gradients. |
| **E. High-Resolution Window / Entrance Crops ($800\times 800$)** | `shutter_blind`, `door`, `molding` | Moderate (+0.06 Recall) | Moderate (+12 ms) | Shared backbone | **RECOMMENDED:** Captures fine slats and moldings without retraining. |
| **F. CLAHE Preprocessing on Luminance** | `porch`, `roof_cornice` | Low to Moderate (+0.02 F1) | **+13 to +18 ms (CPU)** | 0 MB GPU | **VALIDATED AS OPTIONAL PROFILE:** Suppresses porch FPs; available in UI. |

---

## 4. Recommended Architectural Roadmap & Data Plan

### 4.1 Short-Term (Immediate Phase 17 / Phase 18 Actions)
1. **Preserve Production Baseline (Model B2):**  
   Maintain `exterior_detector_b2_yolo11s_best.pt` as the default in production to protect application latency (66.12 ms full-pipeline) and prevent false-alarm regressions on clean buildings.
2. **Promote Model B6 to Primary Experimental Candidate:**  
   Register Model B6 in the UI for users needing superior coverage of doors (+63 TPs), columns (+80 TPs), and moldings (+143 TPs).
3. **Maintain CLAHE as an Optional UI Toggle:**  
   Allow users analyzing shadowed or night scenes to toggle CLAHE contrast enhancement (`exterior_b6_clahe`), documenting the ~18 ms CPU transform overhead.
4. **Enforce Test Set Freeze:**  
   Quarantine the 154 test scenes to prevent over-fitting.

### 4.2 Medium-Term (Phase 18+ Data Expansion Plan)
Before training any new model, the team must execute targeted data collection:
- **Sample Target:** 500+ novel high-resolution ($1920\times 1080$ or higher) street-level entrance scenes.
- **Annotation Guidelines:**
  1. **Ground-Floor Entrance Co-occurrence:** Annotate `porch`, `front_door`, and `support_column` as a cohesive semantic group.
  2. **Column vs Pilaster Disambiguation:** Explicitly exclude flat wall moldings and non-structural pilasters from the `pillar_column` label.
  3. **Multi-Storey Porch Rules:** Document unambiguous guidelines for multi-storey verandas (ground-floor covered entrance vs second-floor open balcony).
- **Format:** Store in canonical YOLO format with byte-for-byte SHA256 checksums and group-aware train/val/test splits based on architectural scene locations.

### 4.3 Long-Term Architecture Evolution: Unified Multi-Task Network
Rather than running $N$ parallel YOLO models on GPU:
$$\text{Input Image} \longrightarrow \text{Shared Backbone (YOLO11s or SegFormer)} \longrightarrow \begin{cases} \text{Detection Head} & (\text{windows, doors, roofs, balconies}) \\ \text{Segmentation Head} & (\text{facades, walls}) \\ \text{High-Res Entrance Crop} & (\text{porch + door co-occurrence}) \end{cases}$$
- **Latency Budget:** Total end-to-end latency remains **$< 60\text{ ms}$** on an RTX 4060 GPU.
- **Memory Footprint:** Peak allocated VRAM capped at **$< 180\text{ MB}$**.
- **Acceptance Criterion:** Porch F1 $\ge 0.35$, Pillar/Column F1 $\ge 0.45$, Facade mIoU $\ge 0.60$.

---

## 5. Summary of Deliverables & Verification Status

| Deliverable | Path | Status |
|---|---|:---:|
| **Final Test Evaluation Report** | [reports/phase17_final_test_evaluation.md](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_final_test_evaluation.md) | Verified |
| **Final Test Metrics JSON** | [reports/phase17_final_test_metrics.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_final_test_metrics.json) | Verified |
| **Test Error Gallery** | [reports/phase17_final_test_error_gallery/](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_final_test_error_gallery/) | Verified (12 images) |
| **Specialist Recommendation Report** | [reports/phase17_specialist_recommendation.md](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase17_specialist_recommendation.md) | Verified |
| **Evaluation Script** | [scripts/evaluate_phase17_final_test.py](file:///C:/Users/apaar/Desktop/RoomAI/scripts/evaluate_phase17_final_test.py) | Verified |
| **Full Pytest Regression Suite** | `pytest tests/` | Verified (102 passed) |
