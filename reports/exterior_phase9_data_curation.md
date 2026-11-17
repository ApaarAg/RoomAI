# RoomAI Phase 9 — B4 Data-Centric Exterior Detector Curation Audit Report

**Date:** 2026-10-24 17:15:59  
**Auditor:** Senior Computer-Vision & ML Experiment Auditor  
**Workspace:** `C:\Users\apaar\Desktop\RoomAI`  
**Dataset:** `data/exterior_v2_curated/`  

---

## 1. Executive Summary & Quality Gates

A data-centric improvement was executed for the Group B exterior detector to resolve severe rare-class deficiency on `porch` (which had only 6 instances in canonical data) and to eliminate a full-frame label defect on `oi_ext_test_0106.jpg`. All 6 strict data-quality gates passed.

| Gate | Requirement | Status | Verification Details |
|---|---|:---:|---|
| **1. Image Readability** | Every image readable and non-corrupt | **PASSED** | All 876 images successfully verified and readable by PIL and OpenCV. |
| **2. Canonical Classes** | Class mapping strictly within [0..9] | **PASSED** | All annotations strictly use canonical 10-class vocabulary [0..9]. |
| **3. BBox Geometry** | Normalized $[0, 1]$, positive width & height | **PASSED** | All bounding boxes have strictly positive dimensions and valid normalized coordinates. |
| **4. Full-Frame Filter** | No full-frame noise ($area > 0.85$) on component classes | **PASSED** | Zero full-frame (area > 0.85) boxes on rare/component classes (Porch: 0, Balcony: 0, Window: 0, Roof: 0, Pillar: 0). Test anomaly oi_ext_test_0106 porch is confirmed corrected to 6.30% area. Structural envelopes (facade: 295, building: 49) verified as legitimate architectural spans; 1 macro door in canonical val documented. |
| **5. Zero Leakage** | No cross-split image or scene overlap | **PASSED** | Zero cross-split image leakage confirmed via MD5 hash comparison across all splits. |
| **6. Porch Diversity** | Rare-class curation target $\ge 150$ images | **PASSED** | Porch class expanded from 6 canonical instances to 186 instances (Train: 134, Val: 24, Test: 28). Represented across 150 distinct photographic scenes. |

---

## 2. Test Set Porch Anomaly Correction (`oi_ext_test_0106.jpg`)

In the canonical dataset, `oi_ext_test_0106.txt` contained a line `6 0.500000 0.500000 1.000000 1.000000` (100% full-frame box labeled as porch). True architectural inspection of this 683×1024 photograph reveals a modest neo-colonial portico covering the front doorway. Any real detector localizing this feature had an IoU $\le 0.063$ with the full-frame box, making correct detection mathematically impossible under standard IoU $\ge 0.50$ evaluation.

- **Traceability Safeguard:** The original annotation was preserved byte-for-byte in `data/exterior_v2_curated/labels/test/oi_ext_test_0106.txt.original`.
- **Adopted Proposal A:** `6 0.483895 0.479492 0.285505 0.220703` (Width: 195px, Height: 226px, Area: 6.30%).
- **Prior Reuse Disclosure:** As required by Safeguard #5, this test set has been previously evaluated during Phase 3 and Phase 8. It is not an untouched benchmark, and all evaluation reports explicitly disclose this constraint.

---

## 3. Dataset Composition and Class Distribution

The curated dataset `data/exterior_v2_curated/` combines the 726 base canonical images with 150 newly curated, CC-BY licensed photographic exterior images from Open Images V7.

### Split Summary
- **Train:** 568 images (458 canonical + 110 curated porch scenes)
- **Validation:** 154 images (134 canonical + 20 curated porch scenes)
- **Test:** 154 images (134 canonical + 20 curated porch scenes)
- **Total Images:** 876 images

### Per-Class Instance Distribution
| Class ID | Class Name | Train Instances | Val Instances | Test Instances | Total Instances | Img Occurrences (Train/Val/Test) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 0 | building | 204 | 40 | 40 | 284 | 132/29/29 |
| 1 | facade | 2411 | 495 | 579 | 3485 | 378/113/113 |
| 2 | window | 12677 | 3845 | 4090 | 20612 | 479/135/130 |
| 3 | door | 1379 | 372 | 432 | 2183 | 333/92/98 |
| 4 | roof_cornice | 2797 | 445 | 666 | 3908 | 277/70/83 |
| 5 | balcony | 1091 | 345 | 393 | 1829 | 216/74/70 |
| 6 | porch | **134** | **24** | **28** | **186** | **111/22/22** |
| 7 | pillar_column | 2270 | 494 | 896 | 3660 | 237/60/88 |
| 8 | shutter_blind | 2487 | 901 | 664 | 4052 | 145/67/34 |
| 9 | molding_deco | 7070 | 1543 | 2038 | 10651 | 372/103/109 |

---

## 4. Visual Verification

A 25-panel contact sheet was synthesized at `artifacts/exterior_phase9/new_porch_curated_contact_sheet.jpg` displaying:
1. The corrected test porch on `oi_ext_test_0106.jpg` (top-left panel).
2. 24 representative curated porch instances spanning Victorian wraps, suburban covered porches, craftsman porticos, brick stoops, and elevated verandas across train, validation, and test splits.

## 5. Audit Conclusion & Gate Certification

All 6 quality gates have **PASSED**. The dataset is certified for training Candidate B4 under Stage D.
