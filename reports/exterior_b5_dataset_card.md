# Dataset Card: RoomAI B5 Unified Exterior Dataset (`exterior_b5_unified`)

**Version:** 1.0.0 (B5 Large-Scale Transfer Learning Release)  
**Creation Date:** 2026-10-24  
**Curator:** Senior Computer Vision Engineer & Object Detection Researcher  
**Storage Location:** `data/exterior_b5_unified/`  
**Configuration File:** `data/exterior_b5_unified/dataset.yaml`  
**Provenance Manifest:** `data/exterior_b5_unified/b5_manifest.json`  

---

## 1. Dataset Summary & Motivation

The `exterior_b5_unified` dataset is engineered specifically for transfer learning with Model B5 to improve precision, recall, and generalization across all 10 canonical building exterior classes in RoomAI. It harmonizes high-density rectified architectural facade segmentations (CMP Facade Database) with natural real-world street and exterior photography (Google Open Images V7).

### Key Dataset Metrics
- **Total Images:** 1,211 images (903 train, 154 val, 154 test)
- **Total Bounding Boxes:** 52,280 annotations
- **Resolution:** Up to $1024\times768$ (rectified and unrectified natural photography)
- **Storage Size:** ~310 MB images + labels
- **Format:** Standard YOLO format (`class_id x_center y_center width height` normalized to $[0.0, 1.0]$)

---

## 2. Constituent Data Sources & Provenance

| Source Repository | License | Original Split Used | New B5 Split Assignment | Images Added | Primary Classes Contributed |
|---|---|---|---|---:|---|
| **CMP Facade Database** (CTU Prague) | CC-BY-SA 4.0 | Train (378), Eval (114), Test (114) | Preserved: 378 train, 114 val, 114 test | 606 | facade, roof_cornice, balcony, pillar_column, shutter_blind, molding_deco, window, door |
| **Open Images V7 Canonical** | CC-BY 2.0 | Canonical 80 train, 20 val, 20 test | Preserved: 80 train, 20 val, 20 test | 120 | building, house, porch, door, window |
| **Open Images V7 Curated (Phase 9)** | CC-BY 2.0 | 110 train, 20 val, 20 test | Preserved: 110 train, 20 val, 20 test | 150 | porch, building, house, window, door |
| **Open Images V7 B5 Extension** | CC-BY 2.0 | Public AWS S3 Mirror (Val & Test pools) | **Strictly Train Split Only** | 335 | building (+287 boxes), door (+228 boxes), window (+848 boxes), shutter_blind (+67 boxes) |

### Non-Negotiable Split Protection
- **Locked Evaluation Splits:** The 154 validation and 154 test images from `exterior_v2_curated` were preserved byte-for-byte. No image from the validation or test splits was moved to training.
- **Leakage Prevention:** 434 candidate Open Images scenes were screened against existing validation and test sets using MD5 hashes and 64-bit DCT perceptual hashing (pHash). Two near-duplicate candidate images (`358a88828eaca3ad` and `1b3247da6dd5d578`) were identified and rejected (Hamming distance $< 5$). Zero leakage into evaluation splits was verified.

---

## 3. Class Taxonomy & Distribution

| Class ID | Canonical Class Name | Open Images & CMP Source Mapping | Train Boxes | Val Boxes | Test Boxes | Total Annotations |
|---|---|---|---:|---:|---:|---:|
| 0 | **building** | Open Images: `/m/0cgh4` (Building), `/m/03jm5` (House), `/m/021sj1` (Office building) | 491 | 40 | 40 | 571 |
| 1 | **facade** | CMP Facade: class index 2 | 2,411 | 495 | 579 | 3,485 |
| 2 | **window** | Open Images: `/m/0d4v4`, CMP: class index 3 | 13,525 | 3,845 | 4,090 | 21,460 |
| 3 | **door** | Open Images: `/m/02dgv`, CMP: class indices 4 & 12 (shop) | 1,607 | 372 | 432 | 2,411 |
| 4 | **roof_cornice** | CMP Facade: class index 5 (cornice) | 2,797 | 445 | 666 | 3,908 |
| 5 | **balcony** | CMP Facade: class index 7 (balcony) | 1,091 | 345 | 393 | 1,829 |
| 6 | **porch** | Open Images: `/m/04m6gz` (Porch) | 134 | 24 | 28 | 186 |
| 7 | **pillar_column** | CMP Facade: class index 9 (pillar) | 2,270 | 494 | 896 | 3,660 |
| 8 | **shutter_blind** | Open Images: `/m/031b6r` (Window blind), CMP: class index 8 | 2,554 | 901 | 664 | 4,119 |
| 9 | **molding_deco** | CMP Facade: class indices 6 (sill), 10 (deco), 11 (molding) | 7,070 | 1,543 | 2,038 | 10,651 |
| **Total** | — | — | **33,950** | **8,504** | **9,826** | **52,280** |

---

## 4. Annotation Quality & Noise Filtering

1. **Full-Frame Box Suppression:** In Open Images, background building instances often span the entire photograph ($1.0 \times 1.0$), suppressing local feature learning. All candidate boxes with area $> 0.85$ ($85\%$ of frame area) were discarded.
2. **Sub-Pixel Noise Removal:** Boxes with width or height $< 0.005$ ($0.5\%$ of image dimension) were discarded.
3. **Clamping & Normalization:** All coordinates were strictly validated to ensure $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0.0, 1.0]$ and $x_{\max} \ge x_{\min}$, $y_{\max} \ge y_{\min}$.

---

## 5. Licensing, Attribution, & Provenance

1. **CMP Facade Database:**
   - Authors: Radim Tyleček and Radim Šára, Center for Machine Perception, Czech Technical University in Prague.
   - License: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0).
   - Citation: *Tyleček, R., & Šára, R. (2013). Spatial Pattern Templates for Recognition of Objects with Regular Structure. In GCPR 2013.*
2. **Open Images V7:**
   - Authors: Google LLC and contributors.
   - License: Creative Commons Attribution 2.0 Generic (CC-BY 2.0).
   - Annotations: Licensed under CC-BY 4.0.
   - S3 Mirror: `https://s3.amazonaws.com/open-images-dataset/`
3. **Internal Provenance Manifest:**
   - Complete record with image IDs, source splits, MD5 hashes, and per-image bounding-box arrays saved in `data/exterior_b5_unified/b5_manifest.json`.
