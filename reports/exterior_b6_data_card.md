# RoomAI — Dataset Card: Model B6 Targeted Data Expansion

**Date:** October 25, 2026  
**Curator:** Senior Computer Vision Engineer & Dataset Auditor  
**Dataset Version:** `data/exterior_b6_expanded/`  
**Dataset YAML:** [dataset.yaml](file:///C:/Users/apaar/Desktop/RoomAI/data/exterior_b6_expanded/dataset.yaml)  
**Manifest:** [b6_manifest.json](file:///C:/Users/apaar/Desktop/RoomAI/data/exterior_b6_expanded/b6_manifest.json)  

---

## 1. Dataset Overview & Motivation

Model B6 data expansion addresses the acute class sparsity and distribution shifts identified during the Stage A audit of Model B5. The expansion specifically targets:
1. **`porch`**: Grew from 134 boxes in 111 images to **509 boxes across 361 images** ($+280\%$ increase in instances, $+225\%$ increase in distinct scenes).
2. **`shutter_blind`**: Scene diversity expanded from 184 images to **333 images** (+227 boxes), introducing diverse exterior louvered, board-and-batten, and paneled shutters.
3. **`building`**: Grew from 491 boxes in 275 images to **777 boxes across 438 images** (+286 boxes).
4. **`door`**: Expanded to 1,633 boxes across 506 distinct images.

Validation (154 images, 8,504 annotations) and test splits (154 images, 9,826 annotations) remain **completely locked, untouched, and unpolluted**.

---

## 2. Dataset Split Sizes & Instance Counts

| Split | Images | Annotations | Provenance Sources | Integrity Guarantee |
|---|:---:|:---:|---|---|
| **Train** | **1,302** | **35,515** | CMP Facade (378) + OI V7 Val/Test Curated (335) + OI V7 Train Curated (399) | Zero collision with val/test |
| **Validation** | **154** | **8,504** | CMP Facade Eval (114) + Curated Open Images (40) | Locked benchmark split |
| **Test** | **154** | **9,826** | CMP Facade Test (114) + Curated Open Images (40) | Untouched evaluation split |
| **Total** | **1,610** | **53,845** | Harmonized multi-source architectural imagery | Full cross-split verification |

---

## 3. Class Distribution Comparison: B5 Unified vs. B6 Expanded (Train Split)

| Canonical Class | B5 Train Boxes | B5 Train Images | **B6 Train Boxes** | **B6 Train Images** | **Net Change (Boxes)** | **Net Change (Images)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `building` | 491 | 275 | **777** | **438** | **+286 (+58.2%)** | **+163 (+59.3%)** |
| `facade` | 2,411 | 378 | **2,411** | **378** | +0 | +0 |
| `window` | 13,525 | 635 | **14,176** | **752** | **+651 (+4.8%)** | **+117 (+18.4%)** |
| `door` | 1,607 | 483 | **1,633** | **506** | **+26 (+1.6%)** | **+23 (+4.8%)** |
| `roof_cornice` | 2,797 | 277 | **2,797** | **277** | +0 | +0 |
| `balcony` | 1,091 | 216 | **1,091** | **216** | +0 | +0 |
| `porch` | **134** | **111** | **509** | **361** | **+375 (+279.9%)** | **+250 (+225.2%)** |
| `pillar_column` | 2,270 | 237 | **2,270** | **237** | +0 | +0 |
| `shutter_blind` | 2,554 | 184 | **2,781** | **333** | **+227 (+8.9%)** | **+149 (+81.0%)** |
| `molding_deco` | 7,070 | 372 | **7,070** | **372** | +0 | +0 |
| **Total Annotations** | **33,950** | **903** | **35,515** | **1,302** | **+1,565 (+4.6%)** | **+399 (+44.2%)** |

### Shortfall Reporting on Non-Boxable Classes:
As mandated by Stage B, Open Images V7 600 boxable class taxonomy was queried. Classes `pillar_column`, `facade`, `roof_cornice`, `balcony`, and `molding_deco` do not have corresponding boxable categories in Open Images V7. Rather than inventing pseudo-labels or applying unreliable heuristics, the shortfall is reported transparently. All 606 CMP Facade ground-truth images remain the authoritative reference for these structural architectural features.

---

## 4. Quality Gates & Leakage Audit

1. **Pre-Ingestion Screening:**
   - 400 candidate scenes from Open Images V7 Train split were downloaded and screened.
   - 1 image failed integrity decoding and was discarded.
   - 399 images satisfied image decoding, dimension, and coordinate validation ($0.0 \le x, y, w, h \le 1.0$).
2. **Deduplication:**
   - MD5 collision check against all 1,211 existing images in B5 unified splits: **0 collisions**.
   - 64-bit DCT perceptual hash (pHash) screening against all 308 validation and test images: **0 near-duplicates** (Hamming distance $d \ge 5$ enforced for all accepted images).
3. **Split Integrity:**
   - Train vs. Val MD5 collisions: **0**
   - Train vs. Test MD5 collisions: **0**
   - Val vs. Test MD5 collisions: **0**
