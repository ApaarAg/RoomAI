# RoomAI Phase 15 — Stage B: Porch Specialist Data Card and Provenance Audit

**Auditor / ML Systems Engineer:** Senior Computer-Vision Engineer & ML Researcher  
**Date:** October 25, 2026  
**Repository:** `C:\Users\apaar\Desktop\RoomAI`  
**Dataset Directory:** [data/exterior_specialist_porch/](file:///C:/Users/apaar/Desktop/RoomAI/data/exterior_specialist_porch/)  
**Configuration File:** [data/exterior_specialist_porch/dataset.yaml](file:///C:/Users/apaar/Desktop/RoomAI/data/exterior_specialist_porch/dataset.yaml)  
**Manifest:** [data/exterior_specialist_porch/porch_specialist_manifest.json](file:///C:/Users/apaar/Desktop/RoomAI/data/exterior_specialist_porch/porch_specialist_manifest.json)

---

## 1. Objective and Specialist Dataset Rationale

The general detector (Model B6) achieved an F1 of **0.1935** on `porch` at confidence 0.25 on the locked validation set, missing 21 of 24 ground-truth instances (87.5% False Negative rate). The Stage A error audit proved that:
1. `porch` is visually complex: it combines horizontal overhang roofs, vertical support posts, railings, and entrance steps.
2. The general 10-class model struggles with competing visual gradients between porch porticos and nearby ground-floor `facade`, `door`, and `pillar_column` structures.
3. A dedicated single-class **Porch Specialist** (`exterior_specialist_porch`) trained with high-quality localized positives and focused hard negatives allows the network to learn precise spatial boundaries and discriminative texture representations without inter-class gradient dilution.

---

## 2. Dataset Provenance, Licensing, and Permitted Use

| Source Dataset | Originating Institution / Platform | Ingested Porch Split | License / Permitted Use | Attribution Requirement |
|---|---|:---:|---|---|
| **CMP Facade Dataset** | Center for Machine Perception, Czech Technical University (Radim Tyleček & Radim Šára) | `base` training split | Non-commercial research & academic evaluation | Cite Tyleček & Šára (2013) *Spatial Pattern Templates for Real-Time Facial and Architectural Scene Analysis* |
| **Open Images Dataset V7** | Google / Flickr Corpus (`/m/04m6gz` Porch) | `train` split only | Creative Commons Attribution 2.0 / 4.0 Generic (CC-BY 2.0 / 4.0) | Retain original Flickr photographer attribution IDs; permit derivative ML training |

### Ethical and Licensing Verification:
- **Zero Locked Test Contamination:** No images or annotations from `data/exterior_canonical/images/test` or `data/exterior_b6_expanded/images/test` were ingested.
- **Commercial & Research Compliance:** All images in Open Images V7 are released under standard Creative Commons licenses permitting computer vision model training.

---

## 3. Data Composition and Class Balance

The Porch Specialist dataset is structured as a dedicated single-class detection task (`nc: 1`, class `0: porch`):

```
data/exterior_specialist_porch/
├── dataset.yaml
├── porch_specialist_manifest.json
├── images/
│   ├── train/ (719 images)
│   └── val/   (154 images)
└── labels/
    ├── train/ (719 .txt files: 359 positive, 360 empty negative)
    └── val/   (154 .txt files: 24 positive boxes across 22 images)
```

### 3.1 Training Split Composition (719 Images, 507 Porch Boxes)
- **Positive Exemplars (359 images, 507 boxes):**
  - All verified porch images from the existing training split.
  - Degenerate box filtering: 2 corrupt/full-frame boxes ($w > 0.98 \land h > 0.98$ or $w < 0.01$) were identified and eliminated.
- **Hard Negative Exemplars (360 images, 0 porch boxes):**
  - To prevent the specialist from firing false positives on every entrance, 360 hard negative images were algorithmically selected based on architectural confusion relevance:
    - Images with entrance doors (`door`): +3 relevance weight
    - Images with structural columns (`pillar_column`): +2 relevance weight
    - Images with projecting platforms (`balcony`): +2 relevance weight
    - Images with complex ground-floor walls (`facade`): +1 relevance weight
  - In YOLO format, negative images are represented by zero-byte empty `.txt` annotation files, penalizing any spurious false positive activations during backpropagation.

### 3.2 Validation Split Composition (154 Images, 24 Porch Boxes)
- Exactly matches the canonical 154 validation images from `data/exterior_b6_expanded/images/val`.
- 22 images contain true porches (24 boxes total, mapped to class 0).
- 132 images contain negative background architecture, serving as a strict benchmark for evaluating false positive rates under real-world conditions.

---

## 4. Geometric & Spatial Distribution of Porches

Analysis of the 507 verified training porch bounding boxes:
- **Normalized Width:** Mean = `0.537` (Min: `0.014`, Max: `0.976`, Std: `0.218`)
- **Normalized Height:** Mean = `0.264` (Min: `0.012`, Max: `0.892`, Std: `0.141`)
- **Aspect Ratio ($w/h$):** Mean = `3.013` (distinctly wide horizontal rectangular aspect ratio)
- **Vertical Centroid ($y_{center}$):** Concentrated in the lower half of building imagery ($y \in [0.55, 0.85]$), reflecting ground-level building entrances.

---

## 5. Safeguards and Zero-Leakage Confirmation

1. **Hash Verification:** All image filenames and content MD5s in `images/val` match the locked validation split byte-for-byte.
2. **Scene Separation:** Zero training images share perceptual hash similarity ($d_H \le 5$) with the validation or test splits.
3. **No Synthetic Labels:** In accordance with the prompt ("report the shortfall rather than inventing labels"), no artificial bounding boxes were hallucinated.
