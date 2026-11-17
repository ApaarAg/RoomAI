# RoomAI — Phase 5B: Digital Image Processing (DIP) for Detection Refinement Report

**Date:** October 25, 2026  
**Role:** Digital Image Processing Engineer & Senior Computer Vision Auditor  
**Scope:** Architectural Exterior Object Detection & Localization Refinement  
**Base Model:** Model B5 (`exterior_detector_b5_transfer_best.pt`, MD5: `e12b4181fee244a3569dfa782ba3c3c3`)  
**Evaluation Set:** Unified Validation Split (`data/exterior_b5_unified/`, 154 images, 8,504 annotations)  
**Deliverable Document:** `reports/exterior_b5_digital_image_processing.md`  
**Structured Metrics:** [exterior_b5_dip_metrics.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/exterior_b5_dip_metrics.json)  

---

## 1. Executive Summary & Final Decision

This investigation experimentally evaluated whether classical Digital Image Processing (DIP) techniques—spanning edge and gradient extraction, morphological transformations, spatial contrast enhancement, and boundary-guided coordinate snapping—can improve Model B5's detection accuracy, boundary localization quality, and visual clarity without replacing the deep-learning detector.

Every technique was subjected to rigorous, controlled benchmarking against the unmodified B5 baseline on 154 validation images under identical evaluation parameters ($640\times640$ resolution, confidence threshold $0.25$, IoU matching threshold $0.50$).

### Key Findings:
1. **Experiment 1 (CLAHE in LAB Color Space):** While visually enhancing contrast in underexposed and shadowed regions, CLAHE produced **no net measurable benefit across the full distribution** (Macro-F1 changed from `0.4960` to `0.5014` [+0.0054], but Micro-F1 regressed from `0.6371` to `0.6272` [-0.0099] due to a surge in false-positive window and molding detections).
2. **Experiment 2 (Gradient-Guided Bounding-Box Snapping):** Snapping predicted box coordinates to local Sobel/Canny gradient peaks caused a **severe regression across all metrics** (Macro-F1 dropped by `-0.0219`, Micro-F1 dropped by `-0.0371`, Mean IoU degraded from `0.8235` to `0.7690` [-0.0545], and latency increased by `+79.2%` from $52.59$ ms to $94.25$ ms per image).
3. **Morphological Hole Filling:** Validated the theoretical principle that morphological hole filling and contour closing operate strictly on continuous spatial regions (segmentation masks) and are mathematically inapplicable to rectangular bounding boxes.

### Final Decision Rule:
> **DECISION: DIP COMPONENTS REMAIN DISABLED BY DEFAULT.**  
> Classical DIP does not outperform the unmodified YOLO11s feature representations. Deep learning detectors learn superior multi-scale contextual filters that reject interior texture gradients that deceive classical filters. B5 will operate directly on standard RGB inputs in production.

---

## 2. Conceptual Foundation: Classical DIP vs. Detection vs. Instance Segmentation

A critical engineering responsibility is maintaining precise distinctions between visual representations:

```
+---------------------------------------------------------------------------------------+
|                                VISUAL REPRESENTATIONS                                  |
+---------------------------------------------------------------------------------------+
|  1. Classical DIP (Pixel Grids)                                                       |
|     - Transforms intensity arrays: I(x, y) -> I'(x, y)                                |
|     - Local kernel convolutions (Sobel, Canny, Gaussian, Laplacian)                    |
|     - No high-level semantic awareness; cannot differentiate texture from boundary    |
+---------------------------------------------------------------------------------------+
|  2. Object Detection (Bounding Boxes)                                                 |
|     - Predicts discrete rectangular tuples: B_i = [x_center, y_center, width, height] |
|     - Non-spatial inside: region is assumed uniformly occupied by class label C_i     |
|     - Concept of "holes", "connected components", or "morphological opening" is       |
|       mathematically undefined on rectangular coordinate tuples                       |
+---------------------------------------------------------------------------------------+
|  3. Instance Segmentation (Pixel-Level Masks)                                         |
|     - Predicts boolean spatial occupancy masks: M_i in {0, 1}^(H x W)                 |
|     - Morphological operations (dilation, erosion, closing, hole filling) directly    |
|       applicable to close small gaps and remove isolated mask fragments               |
+---------------------------------------------------------------------------------------+
```

### Why Edges Do Not Equal Object Boundaries
In architectural imagery, classical gradient operators (Sobel, Canny, Laplacian, morphological gradient) respond to any high-frequency spatial variation in luminance:
- **Interior Texture Edges:** Window mullions, glazing bars, brick mortar joints, shutter louvers, and decorative stonework generate gradient responses equal to or stronger than the exterior building silhouette.
- **Illumination Edges:** Cast shadows from eaves, balconies, trees, and adjacent structures create intense gradient lines that cross-cut architectural boundaries.
- **Conclusion:** A classical edge detector cannot distinguish an object boundary from surface texture or illumination discontinuities. Attempting to align object bounding boxes to naive edge maps introduces severe localization errors.

---

## 3. Controlled Experimental Evaluation

### Experimental Protocol:
- **Model Checkpoint:** `exterior_detector_b5_transfer_best.pt` (MD5: `e12b4181fee244a3569dfa782ba3c3c3`)
- **Evaluation Dataset:** 154 validation images from `data/exterior_b5_unified/` (8,504 ground-truth bounding boxes)
- **Inference Resolution:** $640 \times 640$ pixels
- **Operating Thresholds:** Confidence = $0.25$, IoU match = $0.50$
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), CUDA 12.8, PyTorch 2.11.0

### Benchmark Comparison Table

| Condition | Macro-F1 | Micro-F1 | Micro-P | Micro-R | Mean IoU | Latency (ms/img) | Outcome |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline: Unmodified B5** | **0.4960** | **0.6371** | **0.6019** | **0.6766** | **0.8235** | **52.59 ms** | **Benchmark Standard** |
| **Exp 1: CLAHE Preprocessing** | 0.5014 | 0.6272 | 0.5841 | 0.6772 | 0.8214 | 51.23 ms | **No Measurable Benefit** |
| **Exp 2: Gradient Box Snapping** | 0.4741 | 0.6000 | 0.5601 | 0.6461 | 0.7690 | 94.25 ms | **Severe Regression** |

---

## 4. Per-Class Quantitative Breakdown

| Class | GT Count | Baseline F1 | CLAHE F1 | Snapped F1 | CLAHE vs Base Δ | Snapped vs Base Δ | Mean IoU (Base) | Mean IoU (Snap) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `building` | 40 | 0.4675 | **0.5455** | 0.4935 | +0.0780 | +0.0260 | 0.8511 | 0.8289 |
| `facade` | 495 | 0.3619 | **0.3743** | 0.3619 | +0.0124 | +0.0000 | 0.8406 | 0.8279 |
| `window` | 3,845 | **0.7505** | 0.7318 | 0.7305 | -0.0187 | -0.0200 | 0.8430 | 0.7813 |
| `door` | 372 | 0.4879 | **0.4974** | 0.4751 | +0.0095 | -0.0128 | 0.8002 | 0.7797 |
| `roof_cornice` | 445 | 0.6281 | **0.6351** | 0.5902 | +0.0070 | -0.0379 | 0.7844 | 0.7490 |
| `balcony` | 345 | 0.6506 | **0.6547** | 0.6335 | +0.0041 | -0.0171 | 0.7961 | 0.7567 |
| `porch` | 24 | **0.0690** | 0.0625 | **0.0690** | -0.0065 | +0.0000 | 0.6713 | 0.6757 |
| `pillar_column` | 494 | **0.3321** | 0.3295 | 0.2994 | -0.0026 | -0.0327 | 0.7625 | 0.7094 |
| `shutter_blind` | 901 | **0.6442** | 0.6333 | 0.6390 | -0.0109 | -0.0052 | 0.8491 | 0.8092 |
| `molding_deco` | 1,543 | **0.5686** | 0.5497 | 0.4493 | -0.0189 | -0.1193 | 0.7788 | 0.6989 |
| **Macro Average** | **8,504** | **0.4960** | **0.5014** | **0.4741** | **+0.0054** | **-0.0219** | **0.8235** | **0.7690** |

---

## 5. In-Depth Error Analysis & Technical Audit

### Experiment 1: CLAHE Preprocessing (LAB Color Space)
- **Algorithm:** Image converted from BGR to LAB color space; Contrast Limited Adaptive Histogram Equalization applied to the luminance channel ($L$) with `clipLimit=2.0` and `tileGridSize=(8, 8)`; re-merged and mapped back to BGR.
- **Positive Effects:** Low-contrast building silhouettes against overcast skies and deeply shadowed ground-floor portals gained contrast. Recall improved on `building` ($0.4500 \to 0.5250$) and `facade` ($0.3030 \to 0.3232$).
- **Negative Effects (High-Frequency Noise):** CLAHE amplifies high-frequency texture noise across window panes and decorative stonework. False positives surged on `window` from $1,086$ to $1,189$ (+103 FPs) and on `molding_deco` from $700$ to $731$ (+31 FPs). Micro-precision dropped from $0.6019$ to $0.5841$.
- **Conclusion:** **No Measurable Benefit.** While it produces subjectively "crisper" images, it degrades precision on the most frequent architectural classes.

### Experiment 2: Gradient-Guided Bounding-Box Snapping
- **Algorithm:** For each predicted box $[x_1, y_1, x_2, y_2]$, computes local horizontal and vertical Sobel gradients within a search radius of $r=4$ pixels. If a local gradient maximum exceeds threshold $T=25.0$, the coordinate snaps to the gradient peak.
- **Failure Mode 1 (Interior Texture Snapping):** For windows with multiple panes, the box border snapped to internal mullions rather than the outer frame, shrinking the predicted box and causing IoU to drop below the $0.50$ matching threshold. Window Mean IoU collapsed from $0.8430$ to $0.7813$.
- **Failure Mode 2 (Shadow Line Deviation):** Deep cast shadows under roof cornices and balconies have steeper gradient magnitude than the architectural boundary. The coordinate snapped to the shadow line, shifting boxes vertically and creating false negatives. Cornice F1 dropped from $0.6281$ to $0.5902$.
- **Failure Mode 3 (Molding Catastrophe):** Thin moldings and decorative friezes have dense parallel edge profiles. Snapping caused coordinate collapse, resulting in an extreme F1 regression from $0.5686$ to $0.4493$ (-0.1193).
- **Latency Penalty:** Computing Sobel and Canny pyramids per image added $+41.66$ ms per frame (+79.2% latency overhead), violating the real-time application budget.
- **Conclusion:** **Severe Regression.** Gradient snapping is fundamentally unsuitable for bounding-box object detection in textured architectural scenes.

---

## 6. Visualization & Reproducibility Gallery

Side-by-side verification panels have been generated and archived in `reports/visualizations/dip_refinement/`:

1. **Edge & Boundary Decomposition Panels:**
   - [dip_edges_cmp_val_0001.jpg](file:///C:/Users/apaar/Desktop/RoomAI/reports/visualizations/dip_refinement/dip_edges_cmp_val_0001.jpg): Original vs. Canny ($50, 150$) vs. Sobel Gradient Magnitude vs. Morphological Gradient ($3\times3$ Rect). Demonstrates dense interior texture edges across classical facades.
   - [dip_edges_oi_curated_porch_val_0110.jpg](file:///C:/Users/apaar/Desktop/RoomAI/reports/visualizations/dip_refinement/dip_edges_oi_curated_porch_val_0110.jpg): Demonstrates edge clutter caused by porch railings, shadow lines, and background vegetation.
2. **Preprocessing Comparison Panels:**
   - [dip_preprocessing_cmp_val_0001.jpg](file:///C:/Users/apaar/Desktop/RoomAI/reports/visualizations/dip_refinement/dip_preprocessing_cmp_val_0001.jpg): Original vs. CLAHE ($clip=2.0$) vs. Gamma Correction ($\gamma=0.8$). Highlights texture amplification in window regions.
   - [dip_preprocessing_oi_curated_porch_val_0110.jpg](file:///C:/Users/apaar/Desktop/RoomAI/reports/visualizations/dip_refinement/dip_preprocessing_oi_curated_porch_val_0110.jpg): Compares shadow recovery on porch and overhang structures.
3. **Detection Boundary Snapping Panels:**
   - [dip_detection_snap_cmp_val_0001.jpg](file:///C:/Users/apaar/Desktop/RoomAI/reports/visualizations/dip_refinement/dip_detection_snap_cmp_val_0001.jpg): Unmodified B5 detections (green) vs. Gradient-snapped detections (yellow). Shows boundary contraction on molding and cornice boxes.
   - [dip_detection_snap_oi_curated_porch_val_0110.jpg](file:///C:/Users/apaar/Desktop/RoomAI/reports/visualizations/dip_refinement/dip_detection_snap_oi_curated_porch_val_0110.jpg): Shows displacement of door and porch boxes toward cast shadow edges.

---

## 7. Conclusions & Architectural Recommendations

1. **Retain Deep Learning Feature Invariance:** Modern convolutional and attention layers (such as YOLO11's C3k2 and C2PSA blocks) already learn robust edge and contrast representations from large-scale pretraining. Applying classical filters prior to inference removes the natural luminance statistics the network was optimized for.
2. **Segmentation Mask Refinement as Future Work:** Morphological operations (erosion, dilation, hole-filling) should be reserved for pixel-level instance segmentation (e.g., YOLO11-seg), where connected-component analysis can close segmentation mask gaps without corrupting bounding boxes.
3. **Application State:** In compliance with the Final Decision Rule, all DIP preprocessing and postprocessing modules remain **disabled** in `config/model_config.yaml`.
