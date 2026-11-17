# RoomAI — Visual Segmentation & Depth Failure Analysis Report

**Date:** 2026-10-31  
**Author:** Principal Computer Vision Engineer & ML Systems Architect  
**Scope:** Root-cause diagnosis of visual quality degradation, mask inaccuracy, duplicate entries, and depth visualization flaws in RoomAI.

---

## Executive Summary

A comprehensive architectural audit of RoomAI's segmentation, depth estimation, image processing, and UI presentation layers revealed the root causes for the observed visual degradation:

1. **COCO Class Confusion:** The current live segmenter (`yolo11n-seg.pt`) relies on generic MS COCO 80-class pretraining. COCO models frequently misclassify indoor furniture (e.g. classifying beds as couches/sofas) and lack coverage for custom indoor categories (`cabinet#shelf`, `lamp`, `door`, `window`, `curtain`, `decoration`).
2. **Duplicate Category Entries in UI:** `models/segmentation_model.py` creates a separate `SegmentationMask` item for every detected instance without aggregation. The UI (`ui/analysis_view.py`) loops over `resp.masks` and renders a separate progress bar and category row for every instance, leading to repeated "Couch", "Chair", or "Table" entries.
3. **Lack of Fine-grained Mask Refinement:** Masks from generic YOLO-seg are low-resolution feature maps ($160 \times 160$) upsampled with nearest-neighbor interpolation, resulting in blocky, imprecise object boundaries (such as TV and bed masks).
4. **Disconnection between A3 Detector and Segmentation:** RoomAI possesses a specialized 12-class Indoor Detector A3 (`indoor_detector_a3_yolo11s_800px_best.pt`), but it is currently used ONLY in the Object Detection tab. The Segmentation tab operates independently using generic COCO `yolo11n-seg.pt`.
5. **SegFormer-B1 and FastSAM Disconnection:** High-accuracy candidates benchmarked in `scripts/benchmark_cv_models.py` (SegFormer-B1 for semantic segmentation and A3 + FastSAM-s for box-prompted instance segmentation) exist in the codebase but were never connected to `models/segmentation_model.py` or the UI.
6. **Depth Map Color Flattening:** Depth Anything V2 Small produces valid continuous relative depth, but linear min-max normalization without contrast enhancement, percentile clamping, or multi-channel colormapping results in broad, visually flat color regions with weak foreground/background contrast.

---

## Detailed Audit & Diagnosis (10 Technical Questions)

### 1. Which checkpoint produces the current segmentation output?
- **Checkpoint:** `checkpoints/yolo11n-seg.pt` (Ultralytics YOLO11n-Seg pretrained on MS COCO 80 classes).
- **Model Name in UI:** `Ultralytics YOLO-Seg (yolo11n-seg.pt)`.

### 2. Which checkpoint produces the depth output?
- **Checkpoint:** `checkpoints/depth_anything_v2_small/` (`depth-anything/Depth-Anything-V2-Small-hf`).
- **Model Name in UI:** `Depth Anything V2 Small (DINOv2 ViT-S + DPT Head)`.

### 3. Is the A3 detector involved in the current segmentation workflow?
- **No.** The production Indoor Detector A3 (`indoor_detector_a3_yolo11s_800px_best.pt`) is isolated to `DetectionService` and `tab_detect`. The current segmentation pipeline calls `CustomRoomSegmenter` directly, which runs `yolo11n-seg.pt` independently.

### 4. Is FastSAM integrated or only benchmarked?
- **Benchmarked only.** FastSAM (`FastSAM-s.pt`) is benchmarked in `scripts/benchmark_cv_models.py` alongside A3, but is not instantiated or wired into `models/segmentation_model.py` or `services/segmentation_service.py`.

### 5. Is SegFormer-B1 integrated or only benchmarked?
- **Benchmarked only.** SegFormer-B1 (`nvidia/segformer-b1-finetuned-ade-512-512`) achieves 0.5432 mIoU on ADE20K indoor categories in `scripts/benchmark_cv_models.py`, but is not imported or available in `models/segmentation_model.py`.

### 6. Is the current depth output relative or metric?
- **Relative inverse depth.** It outputs affine-invariant continuous values in $[0.0, 1.0]$ where higher values indicate proximity to camera. It is non-metric and uncalibrated for physical distance.

### 7. How are masks resized, thresholded, combined, and visualized?
- **Resize:** Low-res raw model masks ($160 \times 160$) are upsampled to image dimensions using `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` and thresholded at $> 0.5$.
- **Combine:** Instance masks overwrite a single 2D uint8 numpy array `combined_mask[binary_mask] = mask_id`. Later instances completely overwrite overlapping regions of earlier instances.
- **Visualization:** `create_segmentation_overlay` maps non-zero IDs to `PALETTE[(uid - 1) % len(PALETTE)]` and alpha-blends with the RGB image at fixed $\alpha = 0.45$.

### 8. How are object labels and surface coverage percentages calculated?
- **Coverage:** `coverage = round((pixel_count / total_pixels) * 100.0, 2)`.
- **Labels:** Direct mapping from COCO class indices (`res.names`).

### 9. Why do duplicate category entries appear in the UI?
- `models/segmentation_model.py` generates a `SegmentationMask` entry for *every instance*. If 3 couches or 2 tables are detected, 3 separate `SegmentationMask` items with `category: "couch"` are placed in `resp.masks`.
- `ui/analysis_view.py` iterates over `resp.masks` directly, creating a separate progress bar and text label for each instance instead of grouping by category or distinguishing instance IDs.

### 10. Are image resizing, RGB/BGR conversions, or coordinate transformations incorrect?
- **PIL/OpenCV convention:** `Image.size` returns `(width, height)` while numpy arrays have shape `(height, width)`. `cv2.resize` accepts `(width, height)`. In `segmentation_model.py`, `m_resized = cv2.resize(raw_m.astype(np.float32), (w, h))` correctly uses `(w, h)`, but the shape check `raw_m.shape != (h, w)` compares `(mask_h, mask_w)` to `(height, width)` rather than `(height, width)`.
- **Depth Contrast:** `create_depth_heatmap` uses simple linear min-max scaling without percentile clipping or histogram adjustments. Outlier depth predictions compress the bulk of the room into narrow, low-contrast color bands.

---

## Action Plan for Remediation

### Phase 2: Primary Semantic Segmentation (SegFormer-B1)
- Integrate `nvidia/segformer-b1-finetuned-ade-512-512` into `models/segmentation_model.py` as the primary semantic segmenter.
- Map ADE20K 150 classes to RoomAI's unified room taxonomy (wall, floor, ceiling, bed, sofa, chair, table, cabinet, door, window).
- Preserve full image resolution with bilinear logit interpolation before argmax.

### Phase 3: Instance Segmentation Refinement (A3 + FastSAM-s Prompted Ensemble)
- Integrate A3 detector + FastSAM box prompting as an experimental instance segmentation candidate.
- Preserve A3 bounding boxes, class labels, and confidence scores.
- Filter invalid masks, crop/soft-refine masks within bounding boxes, and remove duplicate overlapping masks using Mask IoU suppression.

### Phase 4: Correct Depth Pipeline & Visualizations
- Support both Relative Depth (`Depth Anything V2 Small`) and Metric Indoor Depth (`Depth Anything V2 Metric Indoor Small`).
- Enhance contrast using 2nd–98th percentile clipping and robust min-max normalization.
- Provide continuous grayscale, colorized heatmap (using Turbo/Inferno colormaps), and thresholded foreground/background segmentation modes.

### Phase 5: UI & Schema Refinement
- Update `ui/analysis_view.py` and `schemas/responses.py` to support dual views:
  1. **Semantic Segmentation View:** Clean category surface coverage summary (aggregated, no duplicates).
  2. **Instance Segmentation View:** Distinct instance overlays, unique instance IDs, confidence scores, and individual bounding box boundaries.
- Update depth view to clearly distinguish relative vs. metric distance modes.

### Phase 6: Empirical Benchmarking & Verification
- Benchmark updated semantic, instance, and depth pipelines on 30+ diverse indoor validation scenes.
- Verify zero regression across all 146 existing unit and integration tests.
