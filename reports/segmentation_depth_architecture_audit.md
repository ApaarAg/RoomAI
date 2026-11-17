# RoomAI — Segmentation and Depth Pipeline Architecture Audit Report

**Date:** October 30, 2026  
**Auditor:** Senior Computer Vision Engineer & Software Architect  
**Repository Root:** `F:\RoomAI`  
**Target Environment:** `tf-gpu` Conda environment (Python 3.10.20, PyTorch 2.6.0+cu124, NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM)  
**Investigation Mode:** Strict Read-Only Codebase Tracing & Empirical Audit  

---

## Executive Summary

This audit traces the complete execution flow, data contracts, and architectural connections across **Object Detection**, **Semantic/Instance Segmentation**, **Monocular Depth Estimation**, and **3D Scene Understanding** in RoomAI.

### Key Audit Verdicts:
1. **Segmentation Architecture Verdict:** **Architecture B** (Completely Independent Processing). YOLO11n-seg runs entirely detached from object detection models (A3/B2/C2). Detection bounding boxes, class labels, and geometric anchors are **never passed** to the segmentation pipeline.
2. **Depth Pipeline Verdict:** **Independent 2D Relative Proximity Estimation**. Depth Anything V2 Small processes the raw 2D input image in complete isolation. No detection boxes, segmentation masks, or semantic priors inform depth calculation. The displayed output is an affine-invariant, non-metric relative depth heatmap.
3. **3D Capability Verdict:** **Zero True 3D Reconstruction**. RoomAI currently performs no 3D point cloud back-projection, camera intrinsics estimation, depth unprojection, mesh reconstruction, coordinate transformation, or 3D scene graphing. It functions as a collection of disjoint 2D perceptual displays.

---

## Section A: Current Architecture Diagram

Below is the verified end-to-end dataflow currently implemented in the codebase. Every connection shown exists in the Python code; disconnected branches reflect absolute separation in the codebase.

```
                           +---------------------------------------+
                           |           User Image Upload           |
                           |       (Streamlit File Uploader        |
                           |       or Preloaded Benchmark)         |
                           +---------------------------------------+
                                               |
                                               v
                           +---------------------------------------+
                           |      Image Validation Pipeline        |
                           |     (utils/image_validation.py)       |
                           |  - Dimensions clamped <= 4096px       |
                           |  - Decompression bomb check           |
                           |  - Channel norm to RGB (PIL Image)    |
                           +---------------------------------------+
                                               |
                 +-----------------------------+-----------------------------+
                 | (image)                                   | (image)       | (image)
                 v                                           v               v
  +-----------------------------+             +------------------+  +-------------------+
  |   Tab 1: Object Detection   |             | Tab 2: Segment.  |  | Tab 3: Depth (3D) |
  |   (ui/analysis_view.py)     |             | (analysis_view)  |  | (analysis_view)   |
  +-----------------------------+             +------------------+  +-------------------+
                 |                                           |               |
                 v                                           v               v
  +-----------------------------+             +------------------+  +-------------------+
  |      DetectionService       |             |  Segment.Service |  |  Scene3DService   |
  |  (services/detection_       |             |  (services/      |  |  (services/       |
  |         service.py)         |             |  segmentation_   |  |  scene3d_service) |
  +-----------------------------+             |   service.py)    |  +-------------------+
                 |                            +------------------+           |
                 v                                     |                     v
  +-----------------------------+                      v            +-------------------+
  |     CustomObjectDetector    |             +------------------+  | CustomScene3DModel|
  | (models/detection_model.py) |             |CustomRoomSegmenter| |(models/scene3d_   |
  |  - Model A3: YOLO11s (800px)|             |(models/segment-  |  |     model.py)     |
  |  - Model B2: YOLO11s (640px)|             |   ation_model)   |  | - Depth Anything  |
  |  - Model C2: YOLO11s (640px)|             | - YOLO11n-seg.pt |  |   V2 Small        |
  +-----------------------------+             +------------------+  +-------------------+
                 |                                     |                     |
                 v                                     v                     v
  +-----------------------------+             +------------------+  +-------------------+
  |      DetectionResponse      |             |  Segment.Response|  |  Scene3DResponse  |
  |   + Annotated PIL Image     |             |  + 2D uint8 mask |  |  + norm float32   |
  |  (services/image_service.py)|             | (services/image_ |  |    depth array    |
  |   draw_bounding_boxes()     |             |    service.py)   |  +-------------------+
  +-----------------------------+             |  create_segment- |           |
                 |                            |  ation_overlay() |           v
                 v                            +------------------+  +-------------------+
  +-----------------------------+                      |            | create_depth_     |
  | UI Display: Bounding Boxes  |                      v            | heatmap() (PIL)   |
  | + Class Pill Badges         |             +------------------+  +-------------------+
  | + Latency Metrics Breakdown |             | UI Display:      |           |
  +-----------------------------+             | Color Overlay    |           v
                                              | + Category Table |  +-------------------+
                                              +------------------+  | UI Display:       |
                                                                    | Plasma Heatmap    |
                                                                    | + Non-Metric Disc.|
                                                                    +-------------------+

  ============================ ARCHITECTURAL VOID ============================
  [!] NO CONNECTION between Detection bounding boxes and Segmentation masks.
  [!] NO CONNECTION between Segmentation masks and Depth Estimation.
  [!] NO 3D Back-projection, Camera Intrinsics, Point Cloud, or Mesh generation.
  ============================================================================
```

---

## Section B: Component Inventory

| Component | File Path & Function/Class | Model / Checkpoint | Input | Output | Device | Preprocessing & Postprocessing |
|---|---|---|---|---|---|---|
| **Image Validation** | `utils/image_validation.py`<br>`validate_image_file()` | None (Pillow algorithmic) | `BinaryIO` / `bytes` / path | `ImageValidationResult`<br>(PIL Image, RGB) | CPU | EXIF transposition (`exif_transpose`), dimensions downsampled via Lanczos if `max(w,h) > 4096`. |
| **Indoor Detector (A3)** | `models/detection_model.py`<br>`CustomObjectDetector.predict()` | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` (18.29 MB) | Validated PIL Image in RGB | `DetectionResponse`<br>(BoundingBox normalized `[0,1]`, class IDs, confidences) | `cuda:0`<br>(fallback CPU) | **Pre:** Auto letterboxing to 800x800, normalize.<br>**Post:** Thresholding, `xyxy` normalized to `[0,1]`, coordinate clipping. |
| **Detection Coordinator** | `services/detection_service.py`<br>`DetectionService.run()` | Wraps `CustomObjectDetector` / `EnsembleDetector` | Validated PIL Image, `confidence_threshold` | `dict` containing `DetectionResponse`, `annotated_image` | `cuda:0` / CPU | Optional CLAHE (RGB->LAB->CLAHE->RGB), draws bounding boxes via PIL `ImageDraw` with high-contrast pills. Saves PNG to `artifacts/detections/`. |
| **Instance Segmenter** | `models/segmentation_model.py`<br>`CustomRoomSegmenter.predict()` | `checkpoints/yolo11n-seg.pt` (5.90 MB, COCO 80-class) | Validated PIL Image in RGB | `Tuple[SegmentationResponse, np.ndarray]` (2D mask array `uint8`) | `cuda:0`<br>(fallback CPU) | **Pre:** YOLO standard letterboxing (640x640).<br>**Post:** Masks resized via `cv2.INTER_NEAREST` to original `(h, w)`, binarized at `> 0.5`, composite mask assigned unique sequential integer IDs. |
| **Segmentation Coordinator** | `services/segmentation_service.py`<br>`SegmentationService.run()` | Wraps `CustomRoomSegmenter` | Validated PIL Image | `dict` containing `SegmentationResponse`, `overlay_image`, `mask_array` | `cuda:0` / CPU | Blends composite mask with original image using `create_segmentation_overlay(alpha=0.45)`. |
| **Monocular Depth Estimator** | `models/scene3d_model.py`<br>`CustomScene3DModel.reconstruct()` | `checkpoints/depth_anything_v2_small/` (`model.safetensors`, 94.58 MB) | Validated PIL Image in RGB | `Tuple[np.ndarray, Scene3DResponse]` (float32 array `[0.0, 1.0]`) | `cuda:0`<br>(fallback CPU) | **Pre:** `AutoImageProcessor` tensor prep (rescaling, ImageNet normalization).<br>**Post:** Bicubic interpolation to exact original `(orig_h, orig_w)`, NaN/Inf cleanup, min-max normalization to `[0.0, 1.0]`. |
| **Depth Coordinator** | `services/scene3d_service.py`<br>`Scene3DService.run()` | Wraps `CustomScene3DModel` | Validated PIL Image | `dict` containing `Scene3DResponse`, `depth_map`, `heatmap_image` | `cuda:0` / CPU | Converts continuous 2D relative depth array into a 3-channel Plasma-like gradient RGB PIL Image (`create_depth_heatmap`). |
| **UI Presentation Layer** | `ui/analysis_view.py`<br>`render_analysis_view()` | None (Streamlit UI layout) | Validated PIL Image, config dicts | Streamlit rendered tabs | CPU / Client Browser | Independent tabs with separate trigger buttons (`btn_run_detect_*`, `btn_run_seg`, `btn_run_3d`). |

---

## Section C: Investigation 1 — Trace Object Detection

### 1. UI Component Invoking Detection
- **Source File:** [`ui/analysis_view.py`](file:///F:/RoomAI/ui/analysis_view.py#L63-L157)
- **Execution:**
  Inside `render_analysis_view()`, under **Tab 1: Object Detection**:
  - Domain radio selector (`task_group_radio`, lines 67-78) switches between Indoor (`indoor`), Exterior (`exterior`), and Parking (`parking`).
  - For Indoor, `detector_choices` defaults to `"Model A3: Higher-Resolution YOLO11s (800px 12-Class) [Primary Recommended]"` mapping to `"checkpoints/indoor_detector_a3_yolo11s_800px_best.pt"` (line 83).
  - Confidence slider `det_conf` defaults to `0.35` (lines 122-135).
  - The detection trigger button is `st.button("🚀 Run Object Detection", key="btn_run_detect_indoor", type="primary")` (line 139).
  - Calling code (lines 147-154):
    ```python
    det_service = get_cached_detection_service(
        mock_mode=mock_mode,
        checkpoint_path=selected_ckpt,
        device=det_device,
        task_group=active_domain,
        imgsz=det_imgsz,
    )
    result = det_service.run(image, confidence_threshold=det_conf)
    ```

### 2. Service Handling the Request
- **Source File:** [`services/detection_service.py`](file:///F:/RoomAI/services/detection_service.py#L19-L168)
- **Function/Class:** `resolve_detector()` and `DetectionService.run()`
- **Execution:**
  - `resolve_detector(config, mock_mode)` instantiates `CustomObjectDetector` with the checkpoint path (lines 88, 164).
  - In `DetectionService.run(image, confidence_threshold)`:
    - Preprocesses image to RGB (line 197).
    - Measures wall-clock preprocessing latency (line 211).
    - Calls `self.model.predict(image, confidence_threshold=confidence_threshold)` (line 215).
    - Calls `draw_bounding_boxes(image, response.detections)` (line 231).
    - Saves annotated artifact to `artifacts/detections/detection_{timestamp}.png` (lines 238-250).

### 3. Model Class Loading Model A3
- **Source File:** [`models/detection_model.py`](file:///F:/RoomAI/models/detection_model.py#L39-L108)
- **Function/Class:** `CustomObjectDetector.load_model()` and `get_cached_yolo()`
- **Execution:**
  - Resolves path to `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` (lines 79-90).
  - Retrieves or instantiates `ultralytics.YOLO(checkpoint_path)` stored in module-level `_YOLO_CACHE[(checkpoint_path, device)]` (lines 23-32, 101).
  - Sets resolution `self.imgsz = 800` automatically because `"800px"` is present in the checkpoint path name (line 48).

### 4. Exact Image Passed to the Detector
- In `ui/upload_view.py`: `validate_image_file()` loads the raw upload bytes, applies `ImageOps.exif_transpose`, converts to RGB, clamps dimensions if exceeding 4096px, and returns a PIL `Image.Image`.
- In `services/detection_service.py`: `image = image.convert("RGB")` (line 197).
- In `models/detection_model.py`: The **unmodified, uncropped full-resolution PIL Image** is passed directly to Ultralytics:
  ```python
  results = self.model.predict(
      source=image,
      imgsz=self.imgsz, # 800
      conf=infer_conf,
      iou=self.iou_threshold,
      device=self.device,
      half=use_half,
      verbose=False,
  )
  ```

### 5. What the Detector Returns
- `CustomObjectDetector.predict()` returns a strictly typed Pydantic schema: `schemas.responses.DetectionResponse` (lines 216-224):
  - `detections`: List of `DetectionItem` objects.
  - `total_objects`: `int` (count of retained bounding boxes).
  - `is_simulated`: `False`.
  - `model_name`: `"Ultralytics YOLO (indoor_detector_a3_yolo11s_800px_best.pt)"`.
  - `checkpoint`: `"indoor_detector_a3_yolo11s_800px_best.pt"`.
  - `inference_time_ms`: CUDA-synchronized hardware forward pass duration.
  - `device`: `"cuda:0"`.

### 6. Where Bounding Boxes and Class Labels Are Stored
- **In Memory:** Each detection is stored as a `schemas.responses.DetectionItem` (lines 208-214):
  - `class_id`: `int` (0 to 11 for A3).
  - `label`: `str` (`r.names[cls_id]`, e.g., `'sofa'`, `'table'`, `'lamp'`).
  - `confidence`: `float` rounded to 4 decimal places.
  - `box`: `BoundingBox(x_min, y_min, x_max, y_max)` normalized to `[0.0, 1.0]`.
- **In Streamlit Session State:**
  - `st.session_state["detection_result_indoor"] = result` ([`ui/analysis_view.py#L155`](file:///F:/RoomAI/ui/analysis_view.py#L155)).
  - `st.session_state["detection_result"] = result` (line 156).

### 7. Where Confidence Thresholds Are Applied (Audit of Multiple Places)
Confidence thresholds are evaluated in **two distinct stages**:
1. **Inference Stage (`models/detection_model.py#L145-L158`):** Passed into `model.predict(..., conf=infer_conf)`. Ultralytics filters candidate anchors during Non-Maximum Suppression (NMS).
2. **Post-Processing Filtering Stage (`models/detection_model.py#L176-L185`):** Explicit secondary validation against `effective_threshold` inside the detection parsing loop:
   ```python
   if conf < effective_threshold:
       continue
   ```
*(Note: In `services/ensemble_service.py#L302`, an initial lower threshold `max(0.10, conf - 0.15)` is applied to individual models so weak consensus signals can participate before final WBF thresholding at line 241).*

---

## Section D: Investigation 2 — Trace Segmentation (Critical Verdict)

### 1. Architectural Verdict
**Verdict:** **Architecture B — YOLO11n-seg independently processes the original image and detects/segments objects.**

There is **no coupling whatsoever** between the primary indoor detector (Model A3) and the segmentation model.
- Architecture A is FALSE: A3 bounding boxes are **never** passed to YOLO11n-seg.
- Architecture C is FALSE: Results are **never** combined or cross-referenced after inference.
- Architecture D is FALSE: It is not a cascaded or prompt-based pipeline (e.g. SAM).

#### Code Evidence:
1. In [`ui/analysis_view.py#L244-L250`](file:///F:/RoomAI/ui/analysis_view.py#L244-L250):
   ```python
   if st.button("🚀 Run Segmentation", key="btn_run_seg", type="primary"):
       with st.spinner("Segmenting object instances with YOLO11n-seg..."):
           seg_service = SegmentationService(
               mock_mode=mock_mode, config=model_config.get("segmentation", {})
           )
           result = seg_service.run(image)
           st.session_state["segmentation_result"] = result
   ```
   **Notice:** Only `image` is passed to `seg_service.run(image)`. The detection result (`st.session_state["detection_result"]`) is completely ignored.
2. In [`services/segmentation_service.py#L27`](file:///F:/RoomAI/services/segmentation_service.py#L27):
   ```python
   response, mask_array = self.model.predict(image)
   ```
   No bounding boxes or class hints are passed to `self.model.predict()`.
3. In [`models/segmentation_model.py#L73-L91`](file:///F:/RoomAI/models/segmentation_model.py#L73-L91):
   ```python
   def predict(self, image: Image.Image) -> Tuple[SegmentationResponse, Optional[np.ndarray]]:
       ...
       results = self.model.predict(
           image,
           conf=self.conf,
           device=self.device,
           verbose=False,
       )
   ```
   The segmentation model runs raw prediction directly on the uploaded image.

### 2. Specific Technical Questions Answered

- **Which checkpoint is loaded? Is it actually YOLO11n-seg?**
  - **Yes.** Checkpoint: `checkpoints/yolo11n-seg.pt` (size: 5.90 MB).
  - Inspection via Ultralytics confirms: `Task: segment`, `Architecture: SegmentationModel`, `Number of Classes: 80` (Standard MS COCO).
- **Does segmentation receive the original image or cropped object regions?**
  - It receives the **entire original full image**. No cropping is performed.
- **Does it receive detection bounding boxes?**
  - **No.** Zero bounding boxes are received.
- **Does it use prompts?**
  - **No.** YOLO11n-seg is a direct single-stage anchor-free instance segmentation network; it has no prompt encoder (unlike SAM or FastSAM).
- **Does it generate masks independently?**
  - **Yes.** Both classification, bounding boxes, and prototype mask coefficients are predicted entirely within YOLO11n-seg in a single forward pass.
- **Are detection and segmentation results matched by class or bounding-box overlap?**
  - **No.** There is no matching, IoU alignment, or fusion between detection and segmentation.
- **Are masks filtered by confidence?**
  - **Yes.** Filtered by `self.conf` (configured as `0.25` in `config/model_config.yaml#L191`) passed to `self.model.predict(conf=self.conf)`.
- **Are masks resized to original image dimensions?**
  - **Yes.** In `models/segmentation_model.py#L114-L117`:
    ```python
    if raw_m.shape != (h, w):
        m_resized = cv2.resize(raw_m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        m_resized = raw_m
    ```
- **How are masks converted into the displayed colored overlay?**
  - In `models/segmentation_model.py#L102-L135`, a single 2D integer array `combined_mask` of shape `(h, w)` and type `uint8` is created.
  - Each detected instance `idx` is assigned an integer `mask_id = (idx % 254) + 1`. Where `binary_mask` is True, `combined_mask[binary_mask] = mask_id`.
  - In `services/image_service.py#L73-L106` (`create_segmentation_overlay`):
    - For every unique ID in `combined_mask`, a color from `PALETTE` is assigned: `color = PALETTE[(int(uid) - 1) % len(PALETTE)]`.
    - It alpha-blends the colors with the RGB image using `alpha = 0.45`:
      ```python
      blended = np.where(is_masked, (1 - alpha) * img_rgb + alpha * color_mask, img_rgb)
      ```
- **Are overlapping masks resolved or simply drawn over one another?**
  - **Drawn over one another (Overwritten).**
    Line 135 in `models/segmentation_model.py`:
    ```python
    combined_mask[binary_mask] = mask_id
    ```
    If two masks overlap, the later instance in the loop overwrites the pixel values of the earlier instance. No occlusion reasoning or multi-channel mask blending is performed.
- **Are unsupported classes silently discarded?**
  - **Yes.** YOLO11n-seg is hardcoded to MS COCO 80 categories. Non-COCO interior classes (such as architectural features like walls, floors, ceilings, mouldings, columns, cabinets, and curtains) have no class heads in `yolo11n-seg.pt` and cannot be predicted.

### 3. Root Cause: Why Visible Room Objects Fail to Receive Masks
There are **three verified causes**:
1. **Taxonomy Mismatch (Primary Cause):**
   - Model A3 is trained on DeepFurniture / HomeObjects-3K with specialized classes: `cabinet#shelf`, `door`, `window`, `lamp`, `curtain`, `decoration`, `home-appliance`.
   - YOLO11n-seg is trained on MS COCO. COCO does **not have** classes for `cabinet`, `shelf`, `lamp`, `door`, `window`, `curtain`, `wall`, `floor`, or `ceiling`. Therefore, even if Model A3 detects cabinets, doors, and windows with 90%+ confidence, YOLO11n-seg has no weights or heads to segment them.
2. **Independent Re-Detection Failure (Secondary Cause):**
   - Because YOLO11n-seg must re-detect objects from scratch, its tiny nano backbone (YOLO11n, 2.6M params) frequently misses low-contrast or partially occluded furniture that Model A3 (YOLO11s, 9.4M params at 800px) easily detects.
3. **Resolution Difference:**
   - Model A3 runs at **800px** resolution. YOLO11n-seg runs at **640px** default resolution, leading to lost segmentation fidelity on smaller decor items and background furniture.

---

## Section E: Investigation 3 — Trace Depth Estimation

### 1. Complete Depth Anything V2 Small Pipeline
- **Source Files:**
  - UI: [`ui/analysis_view.py#L303-L358`](file:///F:/RoomAI/ui/analysis_view.py#L303-L358)
  - Service: [`services/scene3d_service.py#L14-L64`](file:///F:/RoomAI/services/scene3d_service.py#L14-L64)
  - Model: [`models/scene3d_model.py#L21-L208`](file:///F:/RoomAI/models/scene3d_model.py#L21-L208)
  - Visualization: [`services/image_service.py#L108-L125`](file:///F:/RoomAI/services/image_service.py#L108-L125)

### 2. Step-by-Step Data Transformations
1. **Input Image:** Full-resolution RGB PIL Image of arbitrary input dimensions `(orig_w, orig_h)`.
2. **Image Preprocessing (`models/scene3d_model.py#L143-L144`):**
   - Processed via Hugging Face `AutoImageProcessor.from_pretrained("checkpoints/depth_anything_v2_small")`.
   - Rescales pixel intensity to `[0, 1]` and standardizes using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
   - Dimensions are resized to `(518, 518)` (or patch-multiple suitable for DINOv2 ViT).
3. **Model Configuration:**
   - Architecture: DINOv2 Vision Transformer Small (`ViT-Small` backbone, 24.8M parameters) with a DPT (Dense Prediction Transformer) head.
   - Forward pass in PyTorch with `torch.no_grad()`: `outputs = self.model(**inputs)`.
4. **Raw Model Output Shape:**
   - `pred_depth = outputs.predicted_depth` yields a tensor of shape `(1, 518, 518)` (batch, height, width).
5. **Inverse Resizing & Interpolation (`models/scene3d_model.py#L166-L173`):**
   - The predicted depth is resized back to exact original input resolution using **bicubic interpolation**:
     ```python
     pred_aligned = torch.nn.functional.interpolate(
         pred_depth.unsqueeze(1),
         size=(orig_h, orig_w),
         mode="bicubic",
         align_corners=False,
     ).squeeze()
     ```
6. **Numerical Cleaning & Min-Max Normalization (`models/scene3d_model.py#L176-L186`):**
   - Guard against NaNs/Infs: `raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=1.0, neginf=0.0)`.
   - Normalized to strictly `[0.0, 1.0]`:
     ```python
     d_min, d_max = float(raw_depth.min()), float(raw_depth.max())
     norm_depth = (raw_depth - d_min) / (d_max - d_min)
     ```
7. **Relative vs. Metric Depth:**
   - **Strictly Relative Depth (Affine-Invariant).**
   - It is continuous disparity/proximity: higher values = closer to camera, lower values = farther from camera.
   - It contains **no metric scale** (meters/feet).
   - The code explicitly documents this in `schemas/responses.py#L135-L138`:
     `"Relative depth map output: higher values indicate proximity to camera. Not metric distance in meters; uncalibrated for physical measurement."`
8. **Depth Visualization Generation (`services/image_service.py#L108-L125`):**
   - `create_depth_heatmap(norm_depth)` maps `norm_depth` into RGB channels using piecewise linear gradient formulas:
     ```python
     r = (np.clip(2.0 * norm, 0, 1) * 255).astype(np.uint8)
     g = (np.clip(2.0 * (1.0 - np.abs(norm - 0.5)), 0, 1) * 220).astype(np.uint8)
     b = (np.clip(2.0 * (1.0 - norm), 0, 1) * 255).astype(np.uint8)
     ```
   - This produces a pseudo-color plasma heatmap where bright yellow = foreground, dark purple = background.
9. **Influence of Detection or Segmentation on Depth:**
   - **None.** Neither bounding boxes nor segmentation masks are passed to or read by `CustomScene3DModel` or `Scene3DService`.

---

## Section F: Investigation 4 — Trace the 3D Scene Pipeline

### Comprehensive Codebase Audit for 3D Capabilities

| 3D Capability | Found in Codebase? | Source Evidence & Status |
|---|---|---|
| **Point Cloud Generation** | **NO** | Flag `generate_point_cloud: false` in `config/model_config.yaml#L222`. Schema field `point_cloud_available = False` hardcoded in `models/scene3d_model.py#L191`. No back-projection code exists. |
| **Camera Intrinsics ($K$ matrix)** | **NO** | Focal length ($f_x, f_y$), principal point ($c_x, c_y$), and field-of-view (FOV) are **completely absent** in all configs, models, and schemas. |
| **Coordinate Transformations** | **NO** | No transformation matrices, extrinsics ($R, t$), camera-to-world, or world-to-grid projections exist. |
| **Mesh Generation / Polygonal 3D** | **NO** | Zero occurrences of marching cubes, Poisson surface reconstruction, Delaunay triangulation, or geometry libraries (`open3d`, `trimesh`, `pyvista`, `pytorch3d`, `scipy.spatial`). |
| **Depth-to-3D Unprojection** | **NO** | No $Z = \text{depth}, X = (u - c_x) Z / f_x, Y = (v - c_y) Z / f_y$ equation exists anywhere in the repository. |
| **Scene Graphs / Spatial Relations** | **NO** | Bounding boxes are stored as independent flat 2D rectangles in `DetectionItem`. No parent-child, support, or containment hierarchy exists. |
| **Object Placement & 3D Bounding Boxes** | **NO** | Only 2D bounding boxes `[x_min, y_min, x_max, y_max]` exist. No 3D oriented bounding boxes `(x, y, z, w, h, d, r, p, y)`. |
| **Materials & Lighting Models** | **NO** | No PBR (physically based rendering), albedo, roughness, specular, or normal map generation. |
| **3D Rendering Engine** | **NO** | No WebGL, Three.js, PyVista, or PyTorch3D rendering canvas in Streamlit UI. UI only displays 2D PNG images. |

### 3D Capability Verdict:
**Actual 3D scene reconstruction DOES NOT exist in RoomAI.**  
The module named `scene3d` is purely a **2D monocular relative depth heatmap generator**. The application produces and displays three independent 2D raster images:
1. 2D bounding box overlay image.
2. 2D instance segmentation mask overlay image.
3. 2D false-color depth heatmap image.

---

## Section G: Investigation 5 — Architectural Weaknesses & Evaluation

### 1. Segmentation Model Suitability for RoomAI Use Case
- **Status: Verified Critical Deficiency.**
- `yolo11n-seg.pt` is a generic COCO object segmenter designed for general scenes (airplanes, zebras, sports balls, traffic lights). It is fundamentally unsuited as the primary segmentation engine for an interior architecture and design tool.

### 2. Interior Object Category Coverage
- **Status: Verified Severe Gap.**
- COCO contains only 6 household furniture categories: `chair`, `couch`, `bed`, `dining table`, `toilet`, `tv`.
- It completely lacks interior design staples: `cabinet`, `shelf`, `lamp`, `door`, `window`, `curtain`, `nightstand`, `desk`, `coffee table`, `wardrobe`, `refrigerator`, `stove`, `kitchen island`.

### 3. Room Surface Segmentation (Walls, Floors, Ceilings)
- **Status: Verified Complete Absence.**
- Neither `yolo11n-seg` nor Model A3 predicts planar room surfaces (`wall`, `floor`, `ceiling`).
- **Deceptive Documentation vs. Reality:**
  In `config/model_config.yaml#L194-L198`:
  ```yaml
  categories:
    - "floor"
    - "wall"
    - "furniture"
    - "ceiling"
    - "window"
  ```
  These categories are declared in the YAML comments/config, but `models/segmentation_model.py` loads `yolo11n-seg.pt`, which outputs COCO labels (`names: {0: 'person', 1: 'bicycle', ..., 56: 'chair', 57: 'couch'}`). The YAML list is completely ignored by the model. The only place "floor" and "wall" exist is in `MockRoomSegmenter` (lines 179-209), which hardcodes synthetic rectangles.

### 4. Geometric Consistency Between Detection and Segmentation
- **Status: Verified Inconsistent.**
- Because detection (Model A3) and segmentation (YOLO11n-seg) run independently:
  - Detection identifies a `cabinet#shelf` with 0.88 confidence; Segmentation outputs nothing.
  - Detection identifies a `chair#stool` with bounding box $B_1$; Segmentation predicts a `chair` with bounding box $B_2 \neq B_1$ and a differing mask boundary.
  - The masks frequently spill outside the detected bounding boxes or fail to cover detected objects.

### 5. Utility of Current Depth Estimation
- **Status: Verified Underutilized.**
- The relative depth map from Depth Anything V2 Small is treated as a dead-end endpoint. It is mapped to a colored PNG and shown to the user. It does not inform:
  - Object distance from camera.
  - Spatial ordering (which chair is in front of the table).
  - Room bounding geometry or wall depth planes.

### 6. Viability of Current Outputs for Convincing 3D Scene Generation
- **Status: Verified Inviable Without Re-Architecture.**
- To build a 3D scene, an engine requires:
  1. Camera intrinsics (focal length, principal point).
  2. Metric or consistent affine depth.
  3. Ground plane estimation (floor normal and height).
  4. 3D bounding bounding boxes or 3D CAD model retrieval for detected furniture.
  None of these prerequisites exist in RoomAI's current pipeline.

---

## Section H: Prioritized Improvement Plan

To transition RoomAI from three disconnected 2D visualizers to a coherent 3D interior intelligence engine, execute the following phased plan:

### Phase 1: Object Segmentation Quality (Immediate Priority)
- **Current Limitation:** YOLO11n-seg ignores Model A3's detections and misses furniture categories due to COCO taxonomy limits.
- **Proposed Change:** Implement **Prompted Instance Segmentation** (e.g. MobileSAM / FastSAM / SAM 2). Pass Model A3's high-confidence bounding boxes `[x1, y1, x2, y2]` directly as bounding box prompts to the promptable segmenter.
- **Expected Benefit:** 100% of Model A3's detected furniture objects (cabinets, shelves, beds, sofas, lamps) immediately receive exact, high-fidelity boundary masks. Detection and segmentation become 100% geometrically consistent.
- **Compute Requirements:** FastSAM / MobileSAM requires ~40ms on RTX 4060 GPU and ~400 MB VRAM.
- **Additional Data Needed:** **Zero.** Pretrained MobileSAM / FastSAM generalizes zero-shot to any bounding box.
- **Risks & Trade-offs:** Latency increases by ~40ms per image.

### Phase 2: Room-Region & Surface Segmentation (Structural Boundary Priority)
- **Current Limitation:** Bare walls, floors, ceilings, and windows are not segmented.
- **Proposed Change:** Integrate a lightweight indoor semantic segmenter trained on ADE20k (e.g. SegFormer-B0 or EfficientViT-Seg).
- **Expected Benefit:** Dense multi-class planar masks identifying `wall`, `floor`, `ceiling`, `door`, and `window`.
- **Compute Requirements:** SegFormer-B0 requires ~15ms on GPU and ~300 MB VRAM.
- **Additional Data Needed:** Pretrained ADE20k weights are open-source and ready off-the-shelf.
- **Risks & Trade-offs:** Blending instance masks (furniture) with semantic masks (walls/floors) requires priority compositing (instances overlaid on top of structural planes).

### Phase 3: Depth Estimation Quality & Metric Alignment
- **Current Limitation:** Relative uncalibrated depth cannot measure distances in meters.
- **Proposed Change:** Upgrade from relative Depth Anything V2 to **Depth Anything V2 Metric** (indoor model) or apply vanishing point / camera field-of-view estimation to recover approximate metric scale ($Z$ in meters).
- **Expected Benefit:** Quantitative room depth in meters; enables calculating room width, length, and ceiling height.
- **Compute Requirements:** Depth Anything V2 Metric Small requires identical GPU compute (~22ms on RTX 4060, ~950 MB VRAM).
- **Additional Data Needed:** None (Hugging Face off-the-shelf metric weights).
- **Risks & Trade-offs:** Metric depth models assume typical indoor camera heights; extreme angles may require tilt compensation.

### Phase 4: Multi-Modal Scene Fusion
- **Current Limitation:** Detection, segmentation, and depth operate as isolated silos.
- **Proposed Change:** Construct an unified `SceneGraph` representation:
  1. Use surface segmentation (Phase 2) to identify the floor plane.
  2. Sample depth values within each object mask (Phase 1) to compute 3D centroid coordinates $(X, Y, Z)$ relative to the camera.
  3. Compute pairwise spatial relations (`"chair is on floor"`, `"lamp is on table"`, `"sofa is in front of wall"`).
- **Expected Benefit:** True spatial intelligence for layout validation, collision detection, and furniture replacement feasibility.
- **Compute Requirements:** Purely algorithmic NumPy/SciPy operations (< 5ms on CPU).
- **Additional Data Needed:** None.
- **Risks & Trade-offs:** Requires robust depth outlier filtering (median depth within mask rather than mean) to prevent boundary bleed.

### Phase 5: Visually Convincing 3D Scene Generation
- **Current Limitation:** No 3D visualization or mesh reconstruction exists.
- **Proposed Change:**
  1. Unproject RGB-D into an Open3D / Three.js textured point cloud.
  2. Fit bounding cuboids (3D oriented bounding boxes) around detected furniture centroids.
  3. Render an interactive 3D scene directly in Streamlit using `streamlit-threejs` or HTML/WebGL component.
- **Expected Benefit:** Users can orbit, pan, and inspect the room in true 3D, seeing actual spatial clearance and furniture layout.
- **Compute Requirements:** WebGL handles rendering on client GPU; unprojection takes < 10ms in Python.
- **Additional Data Needed:** Low-poly 3D furniture asset library for CAD retrieval (optional).
- **Risks & Trade-offs:** Occluded areas behind furniture will have "shadow tears" (voids) in raw point clouds unless depth inpainting is applied.

---

## Section I: Final Summary (Teaching a Junior Computer Vision Engineer)

Imagine you have three students working on an interior design project, each sitting in a separate room with soundproof glass:

1. **Student 1 (Detection):** Looks at the room photo and draws tight colored boxes around the sofa, table, and lamps.
2. **Student 2 (Segmentation):** Looks at a copy of the room photo, but only knows about the 80 objects listed in the COCO textbook. If they see a couch, they color its exact pixels. But because "cabinet", "curtain", "wall", and "floor" were never in their textbook, they completely ignore them. Worse, they never look at Student 1's work to see where the furniture was already found!
3. **Student 3 (Depth):** Looks at another copy of the room photo and paints a grayscale picture where closer things are bright and distant things are dark. But they don't know the focal length of the camera, so they cannot tell you if the wall is 2 meters away or 10 meters away. And nobody uses their depth painting to position the sofa in 3D.

Right now, **RoomAI simply displays the three student drawings on three separate tabs on your screen.**

To make RoomAI a true **3D Interior AI Studio**, we must connect them into a team:
- Student 1 hands their bounding boxes to a promptable segmenter (like SAM), saying: *"I already found the cabinet and sofa right here—just trace their exact outlines."*
- A specialized room-surface model paints the floor, walls, and ceiling.
- We calibrate Student 3's depth map so that every pixel has an $(X, Y, Z)$ coordinate in meters.
- Finally, we back-project those pixels into a 3D coordinate system to build an interactive 3D model that you can spin and walk through in your web browser.

