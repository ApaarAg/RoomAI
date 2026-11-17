# RoomAI — Phase 7: Initial Architecture Audit

**Date:** November 5, 2026  
**Auditor:** Principal Computer Vision & 3D Reconstruction Engineer  
**System:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), Windows 11, Conda `tf-gpu` (Python 3.10.20, PyTorch 2.11.0+cu128)  
**Status:** Complete  

---

## 1. Executive Summary

Phase 7 initiates the transformation of RoomAI from an image analysis tool into a complete room-understanding and 3D scene reconstruction system. Before designing and deploying the new reconstruction pipeline, an exhaustive forensic audit of the existing codebase was performed to diagnose active runtime exceptions, evaluate architectural constraints, verify service contracts, and plan the user interface redesign.

---

## 2. Root Cause Analysis: The 3D Scene AttributeError

### 2.1 Exception Traceback and Reproduction
When running 3D Scene generation with real inference enabled (`mock_mode: false`), the application crashes with:
```text
AttributeError: 'CustomRoomSegmenter' object has no attribute 'segmentation_mode'
  File "F:\RoomAI\services\scene_fusion_service.py", line 101, in fuse
    if self.segmenter.segmentation_mode == "instance":
```

### 2.2 Forensic Analysis
Investigation of `models/interfaces.py`, `models/segmentation_model.py`, and `services/scene_fusion_service.py` revealed four interconnected defects:

1. **Missing Abstract Property in Base Class Contract:**
   `BaseSegmenter` in `models/interfaces.py` specifies only `predict(image) -> Tuple[SegmentationResponse, Optional[np.ndarray]]`. It defines no `segmentation_mode` property or attribute.

2. **Uninitialized Property in Concrete Segmenter:**
   `CustomRoomSegmenter` in `models/segmentation_model.py` wraps Ultralytics YOLO-Seg (`checkpoints/yolo11n-seg.pt`). While YOLO-Seg is inherently an instance segmentation architecture, `CustomRoomSegmenter.__init__` did not declare or expose `self.segmentation_mode`.

3. **Inappropriate Property Access in Caller:**
   `SceneFusionService.fuse()` in `services/scene_fusion_service.py` directly performed `if self.segmenter.segmentation_mode == "instance":` without using `getattr()` or inspecting configuration (`config.get("segmentation", {}).get("mode")`).

4. **Redundant Inference Execution:**
   In `SceneFusionService.fuse()`, line 96 executed `sem_resp, sem_mask = self.segmenter.predict(image)`, and then line 102 attempted to execute `ins_resp, ins_mask = self.segmenter.predict(image)` a second time. This unnecessarily doubled inference latency and GPU memory allocations.

5. **Schema Incompatibility in `SegmentationMask`:**
   `schemas/responses.py` defined `SegmentationMask` with `category`, `coverage_percentage`, `mask_shape`, and `color_hex`. However, `_extract_objects_from_instance` in `services/scene_fusion_service.py` attempted to access `m.instance_id`, `m.mask_points_3d`, and `m.mask_points_colors`. Because Pydantic v2 enforces schema validation, missing fields would cause subsequent runtime failures if instance objects were processed.

---

## 3. UI Redesign Audit: Navigation and Deprecations

### 3.1 Current User-Facing Navigation
In `ui/analysis_view.py`, the analysis view currently defines 5 tabs:
1. `🔍 Object Detection`
2. `🧩 Region Segmentation`
3. `🏷️ Style Classification`
4. `💬 Vision Q&A`
5. `📐 3D Scene (Experimental)`

### 3.2 Deprecation Plan
Per Phase 7 requirements:
- **Remove from UI:**
  - Tab 2: Region Segmentation (semantic mask overlays and category tables).
  - Tab 3: Style Classification (room type and architectural style distributions).
- **Consolidate into 3 Streamlined Sections:**
  1. `🔍 Object Detection` (Featuring production models: A3 800px Indoor, B2 Exterior, C2 Parking).
  2. `💬 Vision Q&A` (Multimodal vision-language understanding).
  3. `🧊 3D Scene Reconstruction` (Upgraded from experimental viewer to holistic room reconstruction).
- **Backend Preservation Guarantee:**
  - `SegmentationService`, `CustomRoomSegmenter`, `SegFormer-B1`, and `FastSAM` remain fully preserved and functional in `models/` and `services/`.
  - These perception modules will be repurposed as internal pipeline components for room surface extraction, object masking, and 3D bounding box estimation.

---

## 4. Hardware and Environment Constraints

| Parameter | Specification | Status / Constraint |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU | 8,187.5 MB Total VRAM (~7,200 MB usable) |
| **CUDA / Driver** | CUDA 12.8 / Driver 572.70 | Fully supported by PyTorch 2.11.0+cu128 |
| **Host Memory** | 24 GB System RAM | Sufficient for model weights and mesh buffers |
| **Python Environment** | Conda `tf-gpu` (Python 3.10.20) | Active verified environment |
| **VRAM Budget Rule** | Max peak allocated < 6.5 GB | Staged execution required; no concurrent monolithic loading |

---

## 5. Remediation Roadmap

1. **Step 1:** Fix interface contract between `BaseSegmenter`, `CustomRoomSegmenter`, and `SceneFusionService`.
2. **Step 2:** Update `SegmentationMask` schema to support `instance_id`, `confidence`, and 3D points.
3. **Step 3:** Reorganize `ui/analysis_view.py` to present the 3 streamlined tabs, removing user-facing segmentation and classification while preserving backend services.
4. **Step 4:** Research and benchmark whole-room 3D reconstruction models against local hardware limits.
5. **Step 5:** Implement and evaluate metric depth estimation with `Depth Anything V2 Metric Indoor Small`.
6. **Step 6:** Implement room geometry estimation (floor/wall planes, room bounding box) and 3D object bounding boxes.
7. **Step 7:** Upgrade interactive 3D viewer with room structure rendering, furniture visibility toggles, object inspection cards, and GLB/PLY export.
8. **Step 8:** Validate on 20 representative indoor images and run the full 155+ test regression suite.

