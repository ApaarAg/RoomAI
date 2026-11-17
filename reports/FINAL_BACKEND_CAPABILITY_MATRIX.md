# ROOMAI — FINAL BACKEND CAPABILITY MATRIX

**Date:** 2026-11-06  
**Auditor:** Principal Software Architect & QA Lead  
**Scope:** Backend Subsystem Classification & Frontend Capability Mapping  

---

## 1. Subsystem Capability Classification Matrix

Each backend feature in RoomAI is strictly categorized according to empirical verification and operational stability:
- **`PRODUCTION_READY`**: Fully implemented, verified with authentic checkpoints on local GPU, passes automated test coverage, and exhibits stable real-time performance.
- **`EXPERIMENTAL`**: Algorithmic pipeline implemented and functional, but lacks empirical validation against authentic ground-truth benchmark datasets or real multi-view physical scans.
- **`UNAVAILABLE`**: Offline or unhosted locally to safeguard hardware resources (e.g. 8 GB VRAM preservation); returns structured degradation cards without crashing.
- **`BLOCKED`**: Dependent on missing third-party APIs or unresolvable environment constraints.
- **`NOT_IN_SCOPE`**: Features intentionally excluded from the current product specification.

| Subsystem / Capability | Status | Model / Architecture | Checkpoint Location | Verification Evidence | Operational Limitations & Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Indoor Object Detection** | **`PRODUCTION_READY`** | YOLO11s (A3 Fine-Tuned 800px) | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | mAP50=0.512 on indoor validation set; 24.8 ms warm latency; verified real detections. | Limited to trained indoor classes (beds, chairs, tables, sofas, lamps, etc.). |
| **Exterior Detection** | **`PRODUCTION_READY`** | YOLO11s (B2 Fine-Tuned) | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | Passes Level 4 real inference; 23.4 ms latency. Supports CLAHE contrast enhancement. | Specialized for exterior building components and facades. |
| **Parking Detection** | **`PRODUCTION_READY`** | YOLO11s (C2 Pilot) | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | Passes Level 4 real inference; 23.7 ms latency. | Specialized for outdoor parking spaces and vehicles. |
| **Instance Segmentation** | **`PRODUCTION_READY`** | YOLO11n-Seg | `checkpoints/yolo11n-seg.pt` | Real inference verified; 172.3 ms latency; produces polygon masks & overlay images. | Small model optimized for VRAM budget; complex occlusions may merge masks. |
| **Metric Monocular Depth** | **`PRODUCTION_READY`** | Depth Anything V2 Metric Indoor Small | `checkpoints/depth_anything_v2_metric_indoor_small` | Real inference verified; 518.5 ms latency; calibrated in metric meters. | Monocular scale uncertainty on non-standard focal lengths; guided refinement applied. |
| **Relative Depth Estimation** | **`PRODUCTION_READY`** | Depth Anything V2 Small Relative | `checkpoints/depth_anything_v2_small` | Real inference verified; 521.8 ms latency; normalized inverse depth [0, 1]. | Unscaled; suitable for relative depth ordering rather than physical CAD sizing. |
| **Single-View 3D Reconstruction** | **`PRODUCTION_READY`** | RoomAI Unified Scene Fusion (A3 + YOLO11-Seg + Depth + RANSAC) | Composite local models | 1,546.55 ms end-to-end latency; produces 200,000 metric 3D points, floor plane, and 3D boxes. | Dependent on camera angle; extreme top-down or fish-eye views may reject floor plane. |
| **Room Layout & Floor Plane** | **`PRODUCTION_READY`** | RANSAC Plane Estimator + Gravity Prior | Algorithmic (`services/room_layout_service.py`) | Tested across synthetic and real rooms; rejects non-horizontal planes (gravity threshold). | If floor is severely occluded by carpeting or clutter, defaults to bounding box lower bound. |
| **Multi-View 3D Scene Fusion** | **`EXPERIMENTAL`** | SIFT + FLANN + Umeyama RANSAC + Voxel Downsampling | Algorithmic (`services/multiview_fusion_service.py`) | Mathematical registration verified in tests; clean fallback when consensus fails. | **UNVALIDATED** against authentic multi-view room scans. When consensus fails, gracefully falls back to View 0 reference geometry. |
| **glTF 2.0 (GLB) Export** | **`PRODUCTION_READY`** | Custom Binary glTF 2.0 Serializer | `utils/export_3d.py` | 125.3 ms latency; verified via `validate_glb_bytes` (3 valid meshes in test room). | Point clouds exported in POSITION buffer with mode 0 (POINTS); boxes exported as index buffers. |
| **ASCII PLY Export** | **`PRODUCTION_READY`** | Standard ASCII PLY Serializer | `utils/export_3d.py` | Verified via `validate_ply_bytes` (40,000 vertices formatted with RGB). | Max points capped at 40,000 by default for viewer responsiveness. |
| **Vision-Language Q&A** | **`UNAVAILABLE`** | Qwen2-VL / MiniCPM-V | None (Offline) | Graceful degradation verified; returns `is_implemented=False` and status `unsupported`. | Offline to preserve 8 GB VRAM budget. Must not be advertised as active in frontend UI without external API. |
| **Generative Room Redesign** | **`UNAVAILABLE`** | SDXL / ControlNet Interior | None (Offline) | Graceful degradation verified; returns `is_implemented=False` and status `unsupported`. | Offline to preserve 8 GB VRAM budget. Must be disabled or clearly marked as coming soon in UI. |
| **Room Classification** | **`UNAVAILABLE`** | None | None | Handled via Detection & SceneGraph taxonomy. | Standalone classifier not deployed; indoor scene taxonomy inferred from detected furniture. |

---

## 2. Frontend Developer Implementation Guidance

### 2.1 Display Guidelines for `PRODUCTION_READY` Features
- **Object Detection & 3D Viewer:** Primary features. Expose directly in the main workspace. Display confidence scores, bounding boxes, 3D point clouds, and downloadable GLB/PLY files.
- **Metric Dimensions:** Present bounding box dimensions in meters ($W \times H \times D$) with a note indicating estimated physical bounds derived from monocular metric depth.

### 2.2 Display Guidelines for `EXPERIMENTAL` Features
- **Multi-View Upload:** The UI should allow uploading multiple room angles (2–10 images), but MUST display an **`[EXPERIMENTAL]`** badge.
- If feature consensus fails (common if images depict different rooms or lack sufficient overlap), the UI must show the backend warning: *"Consensus unachieved across viewpoints; displaying high-fidelity single-view reference geometry."*

### 2.3 Display Guidelines for `UNAVAILABLE` Features
- **VLM & AI Redesign:** The frontend should NOT present active interactive chat or redesign buttons as functional features. If included, they should be rendered as disabled or labeled *"Requires cloud endpoint or 16+ GB VRAM"*, and clicking them should display the backend's degradation status card without error crashes.

