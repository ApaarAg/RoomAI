# Project: RoomAI Professional Polish & Real Inference Transformation

## Architecture
- **Presentation Layer**: Streamlit (`app.py`, `ui/components.py`, `ui/upload_view.py`, `ui/analysis_view.py`, `ui/three_d_viewer.py`)
  - Elevated 3 dedicated workspaces: Object Detection, Vision Q&A, and 3D Scene Reconstruction.
  - Internalized models: CustomRoomSegmenter (`checkpoints/yolo11n-seg.pt`) used internally by `SceneFusionService`.
  - Application header: RoomAI branding, official tagline ("Understand Your Space. Visualize It in 3D."), product summary, and live status badges (Real Mode active, GPU device, VRAM allocation, model checkpoints).
  - Invalidation engine: SHA-256 byte hashing of image uploads + full purge on clear/remove + strict preservation of all 11 `INFERENCE_RESULT_KEYS`.
- **Inference & Perception Layer**: PyTorch / Ultralytics / HuggingFace Transformers
  - Production Detectors: Model A3 Indoor (`indoor_detector_a3_yolo11s_800px_best.pt`, default), Model B2 Exterior (`exterior_detector_b2_yolo11s_best.pt`), Model C2 Parking (`parking_detector_c2_yolo11s_pilot_best.pt`).
  - Depth Estimation: Depth Anything V2 Metric Small (`checkpoints/depth_anything_v2_metric_indoor_small`) for true 3D spatial unprojection, with explicit metric vs relative labeling.
  - Vision Q&A: Honest offline / GPU VRAM preservation reporting with explicit opt-in toggle to mock mode.
  - Scene Fusion: Surface extraction, RANSAC horizontal floor plane estimation, room dimensions, oriented 3D bounding boxes.
- **Export & 3D Visualization Layer**: Plotly 3D + `utils/export_3d.py`
  - Interactive 3D point cloud & bounding box viewer with camera reset and aspect ratio preservation.
  - GLB and PLY export generators with honest labeling: point clouds and bounding boxes, not textured polygonal meshes.
  - Lazy / cached export serialization to eliminate UI lag.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Workspace Consolidation | Consolidate navigation to exactly 3 workspaces (Object Detection, Vision Q&A, 3D Scene Reconstruction) | M1 | ORIGINAL_REQUEST R1 (DONE) |
| F2 | Professional Header & Branding | RoomAI branding, official tagline, product summary, Real/Mock badge, GPU/VRAM status | M1 | ORIGINAL_REQUEST R1 (DONE) |
| F3 | Strict Session-State Invalidation | SHA-256 image byte hash tracking, purge on file removal, preserve 11 AST keys | M1 | ORIGINAL_REQUEST R1 (DONE) |
| F4 | Object Detection Polish | Real confidence, detector selection (A3 default, B2, C2), checkpoint metadata, latency breakdown | M2 | ORIGINAL_REQUEST R2 |
| F5 | Vision Q&A Grounding | Image preview, honest VLM unavailable card (VRAM preservation), explicit mock opt-in toggle | M2 | ORIGINAL_REQUEST R2 |
| F6 | 3D Reconstruction Polish | Single metric depth pass, explicit relative vs metric labels, room dimensions, estimation assumptions | M2 | ORIGINAL_REQUEST R2 |
| F7 | 3D Viewer & Honest Export | Camera reset, aspect ratio preservation, PLY/GLB export descriptions (point clouds/boxes) | M2 | ORIGINAL_REQUEST R2 |
| F8 | Real-Mode Enforcement & Caching | Real mode default, standardize constructor signatures, cache models, eliminate export re-serialization | M2 | ORIGINAL_REQUEST R3 |
| F9 | ADE20K 5-Scene Visual Validation | Real model validation across 5 ADE20K room types (Living, Bedroom, Kitchen, Dining, Office) | M3 | ORIGINAL_REQUEST R4 |
| F10 | Screenshots & Visual Assets | Capture real application screenshots in docs/assets/product/ (main, detect, vlm, depth, 3d, export) | M3 | ORIGINAL_REQUEST R4 |
| F11 | LinkedIn Product Demo Plan | 2-minute step-by-step demonstration sequence and narration in docs/PRODUCT_DEMO_PLAN.md | M3 | ORIGINAL_REQUEST R4 |
| F12 | Comprehensive Phase Reports | 6 required reports under reports/ with executive summaries and empirical results | M4 | ORIGINAL_REQUEST R5 |
| F13 | Pytest Regression Verification | Full suite passes (226 tests) with 0 regressions, clean exit code | M5 | ORIGINAL_REQUEST R5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | UI Redesign, Identity & State Invalidation | `app.py`, `ui/components.py`, `ui/upload_view.py` (Header, 3 workspaces, SHA-256 state invalidation) | none | DONE |
| M2 | Workspace Feature Polish & Backend Hardening | `ui/analysis_view.py`, `ui/three_d_viewer.py`, `services/` (A3/B2/C2 metrics, VLM card, metric depth, 3D viewer & export polish, cache optimization) | M1 | DONE |
| M3 | Visual Validation, Screenshots & Demo Plan | 5 ADE20K rooms, `docs/assets/product/` screenshots, `docs/PRODUCT_DEMO_PLAN.md` | M2 | IN_PROGRESS |
| M4 | Documentation Deliverables & Final Audit | 6 Phase 7.2A reports in `reports/` with GO/NO-GO gate decision | M3 | PLANNED |
| M5 | Final E2E Regression Gate & Adversarial Coverage | 100% pass across all 242 pytest tests and E2E validation | M4 | PLANNED |

## Interface Contracts
### `ui/upload_view.py` ↔ `app.py`
- `render_upload_view()` returns `Optional[ImageValidationResult]`.
- Active image ID tracked via `st.session_state["active_image_id"] = hashlib.sha256(img_bytes).hexdigest()`.
- When image is cleared (`uploaded_file is None`) or changed, strictly pop all 11 keys in `INFERENCE_RESULT_KEYS`:
  `["detection_result", "detection_result_indoor", "detection_result_exterior", "detection_result_parking", "segmentation_result", "scene3d_result", "scene_fusion", "last_scene3d_error", "classification_result", "vlm_result", "design_result"]`.
- Retain exact AST clear button loop pattern with `st.rerun()`.

### `ui/analysis_view.py` ↔ `services/scene_fusion_service.py`
- Instantiates `SceneFusionService(mock_mode=mock_mode, config=model_config)` directly.
- Uses `scene.depth_map` (metric depth in meters) for visualization and unprojection.
- Does not run redundant relative depth pass before fusion.
- Accurately renders metric depth heatmap (meters) and relative depth comparison.

### `ui/three_d_viewer.py` ↔ `utils/export_3d.py`
- Download buttons generate GLB and PLY binary/ASCII buffers only when clicked or via cached generation.
- Clear tooltips explain contents: "Download 3D Scene (.GLB) — Colored 3D point cloud and bounding boxes (not a textured polygon mesh)".

## Code Layout
- `app.py`: Main application entrypoint, workspace routing, device/mock mode configuration.
- `ui/components.py`: Header, branding, status indicators, CSS typography and palette.
- `ui/upload_view.py`: Image upload, metadata preview, SHA-256 session state invalidation.
- `ui/analysis_view.py`: 3 consolidated workspaces (Object Detection, Vision Q&A, 3D Scene Reconstruction).
- `ui/three_d_viewer.py`: Interactive Plotly 3D visualizer, camera controls, GLB/PLY exports.
- `services/`: Detection, Scene Fusion, Depth, VLM, and Segmentation services.
- `models/`: YOLO wrappers, Depth Anything V2 wrappers, VLM stubs.
- `utils/export_3d.py`: GLB and PLY serialization utilities.
- `tests/`: 22 test suites comprising 226 tests (including 58 E2E tests).
- `archive (2)/ADEChallengeData2016/images/validation/`: 2,000 ADE20K validation images.
- `reports/`: Phase 7.2A audit and validation documentation deliverables.
- `docs/assets/product/`: Application screenshots.
- `docs/PRODUCT_DEMO_PLAN.md`: LinkedIn demonstration script and narration.
