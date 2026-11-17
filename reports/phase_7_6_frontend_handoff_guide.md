# RoomAI Phase 7.6 — Frontend Handoff Guide

## Role & Purpose
**Audience:** Frontend Engineers, Streamlit Developers, UI/UX Integrators.  
**Author:** Principal ML Systems Architect & Senior Computer Vision Researcher.  
**System Target:** Windows 11, Python 3.10.20 (`tf-gpu` conda environment), NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM).

This guide provides frontend developers with a complete, stable, clean, and decoupled backend API.
**You do NOT need to load checkpoints, import PyTorch, manage CUDA tensors, or perform 3D coordinate arithmetic.** All capabilities are encapsulated in `RoomAIBackend` (`services/roomai_backend.py`).

---

## 1. Quick Start

```python
from PIL import Image
from services.roomai_backend import RoomAIBackend

# Initialize the backend (uses config/model_config.yaml automatically)
backend = RoomAIBackend()

# 1. Check system hardware health & GPU headroom
health = backend.get_system_health()
print(f"Device: {health['device_name']} | Headroom: {health['vram_headroom_mb']} MB")

# 2. Run object detection on an image
image = Image.open("path/to/room.jpg")
det_result = backend.detect_objects(
    image,
    task_group="Group A: Indoor Furniture & Household Objects",
    confidence_threshold=0.25,
)
if det_result["success"]:
    annotated_img = det_result["annotated_image"]  # PIL.Image ready for st.image
    objects = det_result["response"].detections

# 3. Run full 3D room reconstruction (supports 1 to 10 images)
recon_result = backend.reconstruct_room_3d(image, refine_depth=True)
if recon_result["success"]:
    scene = recon_result["scene_graph"]
    point_cloud = recon_result["point_cloud"]      # {"xyz": [...], "rgb": [...]}
    furniture = recon_result["objects"]            # List of SceneObject with 3D bounds
    layout = recon_result["room_layout"]          # Dimensions: width, length, height

# 4. Generate standalone 3D export downloads (GLB & PLY)
export_data = backend.export_3d_assets(recon_result["scene_graph"])
glb_bytes = export_data["glb"]  # glTF 2.0 binary bytes for st.download_button
ply_bytes = export_data["ply"]  # ASCII PLY bytes for st.download_button
```

---

## 2. API Reference: `RoomAIBackend`

### `get_system_health() -> Dict[str, Any]`
Returns current GPU status, VRAM consumption, and checkpoint presence on disk.

**Return Dictionary Keys:**
- `cuda_available` (*bool*): True if CUDA is active on the RTX 4060.
- `device_name` (*str*): GPU device string (e.g., `'NVIDIA GeForce RTX 4060 Laptop GPU'`).
- `total_vram_mb` (*float*): Total dedicated VRAM (e.g., `8187.5`).
- `allocated_vram_mb` (*float*): Currently active model tensor memory in MB.
- `reserved_vram_mb` (*float*): PyTorch cache memory in MB.
- `vram_headroom_mb` (*float*): Available capacity before hitting the 8 GB budget.
- `checkpoints_status` (*Dict[str, bool]*): Presence of production weights (`indoor_a3`, `exterior_b2`, `parking_c2`, `yolo11_seg`, `depth_anything_v2_metric`, `depth_anything_v2_relative`).
- `registered_models` (*List[str]*): IDs of all registered models in `ModelRegistry`.

---

### `detect_objects(image, task_group=..., confidence_threshold=0.25, use_clahe=False, mock_mode=False) -> Dict[str, Any]`
Runs object detection using production YOLO11 models.

**Arguments:**
- `image` (*PIL.Image.Image*): Input image in RGB mode.
- `task_group` (*str*):
  - `"Group A: Indoor Furniture & Household Objects"` (Model A3 YOLO11s 800px — Primary).
  - `"Group B: Exterior and Building Components"` (Model B2 YOLO11s 640px).
  - `"Group C: Parking & Outdoor Property"` (Model C2 YOLO11s 640px).
- `confidence_threshold` (*float*): Minimum detection confidence (default: `0.25`).
- `use_clahe` (*bool*): Contrast enhancement for outdoor/exterior imagery (default: `False`).
- `mock_mode` (*bool*): Enable only when simulated outputs are explicitly requested (default: `False`).

**Return Structure:**
```python
{
    "success": bool,
    "response": DetectionResponse,  # Pydantic schema with .detections, .total_objects, .device, .inference_time_ms
    "annotated_image": PIL.Image.Image,  # Rendered image with bounding boxes & labels
    "saved_image_path": Optional[str],
    "timing_metrics": Dict[str, Any],
    "error_message": Optional[str],
}
```

---

### `reconstruct_room_3d(images, refine_depth=True, mock_mode=False) -> Dict[str, Any]`
End-to-end 3D scene reconstruction. Seamlessly switches between single-view and multi-view pipelines.

**Arguments:**
- `images` (*PIL.Image.Image* or *List[PIL.Image.Image]*): Single photo or list of 2–10 photos of the room.
- `refine_depth` (*bool*): Enables guided bilateral smoothing and flying-pixel edge filtering (default: `True`).
- `mock_mode` (*bool*): Synthetic reconstruction for offline UI testing (default: `False`).

**Return Structure:**
```python
{
    "success": bool,
    "scene_graph": SceneGraph,             # Complete unified 3D representation
    "point_cloud": Dict[str, List],        # {"xyz": [[x,y,z], ...], "rgb": [[r,g,b], ...]}
    "objects": List[SceneObject],          # Furniture instances with 3D centroid & extents
    "surfaces": List[RoomSurface3D],       # Semantic planes (walls, floors, ceilings)
    "room_layout": Dict[str, Any],         # {"floor_plane_detected": bool, "room_width_m": float, ...}
    "relationships": List[SpatialRelationship],  # Pairwise spatial relations ('near', 'on_floor', etc.)
    "telemetry": List[Dict[str, Any]],     # Multi-view registration stats (RMSE, inliers, rotation, baseline)
    "per_view_provenance": List[Dict],     # Audit trail for each viewpoint
    "validation_warnings": List[str],      # Any blur, resolution, or duplicate warnings
    "error_message": Optional[str],
}
```

---

### `export_3d_assets(scene_graph, max_points=40000) -> Dict[str, bytes]`
Generates standard 3D file formats ready for user downloads.

**Arguments:**
- `scene_graph` (*SceneGraph*): The scene graph object from `reconstruct_room_3d`.
- `max_points` (*int*): Point cloud subsampling limit for fast viewer loading (default: `40000`).

**Return Structure:**
- `{"glb": bytes, "ply": bytes}`
  - `"glb"`: Binary glTF 2.0 file containing the metric point cloud and 3D furniture bounding boxes.
  - `"ply"`: Standard ASCII Stanford PLY file containing colored point cloud vertices.

---

### `ask_vlm(image, question, mock_mode=False) -> Dict[str, Any]`
Queries the Vision-Language Model.

**Behavior:**
- When real weights are kept offline (to preserve the 8 GB VRAM budget for CV models), this method returns `{"is_implemented": False, "success": False, "error_message": ...}`.
- **Frontend Guidance:** Display the informative status card explaining VRAM preservation. Do NOT fall back to synthetic mock answers unless `mock_mode=True` is explicitly toggled by the user.

---

## 3. Data Contracts & Pydantic Schemas

All backend responses use validated Pydantic V2 models located in `schemas/responses.py` and `schemas/scene.py`.

### `SceneObject`
- `object_id`: Unique identifier (e.g. `'chair#stool_001'`).
- `semantic_class`: Canonical indoor label from the 12-class taxonomy.
- `detection_confidence`: Confidence score in `[0.0, 1.0]`.
- `bounding_box`: Normalized 2D box `(x_min, y_min, x_max, y_max)`.
- `geometry`: `Object3DGeometry` containing:
  - `centroid`: `(x, y, z)` in meters. Camera frame: $+X$ right, $+Y$ down, $+Z$ forward.
  - `extent`: `(width, height, depth)` in meters.
  - `is_dimension_reliable`: `bool` indicating whether the dimensions are physically plausible for domestic furniture.
  - `dimension_warning`: Human-readable explanation if dimensions exceed domestic envelopes.
  - `raw_floor_contact_delta_m`: Raw physical offset between object bottom and estimated floor before snapping.
  - `post_floor_contact_delta_m`: Remaining offset after grounding ($< 0.005\text{ m}$).

### `SceneGraph`
- `width`, `height`: Image resolution.
- `depth_mode`: `'metric'` (meters) or `'relative'`.
- `depth_min_m`, `depth_max_m`: Valid physical depth bounds in meters.
- `point_cloud`: Dict with `'xyz'` and `'rgb'` coordinates.
- `depth_map`: 2D numpy array of metric depths in meters.
- `heatmap_image`: Rendered PIL Image of depth heatmap.
- `multiview_telemetry`: Pairwise multi-view registration metrics.
- `multiview_per_view_provenance`: Per-view contribution records.

---

## 4. Architectural Rules for UI Developers

1. **Real Inference Default:** Never set `mock_mode=True` in production code or on startup. Mock mode is strictly an opt-in toggle for users without GPU hardware or for rapid UI mockups.
2. **Honesty Disclaimers:** When presenting 3D room dimensions, always include the standard architectural disclosure: camera perspective backprojection relies on nominal intrinsics ($f_x = f_y = 1000\text{ px}$, $c_x = c_y = 0.5$).
3. **No Silent Mock Fallback:** When a model fails or weights are unavailable, surface the explicit error or status card. Never display mock synthetic predictions as real model outputs.
4. **VRAM Preservation:** The backend enforces lazy loading and automatic caching. Avoid importing heavy deep learning libraries (`torch`, `ultralytics`, `transformers`) directly in frontend view scripts.

