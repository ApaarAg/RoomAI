# ROOMAI — FINAL FRONTEND INTEGRATION CONTRACT & DEVELOPER GUIDE

**Version:** 1.0 (Production Handoff)  
**Date:** 2026-11-06  
**Target Audience:** Frontend Engineers, Streamlit Developers, Web / Full-Stack Engineers  

---

## 1. Quick Start & Architecture Overview

The RoomAI backend is encapsulated inside a single production facade: `RoomAIBackend`. Frontend developers do not need to manage model loading, PyTorch CUDA tensors, or 3D coordinate geometry.

### 1.1 The Golden Import
```python
from services.roomai_backend import BACKEND

# Or instantiate custom:
# from services.roomai_backend import RoomAIBackend
# backend = RoomAIBackend()
```

---

## 2. Standard Response Envelope Contract

Every backend method returns a standardized, typed dictionary envelope.

```python
{
    "status": "success" | "error" | "unsupported" | "warning",
    "capability_status": "PRODUCTION_READY" | "EXPERIMENTAL" | "UNAVAILABLE",
    "data": Any,                        # Payload dict containing result objects
    "error": Optional[Dict[str, str]],  # {"code": "...", "message": "...", "remediation": "..."}
    "warnings": List[str],              # Informational warnings for UI display
    "provenance": Dict[str, Any],       # Model name, execution metadata, view count
    "timing": Dict[str, float],         # {"inference_time_ms": float, "total_time_ms": float}
    
    # Backwards-compatibility fields:
    "success": bool,                    # True if status == "success"
    # Plus method-specific convenience attributes (e.g., res["scene_graph"], res["annotated_image"])
}
```

---

## 3. Complete API Reference

### 3.1 Capabilities & Hardware Telemetry

#### `BACKEND.get_capabilities() -> Dict[str, Any]`
Returns the operational capability matrix.
```python
capabilities = BACKEND.get_capabilities()
# Check if feature is production ready before rendering UI controls
if capabilities["indoor_detection"]["status"] == "PRODUCTION_READY":
    render_indoor_detection_panel()
```

#### `BACKEND.get_system_health() -> Dict[str, Any]`
Returns real-time GPU VRAM utilization and checkpoint status.
```python
health = BACKEND.get_system_health()
print(f"GPU: {health['device_name']} | Headroom: {health['vram_headroom_mb']} MB")
```

---

### 3.2 Object Detection

#### `BACKEND.detect_objects(image, task_group=..., confidence_threshold=0.25) -> Dict[str, Any]`
- **`image`**: PIL Image in RGB format.
- **`task_group`**: One of:
  - `"Group A: Indoor Furniture & Household Objects"` (Model A3 800px)
  - `"Group B: Exterior and Building Components"` (Model B2)
  - `"Group C: Parking & Outdoor Property"` (Model C2)
- **`confidence_threshold`**: Float in `[0.0, 1.0]`.

**Frontend Example:**
```python
result = BACKEND.detect_objects(pil_image, task_group="Group A: Indoor Furniture & Household Objects")

if result["status"] == "success":
    st.image(result["annotated_image"], caption="Detections")
    for det in result["data"]["detections"]:
        st.write(f"{det.label}: {det.confidence:.2f}")
else:
    st.error(f"Detection error: {result['error']['message']}")
```

---

### 3.3 Semantic & Instance Segmentation

#### `BACKEND.segment_scene(image) -> Dict[str, Any]`
- **`image`**: PIL Image in RGB format.
- Returns segmentation masks and blended RGB overlay image.

**Frontend Example:**
```python
result = BACKEND.segment_scene(pil_image)
if result["status"] == "success":
    st.image(result["overlay_image"], caption="Segmentation Overlay")
```

---

### 3.4 Monocular Depth Estimation

#### `BACKEND.estimate_depth(image, mode="metric") -> Dict[str, Any]`
- **`image`**: PIL Image in RGB format.
- **`mode`**: `"metric"` (calibrated physical depth in meters) or `"relative"` (normalized [0, 1]).

**Frontend Example:**
```python
result = BACKEND.estimate_depth(pil_image, mode="metric")
if result["status"] == "success":
    st.image(result["heatmap_image"], caption="Metric Depth Map")
    depth_array = result["data"]["depth_map"] # numpy float32 array
```

---

### 3.5 3D Room Scene Reconstruction (Single-View & Multi-View)

#### `BACKEND.reconstruct_room_3d(images, refine_depth=True) -> Dict[str, Any]`
- **`images`**: A single PIL Image or a list of 2–10 PIL Images.
  - If single image: executes **PRODUCTION_READY** single-view reconstruction.
  - If multiple images: executes **EXPERIMENTAL** multi-view registration and fusion.
- **`refine_depth`**: Apply edge-preserving bilateral depth refinement (default `True`).

**Frontend Example:**
```python
result = BACKEND.reconstruct_room_3d(uploaded_image)

if result["status"] == "success":
    scene = result["data"]["scene_graph"]
    st.success(f"Reconstructed {scene.n_valid_points:,} 3D points and {len(scene.objects)} objects.")
    
    # Check if any warnings (e.g. experimental multi-view consensus fallback)
    for warning in result["warnings"]:
        st.warning(warning)
        
    # Export downloadable assets
    export_result = BACKEND.export_3d_assets(scene)
    st.download_button("Download GLB", export_result["glb"], file_name="room.glb", mime="model/gltf-binary")
    st.download_button("Download PLY", export_result["ply"], file_name="room.ply", mime="text/plain")
```

---

### 3.6 3D Asset Export

#### `BACKEND.export_3d_assets(scene_graph, max_points=40000) -> Dict[str, Any]`
- **`scene_graph`**: Valid `SceneGraph` instance from `reconstruct_room_3d`.
- **`max_points`**: Integer point cap for WebGL viewer performance (default `40000`).
- **Returns:**
  - `result["glb"]`: Binary glTF 2.0 file bytes.
  - `result["ply"]`: ASCII PLY file bytes.
  - `result["glb_valid"]`: `True` if validated.
  - `result["ply_valid"]`: `True` if validated.

---

### 3.7 Offline Subsystems (Graceful Degradation)

#### `BACKEND.ask_vlm(image, question) -> Dict[str, Any]`
#### `BACKEND.generate_redesign(image, target_style=...) -> Dict[str, Any]`
Because weights are offline to preserve the 8 GB VRAM budget, these endpoints return `status="unsupported"` with structured error messages:
```python
res = BACKEND.ask_vlm(image, "What is the floor material?")
if res["status"] == "unsupported":
    st.info(f"Feature Offline: {res['error']['message']}")
```

---

## 4. Error Code Registry

| Error Code | Meaning | Remediation for User / UI |
| :--- | :--- | :--- |
| `INVALID_INPUT_IMAGE` | Input was `None`, not a PIL Image, or exceeds 8192px. | Prompt user to upload a valid JPEG/PNG under 8192px. |
| `NO_IMAGES_PROVIDED` | An empty list was passed to 3D reconstruction. | Ensure at least 1 image is selected. |
| `DETECTION_FAILED` | Detector could not run on the image. | Check image channels and resolution. |
| `VLM_OFFLINE` | VLM model weights are offline. | Inform user that VLM requires external endpoint or 16+ GB VRAM. |
| `GENERATION_OFFLINE` | Diffusion redesign weights are offline. | Inform user that Generative Redesign is currently offline. |
| `RECONSTRUCTION_EXCEPTION`| Unhandled error during 3D reconstruction. | Check console logs and GPU VRAM headroom. |

---

## 5. WebGL / 3D Viewer Integration

### Rendering GLB in Three.js:
```javascript
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.parse(glbArrayBuffer, '', (gltf) => {
    scene.add(gltf.scene);
    // Point cloud is stored with POINTS primitive mode (mode 0)
    // Furniture bounding boxes are stored as wireframe box meshes
});
```

### Rendering in Plotly (Streamlit):
```python
import plotly.graph_objects as go

cloud = scene.point_cloud
xyz = np.array(cloud["xyz"])
rgb = np.array(cloud["rgb"])

fig = go.Figure(data=[
    go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(size=2, color=rgb)
    )
])
st.plotly_chart(fig, use_container_width=True)
```

