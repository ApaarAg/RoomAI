# RoomAI — Phase 5A: 3D Capability Audit Report

**Date:** 2026-10-31  
**Author:** Principal 3D Computer Vision Engineer & Spatial AI Researcher

---

## 1. Implemented Features

### 1.1 Point-Cloud Generation ✅
- **Status:** IMPLEMENTED
- **Method:** `CustomScene3DModel._unproject_depth_to_pointcloud()` in `models/scene3d_model.py`
- **Algorithm:** Inverse depth → 3D coordinates using pinhole camera model
- **Input:** Raw depth map (metric or relative), RGB image
- **Output:** Dict with `xyz` (N×3), `rgb` (N×3), `depth` (N,)

### 1.2 Camera Intrinsics Handling ✅
- **Status:** IMPLEMENTED
- **Location:** `CAMERA_PRESETS` dict in `models/scene3d_model.py` and `config/model_config.yaml`
- **Parameters:** `fx`, `fy`, `cx`, `cy` (focal lengths and principal point)
- **Formats:** Supports both pixels and normalized ratios

### 1.3 RGB-to-Point-Cloud Color Assignment ✅
- **Status:** IMPLEMENTED
- **Method:** Direct pixel lookup using valid depth coordinates
- **Implementation:** `rgb = img_arr[vv, uu]` in `_unproject_depth_to_pointcloud()`

### 1.4 Depth-to-Coordinate Conversion ✅
- **Status:** IMPLEMENTED
- **Convention:** Right-handed camera space (X right, Y down, Z forward)
- **Formula:** `x = (u - cx) * z / fx`, `y = (v - cy) * z / fy`
- **Interpolation:** Bicubic from 518×518 to original resolution

### 1.5 Semantic Label Association ✅
- **Status:** PARTIALLY IMPLEMENTED
- **Current:** SegFormer-B1 produces semantic masks; masks not yet attached to 3D points
- **Gap:** Need to compute per-point semantic class IDs

### 1.6 Instance ID Association ✅
- **Status:** PARTIALLY IMPLEMENTED
- **Current:** A3 + FastSAM produce instance masks; masks not yet unprojected to 3D
- **Gap:** Need to compute per-point instance IDs

### 1.7 Scene Fusion
- **Status:** NOT YET IMPLEMENTED
- **Gap:** No unified scene graph data structure exists
- **Plan:** Phase 5B will create this

### 1.8 3D Visualization
- **Status:** NOT YET IMPLEMENTED
- **Gap:** Streamlit has no 3D viewer component
- **Plan:** Phase 5D will implement Plotly-based viewer

### 1.9 Object Spatial Statistics
- **Status:** NOT YET IMPLEMENTED
- **Gap:** Depth statistics per object not computed
- **Plan:** Phase 5B will add this

---

## 2. Summary: Implemented vs. Needed

| Feature | Status | Implementation Location | Phase Needed |
| :--- | :--- | :--- | :--- |
| Point-cloud generation | ✅ | `scene3d_model.py:_unproject_depth_to_pointcloud()` | Done (Phase 4E) |
| Camera intrinsics | ✅ | `CAMERA_PRESETS`, `model_config.yaml` | Done (Phase 4E) |
| RGB color mapping | ✅ | `_unproject_depth_to_pointcloud()` | Done (Phase 4E) |
| Depth-to-3D conversion | ✅ | `_unproject_depth_to_pointcloud()` | Done (Phase 4E) |
| Semantic-to-3D association | ⚠️ Partial | None | Phase 5B |
| Instance-to-3D association | ⚠️ Partial | None | Phase 5B |
| Scene graph/fusion | ❌ | None | Phase 5B |
| Interactive 3D viewer | ❌ | None | Phase 5D |
| Object spatial stats | ❌ | None | Phase 5B |

---

## 3. Recommendations for Phase 5

1. **Create unified scene representation schema** (`schemas/scene.py` or extend existing)
2. **Implement point-wise semantic/instance lookup** in unprojection
3. **Build scene fusion service** to combine all modalities
4. **Add 3D viewer component** to Streamlit UI