# RoomAI — Phase 5.7: Real-Mode 3D Viewer Integration Audit

**Date:** 2026-10-31  
**Author:** Senior Computer Vision Engineer & QA Reviewer  

---

## 1. Executive Summary

**Status: NO-GO for Real-Mode 3D Viewer**

The Phase 5.6 3D viewer implementation is **mock-only**. Real-mode inference cannot generate valid point clouds because the underlying `CustomScene3DModel` is not implemented. The viewer infrastructure is complete, but the engine is missing.

---

## 2. Real Inference Pipeline Trace

### Expected Flow (Per Phase 5.6 Requirements)

```
Uploaded Image
      ↓
[1] A3 Detection (models/detection_model.py)
      ↓
[2] SegFormer Semantic Seg (models/segmentation_model.py)  
      ↓
[3] FastSAM Instance Seg (models/segmentation_model.py)
      ↓
[4] Depth Anything Metric (models/scene3d_model.py - NOT IMPLEMENTED)
      ↓
[5] Point Cloud Generation (models/scene3d_model.py)
      ↓
[6] Scene Fusion (services/scene_fusion_service.py)
      ↓
[7] Plotly Viewer (ui/three_d_viewer.py)
```

### Execution Trace

| Stage | File | Function | Status | Notes |
|-------|------|----------|--------|-------|
| Detection | detection_model.py | `CustomObjectDetector.predict()` | ✅ IMPLEMENTED | Real model available |
| Semantic Seg | segmentation_model.py | `CustomRoomSegmenter.predict()` | ✅ IMPLEMENTED | SegFormer-B1 loaded |
| Instance Seg | segmentation_model.py | Via detection fallback | ⚠️ PARTIAL | Detection fallback used |
| Depth Estim | scene3d_model.py | `CustomScene3DModel.reconstruct()` | ❌ NOT IMPLEMENTED | Raises `ModelNotImplementedError` |
| Point Cloud | scene3d_model.py | `_unproject_depth_to_pointcloud()` | ❌ DEAD CODE | Never executed |
| Scene Fusion | scene_fusion_service.py | `SceneFusionService.fuse()` | ✅ INTEGRATION | Passes but depth returns None |
| 3D Viewer | three_d_viewer.py | `render_3d_viewer()` | ✅ INTEGRATION | Works with mock data only |

---

## 3. Critical Finding: CustomScene3DModel NOT IMPLEMENTED

### File: `models/scene3d_model.py`

```python
class CustomScene3DModel(BaseScene3D):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None

    def load_model(self) -> None:
        raise ModelNotImplementedError("CustomScene3DModel", __file__)

    def reconstruct(self, image: Image.Image):
        if not self.is_ready():
            raise ModelNotImplementedError("CustomScene3DModel", __file__)
        raise ModelNotImplementedError("CustomScene3DModel", __file__)
```

**Result:** Any attempt at real inference fails immediately at `load_model()` or `reconstruct()`.

---

## 4. Mock Model Status

### File: `models/scene3d_model.py`

```python
class MockScene3DModel(BaseScene3D):
    def reconstruct(self, image: Image.Image):
        depth_map = np.tile(y[:, np.newaxis], (1, width))
        point_cloud = self._generate_mock_point_cloud(image)
        response = Scene3DResponse(
            point_cloud=point_cloud,
            is_simulated=True,
        )
        return depth_map, response
```

**Status:** ✅ Functional for testing
**Limitations:** Synthetic, not real model output

---

## 5. Point Cloud Data Structure Verification

### Schema: `schemas/responses.py`

```python
class Scene3DResponse(BaseModel):
    point_cloud: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Point cloud data: xyz, rgb, depth lists"
    )
```

The schema accepts the correct structure:
- `xyz`: List of [x, y, z] coordinates
- `rgb`: List of [r, g, b, b] colors (normalized)
- `depth`: List of depth values

### Mock Generation: `MockScene3DModel._generate_mock_point_cloud()`

```python
def _generate_mock_point_cloud(self, image: Image.Image) -> Dict[str, List]:
    # Generates synthetic depth gradient
    # Unprojects using camera intrinsics
    # Returns dict with xyz, rgb, depth as lists
    return {
        "xyz": xyz.tolist(),
        "rgb": rgb.tolist(), 
        "depth": depth_map.reshape(-1).tolist(),
    }
```

**Status:** ✅ Correct structure for mock mode

---

## 6. Viewer Integration Verification

### File: `ui/analysis_view.py`

The 3D tab correctly:
1. ✅ Creates Scene3DService with `mock_mode` flag
2. ✅ Calls `run()` to get results
3. ✅ Passes point cloud to `render_3d_viewer()`
4. ✅ Displays 3D viewer when data available

### File: `ui/three_d_viewer.py`

Viewer successfully:
1. ✅ Renders XYZ coordinates as 3D scatter
2. ✅ Supports RGB coloring
3. ✅ Provides color mode switching
4. ✅ Shows coordinate axes
5. ✅ Handles empty point clouds

**Status:** ✅ Integration works (with mock data)

---

## 7. Blockers for Real-Mode

### Blocker 1: Depth Model Not Loaded

```
Error: 'CustomScene3DModel' is not yet implemented or weights are missing.
```

The student task for Phase 5 (implementing Depth Anything V2) was not completed in the actual codebase.

### Blocker 2: No Point Cloud Generation

Even if depth worked, `CustomScene3DModel` lacks:
- `AutoImageProcessor` loading
- `AutoModelForDepthEstimation` inference
- `_unproject_depth_to_pointcloud()` implementation

### Blocker 3: Real Checkpoint Missing

The checkpoint `depth_anything_v2_metric_small` is not available locally; requires HuggingFace download which may not work in this environment.

---

## 8. Conclusion

**Real-mode 3D viewer cannot be validated because the core depth estimation model is not implemented.**

The viewer infrastructure is complete and functional in mock mode, but real inference requires:
1. Implementing `CustomScene3DModel.load_model()`
2. Implementing `CustomScene3DModel.reconstruct()`
3. Adding point cloud generation to the reconstruction

This is a **prerequisite for Phase 6**, not a Phase 5.7 outcome.