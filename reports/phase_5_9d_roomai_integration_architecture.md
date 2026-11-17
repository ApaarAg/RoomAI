# RoomAI — Phase 5.9D: Generative 3D Furniture Integration Architecture

**Date:** 2026-11-05  
**Author:** Principal Software Architect & 3D Spatial Systems Engineer  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Phase:** Phase 6A Implementation Blueprint

---

## 1. Executive Summary

This architecture specification defines the modular, end-to-end integration pipeline for incorporating single-view generative 3D furniture reconstruction into RoomAI. 

The architecture strictly adheres to three design laws:
1. **Zero Degradation of Production Perception:** Existing detectors (A3/B2/C2) and segmentation models remain unaltered.
2. **Deterministic Memory Staging:** Perception and 3D generation run in decoupled, sequential stages to honor the RTX 4060 8 GB VRAM budget.
3. **Rigorous Epistemological Separation:** Observed visual evidence is never conflated with hallucinated 3D geometry.

---

## 2. End-to-End Pipeline Data Flow

```
                      Uploaded Room Image (RGB)
                                 │
                                 ▼
                   [STAGE 1: 2D PERCEPTION]
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
       Detector A3 (YOLO11s)              Depth Anything V2 Small
   (12 Indoor Furniture Classes)         (Monocular Relative Depth)
               │                                   │
               ▼                                   ▼
        Bounding Boxes                     Dense Depth Tensor
               │                                   │
               ▼                                   │
       FastSAM-s Prompting                         │
     (Per-box binary masks)                        │
               │                                   │
               ▼                                   │
   [STAGE 2: EXTRACTION & PADDING]                 │
  • FastSAM alpha masking                          │
  • 3x3 Gaussian edge feathering                   │
  • Square canvas + 15% margin                     │
  • 512x512 RGBA export                            │
               │                                   │
               ▼                                   │
   [STAGE 3: VRAM CACHE EVICTION]                  │
  • Free detector/segmenter/depth                  │
  • torch.cuda.empty_cache()                       │
               │                                   │
               ▼                                   │
  [STAGE 4: 3D GENERATIVE RECONSTRUCTION]          │
  • TripoSR (fp16 mode)                            │
  • Feed-forward triplane NeRF                     │
  • Marching Cubes mesh extraction                 │
  • Mesh cleaning & validation                     │
  • Export .obj / .glb                             │
               │                                   │
               ▼                                   │
   [STAGE 5: 3D SCENE FUSION & GRAPH]              │
  • Combine unprojected room point cloud ◄─────────┘
  • 9-DoF bounding box scaling & pose
  • Associate generative meshes to SceneObjects
               │
               ▼
   [STAGE 6: INTERACTIVE 3D VIEWER]
  • Streamlit WebGL / Plotly rendering
  • Toggle point clouds, surfaces, and furniture meshes
  • Color by semantic, instance, or generative source
```

---

## 3. Epistemological Separation Schema

To maintain scientific integrity and prevent misleading users, RoomAI's data models must strictly differentiate between four categories of spatial information:

```
                                    Scene Data
                                         │
        ┌────────────────────┬───────────┴───────────┬────────────────────┐
        ▼                    ▼                       ▼                    ▼
   [Observed]         [Depth-Derived]        [Generative Prior]       [Unknown]
  • 2D RGB Pixels    • Unprojected Points   • Extrapolated Mesh      • Occluded Floor
  • Bounding Boxes   • Surface Planes       • Hallucinated Legs      • Interior Cavities
  • Class Labels     • Relative Distance    • Inferred Textures      • Metric Dimensions
```

### Schema Extension for `schemas/scene.py`:
In Phase 6A, `SceneObject` will be extended with explicit provenance metadata:

```python
class MeshProvenance(str, Enum):
    OBSERVED_POINTCLOUD = "observed_depth_pointcloud"
    GENERATIVE_PRIOR = "generative_synthesized_mesh"
    CAD_CATALOG_MATCH = "cad_catalog_match"
    PROXY_BOUNDING_BOX = "proxy_bounding_box"

class FurnitureMesh3D(BaseModel):
    mesh_path: str
    format: str = "glb"  # "glb" or "obj"
    provenance: MeshProvenance = MeshProvenance.GENERATIVE_PRIOR
    is_metric_verified: bool = False
    vertex_count: int
    face_count: int
    centroid_camera_frame: List[float]
    scale_xyz: List[float]
    disclaimer: str = (
        "Generatively reconstructed shape based on single-view deep learning priors. "
        "Hidden geometry, back faces, and dimensions are synthesized estimates."
    )
```

---

## 4. Software Service Architecture

Phase 6A will introduce two new isolated modules:

### 1. Model Class: `models/furniture_3d_model.py`
- Inherits from `models/base.py:BaseModel`.
- Encapsulates TripoSR weights loading, GPU execution, Marching Cubes grid extraction, and post-extraction Laplacian smoothing.
- Implements `load_model()`, `predict()`, and `unload_model()` to guarantee clean GPU memory release.

### 2. Service Class: `services/furniture_3d_service.py`
- Manages the lifecycle between perception outputs and 3D mesh files.
- Handles image pre-processing (FastSAM alpha feathering and 15% margin square padding).
- Writes exported meshes to session-isolated artifact directories: `artifacts/sessions/{session_id}/meshes/{object_id}.glb`.
- Integrates with `services/scene_fusion_service.py` to embed mesh references into the `SceneGraph`.

---

## 5. UI & 3D Viewer Integration

The existing Plotly/WebGL viewer in [`ui/three_d_viewer.py`](file:///F:/RoomAI/ui/three_d_viewer.py) currently supports scatter3d point clouds. In Phase 6A, it will be upgraded to support dual rendering:
1. **Background Room Shell:** Unprojected point cloud (walls, floor, ceiling) rendered as semi-transparent surface points.
2. **Foreground Furniture:** Rendered as solid 3D polygonal meshes (`Mesh3d` in Plotly or native Three.js via an HTML sidecar widget).
3. **Inspector Controls:** A dedicated UI sidebar toggle allowing users to switch between:
   - *Point Cloud Only* (pure observed depth data).
   - *CAD/Generative Meshes Only* (isolated synthesized furniture).
   - *Composite Scene* (room point cloud + aligned furniture meshes).

