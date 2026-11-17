# RoomAI — Phase 4F: Scene Fusion Report

**Date:** 2026-10-31  
**Author:** ML Systems Architect  
**Scope:** Fusion of semantic segmentation, instance segmentation, and metric depth.

---

## 1. Scene Representation Architecture

Each detected/segmented object is enriched with 3D geometric properties:

| Property | Source | Description |
| :--- | :--- | :--- |
| **Object ID** | Instance segmentation | Unique instance_id per detection |
| **Class** | A3 detector / SegFormer | Semantic category |
| **2D Bounding Box** | A3 detector | [x_min, y_min, x_max, y_max] normalized |
| **Instance Mask** | FastSAM / SegFormer | Binary mask at image resolution |
| **Depth Statistics** | Metric depth | mean, median, min, max depth within mask |
| **3D Centroid** | Point cloud | Mean (x,y,z) of valid points in mask |
| **Spatial Extent** | Point cloud | 3D bounding box of mask points |
| **Confidence** | Detector | A3 confidence score |

---

## 2. Fusion Pipeline

```
Input Image
    │
    ├─► A3 Detector ──► BBoxes + Classes + Confidences
    │                    │
    │                    └─► FastSAM (box prompts) ──► Instance Masks
    │
    ├─► SegFormer-B1 ──► Semantic Mask (wall/floor/ceiling/etc.)
    │
    └─► Depth Anything V2 Metric ──► Raw Depth Map (meters)
                                 │
                                 └─► Unproject ──► Point Cloud (XYZ + RGB)
                                      │
                                      └─► Fuse with Masks ──► Per-Object 3D
```

---

## 3. 3D Viewer (Phase 4G)

Implemented Streamlit-compatible 3D point cloud viewer using Plotly:
- Rotation, zoom, pan controls
- Toggle semantic colors / instance colors
- Per-object inspection on hover
- Independent from 2D image view

---

## 4. Known Limitations

1. **Occlusion handling:** Object depth statistics biased toward visible surfaces only
2. **Instance-to-point-cloud association:** Mask-based lookup may include background points near boundaries
3. **Scale uncertainty:** Metric depth provides meters but absolute scale depends on camera intrinsics
4. **Temporal consistency:** Each frame processed independently; no SLAM or tracking