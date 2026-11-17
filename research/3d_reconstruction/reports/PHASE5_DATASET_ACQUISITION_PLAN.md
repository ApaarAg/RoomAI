# ROOMAI — PHASE 5: GROUND-TRUTH INDOOR DATASET ACQUISITION PLAN

**Document ID:** `research/3d_reconstruction/reports/PHASE5_DATASET_ACQUISITION_PLAN.md`  
**Date:** 2026-11-08  
**Author:** Senior 3D Computer Vision Research Scientist & ML Infrastructure Lead  
**Scope:** Actionable, phased plan for acquiring, validating, and standardizing genuine indoor geometric ground-truth datasets for RoomAI.

---

## 1. Executive Summary

Phase 4 proved that RoomAI's training and evaluation infrastructure functions end-to-end. However, Phase 5 established that the current 11-scene / 39-view dataset relies on **model pseudo-labels (Depth Anything V2)** and **synthetic camera trajectories ($R_y(0.05\cdot v)$)**.

To transition from **pseudo-label distillation** to **rigorous physical 3D reconstruction**, RoomAI must acquire authentic indoor datasets with:
1. **Measured sensor depth (LiDAR / Structured Light)** or **exact CAD synthetic geometry**.
2. **Calibrated 6-DoF camera poses (COLMAP / VIO / SLAM)**.
3. **Reconstructed 3D point clouds or CAD meshes**.
4. **Deterministic, zero-leakage scene splits**.

---

## 2. Target Dataset Evaluation & Prioritization Matrix

| Dataset | Modality | Annotations Provided | Typical Volume | License & Access | Suitability for RoomAI |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Hypersim** | Synthetic Photorealistic | Exact metric depth, surface normals, diffuse reflectance, 6-DoF camera poses, semantic mesh | 461 scenes (73,400 frames, ~1.2 TB full; ~25 GB subset) | Apple Research License (Non-commercial research) | **PRIMARY TARGET (Synthetic)**: Perfect ground truth for multi-task geometry (points, depth, normals). |
| **ScanNet (v2)** | Real RGB-D (Structure Sensor) | Metric depth, calibrated $3\times3$ intrinsics, 6-DoF poses, dense 3D surface meshes, semantics | 1,513 scenes (2.5M frames; ~50 GB subset) | Academic Terms of Use (Requires university/research request) | **PRIMARY TARGET (Real-World)**: Gold-standard for real indoor RGB-D benchmarking. |
| **ARKitScenes** | Real RGB + LiDAR (iPad/iPhone) | High-res RGB, raw LiDAR depth, calibrated ARKit poses, 3D room boxes, textured meshes | 5,048 captures (~100 GB subset) | Apple Open Source License | **SECONDARY TARGET (Mobile LiDAR)**: Exact fit for smartphone indoor scanning use case. |
| **Matterport3D** | Real RGB + Panoramic Depth | Matterport Pro 3D camera, panoramic HDR RGB-D, camera poses, textured 3D meshes | 90 building scans (194,400 images) | Matterport Terms of Agreement | Alternative benchmark for whole-building navigation. |

---

## 3. Phased Acquisition Protocol (Pilot-First Gate)

To prevent unverified multi-hundred-gigabyte downloads that overwhelm local storage:

```
[Phase 5.1: Pilot Ingestion]  -->  [Phase 5.2: Verification Gate]  -->  [Phase 5.3: Benchmark Expansion]
  - 10 scenes Hypersim (~5 GB)       - Verify metric depth in meters      - 100 scenes Hypersim (~35 GB)
  - 10 scenes ScanNet (~5 GB)        - Verify SO(3) pose orthogonality    - 50 scenes ScanNet (~30 GB)
  - 10 scenes ARKitScenes (~5 GB)    - Verify OpenCV camera coordinates   - 50 scenes ARKitScenes (~25 GB)
                                     - Verify zero-leakage scene hash
```

### Phase A: Pilot Acquisition (Target Budget: 15 GB total)
1. **Hypersim Subset (10 scenes):**
   * Download 10 residential indoor scenes (`ai_001_001` through `ai_001_010`).
   * Extract: `color.jpg`, `geometry_hdf5/depth_meters.hdf5`, `geometry_hdf5/normal_cam.hdf5`, `camera_keyframe_positions.hdf5`.
2. **ScanNet Subset (10 scenes):**
   * Extract 10 distinct scenes (`scene0000_00` to `scene0009_00`).
   * Extract: 20 keyframes per scene, depth frames (16-bit PNG, millimeter scale $\to$ meters), camera pose text files.
3. **ARKitScenes Subset (10 scenes):**
   * Download 10 rooms (Living rooms, bedrooms, kitchens).
   * Extract: Low-res wide RGB video frames, aligned LiDAR depth maps, and trajectory poses.

---

## 4. Standardization Contract & Coordinate Transformation

Every acquired scene must be automatically converted into RoomAI's unified storage schema:

```
research/3d_reconstruction/data/ground_truth/<dataset_name>/<scene_id>/
  frame_0000_rgb.png       # 518x518 RGB image
  frame_0000_depth.npy     # float32 metric depth in meters (Z > 0)
  frame_0000_normal.npy    # float32 unit normal vector in OpenCV camera frame
  frame_0000_cam.npz       # intrinsics [3,3] float32, pose [4,4] float32 (C2W)
  mesh.ply                 # Ground-truth 3D surface mesh (where available)
```

### Coordinate System Transformation to OpenCV Standard:
* **OpenCV Convention:** $+X$ Right, $+Y$ Down, $+Z$ Forward (camera view direction).
* **ScanNet Conversion:** ScanNet uses $+X$ Right, $+Y$ Forward, $+Z$ Up (OpenGL/Blender).  
  Transformation matrix:
  $$T_{\text{ScanNet}\to\text{OpenCV}} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
* **Hypersim Conversion:** Hypersim stores depth in planar camera distance. Converted to orthogonal $Z$-depth.

---

## 5. Strict Zero-Data-Leakage Splitting Strategy

Scene assignment must follow a deterministic SHA-256 hash on the global Scene UUID:

```python
def assign_scene_split(scene_uuid: str) -> str:
    h = hashlib.sha256(scene_uuid.encode('utf-8')).hexdigest()
    ratio = int(h[:8], 16) / 0xFFFFFFFF
    if ratio < 0.70:
        return "train"
    elif ratio < 0.85:
        return "val"
    else:
        return "test"
```

### Leakage Prevention Rules:
1. **Scene-Level Exclusivity:** All video frames, viewpoints, and sub-sequences of a given physical room are confined to a single split.
2. **Cross-Room Contamination Check:** Multi-room apartments (e.g. ScanNet `scene0000_00` and `scene0000_01`) share building geometry and must be grouped under a single parent Building ID before hashing.
3. **No Hyperparameter Tuning on Test Scenes:** Test split remains completely frozen until final sign-off.

---

## 6. Execution Gate & Next Actions

1. Acquire official research access credentials for Hypersim and ScanNet.
2. Run pilot download script `research/3d_reconstruction/scripts/download_ground_truth_pilot.py` targeting 10 scenes (~15 GB).
3. Verify coordinate transforms, metric depth ranges, and pose orthogonality.
4. Execute baseline evaluation of WorldMirror-1.0 on the true physical ground truth before launching multi-epoch training.

