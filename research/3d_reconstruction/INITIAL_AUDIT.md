# ROOMAI — INITIAL 3D RECONSTRUCTION REPOSITORY & HARDWARE AUDIT

**Date:** 2026-11-07  
**Role:** Senior 3D Computer Vision Research Engineer, ML Infrastructure Engineer, and Experimental Lead  
**Repository:** `F:\RoomAI`  
**Execution Environment:** Windows 11 (Build 26200), Python 3.10.20, Conda `tf-gpu`, PyTorch 2.11.0+cu128 (CUDA 12.8)  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), 24 GB DDR5 RAM, 394 GB Free Storage on Drive F:\  

---

## 1. Executive Summary

This audit assesses the state of RoomAI's existing 3D reconstruction pipeline, model inventory, hardware constraints, dependencies, and geometric limitations before investigating modern foundation models for 3D indoor scene reconstruction—specifically **HunyuanWorld-Mirror** (Tencent Hunyuan) and **VGGT** (Oxford VGG / Meta AI).

RoomAI currently possesses a functional, lightweight, single-view 3D room proxy pipeline based on 2D monocular depth unprojection and heuristic plane fitting. While fast (1.55s latency) and memory-efficient (368 MB VRAM), it cannot produce true view-consistent multi-view reconstructions, continuous textured 3D representations, 3D Gaussian Splats, or learned camera trajectories. This document establishes the empirical baseline, identifies architectural bottlenecks, and outlines the experimental roadmap for next-generation 3D reconstruction.

---

## 2. Current Architecture & Entry Points

RoomAI's architecture decouples model inference, modality fusion, and presentation into distinct layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Interface (Streamlit)                      │
│                app.py ──▶ ui/analysis_view.py ──▶ ui/three_d_viewer.py │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (In-process Python API)
┌────────────────────────────────────────────────────────────────────────┐
│                  Production Facade: services/roomai_backend.py         │
│          BACKEND.detect_objects() │ BACKEND.reconstruct_room_3d()     │
└────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ DetectionService │      │SegmentationServ. │      │ Scene3DService   │
│ YOLO11s A3 800px │      │   YOLO11n-Seg    │      │Depth Anything V2 │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Modality Fusion & Spatial Reasoning Engine                │
│                                                                        │
│  Single-View: services/scene_fusion_service.py                         │
│  Multi-View:  services/multiview_fusion_service.py (SIFT + Kabsch)    │
│  Floor/Layout:services/room_layout_service.py (RANSAC Plane Fitting)   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Output Schemas & Asset Exporters                   │
│                                                                        │
│  Schema:   schemas/scene.py (SceneGraph: 200k points, objects, layout) │
│  Exporters:utils/export_3d.py (glTF 2.0 GLB binary + ASCII PLY)        │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Entry Points:
1. **Application Entry Point:** [`app.py`](file:///F:/RoomAI/app.py) — Streamlit application initializing sidebar, session state, and mode routers (`Analysis View` and `Design View`).
2. **Backend Service Facade:** [`services/roomai_backend.py`](file:///F:/RoomAI/services/roomai_backend.py) — Singleton facade `BACKEND` exposing typed envelope responses, capabilities discovery, DOS guards, and 3D reconstruction workflows.
3. **Single-View Fusion:** [`services/scene_fusion_service.py`](file:///F:/RoomAI/services/scene_fusion_service.py) — Coordinates 2D detections, instance masks, and monocular depth to unproject points into metric camera space, associate depth to detected objects, and compute 3D bounding boxes.
4. **Multi-View Fusion:** [`services/multiview_fusion_service.py`](file:///F:/RoomAI/services/multiview_fusion_service.py) — Feature matching via OpenCV SIFT + FLANN matcher + RANSAC rigid alignment (Kabsch/Umeyama) + voxel downsampling.
5. **Layout Recovery:** [`services/room_layout_service.py`](file:///F:/RoomAI/services/room_layout_service.py) — RANSAC ground plane fitting with gravity normal vector alignment scoring.
6. **3D Exporters:** [`utils/export_3d.py`](file:///F:/RoomAI/utils/export_3d.py) — Point cloud PLY serializer with RGB channels and binary glTF 2.0 GLB serializer with `POINTS` mode and oriented bounding boxes.

---

## 3. Current Model Inventory & Checkpoints

All active checkpoints are stored in `checkpoints/` and have been verified with cryptographic SHA-256 hashes:

| Checkpoint Name | Relative Path | Architecture / Backbone | Params | Disk Size | Verified Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Indoor Detector (A3)** | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | YOLO11s (800px fine-tuned) | 9.4M | 18.29 MB | Primary indoor furniture detector (mAP50=0.512) |
| **Exterior Detector (B2)** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | YOLO11s (640px fine-tuned) | 9.4M | 18.28 MB | Exterior building & facade detector |
| **Parking Detector (C2)** | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | YOLO11s (640px fine-tuned) | 9.4M | 18.27 MB | Parking facility & vehicle detector |
| **Instance Segmentation** | `checkpoints/yolo11n-seg.pt` | YOLO11n-Seg | 2.9M | 5.90 MB | Fast furniture & boundary segmentation |
| **Metric Depth Small** | `checkpoints/depth_anything_v2_metric_indoor_small` | DINOv2 ViT-Small + DPT Head | 24.8M | 94.58 MB | Metric monocular depth in meters (primary) |
| **Relative Depth Small** | `checkpoints/depth_anything_v2_small` | DINOv2 ViT-Small + DPT Head | 24.8M | 94.58 MB | Normalized inverse depth fallback |
| **SegFormer-B1 (Pilot)** | `checkpoints/experimental/segformer_b1_pilot` | SegFormer-B1 | 13.7M | 52.35 MB | Experimental semantic segmentation |

---

## 4. Current Reconstruction Workflow & Mathematical Steps

1. **2D Image Preprocessing:** Input image is validated against dimension limits (max 8,192px, 100M pixels ceiling) and converted to RGB tensor.
2. **Object & Instance Perception:**
   - Model A3 predicts bounding boxes $[x_1, y_1, x_2, y_2]$ and class labels.
   - YOLO11n-Seg predicts binary instance masks $M_i \in \{0, 1\}^{H \times W}$.
3. **Monocular Depth Inference:**
   - Depth Anything V2 Metric processes image at native aspect ratio.
   - Bilateral edge-preserving guided filter (`utils/depth_refinement.py`) refines depth edges against RGB image gradients.
4. **Point Cloud Unprojection (Pinhole Model):**
   - For pixel $(u, v)$ with depth $Z = D(u, v)$:
     $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$
   - Generates up to 200,000 $(X, Y, Z)$ points with assigned $(R, G, B)$ colors.
5. **Floor Plane & Gravity Estimation:**
   - RANSAC plane fitting on points in the lower quartile of the room: $aX + bY + cZ + d = 0$.
   - Normal vector $[a, b, c]$ is evaluated against gravity prior $[0, -1, 0]$: $\text{score} = |\mathbf{n} \cdot \mathbf{g}|$.
   - If score $> 0.80$, plane is accepted as floor at elevation $Y_{\text{floor}}$.
6. **3D Oriented Bounding Boxes:**
   - For each detected 2D object $i$, points within mask $M_i$ are collected: $\mathcal{P}_i = \{(X, Y, Z)\}$.
   - Depth outliers are filtered (10th to 90th percentile).
   - Centroid $\mathbf{c}_i = \text{median}(\mathcal{P}_i)$ and extents $\mathbf{e}_i = (W_i, H_i, D_i)$ are calculated.
   - Objects are physically grounded to rest on the detected floor plane ($Y_{\text{bottom}} = Y_{\text{floor}}$).
7. **Asset Serialization:**
   - Point cloud is uniformly downsampled to 40,000 points.
   - Exported to ASCII PLY with RGB and binary glTF 2.0 GLB with point cloud primitive (`mode: 0 POINTS`) and box wireframe primitives (`mode: 4 TRIANGLES`).

---

## 5. Hardware Constraints & Telemetry

### Local Hardware Platform:
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (AD107 architecture)
- **VRAM:** 8,187.5 MB (8.0 GB) GDDR6
- **Compute Capability:** 8.9 (CUDA 12.8 compatible)
- **System RAM:** 23.73 GB Total DDR5, ~6.55 GB currently available
- **Storage:**
  - Drive `F:\`: 449.22 GB Total, **394.30 GB Free** (Designated location for research workspaces and dataset caching)
  - Drive `C:\`: 474.87 GB Total, 65.87 GB Free (OS and Conda root)

### Existing Baseline Performance (Empirical Benchmark):
- **A3 Detection Latency:** 24.81 ms (p50) | Peak VRAM: 131.2 MB
- **YOLO11n-Seg Latency:** 172.30 ms (p50) | Peak VRAM: 203.3 MB
- **Metric Depth Latency:** 518.54 ms (p50) | Peak VRAM: 345.4 MB
- **End-to-End Reconstruction:** 1,546.55 ms (p50) | Peak VRAM: 367.7 MB allocated, 486.0 MB reserved
- **3D Asset Export:** 125.33 ms (p50)

---

## 6. Dependency Risks & Environment Isolation Strategy

| Risk Factor | Current RoomAI Setup | HunyuanWorld-Mirror Setup | Isolation & Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **PyTorch / CUDA** | PyTorch 2.11.0 + CUDA 12.8 (`tf-gpu`) | PyTorch 2.4.0 + CUDA 12.4 | **MANDATORY ISOLATION:** Do not install into `tf-gpu`. Create independent Conda environment `hunyuanworld-mirror` targeting PyTorch 2.4.0 + cu124. |
| **CUDA Extensions (`gsplat`)** | None (pure PyTorch & OpenCV) | Requires `gsplat` (compiled CUDA rasterizer for 3DGS) | Windows MSVC 2022 C++ build tools required. Install prebuilt wheel for `pt24cu124` from `docs.gsplat.studio`. |
| **TensorFlow / NumPy Conflict** | `USE_TF=0`, `TRANSFORMERS_NO_TF=1` | Pure PyTorch / Transformers / Diffusers | Maintain explicit environment variables in all research scripts. |
| **VRAM Budget Exceeded** | 368 MB allocated (well within 8 GB) | HunyuanWorld-Mirror requires $\ge 12$ GB for full res video/multi-view | **Low-resolution / cropped inference:** Evaluate at $512 \times 512$ or $384 \times 384$ for local smoke tests. Cloud GPUs (A100 / L40S) required for full-scale fine-tuning. |

---

## 7. Analysis of Missing Capabilities

| Desired Capability | Current RoomAI Implementation | Limitation | How Candidate Models Address It |
| :--- | :--- | :--- | :--- |
| **View-Consistent Multi-View Fusion** | SIFT feature matching + Kabsch rigid alignment | Fails on smooth indoor walls, wide baselines, and lighting changes; falls back to single view | **HunyuanWorld-Mirror / VGGT:** Learned cross-view attention across multiple frames directly outputs aligned point maps and camera poses without SIFT. |
| **Learned Camera Extrinsics / Intrinsics** | Assumes approximate phone intrinsics ($f_x=f_y=1000$); camera pose fixed at origin | Inaccurate physical scale and perspective distortion on varied camera optics | **HunyuanWorld-Mirror / VGGT:** Explicit camera heads predict 6-DoF camera poses ($R, \mathbf{t}$) and focal lengths from visual features. |
| **Dense Surface Normal Maps** | None (plane fitting only on floor) | Cannot estimate wall orientations, ceiling slopes, or surface slant for CAD integration | **HunyuanWorld-Mirror:** Native surface normal prediction head predicts pixel-aligned normal vectors. |
| **Continuous Textured Scene Representation** | Discrete 3D point cloud proxy (no mesh/radiance) | Sparse visual quality; holes when viewing from novel angles | **HunyuanWorld-Mirror:** Predicts 3D Gaussian Splats (3DGS: position, covariance, opacity, spherical harmonics) for photorealistic novel view rendering. |
| **Geometric Prior Conditioning** | None (pipeline is purely feed-forward sequential) | Cannot inject metric depth hints or known floor heights into neural reconstruction | **HunyuanWorld-Mirror:** Multi-modal prior prompting encodes depth, poses, or point maps into structured conditioning tokens. |

---

## 8. Proposed Research Implementation Plan

```
Phase 0: Repository & Hardware Audit ─── [COMPLETE] (INITIAL_AUDIT.md)
   │
   ▼
Phase 1: Model Investigation & Baseline
   ├─ Clone HunyuanWorld-Mirror & VGGT into separate research workspace
   ├─ Establish isolated Conda environment (pt24cu124)
   ├─ Build controlled indoor evaluation set (same-room multi-view, varied lighting/clutter)
   └─ Run baseline comparison under identical inputs ──▶ BASELINE_REPORT.md & MODEL_SELECTION.md
   │
   ▼
Phase 2: Dataset Research & Audit
   ├─ Audit ScanNet, Hypersim, ARKitScenes, Structured3D (100–250 GB budget)
   ├─ Scene-level split enforcement (strict leakage prevention)
   └─ Quality & coordinate convention audit ──▶ DATASET_AUDIT.md
   │
   ▼
Phase 3: RoomAI Dataset Pipeline
   ├─ Structured scene manifests (YAML/JSON: images, poses, intrinsics, depth, normals)
   └─ Automated validation scripts in research/3d_reconstruction/scripts/
   │
   ▼
Phase 4 & 5: Training Infrastructure & Experiments
   ├─ RTX 4060 smoke tests (adapters / lightweight heads)
   ├─ Cloud GPU specification for full foundation model fine-tuning
   └─ Staged experiments A–E ──▶ TRAINING_PLAN.md
   │
   ▼
Phase 6: Comprehensive Evaluation
   ├─ Scale-aware & aligned metrics (RMSE, AbsRel, Chamfer, Normal Angular Error, PSNR)
   └─ Comparative benchmarks on held-out test scenes ──▶ EVALUATION_REPORT.md
   │
   ▼
Phase 7: RoomAI Integration Handoff
   ├─ Non-breaking integration contract with RoomAIBackend & SceneGraph
   └─ Coordinate system adapters ──▶ INTEGRATION_CONTRACT.md
```

