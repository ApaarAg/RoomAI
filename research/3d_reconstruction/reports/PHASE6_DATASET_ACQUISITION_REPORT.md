# ROOMAI — PHASE 6: GROUND-TRUTH DATASET ACQUISITION & GEOMETRIC SUPERVISION REPORT

**Document ID:** `research/3d_reconstruction/reports/PHASE6_DATASET_ACQUISITION_REPORT.md`  
**Date:** 2026-11-08  
**Author:** Senior 3D Computer Vision Research Scientist, Dataset Engineer & ML Infrastructure Lead  
**Scope:** Complete implementation and empirical verification of genuine indoor geometric ground truth for RoomAI.

---

## 1. Executive Summary & Context

Phase 5 concluded with an authoritative verdict: **PSEUDO-LABEL DISTILLATION ONLY**.  
The previous fine-tuning iterations distilled `Depth Anything V2 Metric Indoor Small` predictions into `WorldMirror-1.0`, yielding a 44.7% reduction in pseudo-label discrepancy on typical rooms but triggering severe 3.63x scale divergence on unfamiliar cubicle layouts (`Office`) and leaving 3DGS novel-view rendering unanchored.

Phase 6 completely replaces that pseudo-label-only experimental setup with an **authentic, multi-modal geometric supervision pipeline** combining:
1. **Hypersim (Apple Research):** Photorealistic synthetic CAD ground truth with exact path-traced primary ray distance, CAD surface normals, and keyframe camera poses.
2. **ScanNet (v2):** Real-world indoor RGB-D captures with physical infrared structured-light depth measurements, calibrated camera intrinsics, and BundleFusion SLAM trajectories.
3. **ARKitScenes (Apple Research):** Mobile handheld dToF LiDAR dataset formally investigated and evaluated as an auxiliary mobile benchmark.

All pseudo-label-dependent training is terminated. A verified pilot dataset of **1,090 standardized frames** has been acquired, transformed, and empirically validated with **zero data leakage** across strictly isolated room partitions.

---

## 2. Dataset Acquisition Objectives & Modality Analysis

### 2.1 Hypersim (Synthetic Ground Truth)
* **Official Repository:** `apple/ml-hypersim`
* **Licensing & Access:** Apple Research License (Non-commercial research). Hosted on Apple's Content Delivery Network (`docs-assets.developer.apple.com`).
* **Sensor / Synthesis Engine:** Chaos Group V-Ray photorealistic path tracer over 461 high-poly 3D indoor architectural assets.
* **Provided Modalities:**
  - `color.jpg`: Photorealistic 8-bit sRGB tonemapped preview ($1024 \times 768$).
  - `depth_meters.hdf5`: Float16 Euclidean distance from camera center along each ray ($R = \|\mathbf{X}_c\|_2$).
  - `normal_cam.hdf5`: Float16 unit surface normal vector in OpenGL camera space ($+X$ right, $+Y$ up, $+Z$ back).
  - `camera_keyframe_positions.hdf5`: Camera center in world asset units (scaled by `meters_scale = 0.01`).
  - `camera_keyframe_orientations.hdf5`: Camera-to-world rotation matrix $R_{\text{C2W}}$ in OpenGL conventions ($\det(R) = 1.0, \|R^T R - I\|_F < 10^{-6}$).
  - `metadata_camera_parameters.csv`: Exact camera ray projection matrix $M_{\text{cam\_from\_uv}}$.

### 2.2 ScanNet (Real-World Sensor Measurements)
* **Official Repository:** `ScanNet/ScanNet` (Dai et al., CVPR 2017)
* **Licensing & Access:** Academic Terms of Use. Sourced from verified open community mirrors (`fjd/scannet-processed-test`).
* **Sensor / Hardware Architecture:**
  - **Color Camera:** Apple iPad Air 2 ($1296 \times 968$ downsampled to $640 \times 480$).
  - **Depth Camera:** Occipital Structure Sensor (PrimeSense infrared structured-light projector + IR camera), **NOT LiDAR**.
  - **Tracking System:** Real-time BundleFusion dense RGB-D SLAM.
* **Provided Modalities:**
  - `color/XXXX.jpg`: 8-bit sRGB frames ($640 \times 480$).
  - `depth/XXXX.npy`: 16-bit unsigned integer depth in millimeters ($d_{\text{mm}}$). Value 0 denotes unmeasured infrared reflections (specular surfaces, black absorption, or occlusions).
  - `pose/XXXX.txt`: 4x4 camera-to-world matrix in Stanford CAD world coordinates.
  - `intrinsic/intrinsic_depth.txt`: Calibrated pinhole intrinsic matrix ($f_x \approx 577.6, f_y \approx 578.7, c_x \approx 318.9, c_y \approx 242.7$).
* **Critical Distinction:** ScanNet depth is **direct physical sensor measurement** subject to sensor noise and dropouts; surface normals are **derived annotations**, not CAD ground truth.

### 2.3 ARKitScenes (Handheld Mobile LiDAR Investigation)
* **Official Repository:** `apple-aiml-research/ARKitScenes` (Baruch et al., NeurIPS 2021)
* **Licensing:** Apple Open Source License / CC-BY-NC 4.0.
* **Sensor:** Apple iPad Pro / iPhone Pro with direct Time-of-Flight (dToF) LiDAR.
* **Investigation Findings:**
  - **Resolution:** Raw LiDAR depth is captured at ultra-low native resolution ($256 \times 192$), requiring neural upsampling or depth confidence thresholding.
  - **Trajectory:** Uses Apple ARKit Visual-Inertial Odometry (VIO) stored in axis-angle radians or column-major 4x4 matrices.
  - **Conventions:** Camera coordinate frame has $+X$ right, $+Y$ up, $+Z$ back (OpenGL/ARKit).
  - **Pipeline Policy:** Retained as an optional downstream evaluation benchmark for mobile real-time scanning; excluded from core backpropagation training to avoid low-resolution dToF discretization artifacts.

---

## 3. Mathematical Rigor: Hypersim Ray Distance vs. Planar Z-Depth

A critical failure mode in indoor depth research is treating spherical or conical ray distance $R$ as orthogonal planar depth $Z_c$. In Hypersim, `depth_meters.hdf5` explicitly stores the Euclidean distance from the camera center to the surface point along the ray passing through that pixel:

$$R(u, v) = \|\mathbf{X}_c\|_2 = \sqrt{X_c^2 + Y_c^2 + Z_c^2}$$

### 3.1 Closed-Form Ray-to-Depth Conversion
Using the scene's intrinsic ray projection matrix $M_{\text{cam\_from\_uv}}$ from `metadata_camera_parameters.csv`:

$$\begin{bmatrix} d_x \\ d_y \\ d_z \end{bmatrix} = M_{\text{cam\_from\_uv}} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad u \in [-1, 1], \ v \in [-1, 1]$$

The unit ray vector is:

$$\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2} = \frac{[d_x, d_y, d_z]^T}{\sqrt{d_x^2 + d_y^2 + d_z^2}}$$

Since OpenGL camera space looks along $-Z$, the cosine of the ray angle relative to the optical axis is:

$$\cos(\theta) = \frac{-d_z}{\|\mathbf{d}\|_2} = \frac{-d_z}{\sqrt{d_x^2 + d_y^2 + d_z^2}}$$

The true orthogonal planar Z-depth in OpenCV camera space ($+Z$ forward) is:

$$Z_c = R \cdot \cos(\theta) = R \cdot \frac{-d_z}{\sqrt{d_x^2 + d_y^2 + d_z^2}}$$

### 3.2 Empirical Verification of Distortion
On scene `ai_015_007`:
* **Optical Center:** $\cos(\theta) = 0.999999 \implies Z_c \approx R$ (0.00% distortion).
* **Frame Corners:** $\cos(\theta) = 0.812364 \implies Z_c = 0.812 \cdot R$.
* **Geometric Correction:** Blindly treating $R$ as $Z_c$ introduces up to an **18.76% geometric dilation error** at the image boundaries. Our pipeline applies exact per-pixel ray unprojection, eliminating this distortion.

---

## 4. Coordinate System Standardization to OpenCV Standard

All datasets are converted into RoomAI's unified OpenCV coordinate convention:
* **$+X$:** Points horizontally to the Right.
* **$+Y$:** Points vertically Downward.
* **$+Z$:** Points Forward into the scene along the optical axis.

### 4.1 Transformation Matrices
* **Hypersim Normal Transformation:**
  $$\mathbf{N}_{\text{OpenCV}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix} \mathbf{N}_{\text{OpenGL}}$$
* **Hypersim Extrinsics (C2W):**
  $$R_{\text{C2W\_OpenCV}} = R_{\text{C2W\_OpenGL}} \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad \mathbf{t}_{\text{world}} = \text{position} \cdot 0.01$$
* **ScanNet Extrinsics (C2W):**
  ScanNet camera coordinates already follow $+X$ right, $+Y$ down, $+Z$ forward. The 4x4 pose matrix maps camera points directly into world space:
  $$\mathbf{X}_{\text{world}} = T_{\text{C2W}} \begin{bmatrix} \mathbf{X}_{\text{cam}} \\ 1 \end{bmatrix}$$

---

## 5. Acquired Pilot Dataset & Zero-Leakage Split Partition

The acquired pilot dataset encompasses **13 scene sequences** (5 Hypersim scenes, 8 ScanNet sequences) totaling **1,090 verified frames** under an initial disk footprint of ~4.12 GB (well within the 15 GB ceiling):

### 5.1 Partition Breakdown (Zero Scene Leakage)

| Split | Dataset | Scene / Room ID | Frames | Sequence Type & Modality | Physical Room Classification |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **TRAIN** | Hypersim | `ai_015_007` | 6 | Synthetic CAD Path Tracing | Private Home Office / Study |
| **TRAIN** | Hypersim | `ai_011_010` | 9 | Synthetic CAD Path Tracing | Modern Residential Living Room |
| **TRAIN** | Hypersim | `ai_047_008` | 16 | Synthetic CAD Path Tracing | Contemporary Kitchen & Dining |
| **TRAIN** | ScanNet | `scene0019_00` | 28 | Structure Sensor RGB-D | Multi-view Office / Desk Space |
| **TRAIN** | ScanNet | `scene0019_01` | 47 | Structure Sensor RGB-D | Multi-view Office / Desk Space |
| **TRAIN** | ScanNet | `scene0015_00` | 98 | Structure Sensor RGB-D | Residential Bedroom / Storage |
| **SUBTOTAL TRAIN** | | **5 Rooms** | **204** | | **Zero Leakage to Val/Test** |
| **VAL** | Hypersim | `ai_016_001` | 100 | Synthetic CAD Path Tracing | Open Plan Dining & Living Space |
| **VAL** | ScanNet | `scene0030_00` | 124 | Structure Sensor RGB-D | Residential Living Room |
| **VAL** | ScanNet | `scene0030_01` | 82 | Structure Sensor RGB-D | Residential Living Room |
| **VAL** | ScanNet | `scene0030_02` | 96 | Structure Sensor RGB-D | Residential Living Room |
| **SUBTOTAL VAL** | | **2 Rooms** | **402** | | **Zero Leakage to Train/Test** |
| **TEST** | Hypersim | `ai_013_010` | 100 | Synthetic CAD Path Tracing | Residential Master Suite / Bedroom |
| **TEST** | ScanNet | `scene0011_00` | 178 | Structure Sensor RGB-D | Large Office & Workspace |
| **TEST** | ScanNet | `scene0011_01` | 206 | Structure Sensor RGB-D | Large Office & Workspace |
| **SUBTOTAL TEST** | | **2 Rooms** | **484** | | **Strictly Frozen for Final Gate** |
| **GRAND TOTAL** | | **9 Physical Rooms** | **1,090** | | **100% Empirically Validated** |

### 5.2 Contamination Prevention Audit
* **Rule 1 (Building-Level Grouping):** All sub-scans of ScanNet `scene0019` (`_00`, `_01`) are grouped exclusively in **TRAIN**. All sub-scans of `scene0030` (`_00`, `_01`, `_02`) are grouped exclusively in **VAL**. All sub-scans of `scene0011` (`_00`, `_01`) are grouped exclusively in **TEST**.
* **Overlap Check:**
  $$\text{Train Rooms} \cap \text{Val Rooms} = \emptyset$$
  $$\text{Train Rooms} \cap \text{Test Rooms} = \emptyset$$
  $$\text{Val Rooms} \cap \text{Test Rooms} = \emptyset$$

---

## 6. Standardized Dataset Schema

All 1,090 frames have been formatted into `research/3d_reconstruction/data/ground_truth/standardized/`:

```
research/3d_reconstruction/data/ground_truth/standardized/<dataset>/<split>/<scene_id>/
  frame_0000_rgb.png       # 518x518x3 uint8 sRGB
  frame_0000_depth.npy     # 518x518 float32 planar metric Z-depth in meters (Z > 0)
  frame_0000_normal.npy    # 518x518x3 float32 unit normal vector in OpenCV camera frame
  frame_0000_mask.png      # 518x518 uint8 binary validity mask (255 = valid, 0 = invalid)
  frame_0000_cam.npz       # intrinsics [3,3] float32, pose [4,4] float32 C2W in OpenCV frame
```

Global Manifest: [`indoor_ground_truth_manifest.json`](file:///f:/RoomAI/research/3d_reconstruction/data_manifests/indoor_ground_truth_manifest.json) tracks per-frame paths, depth extrema, median depth, valid pixel ratio, camera intrinsics, and pose determinants.

---

## 7. Empirical Acceptance Audit Results

The validation script [`validate_ground_truth_pipeline.py`](file:///f:/RoomAI/research/3d_reconstruction/scripts/validate_ground_truth_pipeline.py) executed an exhaustive mathematical inspection across all 1,090 standardized frames (5,450 files loaded):

| Validation Gate | Target Specification | Measured Result | Verdict |
| :--- | :--- | :--- | :---: |
| **1. File Completeness** | 1,090 frames $\times$ 5 assets = 5,450 files | 5,450 / 5,450 files readable | **PASSED** |
| **2. Tensor Dimensions** | Exact $518 \times 518$ resolution | 1,090 / 1,090 matching exactly | **PASSED** |
| **3. Metric Depth Range** | Strictly finite, strictly positive ($Z > 0$) | Range: $[0.148\text{m}, 28.299\text{m}]$ | **PASSED** |
| **4. Global Median Depth** | Realistic indoor room scale | **2.138 m** | **PASSED** |
| **5. Surface Normal Unit Length** | $\|\|\mathbf{N}\|\|_2 = 1.0 \pm 10^{-3}$ on valid pixels | Max deviation: **$0.000000$** | **PASSED** |
| **6. Camera Rotation Det(R)** | $\det(R) = 1.0 \pm 10^{-4}$ (Proper SO(3)) | Max deviation: **$0.000002$** | **PASSED** |
| **7. Camera Orthogonality** | $\|\|R^T R - I\|\|_F < 10^{-4}$ | Max error: **$0.000003$** | **PASSED** |
| **8. Zero Scene Leakage** | Zero room overlap across train, val, test | 0 overlap across all pairs | **PASSED** |

---

## 8. Checkpoint Preservation Confirmation

All baseline model weights have been verified and preserved with unaltered SHA-256 hashes:

| Checkpoint Name | File Path | File Size | SHA-256 Hash | Integrity Status |
| :--- | :--- | :---: | :--- | :---: |
| **Production Baseline** | `checkpoints/production_candidate/production_model.ckpt` | 2,707.66 MB | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` | **VERIFIED UNCHANGED** |
| **Stage 1 Distilled** | `checkpoints/stage1/best_stage1.ckpt` | 2,707.64 MB | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` | **VERIFIED UNCHANGED** |
| **Stage 2 3DGS** | `checkpoints/stage2/stage2_real_verified.ckpt` | 2,522.40 MB | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` | **VERIFIED UNCHANGED** |

