# ROOMAI — 3D INDOOR RECONSTRUCTION DATASET AUDIT & INVENTORY

**Date:** 2026-11-07  
**Role:** Senior 3D Computer Vision Research Engineer & ML Infrastructure Lead  
**Target Storage Budget:** 100 GB – 250 GB Maximum  
**Available Drive F:\ Storage:** **394.30 GB Free**  
**Core Objective:** Curate geometrically supervised, multi-view indoor datasets for foundation model evaluation, prior conditioning, and fine-tuning.

---

## 1. Candidate Dataset Audit & Evaluation

| Dataset Name | Source & Organization | License & Terms | Usable Indoor Scenes | Multi-View Coverage | Camera & 3D Annotations | Coordinate System | Recommended Curated Budget |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hypersim** | Apple Machine Learning Research | Apple Non-Commercial Research License | 461 scenes (73,400 frames) | Continuous camera trajectories | RGB, Metric Float32 Depth (m), Surface Normals, Extrinsics, Intrinsics | Right-handed ($X$ right, $Y$ up, $-Z$ look) | **45.0 GB** (~50 scenes, ~7,500 frames) |
| **ScanNet v2** | Stanford / TUM / Princeton | ScanNet Terms of Agreement (Research Only) | 1,513 scans (707 unique rooms) | Dense RGB-D handheld video loops | RGB, 16-bit Depth (mm), 6-DoF poses, 3D surface meshes (`.ply`), CAD boxes | Camera: OpenCV ($X$ right, $Y$ down, $+Z$ forward) | **55.0 GB** (~80 scans, ~12,000 frames) |
| **ARKitScenes** | Apple | Creative Commons / Apple Research License | 1,600+ spaces (5,000+ scans) | Multi-angle room sweeps | Wide RGB, LiDAR Depth, Confidence, ARKit VIO poses, 3D oriented boxes | ARKit ($X$ right, $Y$ up, $-Z$ look) | **35.0 GB** (~60 spaces, ~9,000 frames) |
| **Structured3D**| SenseTime Research | SenseTime Academic Non-Commercial License | 3,500 homes (21,835 rooms) | Multi-view perspective & 360° | RGB, Ground-truth Depth, Surface Normals, Floor/Ceiling/Wall planes | OpenGL / Blender coordinate frame | **30.0 GB** (~100 rooms, ~6,000 frames) |
| **3D-FRONT / 3D-FUTURE** | Alibaba Research | 3D-FRONT Terms of Use | 6,800+ house layouts | Synthetic perspective views | High-quality CAD geometry, exact bounding boxes, rendered depth | Standard graphics coordinates | **15.0 GB** (~40 houses, ~3,000 frames) |
| **Total Curated Mixture** | Multi-Source Harmonized | Academic / Research Compliant | **330 unique physical rooms** | **Balanced multi-view sweeps** | **Metric depth, normals, poses, point clouds** | **Harmonized to OpenCV Pinhole** | **180.0 GB** (Within 250 GB budget) |

---

## 2. Dataset Detailed Profiles

### 2.1 Hypersim (Apple)
- **Official Source:** `https://github.com/apple/ml-hypersim`
- **Strengths:** Fully synthetic, ray-traced with physically based rendering (V-Ray). Zero sensor noise, perfectly accurate float32 metric depth, dense surface normals, and complete camera parameter matrices.
- **Indoor Relevance:** Extremely high. Diverse room typologies (living rooms, bedrooms, kitchens, modern open-concept layouts).
- **Limitations:** Synthetic materials can exhibit unrealistic reflections or lack subtle real-world handheld motion blur.
- **Intended Role in RoomAI:** Primary ground truth benchmark for metric depth and surface normal quantitative evaluation.

### 2.2 ScanNet v2 (Stanford / TUM)
- **Official Source:** `http://www.scannet.org` (Benchmark scripts located in `ScanNet-master/`)
- **Strengths:** Real-world physical capture using iPad + Structure Sensor. Contains real lighting occlusions, clutter, and realistic indoor camera trajectories.
- **Indoor Relevance:** Gold standard for real-world indoor 3D perception and scene understanding.
- **Limitations:** Sensor depth exhibits holes on shiny/black surfaces; camera poses computed via bundle adjustment have occasional drift.
- **Intended Role in RoomAI:** Multi-view real-world camera trajectory and 3D point cloud evaluation.

### 2.3 ARKitScenes (Apple)
- **Official Source:** `https://github.com/apple/arkitscenes`
- **Strengths:** Hardware LiDAR sensor on consumer iPhone/iPad devices. Includes oriented 3D bounding box annotations for indoor furniture.
- **Indoor Relevance:** Directly matches real-world end-user capture conditions for mobile room scanning.
- **Limitations:** Low-resolution LiDAR depth requires upsampling/alignment with RGB frames.
- **Intended Role in RoomAI:** Evaluation of 3D furniture bounding box localization and floor plane estimation.

---

## 3. Coordinate Systems & Depth Scale Harmonization

A major source of failure in multi-dataset pipelines is coordinate mismatch. All incoming datasets are harmonized to the **Standard OpenCV Pinhole Camera Convention** required by HunyuanWorld-Mirror and VGGT:

$$\mathbf{p}_{\text{cam}} = \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}, \quad \text{where } +X \text{ is Right}, \ +Y \text{ is Down}, \ +Z \text{ is Forward (Viewing Direction)}$$

### Transformation Mappings:
1. **OpenGL / ARKit ($X$ right, $Y$ up, $-Z$ forward) $\rightarrow$ OpenCV ($X$ right, $Y$ down, $+Z$ forward):**
   $$\mathbf{R}_{\text{cv}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix} \mathbf{R}_{\text{gl}}$$
2. **Depth Unit Normalization:**
   - **ScanNet & ARKitScenes:** Converted from unsigned 16-bit millimeter integers to IEEE 754 float32 in meters:
     $$D_{\text{metric}} = \frac{D_{\text{raw}}}{1000.0}$$
   - **Hypersim:** Converted from camera ray distance to perpendicular optical axis planar depth $Z$:
     $$Z = D_{\text{distance}} \cdot \cos(\theta)$$

---

## 4. Strict Leakage Prevention & Scene-Level Splitting

> [!CAUTION]
> **Zero-Data-Leakage Policy:**  
> Random image-level splitting is strictly prohibited. If image $A$ and image $B$ depict the same physical room, they **MUST NEVER** be split between train and test sets.

### Splitting Protocol:
- **Hash-Based Deterministic Partitioning:** Scene identifier string (e.g. `scene0001_00`, `ai_001_001`) is hashed using SHA-256.
- **Split Proportions (Scene-Level):**
  - **Training Set (Train):** 70% of unique rooms (approx. 230 scenes).
  - **Validation Set (Val):** 15% of unique rooms (approx. 50 scenes).
  - **Held-Out Test Set (Test):** 15% of unique rooms (approx. 50 scenes).
- **Test Set Isolation:** The held-out test split is strictly locked and cannot be used for checkpoint selection, early stopping, or hyperparameter sweeps.

---

## 5. Automated Data Integrity Verification Protocol

Before any scene manifest is added to the training or evaluation pipeline, it must pass automated validation:
1. **Deduplication:** Perceptual hashing (pHash) with Hamming distance threshold $d_H \le 4$ prevents duplicate room scans from inflating metrics.
2. **Finite Coordinate Bounds:** Depth maps and point clouds are checked for NaN/Inf. Outliers $> 20\text{ meters}$ in indoor scenes are masked.
3. **Camera Matrix Validity:** Extrinsic rotation matrices $\mathbf{R}$ must satisfy $\det(\mathbf{R}) = +1$ and $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ within floating-point tolerance ($10^{-5}$).
4. **Image-Depth Pixel Correspondence:** Aspect ratios and pixel dimensions of RGB and depth maps must match exactly.

