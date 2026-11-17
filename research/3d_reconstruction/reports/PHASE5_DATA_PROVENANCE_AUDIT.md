# ROOMAI — PHASE 5: DATASET PROVENANCE AUDIT

**Document ID:** `research/3d_reconstruction/reports/PHASE5_DATA_PROVENANCE_AUDIT.md`  
**Date:** 2026-11-08  
**Auditor:** Senior 3D Computer Vision Research Scientist & Independent Model Evaluation Lead  
**Scope:** Forensic provenance audit of `prepare_indoor_dataset.py`, `indoor_verified_manifest.json`, and all 39 views in `indoor_verified_dataset/`.

---

## 1. Executive Summary

This forensic provenance audit establishes the exact origin, mathematical derivation, and verification status of every annotation in the RoomAI indoor 3D reconstruction dataset (`research/3d_reconstruction/data/indoor_verified_dataset/`).

### Foundational Finding
**The dataset does not contain independently measured physical ground truth.**
* **RGB Images:** 10 scenes (38 views) are official demo photographs bundled in `Tencent-Hunyuan/HunyuanWorld-Mirror/examples/realistic/`. 1 scene (1 view) is an internal RoomAI smartphone photo (`artifacts/sample_room.jpg`).
* **Depth Maps:** Generated entirely by inference from **Depth Anything V2 Metric Indoor Small** (`checkpoints/depth_anything_v2_metric_indoor_small`). **Classification: Model Pseudo-Labels**.
* **Surface Normals:** Mathematically derived via finite-difference tangent cross-products on the Depth Anything V2 pseudo-depth. **Classification: Derived Pseudo-Labels**.
* **Point Maps:** Unprojected from pseudo-depth using an assumed $54^\circ$ pinhole camera model. **Classification: Derived Pseudo-Geometry**.
* **Camera Poses:** Computed via an arbitrary synthetic circular trajectory formula ($R_y(0.05 \cdot v), t_x = 0.1 \cdot v$). **Classification: Synthetic Pseudo-Poses**.

**Scientific Conclusion:** Training on this dataset constitutes **pseudo-label distillation**, NOT physical ground-truth supervision.

---

## 2. Complete Provenance Table (All 11 Scenes, 39 Views)

| Scene ID | Split | Views | Original Image Source | Depth Source & Type | Normal Source & Type | Camera Intrinsics Source | Camera Pose Source | Visual Overlap (ORB Matches) |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bright_Room** | Val | 2 | `examples/realistic/Bright_Room/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Genuine video overlap (220 good matches) |
| **Desk** | Train | 2 | `examples/realistic/Desk/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Wide baseline (33 matches) |
| **Dining_Table** | Train | 4 | `examples/realistic/Dining_Table/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Genuine video panning (228 good matches) |
| **Hall** | Train | 2 | `examples/realistic/Hall/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Genuine video corridor (397 good matches) |
| **Lego_Kitchen** | Train | 10 | `examples/realistic/Lego_Kitchen/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Dense orbit sequence (37-251 matches) |
| **Living_Room** | Train | 2 | `examples/realistic/Living_Room/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Genuine video panning (249 good matches) |
| **Office** | Val | 1 | `examples/realistic/Office/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Single frame (Identity $I_{4\times 4}$) | **Single image (No multi-view baseline)** |
| **Room_Cat** | Train | 8 | `examples/realistic/Room_Cat/` (`no_overlap_*.jpg`) | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | **Disjoint views (Only 29 matches)** |
| **Rooms** | Train | 3 | `examples/realistic/Rooms/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Panning room sequence (118 matches) |
| **Workspace** | Train | 4 | `examples/realistic/Workspace/` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Synthetic ($0.05\cdot v$ rad) | Panning desk sequence (194 matches) |
| **Sample_Room** | Test | 1 | `artifacts/sample_room.jpg` | Depth Anything V2 (Pseudo-label) | Finite diff cross-product (Derived) | Assumed ($f=518$) | Single frame (Identity $I_{4\times 4}$) | **Single image (No multi-view baseline)** |

---

## 3. Deep Dive by Modality

### 3.1 Depth Provenance
* **Generating Code:** `prepare_indoor_dataset.py`, lines 150-160:
  ```python
  inputs = processor(images=img_518, return_tensors="pt").to(device)
  with torch.no_grad():
      d_out = depth_model(**inputs)
      pred_depth = d_out.predicted_depth.squeeze().cpu().numpy()
  ```
* **Base Checkpoint:** `checkpoints/depth_anything_v2_metric_indoor_small` (24.78M parameters, DINOv2-Small backbone).
* **Ground-Truth Classification:** **PSEUDO-LABEL ONLY**.
  * No active optical LiDAR was used.
  * No structured light sensor (e.g. Kinect/RealSense) was used.
  * No photogrammetric 3D mesh projection was used.
  * Depth Anything V2 can exhibit scale distortion, edge fattening, and planar warping on complex reflective surfaces.

### 3.2 Surface Normal Provenance
* **Generating Code:** `prepare_indoor_dataset.py`, lines 52-90:
  ```python
  tu[:, 1:-1] = (pts[:, 2:] - pts[:, :-2]) * 0.5
  tv[1:-1, :] = (pts[2:, :] - pts[:-2, :]) * 0.5
  normals = np.cross(tu, tv)
  normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
  ```
* **Ground-Truth Classification:** **DERIVED PSEUDO-LABEL**.
  * Normals were not captured by photometric stereo, surface scanning, or CAD models.
  * Computed via central finite differences on unprojected Depth Anything V2 depth.
  * Inherits high-frequency gradient noise from depth boundaries, causing a mean angular divergence of ~67° from neural surface normal predictors.

### 3.3 Point-Map Provenance
* **Generating Code:** Unprojected from depth via pinhole projection:
  $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = Z$$
* **Ground-Truth Classification:** **DERIVED PSEUDO-GEOMETRY**.

### 3.4 Camera Intrinsics and Extrinsics Provenance
* **Intrinsics:** Manually fixed to square pinhole camera matrix:
  $$K = \begin{bmatrix} 518 & 0 & 259 \\ 0 & 518 & 259 \\ 0 & 0 & 1 \end{bmatrix}$$
  * Assumes exactly $53.13^\circ$ horizontal Field of View.
  * Does not account for radial distortion ($k_1, k_2$), tangential distortion, or variable focal lengths of the capture devices.
* **Extrinsics:** Heuristic trajectory equation in `prepare_indoor_dataset.py`:
  $$\theta = 0.05 \cdot v, \quad P = \begin{bmatrix} \cos\theta & 0 & \sin\theta & 0.1\cdot v \\ 0 & 1 & 0 & 0 \\ -\sin\theta & 0 & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
  * **Classification:** **SYNTHETIC HEURISTIC POSE**.
  * Neither COLMAP nor visual SLAM was run to solve bundle adjustment.
  * The actual camera motions in the demo sequences were manual handheld video pans, NOT constant $2.86^\circ$ Y-axis rotations with $10\text{ cm}$ lateral translations.

---

## 4. Multi-View Validity Audit

1. **Genuinely Overlapping Multi-View Scenes (5 scenes):**
   * `Bright_Room` (220 good ORB matches)
   * `Dining_Table` (228 good ORB matches)
   * `Hall` (397 good ORB matches)
   * `Living_Room` (249 good ORB matches)
   * `Workspace` (194 good ORB matches)
   These sequences represent the same physical room, but lack calibrated camera poses.
2. **Disjoint / Wide-Baseline Scenes (2 scenes):**
   * `Desk` (33 matches) — wide angular shift across desk.
   * `Room_Cat` (29 matches) — original files named `no_overlap_*.jpg`, representing disjoint corners of a room with negligible overlap.
3. **Single-View Scenes (2 scenes):**
   * `Office` (1 view)
   * `Sample_Room` (1 view)
   These scenes cannot support multi-view evaluation or novel-view synthesis.

---

## 5. Audit Conclusion

The dataset provided a valid substrate for verifying **end-to-end execution of the multi-task training loop, optimizer convergence, and 3DGS splat rasterization**.

However, because the supervisory signals are pseudo-labels and the camera poses are synthetic heuristics:
1. **The dataset cannot validate physical geometric accuracy.**
2. **The dataset cannot validate novel-view radiance synthesis (3DGS generalization).**
3. **Stage 1 training must be formally designated as Pseudo-Label Distillation.**
4. **Stage 2 training must be formally designated as Proof-of-Concept Mechanics Verification.**

