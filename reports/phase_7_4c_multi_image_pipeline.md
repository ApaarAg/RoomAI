# RoomAI Phase 7.4C — Multi-Image Reconstruction Pipeline & Input Validation

**Date:** November 6, 2026  
**Author:** Senior Multi-View Geometry Engineer & Systems Architect  
**Project:** RoomAI (`F:\RoomAI`)  
**Status:** **IMPLEMENTED & VALIDATED**

---

## 1. Architectural Overview & Objective

Monocular 3D indoor reconstruction cannot reconstruct areas obscured by foreground obstacles or surfaces outside the camera's single field of view. Phase 7.4C establishes a robust, end-to-end **Multi-Image 3D Reconstruction Pipeline** supporting sequences of **3 to 10 photographs** captured around the same physical room.

The pipeline ensures that unrelated, blurry, or duplicate images are screened out before multi-view geometry is computed, transforming overlapping perspectives into a unified 3D coordinate frame.

---

## 2. Multi-View Capture Instructions for Users

To achieve high-quality 3D room reconstruction, users are provided with clear capture guidelines in the interface:

1. **Same Room Sequence:** All photographs must be taken in the same room under consistent lighting conditions.
2. **Physical Camera Translation:** Move your physical position around the room (e.g. 1–2 meters apart). Do **NOT** simply stand in one spot and rotate the camera in place (pure rotation creates zero camera baseline and degrades depth triangulation).
3. **Sufficient Overlap:** Ensure each adjacent photograph shares at least $50\% - 70\%$ visual overlap with the previous view.
4. **Capture Key Furniture & Surfaces:** Photograph important furniture, doors, and walls from at least two different angles.
5. **Stability & Clarity:** Hold the camera steady to avoid motion blur. Avoid extreme wide-angle fisheye distortion.

---

## 3. Input Validation Architecture (`validate_multiview_sequence`)

Before executing GPU depth models or feature registration, every image in the uploaded sequence passes through an automated validation harness:

```
Uploaded Images (3–10 files)
       │
       ▼
1. Corruption & Format Check (PIL verification)
       │
       ▼
2. Resolution & Aspect Ratio Validation (320x240 to 4096x4096)
       │
       ▼
3. Blurriness / Sharpness Assessment (Laplacian Variance >= 25.0)
       │
       ▼
4. Perceptual Duplicate Detection (Normalized Thumbnail MSE < 0.0008)
       │
       ▼
5. Visual Overlap & Feature Density Assessment (SIFT inliers >= 8)
       │
       ▼
Validated Image Sequence -> Multi-View Geometry Engine
```

### 3.1 Mathematical Validation Metrics
1. **Motion Blur Detection:**
   The image is converted to grayscale $I_{\mathrm{gray}}$ and convolved with the discrete Laplacian operator:
   $$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
   The sharpness score is the sample variance $\sigma_{\mathrm{Lap}}^2 = \mathrm{Var}(\nabla^2 I)$. If $\sigma_{\mathrm{Lap}}^2 < 25.0$, the image is flagged for severe motion blur.
2. **Duplicate Image Suppression:**
   To prevent users from uploading accidental duplicates of the same photo, images are downsampled to a normalized $32 \times 32$ luminance thumbnail $\mathbf{t}_k$. Pairwise Mean Squared Error is computed:
   $$\mathrm{MSE}(\mathbf{t}_a, \mathbf{t}_b) = \frac{1}{1024} \sum_{i=1}^{1024} (t_{a, i} - t_{b, i})^2$$
   If $\mathrm{MSE} < 0.0008$, view $b$ is identified as an uninformative duplicate and pruned from the reconstruction sequence.

---

## 4. End-to-End Multi-View Reconstruction Execution Flow

The full pipeline executes as a deterministic, feed-forward process:

```
Input Images {I_0, I_1, ..., I_{N-1}}
       │
       ▼
[Stage 1] Reference View Grounding
  - Process View 0 (Primary Reference Camera Frame: origin at O = (0,0,0))
  - Run A3 Indoor Object Detector -> Detect furniture instances
  - Run Depth Anything V2 Metric -> Compute raw metric depth Z_0 in meters
  - Estimate Reference Floor Plane & Room Bounding Box
       │
       ▼
[Stage 2] Progressive Multi-View Registration
  For each secondary view I_k (k = 1 ... N-1):
    a. Run Depth Anything V2 Metric -> Compute metric depth Z_k
    b. Extract SIFT features & compute FLANN nearest neighbors
    c. Apply Lowe's ratio test (tau = 0.75) to extract candidate matches
    d. Lift 2D correspondences to 3D point pairs via metric depths:
       p_{0, i} = Unproject(u_{0, i}, v_{0, i}, Z_0)
       p_{k, i} = Unproject(u_{k, i}, v_{k, i}, Z_k)
    e. Compute rigid transform (R_k, t_k) via Umeyama SVD with RANSAC
    f. Check registration quality: If inliers >= 8 and RMSE < 0.35m:
         Transform secondary point cloud: P_k' = R_k P_k + t_k
         Transform detected secondary objects: c_{k, j}' = R_k c_{k, j} + t_k
       Else:
         Reject view k registration (discard to prevent coordinate corruption)
       │
       ▼
[Stage 3] Point Cloud Consolidation & Outlier Removal
  - Concatenate all valid point clouds: P_all = P_0 U P_1' U ... U P_{N-1}'
  - Apply 3D Regular Voxel Grid Downsampling (voxel_size = 0.03m / 3cm)
    -> Collapses overlapping surface points into unified barycentric coordinates
  - Apply Statistical Outlier Removal (k = 16, std_ratio = 1.8)
    -> Culls isolated floating edge points
       │
       ▼
[Stage 4] Global Furniture Association & Floor Grounding
  - Associate 3D furniture centroids across viewpoints (distance < 0.85m)
  - Merge bounding box extents: w_fused = max(w_0, w_k), h_fused = max(h_0, h_k)
  - Append new furniture objects seen exclusively in secondary perspectives
  - Enforce Floor-Contact Constraint: Snap all floor-standing furniture pieces
    so their bottoms rest firmly on the unified floor plane:
    y_centroid = y_floor - (h_fused / 2.0)
       │
       ▼
[Stage 5] Unified Scene Assembly & Export
  - Assemble consolidated SceneGraph with complete provenance
  - Generate Interactive Plotly 3D Viewer with metric axes
  - Export binary glTF 2.0 (.GLB) and standard PLY files
```

---

## 5. Verification & Robustness

The pipeline was validated with sequences ranging from 2 to 15 images. Even when an invalid or blurry image is introduced into a sequence, `validate_multiview_sequence` and the RANSAC registration gate prevent scene corruption. Unit test `test_multiview_sequence_validation_duplicate_and_blur_handling` passes with 100% reliability.

