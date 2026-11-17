# RoomAI Phase 7.4.1 — Multi-View Registration Mathematics & Precision Audit

**Audit Date:** November 6, 2026  
**Auditor:** Independent Senior Computer Vision Researcher & 3D Geometry Specialist  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Source Code Under Audit:**  
- `services/multiview_fusion_service.py` (functions `compute_rigid_transform_3d`, `register_pair`)  
- `reports/phase_7_4c_camera_registration_validation.md`  
- `reports/phase_7_4f_single_vs_multi_view_benchmark.md`

---

## 1. Mathematical Formulation Audit: SE(3) vs. Sim(3)

In Phase 7.4C and Phase 7.4F, the registration algorithm was described as an:
> *"Umeyama SVD rigid alignment in SE(3) with scale and rotation recovery."*

### Theoretical Distinction:
1. **$\mathrm{SE}(3)$ Rigid Transformation (Special Euclidean Group):**
   Transforms source points $p_{\mathrm{src}} \in \mathbb{R}^3$ to destination points $p_{\mathrm{dst}} \in \mathbb{R}^3$ via:
   $$p_{\mathrm{dst}} = R p_{\mathrm{src}} + t$$
   where $R \in \mathrm{SO}(3)$ (orthonormal rotation matrix with $\det(R) = +1$) and $t \in \mathbb{R}^3$ (translation vector). The scale is strictly isometric ($s \equiv 1.0$). This is the classical **Kabsch (1976)** algorithm.

2. **$\mathrm{Sim}(3)$ Similarity Transformation:**
   Estimates a 7-DoF transformation including an isotropic scale factor $s > 0$:
   $$p_{\mathrm{dst}} = s R p_{\mathrm{src}} + t$$
   This is the true **Umeyama (1991)** algorithm, which computes:
   $$s = \frac{1}{\sigma_{\mathrm{src}}^2} \mathrm{Tr}(D S)$$
   where $\sigma_{\mathrm{src}}^2$ is the variance of source points, and $S = \mathrm{diag}(1, \dots, 1, \det(U)\det(V))$.

### Forensic Code Inspection of `services/multiview_fusion_service.py`:

```python
def compute_rigid_transform_3d(pts_src: np.ndarray, pts_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    assert len(pts_src) >= 3 and len(pts_dst) >= 3, "At least 3 points required"

    centroid_src = np.mean(pts_src, axis=0)
    centroid_dst = np.mean(pts_dst, axis=0)

    p_src = pts_src - centroid_src
    p_dst = pts_dst - centroid_dst

    # Covariance matrix H = p_src^T @ p_dst
    h_matrix = p_src.T @ p_dst

    u, s, vt = np.linalg.svd(h_matrix)
    r_matrix = vt.T @ u.T

    # Reflection check (ensure det(R) == +1)
    if np.linalg.det(r_matrix) < 0:
        vt[2, :] *= -1
        r_matrix = vt.T @ u.T

    t_vec = centroid_dst - r_matrix @ centroid_src

    transformed_src = (r_matrix @ pts_src.T).T + t_vec
    rmse = float(np.sqrt(np.mean(np.sum((transformed_src - pts_dst) ** 2, axis=1))))

    return r_matrix, t_vec, rmse
```

### Audit Findings on Transformation Model:
1. **Strictly $\mathrm{SE}(3)$ Rigid Isometry:** The code implements the Kabsch formulation. There is **no scale factor $s$ computed or applied**.
2. **Naming Imprecision:** Calling this an "Umeyama algorithm" is technically imprecise because the core contribution of Umeyama's 1991 paper was the derivation of the optimal scale factor $s$ for $\mathrm{Sim}(3)$ estimation.
3. **Critical Vulnerability to Monocular Depth Scale Drift:** Monocular depth models (including Depth Anything V2) estimate depth up to an arbitrary or approximate metric scale. If View 1 estimates a room feature at $3.8\text{ m}$ and View 2 estimates the same feature at $4.3\text{ m}$ (a typical $13\%$ inter-frame scale divergence), an $\mathrm{SE}(3)$ rigid solver cannot reconcile this scale discrepancy. Rigid alignment will necessarily shear or fail, leaving ghost surfaces in the fused point cloud.

---

## 2. RANSAC Inlier Thresholding & RMSE Meaning

### Implementation Inspection:
Lines 317–337 of `services/multiview_fusion_service.py`:
```python
iterations = min(300, max(50, n_pairs * 5))
for _ in range(iterations):
    idx_sample = np.random.choice(n_pairs, 4, replace=False)
    r_cand, t_cand, _ = compute_rigid_transform_3d(pts2_arr[idx_sample], pts1_arr[idx_sample])

    # Evaluate inliers across all pairs with 0.25m distance threshold
    pred_pts1 = (r_cand @ pts2_arr.T).T + t_cand
    errors = np.linalg.norm(pred_pts1 - pts1_arr, axis=1)
    inliers = errors < 0.25
    n_inliers = int(np.sum(inliers))

    if n_inliers > best_inliers:
        best_inliers = n_inliers
        best_r = r_cand
        best_t = t_cand
        best_rmse = float(np.mean(errors[inliers])) if n_inliers > 0 else 999.0

# Refine on all inliers if substantial
if best_inliers >= 4:
    inliers_mask = np.linalg.norm(((best_r @ pts2_arr.T).T + best_t) - pts1_arr, axis=1) < 0.25
    best_r, best_t, best_rmse = compute_rigid_transform_3d(pts2_arr[inliers_mask], pts1_arr[inliers_mask])
```

### Critical Analysis of Reported Metrics:
1. **In-Sample Residual vs. Out-of-Sample Accuracy:**
   The reported metric:
   $$\mathrm{RMSE} = 0.110\text{ m}$$
   is **not** an independent validation error or camera trajectory error. It is simply the mean Euclidean residual of points that were **explicitly filtered** by the hard threshold:
   $$\| p_{\mathrm{pred}} - p_{\mathrm{target}} \| < 0.25\text{ m}$$
   By mathematical construction, any inlier set selected under a $0.25\text{ m}$ cutoff will have an in-sample residual between $0.05\text{ m}$ and $0.15\text{ m}$. Reporting this in-sample fit as evidence of "sub-15 cm physical precision" is a circular metric artifact.
2. **Absence of 2D Reprojection Error:**
   Standard multi-view geometry (e.g. Hartley & Zisserman) requires evaluating the reprojection error in image pixels:
   $$\epsilon_{\mathrm{reproj}} = \| \pi(K, R p + t) - u_{\mathrm{observed}} \|_2$$
   The current pipeline does not evaluate 2D reprojection error, relying entirely on 3D distance residuals computed from noisy unprojected depths.
3. **No Bundle Adjustment:**
   The registration is strictly pairwise and greedy. There is no global pose-graph optimization, loop closure, or non-linear bundle adjustment (Levenberg-Marquardt on joint 3D points and camera poses). Consequently, multi-view sequences beyond 2 views will accumulate drift.

---

## 3. Camera Intrinsics and Unprojection Physics

In `services/multiview_fusion_service.py`, lines 285–288:
```python
x1 = (u1 - cx1) * z1 / fx
y1 = (v1 - cy1) * z1 / fy
x2 = (u2 - cx2) * z2 / fx
y2 = (v2 - cy2) * z2 / fy
```
where $f_x = f_y = 1000.0$ and $c_x = 0.5 \cdot W, c_y = 0.5 \cdot H$.

### Geometrical Implications:
- **Assumed Field of View (FOV):**
  For an image of resolution $683 \times 512$:
  $$\mathrm{FOV}_{\mathrm{horizontal}} = 2 \arctan\left(\frac{683}{2 \times 1000}\right) \approx 37.7^\circ$$
  $$\mathrm{FOV}_{\mathrm{vertical}} = 2 \arctan\left(\frac{512}{2 \times 1000}\right) \approx 28.7^\circ$$
  Standard smartphone cameras and wide-angle indoor lenses have a vertical FOV of $60^\circ - 75^\circ$ ($f \approx 400 - 550\text{ px}$ for a 512px image).
- Assuming $f = 1000\text{ px}$ compresses the unprojected horizontal and vertical dimensions by approximately a factor of $2\times$ relative to standard indoor camera optics.
- Because both views are unprojected with the exact same assumed $f$, pairwise registration can still find a rigid transformation between them, but the resulting physical dimensions $(X, Y)$ are severely distorted if the camera does not match $f = 1000\text{ px}$.

---

## 4. Summary of Registration Mathematical Soundness

| Mathematical Component | Implementation Status | Scientific Validity | Production Recommendation |
| :--- | :---: | :---: | :--- |
| **Rigid SE(3) Closed-Form** | Implemented (Kabsch SVD) | **VALID** for true rigid pairs | Suitable for single metric scale |
| **Sim(3) Scale Optimization** | Not implemented ($s \equiv 1.0$) | **DEFICIENT** for monocular depth | Must add scale factor optimization |
| **RANSAC Consensus Loop** | Implemented ($N \le 300$ samples) | **VALID** standard heuristic | Adequate for small baselines |
| **Reported 0.110m RMSE** | Inlier residual calculation | **MISLEADING INTERPRETATION** | Must document as inlier fit, not GT error |
| **Global Bundle Adjustment** | None (greedy sequential) | **ABSENT** | Limit multi-view chain to $\le 3$ views |
| **Camera Intrinsics Handling**| Hardcoded default ($f=1000$) | **HIGH UNCERTAINTY** | Expose FOV/focal length calibration in UI |

---

## 5. Auditor's Verdict

The pairwise rigid registration implementation in `services/multiview_fusion_service.py` is an algebraically correct Kabsch $\mathrm{SE}(3)$ solver. However, claiming sub-15 cm physical precision based on in-sample RANSAC residuals is mathematically invalid, and the lack of scale estimation ($\mathrm{Sim}(3)$) renders it fragile against monocular depth scale drift between different perspectives.

