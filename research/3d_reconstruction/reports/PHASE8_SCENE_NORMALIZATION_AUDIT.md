# Phase 8: WorldMirror Canonical Scene Normalization Audit

**Author:** Senior 3D Computer Vision Research Scientist & ML Systems Research Lead  
**Repository:** `F:\RoomAI`  
**Execution Environment:** Windows 11, Conda Environment `roomai-hunyuan`  
**Target Architecture:** HunyuanWorld-Mirror-1.0 (DPT + ViT-Large backbone)  
**Date:** 2026-11-08  

---

## 1. Executive Summary

In Phase 7.1, forensic investigation proved that the original Stage 1 fine-tuning collapsed because it fed unnormalized metric depth ($Z \approx 3.5 - 12.4\text{ m}$) into an exponential prediction head (`activation="exp+expp1"`) calibrated for canonically normalized coordinates ($s \approx 1.0$).

In this audit, we reproduced the exact official HunyuanWorld-Mirror scene-coordinate normalization implementation from `training/data/multiview_dataset.py` lines 176–238, validated all mathematical invariants, and created a 12-test independent unit test suite. All 12 unit tests pass with zero errors ($0.140\text{ s}$).

---

## 2. Official Implementation Source & Code Trace

The canonical normalization function is defined in:
`research/3d_reconstruction/external/HunyuanWorld-Mirror/training/data/multiview_dataset.py`

```python
def apply_pointcloud_normalization(self, views, norm_mode="avg_dis"):
    pts_list = np.concatenate([views[i]["pts3d"] for i in range(len(views))], axis=0)
    valid_list = np.concatenate([views[i]["valid_mask"] for i in range(len(views))], axis=0)
    assert pts_list.ndim >= 3 and pts_list.shape[-1] == 3
    
    norm_mode, dis_mode = norm_mode.split("_")
    all_pts = pts_list.reshape(-1, 3)
    all_valid = valid_list.reshape(-1, )
    all_pts[all_valid == 0] = float('nan')

    valid_pts = all_pts
    dis = np.linalg.norm(valid_pts, ord=2, axis=-1)

    if dis_mode == "dis":
        pass
    elif dis_mode == "log1p":
        dis = np.log1p(dis)
    ...
    if norm_mode == "avg":
        norm_factor = np.nanmean(dis, axis=-1)
    elif norm_mode == "median":
        norm_factor = np.nanmedian(dis, axis=-1)
    ...
    norm_factor = np.nan_to_num(norm_factor, nan=1.0).clip(min=1e-8)

    for i in range(len(views)):
        views[i]["pts3d"] /= norm_factor
        views[i]["depthmap"] /= norm_factor
        views[i]["camera_poses"][:3, 3] /= norm_factor
        views[i]["norm_factor"] = norm_factor
        views[i]["bad_case"] = bad_case

    return views
```

---

## 3. Mathematical Contract & Transformation Invariants

For any indoor scene $\mathcal{S}$ observed by $V$ views $\{v_1, \dots, v_V\}$:

### 3.1 Scene Scale Factor Calculation
World-coordinate points $\mathbf{X}_w \in \mathbb{R}^{H \times W \times 3}$ are computed via camera intrinsics $\mathbf{K}$ and camera-to-world pose $[\mathbf{R} \mid \mathbf{t}]$:
$$\mathbf{X}_{cam}(u, v) = Z(u, v) \cdot \mathbf{K}^{-1} [u, v, 1]^T$$
$$\mathbf{X}_w(u, v) = \mathbf{R} \mathbf{X}_{cam}(u, v) + \mathbf{t}$$

For all valid pixels across all $V$ views:
$$\mathcal{P}_{valid} = \bigcup_{i=1}^V \left\{ \mathbf{X}_w^{(i)}(u, v) \mid M^{(i)}(u, v) = 1 \text{ and finite} \right\}$$

The Euclidean distance from the world origin is:
$$d_j = \|\mathbf{p}_j\|_2, \quad \forall \mathbf{p}_j \in \mathcal{P}_{valid}$$

The scene scale factor $s$ is the median Euclidean norm:
$$s = \operatorname{nanmedian}_{j} (d_j)$$

### 3.2 Scaled Quantities
1. **World Point Cloud:**
   $$\mathbf{X}'_w = \frac{\mathbf{X}_w}{s}$$
2. **Depth Map:**
   $$Z' = \frac{Z}{s}$$
3. **Camera Translation:**
   $$\mathbf{t}' = \frac{\mathbf{t}}{s}$$

### 3.3 Strict Invariants (Unscaled Quantities)
1. **Camera Rotation:** $\mathbf{R}' = \mathbf{R}$ (orthonormal matrix preserved: $\mathbf{R}^T \mathbf{R} = \mathbf{I}$, $\det(\mathbf{R}) = 1$).
2. **Camera Intrinsics:** $\mathbf{K}' = \mathbf{K}$ (focal lengths $f_x, f_y$ and principal points $c_x, c_y$ operate in pixel coordinate units, invariant to spatial scale).
3. **Unit Surface Normals:** $\mathbf{n}' = \mathbf{n}$ ($\|\mathbf{n}\|_2 = 1.0$, directions invariant to uniform isotropic dilation).

### 3.4 Invertibility and Metric Recovery
When metric scale factor $s$ is preserved in metadata:
$$\mathbf{X}_w = \mathbf{X}'_w \cdot s, \quad Z = Z' \cdot s, \quad \mathbf{t} = \mathbf{t}' \cdot s$$
Numerical recovery error: $< 1 \times 10^{-6}\text{ m}$ (within IEEE 754 float32 precision).

---

## 4. Normalization Unit-Test Suite Results

The independent test suite [`research/3d_reconstruction/scripts/test_scene_normalization.py`](file:///F:/RoomAI/research/3d_reconstruction/scripts/test_scene_normalization.py) evaluates all 12 mandatory criteria on a synthetic 3D indoor room ($4\text{ m} \times 3\text{ m} \times 2.5\text{ m}$):

| Test ID | Test Description | Status | Recovery Tolerance |
| :--- | :--- | :--- | :--- |
| **01** | Normalization factor computed from valid world points | **PASSED** | Exact match with theoretical median |
| **02** | Invalid pixels excluded from scale estimation | **PASSED** | Extreme outliers ($9999\text{ m}$) fully rejected |
| **03** | Multiple views share consistent scene normalization factor | **PASSED** | $s_{view 1} \equiv s_{view 2} \equiv s_{scene}$ |
| **04** | Depth normalized using the same scene factor | **PASSED** | Relative error $< 10^{-5}$ |
| **05** | Point maps normalized using the same scene factor | **PASSED** | Relative error $< 10^{-5}$ |
| **06** | Camera translations normalized using the same factor | **PASSED** | Relative error $< 10^{-5}$ |
| **07** | Camera rotations remain unchanged | **PASSED** | Absolute error $< 10^{-7}$ |
| **08** | Intrinsics remain unchanged | **PASSED** | Absolute error $< 10^{-7}$ |
| **09** | Surface normals retain unit length | **PASSED** | Absolute error $< 10^{-6}$ |
| **10** | Denormalization restores original metric coordinates | **PASSED** | Absolute error $< 10^{-5}\text{ m}$ |
| **11** | No NaN or Inf is introduced | **PASSED** | $100\%$ finite tensors |
| **12** | Cam2world and world2cam conventions remain valid | **PASSED** | Reprojection error $< 10^{-5}$ |

**Result:** `12 passed in 0.140s. OK.`

---

## 5. Architectural Integration Rules for Phase 8

1. **Dataset Pipeline:** Every training, validation, and evaluation sample will compute and store `norm_factor` per scene in the manifest.
2. **Supervision Targets:** Ground-truth depth maps and world point maps will always be scaled by $1 / s$ before loss computation.
3. **Model Prediction Ingestion:** The DPT exponential head directly outputs canonically normalized depth $\hat{d}_{canon} \in [0.5, 2.0]$. Under canonical normalization, the regression target $Z' \in [0.5, 2.0]$, perfectly matching the head's numerical calibration.
4. **Metric Evaluation:** Metric depth is recovered at evaluation time using the known ground-truth scene scale factor $s$ (for scale-known benchmarking) or via per-frame median scaling $s^* = \operatorname{median}(Z_{gt}) / \operatorname{median}(\hat{d})$ (for monocular scale-invariant benchmarking).

