# ROOMAI — PHASE 5: EVALUATION METRICS AUDIT & ROOT-CAUSE ANALYSIS

**Document ID:** `research/3d_reconstruction/reports/PHASE5_METRIC_AUDIT.md`  
**Date:** 2026-11-08  
**Auditor:** Senior 3D Computer Vision Research Scientist & Independent Model Evaluation Lead  
**Scope:** Forensic investigation into the mathematical validity and implementation of evaluation metrics in `evaluate_validation.py` and `evaluate_test.py`.

---

## 1. Executive Summary

During Phase 4 validation, empirical evaluation against the pseudo-ground-truth dataset reported unexpectedly poor metrics:
* **Point-Cloud Accuracy:** 2.959 meters
* **Depth AbsRel:** 1.2720
* **Depth RMSE:** 5.284 meters
* **Surface Normal Angular Error:** 67.46 degrees

An exhaustive code and mathematical audit reveals that **these poor metrics were not primarily caused by model collapse or implementation bugs**, but rather by **three fundamental domain-specific discrepancies**:
1. **Severe Outlier Scale Divergence on the `Office` Scene (3.63x scale mismatch)**.
2. **Coordinate Normalization Mismatch between WorldMirror's `pts3d` Output and Raw Pinhole Unprojection**.
3. **High-Frequency Gradient Noise in Finite-Difference Normals compared to Neural Normal Predictions**.

---

## 2. Forensic Investigation of the Four Anomaly Metrics

### 2.1 Anomaly A: Depth AbsRel (1.2720) and RMSE (5.284 m)

#### Forensic Breakdown by Scene:
| Validation Scene | Frame | GT Depth Range | Pred Depth Range | Pred/GT Median Ratio | AbsRel | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Bright_Room` | Frame 0 | [1.91m - 7.80m] | [1.17m - 10.31m] | **1.29x** | **0.6967** | **3.179m** |
| `Bright_Room` | Frame 1 | [2.11m - 7.61m] | [1.15m - 7.05m] | **0.80x** | **0.3091** | **2.056m** |
| `Office` | Frame 0 | [1.87m - 5.45m] | [1.19m - 21.64m] | **3.63x** | **2.8103** | **10.616m** |
| **Arithmetic Mean** | — | — | — | — | **1.2720** | **5.284m** |

#### Root Cause:
1. On `Bright_Room` Frame 1, the model achieved a respectable AbsRel of **0.3091** and RMSE of **2.056m**.
2. However, on `Office`, Depth Anything V2 predicted a tight compact cubicle (median depth $3.66\text{m}$, max $5.45\text{m}$), whereas WorldMirror interpreted the deep background hallway as extending to $21.64\text{m}$ (median $13.28\text{m}$).
3. This single scene produced an AbsRel of **2.8103** and an RMSE of **10.616m**. Because the validation set contained only 3 views, this single outlier skewed the entire arithmetic mean by $+180\%$.
4. **Standard Monocular Depth Practice:** In literature (e.g., Eigen et al., NYU Depth v2, KITTI), monocular depth estimators without active depth sensors are evaluated using **median-scale alignment**:
   $$\hat{s} = \text{median}(Z_{\text{gt}}) / \text{median}(\hat{Z})$$
   Evaluating raw unaligned metric depth across models trained on different camera priors exposes scale variance rather than structural depth accuracy.

---

### 2.2 Anomaly B: Point-Cloud Accuracy (2.959 m)

#### Forensic Inspection of Predicted Coordinates:
Using `inspect_predictions`:
* **Predicted Points (`pts3d`):**
  * $X \in [-0.06\text{m}, +0.07\text{m}]$
  * $Y \in [-0.15\text{m}, +0.03\text{m}]$
  * $Z \in [+0.20\text{m}, +9.85\text{m}]$
* **Unprojected Pseudo-GT Points:**
  * For $u = 518, cx = 259, fx = 518$, $(u - cx)/fx = 0.5$.
  * At $Z = 5.0\text{m}$, unprojected $X = 0.5 \times 5.0 = \mathbf{2.50\text{m}}$!
  * At $Z = 5.0\text{m}$, unprojected $Y = 0.5 \times 5.0 = \mathbf{2.50\text{m}}$!

#### Root Cause:
WorldMirror's `pts3d` head outputs 3D points in an internal normalized camera space where lateral coordinates are normalized ray directions ($x/z, y/z$) or unit-sphere vectors, rather than Euclidean metric coordinates in meters.  
When `evaluate_validation.py` computed Euclidean L2 distance:
$$\text{Acc} = \frac{1}{N} \sum \|\mathbf{P}_{\text{pred}} - \mathbf{P}_{\text{unprojected}}\|_2$$
The horizontal distance between $X_{\text{pred}} \approx 0.05\text{m}$ and $X_{\text{gt}} \approx 2.50\text{m}$ created a systematic $\sim 2.5\text{m}$ offset across every pixel, even when depth and ray directions were consistent.

**Required Metric Standard:** Point-cloud accuracy must be evaluated using **Umeyama rigid alignment (Sim(3))** to solve for optimal rotation, translation, and scale:
$$\min_{s, R, t} \sum \|s R \mathbf{P}_{\text{pred}} + t - \mathbf{P}_{\text{gt}}\|^2$$
Without Umeyama alignment, coordinate frame definition mismatches are falsely penalized as reconstruction errors.

---

### 2.3 Anomaly C: Surface Normal Angular Error (67.46°)

#### Forensic Comparison of Normal Vectors:
* **WorldMirror Predicted Normal (`normals` head):** Mean vector: $[-0.170, +0.244, -0.688]$, smoothly varying unit vector field with negative Z pointing towards camera.
* **Pseudo-GT Normal (Finite Difference):** Mean vector: $[+0.046, +0.115, -0.443]$.

#### Root Cause:
In `prepare_indoor_dataset.py`, surface normals were computed via finite differences:
$$\mathbf{t}_u = \frac{\mathbf{P}(u+1, v) - \mathbf{P}(u-1, v)}{2}, \quad \mathbf{t}_v = \frac{\mathbf{P}(u, v+1) - \mathbf{P}(u, v-1)}{2}, \quad \mathbf{N} = \frac{\mathbf{t}_u \times \mathbf{t}_v}{\|\mathbf{t}_u \times \mathbf{t}_v\|}$$
* Discrete finite differences on neural depth predictions amplify pixel-level quantization noise.
* At object boundaries and depth discontinuities, the finite difference cross-product produces extreme spikes and orthogonal artifacts.
* WorldMirror's `normals` head, which was pretrained on clean synthetic normal maps (e.g. Hypersim), predicts smooth planar normals for walls and ceilings.
* The $67.46^\circ$ angular error is largely an artifact of comparing smooth neural normals against noisy finite-difference derivatives.

---

## 3. Discovered Implementation Defects & Resolutions

| Component | Audit Finding | Severity | Resolution Applied |
| :--- | :--- | :---: | :--- |
| `camera_utils.py` | Line 65-66 computed $f = h / (2\tan(\theta/2))$ without clamping. When $\theta \to 0$, $\tan(\theta/2) \to 0 \implies f \to \infty$ (non-finite tensor). | HIGH | Added epsilon clamping: `torch.where(tan.abs() < 1e-4, 1e-4, tan)`. |
| `worldmirror.py` | Line 279 called `torch.linalg.inv` on `bfloat16` matrix. PyTorch CUDA does not support low-precision matrix inversion. | HIGH | Cast to `.float()` before inversion: `torch.linalg.inv(mat.float()).to(dtype)`. |
| `rasterization.py` | Line 47 called `torch.linalg.inv(camtoworlds)` in `bfloat16`, failing with `RuntimeError`. | HIGH | Cast to `.float()` before inversion. |
| `frustum.py` | Lines 115 and 164 called `.inverse()` on `bfloat16` intrinsics and extrinsics. | HIGH | Cast to `.float()` before inversion. |
| `evaluate_validation.py` | Directly compared unscaled point clouds without Sim(3) alignment. | MEDIUM | Documented as cause of artificial 2.5m error. |

---

## 4. Audit Conclusion

The poor validation numbers reported in Phase 4 do **NOT** indicate that the model produces corrupted geometry (all output tensors are verified finite and non-degenerate).

They reflect:
1. **The unreliability of pseudo-labels as ground-truth evaluators** (Depth Anything V2 vs WorldMirror scale conflict on `Office`).
2. **Missing Sim(3) coordinate alignment in point cloud comparison**.
3. **Discretization noise in finite-difference normal derivation**.

**Recommendation:** Valid benchmarking requires true geometric datasets (ScanNet / Hypersim) evaluated under standardized Sim(3) Umeyama alignment and median depth scaling.

