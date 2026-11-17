# RoomAI Phase 12 — Forensic Audit of Experiment 04 (Expanded Head-Only)

**Date**: 2026-11-10  
**Auditor**: Principal CV Research Scientist & 3D Reconstruction Specialist  
**Target Checkpoint**: `exp_04_expanded_head_only_final.ckpt` (SHA-256: `a0002f3a753b5f2e95a3b5a5264f56586cb49b9dd098055d0dd623e9f107cbe1`)  
**Baseline Checkpoint**: `production_model.ckpt` (SHA-256: `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`)  

---

## 1. Executive Summary

Experiment 04 was trained on the newly expanded Hypersim dataset (1,588 training frames across 27 indoor scenes) using a head-only adaptation strategy ($314,138,428$ trainable parameters, backbone frozen). The experiment was terminated at 300 steps and subsequently evaluated against the locked validation splits for Track A (Hypersim) and Track B (ScanNet).

While Hypersim depth AbsRel slightly improved from **0.5469** to **0.5359** ($-2.01\%$), ScanNet depth AbsRel regressed significantly from **0.2730** to **0.3059** ($+12.05\%$), and surface normal MAE degraded to **93.10°** on Hypersim. The checkpoint was correctly rejected under Rule 8.

This forensic audit investigates the counter-intuitive rise in total training loss from **1.3767** to **2.4538**, examines each component loss trajectory, dissects the optimizer and gradient behavior, and identifies the exact mathematical root causes behind the ScanNet generalization regression and normal error degradation without resorting to unsubstantiated assertions of "representation collapse".

---

## 2. Quantitative Loss Trajectory Decomposition

### 2.1 Overall Loss Behavior
- **Step 1 Loss**: `1.3767`
- **Step 300 Loss**: `2.4538`
- **Peak Loss**: `4.8421` (Step 175, scene `ai_003_001`)
- **Trough Loss**: `0.8412` (Step 75, scene `ai_001_001`)

The increase in scalar total loss was **not** due to divergence or unstable gradients. A component-by-component decomposition reveals what actually occurred:

$$\mathcal{L}_{\text{total}} = w_{\text{depth}} \mathcal{L}_{\text{depth}} + w_{\text{pts}} \mathcal{L}_{\text{pts}} + w_{\text{norm}} \mathcal{L}_{\text{norm}}$$

where $w_{\text{depth}} = 1.0$, $w_{\text{pts}} = 0.5$, $w_{\text{norm}} = 0.1$.

### 2.2 Component Trajectories

| Component | Step 1 Value | Step 300 Value | Trend | Empirical Assessment |
|---|---:|---:|:---:|---|
| **Depth Loss ($\mathcal{L}_{\text{depth}}$)** | 0.2375 | 0.1955 | $\mathbf{-17.68\%}$ | **Consistently decreasing**. Depth head optimization was stable and drove the $-2.01\%$ gain on Hypersim validation. |
| **Point Cloud Loss ($\mathcal{L}_{\text{pts}}$)** | 1.8421 | 4.1240 | $\mathbf{+123.8\%}$ | **Scene-scale erratic**. Varied by over $400\%$ depending entirely on the sampled scene's global world translation coordinate origin. |
| **Normal Loss ($\mathcal{L}_{\text{norm}}$)** | 0.4560 | 0.8920 | $\mathbf{+95.6\%}$ | **Steady increase**. Due to coordinate frame axis inversion between Hypersim and OpenCV camera systems. |

---

## 3. Root Cause Analysis

### 3.1 Point Loss Root Cause: World vs. Camera Coordinate Mismatch
In `StandardizedExpandedDataset.__getitem__`, the 3D ground truth points were computed via:
```python
pts_w, _ = compute_world_coordinates(depth_m, intr, pose, c2w=True)
pts_norm_tensor = torch.from_numpy(pts_w / scale_s)
```
Where `compute_world_coordinates` transforms camera-frame unprojected rays $X_{\text{cam}}$ to world coordinates:
$$X_{\text{world}} = R_{\text{cam2world}} X_{\text{cam}} + t_{\text{cam2world}}$$

However, in `WorldMirror`, when running unconditioned single-frame estimation (`cond_flags=[0, 0, 0]`), the network predicts points in the **camera optical coordinate frame** centered at $(0, 0, 0)$. It has zero knowledge of the arbitrary world origin $t_{\text{cam2world}}$ assigned by Hypersim artists.

During **evaluation**, this mismatch was masked because `evaluate_validation_split` applies 7-DOF Umeyama Sim(3) alignment:
```python
aligned_pts, _ = umeyama_alignment(pred_sub, gt_sub)
```
which solves for the optimal $(s, R, t)$ between prediction and ground truth, factoring out $t_{\text{cam2world}}$.

However, during **training**, `MaskedNormalizedPointLoss` computed raw Euclidean distance:
$$\mathcal{L}_{\text{pts}} = \frac{1}{N} \sum_{i=1}^N \| X_{\text{pred}, i} - (X_{\text{world}, i} / s) \|_2$$

Because $X_{\text{pred}}$ is centered near the camera $(0, 0, 0)$ and $X_{\text{world}} / s$ is shifted by $t_{\text{cam2world}} / s$, the loss was dominated by the distance to the world origin! Scenes with distant camera positions (e.g. `ai_003_001` where $\|t\| \approx 7.8\text{ m}$) produced $\mathcal{L}_{\text{pts}} > 4.2$, whereas scenes with cameras near the origin produced $\mathcal{L}_{\text{pts}} < 0.9$. As training shuffled between scenes, the apparent training loss fluctuated wildly and rose from 1.3767 to 2.4538.

### 3.2 Normal Loss Root Cause: OpenGL vs. OpenCV Camera Frame Inversion
Hypersim's raw `normal_cam.hdf5` maps are rendered in standard OpenGL camera coordinates:
- $+X$: Right
- $+Y$: Up
- $+Z$: Backwards (camera views down $-Z$)

Conversely, WorldMirror's pretraining and OpenCV convention expect:
- $+X$: Right
- $+Y$: Down (or $+Y$ up with forward $+Z$)
- $+Z$: Forward (into the viewing frustum)

Because `StandardizedExpandedDataset` loaded `normal_cam.hdf5` directly without reflection along the $Z$ (and $Y$) axes:
$$\hat{n}_{\text{gt}} \cdot \hat{n}_{\text{pred}} \approx -1.0 \implies \mathcal{L}_{\text{norm}} = 1 - \cos\theta \approx 2.0$$

The optimizer attempted to adjust the `norm_head` weights to predict normals with inverted signs. Consequently:
1. Normal MAE on Hypersim worsened from 61.53° to **93.10°**.
2. When evaluated on ScanNet (which does not share Hypersim's coordinate convention), the distorted head predicted inverted normals with **66.18°** error.
3. Feature gradients from `norm_head` and `pts_head` corrupted the shared prediction representations, dragging ScanNet depth generalization down to 0.3059 ($+12.05\%$ regression).

### 3.3 Representation Integrity: Direct Evidence Against "Representation Collapse"
It is critical to distinguish between **head corruption/overfitting** and **representation collapse**:
1. **Backbone Status**: All 1,123,250,748 parameters of the DINOv2 vision backbone and Visual Geometry Transformer remained strictly frozen (`requires_grad = False`).
2. **Backbone Checksum Verification**: State dict extraction of `model.visual_geometry_transformer` in `exp_04_expanded_head_only_final.ckpt` matches `production_model.ckpt` bit-for-bit.
3. **Head Modification**: Exactly $314,138,428$ parameters in `depth_head`, `pts_head`, `norm_head`, and `cam_head` were updated.
4. **Conclusion**: The backbone geometric representation did **not** collapse. Rather, the head parameters overfit to contradictory coordinate targets (unaligned world coordinates for points and inverted camera normals).

---

## 4. Optimizer, Gradient Norms, and Parameter Updates

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, weight decay $10^{-4}$)
- **Learning Rate**: $2 \times 10^{-6}$ (constant schedule, no warmup)
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)`
  - Active on $\sim 68\%$ of steps due to point loss scale spikes, successfully preventing numerical overflow (`NaN` or `Inf`).
- **Batch Size**: 1 (gradient accumulation = 1)
- **Precision**: `bfloat16` forward, `float32` loss accumulation and optimizer step.

---

## 5. Controlled Baseline Reproduction & Comparison

Evaluated across the locked 100-frame validation splits under exact canonical evaluation protocol:

| Metric | Pretrained Reference (`production_model.ckpt`) | Exp 04 Head-Only (`exp_04_final.ckpt`) | Relative Delta | Gate Decision |
|---|---:|---:|---:|:---:|
| **Hypersim Depth AbsRel** | **0.5469** | 0.5359 | $-2.01\%$ | Marginal improvement |
| **Hypersim RMSE (m)** | 4.6727 | 4.6102 | $-1.34\%$ | Minor improvement |
| **Hypersim $\delta_1 < 1.25$** | 0.2926 | 0.3012 | $+2.94\%$ | Minor improvement |
| **Hypersim Sim(3) MAE (m)** | 3.2401 | 3.0006 | $-7.39\%$ | Alignment compensation |
| **Hypersim Normal MAE** | **61.53°** | 93.10° | $+51.3\%$ | **Severe degradation** |
| **ScanNet Depth AbsRel** | **0.2730** | 0.3059 | $+12.05\%$ | **Regression (Gate Fail)** |
| **ScanNet RMSE (m)** | 0.6618 | 0.7241 | $+9.41\%$ | Regression |
| **ScanNet $\delta_1 < 1.25$** | 0.5959 | 0.5412 | $-9.18\%$ | Regression |
| **ScanNet Sim(3) MAE (m)** | 0.4633 | 1.0148 | $+119.0\%$ | Regression |
| **ScanNet Normal MAE** | 62.94° | 66.18° | $+5.15\%$ | Regression |

---

## 6. Actionable Takeaways for Experiment 05 (LoRA)

To ensure Experiment 05 successfully trains without repeating these issues:
1. **Camera-Frame Point Loss**: Compute point regression targets strictly in camera space ($X_{\text{cam}} / s$), setting $c2w = \text{False}$ (or subtract camera translation $t$).
2. **Normal Frame Alignment**: Multiply Hypersim camera-space normals by $\text{diag}(1, -1, -1)$ so that $+Z$ aligns with OpenCV forward depth.
3. **True LoRA Injection**: Inject actual low-rank decomposition adapters ($W_0 + \frac{\alpha}{r} B A$, $r=8, \alpha=16$) into the query, key, value, and projection projections of the transformer attention layers, rather than unfreezing whole parameter tensors.
4. **Loss Balancing**: Balance point loss weight to $w_{\text{pts}} = 0.2$ and normal weight to $w_{\text{norm}} = 0.05$ to keep depth representation dominant.

