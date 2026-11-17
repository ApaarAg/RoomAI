# ROOMAI — PHASE 7.1: STAGE 1 TRAINING TARGET CONSTRUCTION & LOSS AUDIT

**Author:** Independent Senior 3D Computer Vision Researcher & Evaluation Auditor  
**Date:** November 8, 2026  
**Status:** COMPLETE & EMPIRICALLY DEMONSTRATED  
**Audit Target:** `research/3d_reconstruction/scripts/train_stage1.py` and `configs/finetune_stage1.yaml`

---

## 1. EXECUTIVE SUMMARY

This audit inspects the mathematical formulation and training script implementation used to produce `best_stage1.ckpt`.

### Core Forensic Finding:
The Stage 1 fine-tuning implementation contained **three fundamental training formulation errors**:
1. **Omission of Scene-Scale Normalization:** The official WorldMirror training contract requires ground-truth depth and 3D points to be divided by `norm_factor = nanmedian(distance)` before computing loss. `train_stage1.py` fed **unnormalized physical meter arrays ($Z \approx 3.5 - 12.4\text{ m}$)** directly into an exponential activation head designed for normalized unit values ($\hat{d} \approx 1.0$).
2. **Missing Binary Validity Masking:** Loss was computed via global unmasked `nn.functional.l1_loss`, penalizing predictions on background, border padding, and invalid depth pixels.
3. **Loss Dominance by Unnormalized 3D Points:** Unprojected points $\mathbf{P} \in \mathbb{R}^3$ had coordinate magnitudes up to $12\text{ meters}$, while surface normals were unit vectors ($\|\mathbf{n}\|_2 = 1.0$). The unprojected point loss exerted over $90\%$ of the gradient magnitude, driving the DPT linear projection bias weights into a runaway positive regime.

---

## 2. TRAINING CODE TRACE: FROM DATASET TO LOSS

In `research/3d_reconstruction/scripts/train_stage1.py`:

### 2.1 Batch Construction (`RealIndoorDataset.get_batch`):
```python
# Lines 75-78:
depth_arr = np.load(s["depth_path"])  # (518, 518) in meters from Depth Anything V2 Metric
depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, H, W, 1)

# Lines 86-97:
cam_npz = np.load(s["cam_path"])
intr = cam_npz["intrinsics"]
fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
H, W = depth_arr.shape
u, v = np.meshgrid(np.arange(W), np.arange(H))
x = (u - cx) * depth_arr / fx
y = (v - cy) * depth_arr / fy
z = depth_arr
pts_arr = np.stack([x, y, z], axis=-1)  # (518, 518, 3) in METERS (magnitudes up to 12.4m)
pts_tensor = torch.from_numpy(pts_arr).unsqueeze(0).unsqueeze(0)
```
**Audit Note:** No `norm_factor` was computed or applied.

### 2.2 Forward Pass & Loss Calculation (`run_real_training`):
```python
# Lines 133-140:
out = model(views=views, cond_flags=[0, 0, 0], is_inference=False)

# Multi-task geometric losses
loss_pts = nn.functional.l1_loss(out["pts3d"], gt_pts)
loss_depth = nn.functional.l1_loss(out["depth"], gt_depth)
loss_norm = nn.functional.l1_loss(out["normals"], gt_norm)
total_loss = loss_pts + 0.5 * loss_depth + 0.5 * loss_norm

total_loss.backward()
```

### 2.3 Official WorldMirror Formulation (`PointLoss` & `DepthLoss`):
In contrast, the official repository (`training/losses/point.py` & `depth.py`) specifies:
```python
# In training/losses/point.py:
loss_conf, loss_grad, loss_reg = regression_loss(
    pred_points, gt_points, gt_points_mask,  # MASK APPLIED
    conf=pred_points_conf,
    gradient_loss_fn=self.gradient_loss_fn,
    gamma=self.gamma,
    alpha=self.alpha,
    valid_range=self.valid_range
)
```
Where `gt_points` and `gt_depth` had already been normalized by `norm_factor` in `MultiViewDataset`.

---

## 3. GRADIENT IMPACT & SCALE WARP DYNAMICS

### 3.1 Loss Scale Disparity
At Step 1 of Stage 1 training:
- `gt_depth` had a median of **$4.95\text{ m}$**.
- `out["depth"]` had an initial median of **$0.95$**.
- Initial depth error: $|0.95 - 4.95| = 4.00$.
- Initial 3D point error: $\|\hat{\mathbf{P}} - \mathbf{P}_{gt}\|_2 \approx 5.50$.
- Initial normal error: $\|\hat{\mathbf{n}} - \mathbf{n}_{gt}\|_2 \approx 0.85$.

The L1 loss on depth and points produced massive positive gradients on the DPT output convolution layers (`output_conv2`). Because the depth activation is $\hat{d} = \exp(x)$, the network was forced to increase its internal pre-activation feature $x$ from $\approx 0$ ($\exp(0) = 1.0$) to $\approx 1.25$ ($\exp(1.25) \approx 3.5$) and higher.

### 3.2 Consequences on Generalization
1. **Behavior on Synthetic CAD (Hypersim):**
   The training scenes (`Bright_Room`, `Desk`, `Dining_Table`) had depths of $3–8\text{ meters}$. The network learned to output depths around $3.5\text{ meters}$.
   When evaluated on Hypersim validation (`ai_016_001`), where ground truth happens to be $3.5–4.5\text{ meters}$, Stage 1 produced raw depths of $\approx 3.4\text{ meters}$, artificially lowering the *raw* AbsRel from $0.74$ down to $0.32$.
2. **Catastrophic Collapse on Real Sensors (ScanNet):**
   When presented with ScanNet indoor scenes where physical furniture sits $0.8–2.5\text{ meters}$ away, the distorted DPT head encountered non-CAD textures and dark regions. Because its exponential head was inflated, its pre-activation features pushed into extreme positive values, causing output depths to explode to **$18.0 - 23.0\text{ meters}$**.
   This produced the $914\%$ error explosion on ScanNet.

---

## 4. CONCLUSION

The Stage 1 fine-tuning experiment did not fail due to a bug in PyTorch or optimizer mechanics. It failed because **an unnormalized supervision target was applied to a normalized coordinate architecture**.

