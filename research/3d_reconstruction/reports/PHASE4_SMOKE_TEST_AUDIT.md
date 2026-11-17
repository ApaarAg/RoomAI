# ROOMAI — PHASE 4: AUDIT OF PREVIOUS TRAINING SMOKE TEST

**Document ID:** `research/3d_reconstruction/reports/PHASE4_SMOKE_TEST_AUDIT.md`  
**Date:** 2026-11-08  
**Auditor:** Principal 3D Computer Vision Research Engineer & ML Infrastructure Lead  
**Scope:** Technical audit of `scripts/train_stage1.py`, `scripts/verify_checkpoint.py`, and `configs/finetune_stage1.yaml`.

---

## 1. Executive Summary

In Phase 3, a pre-training feasibility gate verified that the official **WorldMirror-1.0** architecture (~1.22 Billion parameters) can instantiate, compute forward representations, backpropagate gradients through geometric heads, execute AdamW optimizer parameter updates, and serialize/deserialize valid checkpoints on local hardware.

However, the Phase 3 smoke test operated exclusively with **synthetic Gaussian random tensors** (`torch.randn`). While mathematically sufficient to verify backward-pass mechanics, computation graph connectivity, and PyTorch optimizer execution, it **did not constitute genuine indoor reconstruction training**.

This audit details the verified capabilities, technical boundaries, and explicit limitations of the smoke test to establish the foundation for real indoor dataset training in Phase 4.

---

## 2. Technical Audit of Inspected Artifacts

### 2.1 `scripts/train_stage1.py`
- **Architecture Instantiation:**  
  Instantiates `WorldMirror` (`img_size=518`, `patch_size=14`, `embed_dim=1024`, `gs_dim=256`, `enable_cam=True`, `enable_pts=True`, `enable_depth=True`, `enable_norm=True`, `enable_gs=False`, `enable_cond=False`, `dpt_gradient_checkpoint=True`).  
  Total parameters: **1,223,250,748** (~1.223B).
- **Backbone Freezing Strategy:**  
  Freezes `visual_geometry_transformer` (ViT backbone, ~1.209B params) to protect local 8 GB physical VRAM.  
  Trainable parameters: **14,411,460** (~14.41M, 1.18% of total).
- **Synthetic Input Mechanics:**  
  Lines 52-55 generate synthetic random inputs:
  ```python
  views = {"img": torch.randn(1, 1, 3, 518, 518, device=device, dtype=torch.bfloat16)}
  gt_pts = torch.randn(1, 1, 518, 518, 3, device=device, dtype=torch.bfloat16)
  gt_depth = torch.randn(1, 1, 518, 518, 1, device=device, dtype=torch.bfloat16)
  gt_norm = torch.randn(1, 1, 518, 518, 3, device=device, dtype=torch.bfloat16)
  ```
  **Audit Finding:** The loss was computed against uncorrelated random noise. While gradients flowed and weights updated ($\Delta W = 8.404$), the network learned no room semantics or geometric priors.
- **Gradient & Optimizer Verification:**  
  Verified finite gradients (`torch.isfinite(p.grad).all()`), applied `clip_grad_norm_(max_norm=1.0)`, and executed `optimizer.step()`. All 1,271 gradient tensors remained finite.
- **Serialization & Reload Verification:**  
  Saved checkpoint `smoke_test_stage1.tmp.ckpt` (2,707.66 MB) and re-loaded into an evaluation session, proving round-trip serialization integrity.

### 2.2 `scripts/verify_checkpoint.py`
- **Integrity Validation:**  
  Computes SHA-256 via 16 MB chunked streaming, validates file size, and counts state-dict keys (1,271 parameter tensors).
- **Empirical Forward Pass:**  
  Instantiates the model architecture on GPU and verifies that the checkpoint weights produce finite prediction tensors for `pts3d`, `depth`, `normals`, and `camera_poses`.
- **Audit Finding:** The verification script is reliable and reusable for arbitrary `.ckpt` and `.safetensors` files.

### 2.3 `configs/finetune_stage1.yaml`
- **Loss Specification:**  
  Defines `LossContainer` with `CameraLoss`, `PointLoss` ($\gamma=1.0, \alpha=0.2$), `DepthLoss` ($\gamma=1.0, \alpha=0.2$), and `NormalLoss` (`AL` angular loss).  
  Weights: `[5.0, 1.0, 1.0, 1.0]`.
- **Target Dataloader Contract:**  
  Specifies `HyperSim_Multi` / `MultiViewDataset` compatibility: expects scenes with `rgb.png`, `depth.npy` (metric float32 in meters), and `cam.npz` (`intrinsics` $3\times3$, `pose` $4\times4$).
- **Hardware Warning:**  
  Explicitly documents the ~11.9 GB peak VRAM footprint of full-model backprop, confirming that local training on an 8 GB RTX 4060 must use backbone freezing or gradient accumulation with system RAM paging.

---

## 3. Discrepancies and Limitations of the Smoke Test

| Attribute | Smoke Test State | Required Phase 4 Real-Training State |
| :--- | :--- | :--- |
| **Input Images** | Gaussian noise (`torch.randn`) | Real photographic indoor room views (RGB, [0, 1]) |
| **Ground-Truth Depth** | Random floats (`torch.randn`) | Calibrated metric depth in meters ($Z > 0$) |
| **Ground-Truth Normals** | Unconstrained 3D vectors | Unit surface normals ($\|\mathbf{n}\| = 1$) in camera frame |
| **Camera Intrinsics** | Unconstrained random prior | Calibrated pinhole camera matrices ($f_x, f_y, c_x, c_y$) |
| **Camera Extrinsics** | Random poses | Rigid Euclidean SE(3) camera-to-world matrices |
| **Data Loader** | In-memory synthetic generator | Disk-backed structured multi-view dataset |
| **Loss Relevance** | Numerical mechanics check | Real geometric error on actual indoor surfaces |

---

## 4. Phase 4 Action Plan

1. **Step 2 (Data Preparation):** Create `research/3d_reconstruction/scripts/prepare_indoor_dataset.py` to extract 11 authentic indoor room scenes (54 total views) from `examples/realistic/` and `artifacts/sample_room.jpg`. Infer metric depth using `Depth Anything V2 Metric Indoor Small`, compute surface normals, format camera poses, and serialize to `HyperSim_Multi` format (`frame_xxxx_{rgb.png,depth.npy,cam.npz,normal.npy}`).
2. **Step 3 (Stage 1 Real Training):** Upgrade `train_stage1.py` with `--real_data` flag to load real indoor batches, compute genuine geometric multi-task loss, backpropagate, update weights, and save verified checkpoints.
3. **Step 4 (Stage 2 3DGS Verification):** Upgrade `train_stage2.py` to train 3D Gaussian Splatting heads on real indoor scenes.
4. **Step 5 (Validation & Verification):** Evaluate on held-out validation rooms, inspect 3D point cloud exports (`.ply`), and confirm finite bounding boxes and metric consistency.

