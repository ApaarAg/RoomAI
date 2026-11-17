# ROOMAI MASTER TRAINING — PHASE 0: REPOSITORY AND CHECKPOINT AUDIT

**Author:** Principal 3D Vision Research Scientist, Deep Learning Engineer & ML Infrastructure Lead  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Phase:** Phase 00 (Repository & Checkpoint Audit)  
**Status:** COMPLETE & VERIFIED  

---

## 1. INFRASTRUCTURE & ENVIRONMENT AUDIT

### 1.1 Git Status & Source Tree
- **Active Branch:** `product-application`
- **Head Commit:** `8f4bf662eb73518866f516710b68a99c6b03f37a` (`feat: real YOLO11 object detection, COCO audit, benchmarks, and fine-tuning experiment`)
- **Worktrees:** 1 active worktree at `F:/RoomAI` (`product-application`)
- **Working Tree State:** Clean separation between production application services and research tree in `research/3d_reconstruction/`. Production application core remains untouched.
- **External Dependencies:**
  - `research/3d_reconstruction/external/HunyuanWorld-Mirror`
  - Submodule/Clone Git Commit: `c61300e47ce06db2f02c8e2551aae68be7ae3a0b`

### 1.2 Hardware & System Compute
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (AD107M)
  - Dedicated VRAM: 8,188 MiB (8.0 GB GDDR6)
  - Driver Version: `572.61` | CUDA API Support: `12.8`
  - Current Idle VRAM Allocation: ~359 MiB (7.8 GB available for training)
- **Host Memory (RAM):**
  - Total Physical RAM: 24,883,064 KB (~24.0 GB)
  - Free Physical Memory: 11,530,872 KB (~11.5 GB)
- **Storage Availability:**
  - Drive `F:\`: **359.93 GB Free** (89.29 GB used of 449.22 GB)
  - Drive `C:\`: 52.06 GB Free
  - Sufficient capacity for multi-gigabyte dataset expansion and checkpoint serialization.

### 1.3 Python & Deep Learning Toolchain
- **Active Conda Environment for 3D Research:** `roomai-hunyuan`
  - Interpreter: `C:\Users\apaar\miniconda3\envs\roomai-hunyuan\python.exe`
  - Python Version: `3.10.16`
  - PyTorch Version: `2.11.0+cu128`
  - CUDA Available: `True` (Device: `NVIDIA GeForce RTX 4060 Laptop GPU`)
  - CuDNN Enabled: `True`

---

## 2. MODEL ARCHITECTURE & CHECKPOINT LOADING VERIFICATION

### 2.1 WorldMirror-1.0 Architecture Configuration
The official WorldMirror model is instantiated via `src.models.models.worldmirror.WorldMirror`:
```python
model = WorldMirror(
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    gs_dim=256,
    enable_cam=True,
    enable_pts=True,
    enable_depth=True,
    enable_norm=True,
    enable_gs=False,
    enable_cond=False,
)
```
- **Total Trainable Parameters:** **1,223,250,748** (~1.22 Billion parameters).
- **Core Components:**
  - `visual_geometry_transformer`: Vision transformer backbone with camera and register tokens.
  - `depth_head`: Predicts metric/canonical scene depth.
  - `pts_head`: Predicts 3D point cloud maps.
  - `norm_head`: Predicts surface normal orientation vectors.
  - `cam_head`: Predicts camera rotation and translation.

### 2.2 Pretrained Checkpoint Loading Fidelity
- **Checkpoint File:** `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt`
- **File Size:** 2,839,184,102 bytes (2,707.66 MB)
- **SHA-256 Digest:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`
- **Parameter State Dict Audit:**
  - Total keys in checkpoint: **1,465**
  - Missing keys when loaded into `WorldMirror`: **0**
  - Unexpected keys when loaded into `WorldMirror`: **0**
- **Conclusion:** 100% parameter loading fidelity verified. The model loads with zero missing and zero unexpected weights.

---

## 3. CHECKPOINT INTEGRITY REGISTRY (SHA-256)

All seven checkpoints across the historical and experimental phases have been verified on disk:

| Checkpoint Identifier | Physical File Path | File Size (MB) | SHA-256 Digest | Status & Role |
| :--- | :--- | :---: | :---: | :--- |
| **`production_model.ckpt`** | `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt` | 2,707.66 MB | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` | **Immutable Baseline & Current Production Winner** |
| **`best_stage1.ckpt`** | `research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt` | 2,707.64 MB | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` | Archived (Phase 4/5 Pseudo-label Distillation) |
| **`stage2_real_verified.ckpt`** | `research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt` | 2,522.40 MB | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` | Archived (Phase 4/5 3DGS Ablation) |
| **`phase8_exp_a`** | `research/3d_reconstruction/experiments/phase8_corrected/experiment_a_hypersim/experiment_a_hypersim_final.ckpt` | 2,333.81 MB | `542c921e68e3a52d78c267c882ea5e44949557ac8166e3acd6caa475ffe56d3a` | Archived (Phase 8 Corrected Losses, Flawed Sampling) |
| **`phase8_exp_b`** | `research/3d_reconstruction/experiments/phase8_corrected/experiment_b_multiview/experiment_b_multiview_final.ckpt` | 2,333.81 MB | `0d27e4705ca411bb658bb991c6c91633d0922b57445eea45bdae12cb4a36a718` | Archived (Phase 8 Multi-view, Flawed Sampling) |
| **`phase9_exp_a`** | `research/3d_reconstruction/experiments/phase9/phase9_exp_a_head_only/phase9_exp_a_head_only_final.ckpt` | 2,333.81 MB | `e83704e3445f31bf339c26ecd478cdc2c73ad9934ad58b1634fb9a5fcd403921` | Evaluated & Rejected (Head-only Adaptation) |
| **`phase9_exp_b`** | `research/3d_reconstruction/experiments/phase9/phase9_exp_b_partial_backbone/phase9_exp_b_partial_backbone_final.ckpt` | 2,333.82 MB | `dfa592066eff02a58b28f8fffe5b47721559992b36cf9ec9ad8176732cb71792` | Evaluated & Rejected (Partial Backbone on Hypersim) |

All files are preserved and unmodified. The original pretrained checkpoint remains the authoritative reference winner.

---

## 4. REQUIRED PHASE 0 INVESTIGATIONS

### 4.1 Investigation A: Baseline Metric Disagreement Reconciliation

#### The Observation
In Phase 9 documentation, two different baseline evaluation numbers were reported for `production_model.ckpt`:
1. `PHASE9_PRETRAINED_BASELINE.md`: Hypersim Scaled AbsRel = **0.5469**, ScanNet Scaled AbsRel = **0.2730**
2. `PHASE9_TRAINING_EXPERIMENTS.md`: Hypersim Scaled AbsRel = **0.6317**, ScanNet Scaled AbsRel = **0.2667**

#### Root Cause Analysis
An algorithmic audit of `research/3d_reconstruction/scripts/evaluate_phase9_baseline.py` vs `research/3d_reconstruction/scripts/train_phase9_experiments.py` confirmed:
1. **Hypersim Evaluation Scope:**
   - In `train_phase9_experiments.py`, line 156 capped evaluation at `max_eval=50`. This evaluated only frames `0` to `49` of scene `ai_016_001`. Frames `0` to `49` depict an open corridor/atrium at long depth range (5m to 25m), resulting in an average Scaled AbsRel of **$0.6317$**.
   - In `evaluate_phase9_baseline.py`, all **100 frames** (`0` to `99`) were evaluated. Frames `50` to `99` navigate close to interior partition walls and furniture (1.5m to 7m), with a lower average Scaled AbsRel of **$0.4621$**.
   - The mathematical average across all 100 frames is:
     $$\text{Mean}_{\text{all 100}} = \frac{0.631737 + 0.462010}{2} = \mathbf{0.546874} \approx \mathbf{0.5469}$$
2. **ScanNet Evaluation Scope:**
   - In `train_phase9_experiments.py`, `val_scannet[:50]` sampled the first 50 frames, all from a single scene (`scene0030_00`), giving **$0.2667$**.
   - In `evaluate_phase9_baseline.py`, stratified sampling was conducted across 3 separate scenes (25 frames from `scene0030_00` [0.2707], 25 from `scene0030_01` [0.2860], and 25 from `scene0030_02` [0.2622]), yielding the true cross-scene average of **$0.2730$**.
3. **Canonical Protocol Defined:**
   Moving forward, all baseline and fine-tuned comparisons must strictly evaluate **complete validation sequences** with stratified multi-scene coverage, not truncated prefixes.

### 4.2 Investigation B: Experiment Dataset Coverage Audit

#### The Finding
In Phase 9, Experiment B was intended to test multi-dataset adaptation (Hypersim + ScanNet). However, an inspection of line 237 of `research/3d_reconstruction/scripts/train_phase9_experiments.py` revealed:
```python
train_dataset = ShuffledCanonicalDataset(MANIFEST_PATH, split="train", dataset_filter="hypersim")
```
- **Filter Hardcoded:** `dataset_filter="hypersim"` was hardcoded for both Experiment A and Experiment B.
- **ScanNet Exclusion:** All 173 ScanNet training frames were filtered out. ScanNet saw 0 training frames.
- **Actual Nature of Experiment B:** Experiment B was solely an ablation on unfreezing the final ViT transformer blocks (`frame_blocks[-1]` and `global_blocks[-1]`) on Hypersim data only.
- **Sample Trace:** Both Experiment A and B executed exactly 62 optimizer updates (31 batches per epoch $\times$ 2 epochs, batch size 1) over identical samples in identical order.

### 4.3 Investigation C: Sampling Bug Correction & Seed Determinism

#### The Test
We audited the shuffling implementation to verify whether `torch.Generator().manual_seed(seed)` provides deterministic, seed-distinct reproducibility across epochs and runs.
- **Execution Output:**
  ```
  Train frames count in V2 manifest: 204
  Determinism test passed! Seed 42 matches itself identically, seed 43 differs.
  First 5 items with seed 42: ['ai_047_008_95', 'scene0015_00_1220', 'ai_015_007_66', 'scene0019_01_600', 'ai_047_008_87']
  First 5 items with seed 43: ['scene0015_00_500', 'scene0019_01_416', 'scene0019_01_192', 'scene0019_00_140', 'scene0015_00_1160']
  ```
- **Result:** The sampling mechanism is strictly deterministic per seed and properly shuffles across datasets when `dataset_filter=None`.

### 4.4 Investigation D: Checkpoint Integrity Confirmation

All 7 checkpoints were verified using SHA-256 hashes against previous experiment ledgers. 
- No bytes have been modified or overwritten.
- `production_model.ckpt` hash `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` is verified identical to its original arrival digest.

---

## 5. RECONCILED CANONICAL BASELINE METRICS

These certified figures serve as the canonical reference for all fine-tuning evaluations:

| Evaluation Track | Primary Metric | Baseline Value | Secondary Metric | Baseline Value |
| :--- | :--- | :---: | :--- | :---: |
| **Track A: Synthetic (Hypersim held-out `ai_016_001`, 100 frames)** | Scaled AbsRel | **0.5469 ± 0.318** | Point Cloud Sim(3) MAE | **3.2401 m** |
| | Scaled RMSE | **4.6727 m** | Surface Normal MAE | **61.53°** |
| | $\delta < 1.25$ Acc | **29.26%** | Failure Rate | **0.00%** |
| **Track B: Real-World (ScanNet held-out 3 scenes, 75 frames)** | Scaled AbsRel | **0.2730 ± 0.100** | Point Cloud Sim(3) MAE | **0.4633 m** |
| | Scaled RMSE | **0.6618 m** | Surface Normal MAE | **62.94°** |
| | $\delta < 1.25$ Acc | **59.59%** | Failure Rate | **0.00%** |
| **Inference Hardware Profile** | Latency (ms) | **377.8 ms** (2.65 FPS) | Peak VRAM | **3,379.8 MB** (41.3% of 8 GB) |

---

## 6. PHASE 0 CONCLUSION & READINESS FOR PHASE 1

Phase 0 is complete with all scientific questions answered, data integrity confirmed, hardware verified, and baseline figures standardized. We now proceed directly to **Phase 1: Large-Scale Dataset Acquisition**.

