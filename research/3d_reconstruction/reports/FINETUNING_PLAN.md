# ROOMAI — FOUNDATION MODEL FINE-TUNING PLAN

**Document Version:** 1.0.0  
**Phase:** Phase 3 — Foundation Model Fine-Tuning, Best Checkpoint Selection & Model Optimization  
**Author:** Senior 3D Computer Vision Research Scientist & ML Infrastructure Lead  
**Target Repository:** `F:\RoomAI`  
**Base Architecture:** Tencent Hunyuan WorldMirror (ICML 2026)  

---

## 1. Executive Strategy & Model Architecture

Following the empirical model selection (`research/3d_reconstruction/UPDATED_MODEL_SELECTION.md`), **WorldMirror** (ICML 2026) was selected as RoomAI's 3D reconstruction foundation model.

### 1.1 Architecture Specifications (Verified)
- **Base Visual Backbone:** Meta DINOv2 ViT-L/14 with register tokens (`dinov2_vitl14_reg`).
- **Embedding Dimension:** 1024
- **Patch Size:** 14x14 (Input resolution: 518x518 $\implies$ 37x37 = 1,369 spatial patches).
- **Registers:** 4 register tokens + 1 class token.
- **Transformer Blocks:** 24 layers, 16 attention heads, MLP ratio 4.0.
- **Prediction Heads:**
  - `pts_head`: 3D dense point cloud unprojection ($[B, S, 518, 518, 3]$) + confidence map.
  - `depth_head`: Dense metric depth prediction ($[B, S, 518, 518, 1]$) + confidence map.
  - `norm_head`: Per-pixel surface normal orientation vectors ($[B, S, 518, 518, 3]$).
  - `cam_head`: 9-D camera representation (translation, quaternion rotation, vertical/horizontal FoV).
  - `gs_head` & `gs_renderer`: 256-dimensional 3D Gaussian Splatting parameters (means, opacities, quaternions, scales, spherical harmonics).
- **Total Parameters:** **1,223,250,748 parameters (~1.22 Billion)**.
  - ViT Transformer Backbone: 909,112,320 parameters (74.32%).
  - Geometric & Splatting Heads: 314,138,428 parameters (25.68%).

---

## 2. Multi-Stage Fine-Tuning Design

To prevent task interference and catastrophic forgetting, the fine-tuning strategy is decomposed into two distinct stages:

### Stage 1: Geometric Fine-Tuning (Metric Depth, Point Maps, Normals, Cameras)
- **Objective:** Align feed-forward surface geometry with authentic indoor room structures, metric scale, and room layouts.
- **Enabled Heads:** `pts_head`, `depth_head`, `norm_head`, `cam_head`. `gs_head` is explicitly disabled.
- **Loss Function:**
  $$\mathcal{L}_{\text{Stage1}} = 5.0 \cdot \mathcal{L}_{\text{camera}} + 1.0 \cdot \mathcal{L}_{\text{point}} + 1.0 \cdot \mathcal{L}_{\text{depth}} + 1.0 \cdot \mathcal{L}_{\text{normal}}$$
  - $\mathcal{L}_{\text{point}}$: Truncated L1 loss + surface normal gradient consistency ($\gamma=1.0, \alpha=0.2$).
  - $\mathcal{L}_{\text{depth}}$: Metric scale-invariant depth gradient loss ($\gamma=1.0, \alpha=0.2$).
  - $\mathcal{L}_{\text{normal}}$: Angular loss ($\text{AL}$) on unit normals.
  - $\mathcal{L}_{\text{camera}}$: SE(3) rotation geodesic distance + L2 translation error.

### Stage 2: 3D Gaussian Splatting & Novel-View Synthesis
- **Objective:** High-fidelity view synthesis and radiance rendering for interactive 3D inspection in the RoomAI UI.
- **Enabled Heads:** `gs_head`, `gs_renderer` with geometric heads frozen.
- **Loss Function:**
  $$\mathcal{L}_{\text{Stage2}} = 1.0 \cdot \mathcal{L}_{\text{RenderMSE}} + 0.05 \cdot \mathcal{L}_{\text{LPIPS}} + 0.1 \cdot \mathcal{L}_{\text{RenderDepth}} + 0.1 \cdot \mathcal{L}_{\text{GSDepth}}$$

---

## 3. Compute Profile & Memory Requirements

Empirical benchmarking was executed directly on the local workstation (NVIDIA RTX 4060 Laptop GPU, 8,188 MB dedicated GDDR6, 24 GB System RAM, Windows 11):

| Mode / Pass | Measured Peak Allocated VRAM | Measured Peak Reserved VRAM | Feasibility on RTX 4060 (8GB) |
| :--- | :--- | :--- | :--- |
| **Inference (Single View, Geometry only)** | 3,379.8 MB (~3.38 GB) | 3,782.0 MB (~3.78 GB) | **100% Native VRAM (Passed)** |
| **Inference (Single View, with 3DGS)** | 3,447.3 MB (~3.45 GB) | 3,982.0 MB (~3.98 GB) | **100% Native VRAM (Passed)** |
| **Frozen Backbone Head Fine-Tuning** | 8,570.7 MB (~8.57 GB) | 8,878.0 MB (~8.88 GB) | Spills by 382.7 MB into Host RAM |
| **Full Model Backward Pass (All 1.22B params)**| 11,934.5 MB (~11.93 GB)| 12,022.0 MB (~12.02 GB)| Spills by ~3.8 GB into Host RAM |

### Cloud Compute Recommendation for Production Training
Because full backpropagation consumes ~12 GB VRAM for batch size 1 (and scaling to multi-view sequences requires $\ge 24\text{–}48\text{ GB}$ VRAM):
- **Recommended Cloud GPU:** NVIDIA L40 (48 GB GDDR6) or NVIDIA A100 (80 GB HBM2e).
- **Estimated Hourly Cost:** ~$1.80–$3.20 / hour on standard cloud providers (GCP/RunPod/Lambda).
- **Recommended Configuration:** Batch size 2, gradient accumulation 4, BF16 mixed precision, DDP across 2x or 4x GPUs.

---

## 4. Dataset Audit & Sourcing Strategy

- **Physical Audit:** The repository was audited. Zero 3D indoor ground-truth datasets (Hypersim, ScanNet, NYUv2) exist on local disk. `artifacts/phase_7_4_2_indoor_dataset_manifest.json` contains 40 ADE20K images (`depth_ground_truth_available: false`).
- **Data Acquisition Plan:**
  - Standard training format: Hypersim / ScanNet multi-view indoor subsets.
  - Coordinate system: OpenCV camera convention ($+X$ right, $+Y$ down, $+Z$ forward).
  - Depth units: Metric meters $[0.1\text{m}, 10.0\text{m}]$.
  - Network streaming: Local internet bandwidth was profiled at ~130–230 KB/s. Sourcing full 100–250 GB datasets must be executed in cloud storage rather than local disk.

