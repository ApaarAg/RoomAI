# ROOMAI — 3D RECONSTRUCTION FOUNDATION MODEL SELECTION REPORT

**Date:** 2026-11-07  
**Role:** Senior 3D Computer Vision Research Engineer & ML Systems Architect  
**Focus:** Comparative Investigation of Foundation Models for Indoor 3D Scene Reconstruction  

---

## 1. Executive Summary & Selection Matrix

To upgrade RoomAI from a heuristic monocular depth unprojection proxy to an authentic, view-consistent 3D indoor scene reconstruction system, four candidate architectures were systematically evaluated:

1. **HunyuanWorld-Mirror (WorldMirror)** — Tencent Hunyuan (Primary Candidate).
2. **VGGT (Visual Geometry Grounded Transformer)** — Oxford VGG & Meta AI (Baseline Candidate).
3. **DUSt3R / Mast3R** — NAVER LABS Europe (Pairwise Geometry Baseline).
4. **RoomAI Existing Pipeline** — YOLO11s + YOLO11n-Seg + Depth Anything V2 Metric + SIFT/Kabsch.

### Comprehensive Candidate Evaluation Matrix

| Evaluation Dimension | Existing RoomAI Pipeline | HunyuanWorld-Mirror | VGGT (Meta/Oxford) | DUSt3R / Mast3R |
| :--- | :--- | :--- | :--- | :--- |
| **Core Architecture** | Sequential Monocular + SIFT | Unified Feed-Forward ViT | Alternating Attention ViT | Asymmetric CroCo ViT |
| **Parameter Count** | ~40M (composite) | ~1.0 Billion | ~1.0 Billion | ~560M (DUSt3R) / 1.1B |
| **Input Views** | 1 to 10 (pairwise) | 1 to 24+ views or video | 1 to 100+ views | 2 views (pairwise) |
| **Multi-View Consistency** | Handcrafted SIFT + RANSAC | Learned Global Attention | Frame + Cross-View Attn | Pairwise Global Alignment |
| **Camera Pose Estimation** | None (fixed origin proxy) | **Learned 6-DoF ($R, \mathbf{t}$)** | **Learned 6-DoF ($R, \mathbf{t}$)** | Pairwise relative poses |
| **Camera Intrinsics** | Assumes $f_x=f_y=1000$ | **Learned Focal Lengths** | **Learned Focal Lengths** | Predicted focal length |
| **Point Cloud / Point Map** | Monocular unprojected proxy | **Dense 3D Point Map** | **Dense 3D Point Map** | Dense pairwise pointmaps |
| **Surface Normal Maps** | None | **Native Normal Head** | None | None (derived via diff) |
| **3D Representation** | Discrete point cloud proxy | **3D Gaussian Splats (3DGS)**| Point maps / tracks only | Point cloud / mesh proxy |
| **Prior Conditioning** | None | **Any-Prior Prompting** | None (image only) | None (image only) |
| **VRAM (FP16 Inference)** | **~368 MB** (Laptop safe) | **~4.5 – 6.5 GB** (Fits 8GB) | **~4.2 – 6.0 GB** (Fits 8GB) | ~5.0 – 7.0 GB |
| **VRAM (Full Training)** | N/A (Pretrained) | $\ge 24\text{ GB}$ (Cloud A100) | $\ge 24\text{ GB}$ (Cloud A100) | $\ge 24\text{ GB}$ |
| **Indoor Suitability** | Low on multi-view walls | **High** (surfaces, normals)| High (general geometry)| Medium (indoor scale drift)|

---

## 2. Deep Dive: HunyuanWorld-Mirror (Primary Candidate)

### 2.1 Architectural Anatomy (*WorldMirror*, arXiv:2510.10726)
HunyuanWorld-Mirror is designed as an all-in-one feed-forward transformer for universal 3D world reconstruction. Rather than relying on slow iterative Structure-from-Motion (SfM) or diffusion sampling loops, it executes dense 3D prediction in a single deterministic pass.

```
Multi-View Images [V, 3, H, W] ──┐
                                  ├──▶ Patchify & Embed ──┐
Optional Depth Prior [V, 1, H, W] ─┤                      │
Optional Poses / Intrinsics ──────┘                      ▼
                                            ┌─────────────────────────┐
                                            │ Unified Vision          │
                                            │ Transformer Backbone    │
                                            │ (Cross-View Attention)  │
                                            └─────────────────────────┘
                                                         │
         ┌───────────────────┬───────────────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                           ▼                   ▼                   ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐
│ Camera Head      ││ Metric Depth Head││ 3D Point Map Head││ Surface Normal   ││ 3DGS Head        │
│ Poses & FoV      ││ Multi-View Depth ││ $(X, Y, Z)$ Cloud││ Normals $(N_x,y,z)$│ Positions/Colors/│
│ $(R, \mathbf{t})$││ $[V, 1, H, W]$   ││ $[V, 3, H, W]$   ││ $[V, 3, H, W]$   │ Covariance/Opacity│
└──────────────────┘└──────────────────┘└──────────────────┘└──────────────────┘└──────────────────┘
```

### 2.2 Why Any-Prior Prompting is a Game-Changer for RoomAI
Unlike black-box multi-view transformers that ignore existing sensors or pre-calibrated models, HunyuanWorld-Mirror features **Any-Prior Prompting**:
- **Depth Prior Injection:** RoomAI already possesses a verified, calibrated monocular metric depth model (`Depth Anything V2 Metric Indoor Small`). In HunyuanWorld-Mirror, this metric depth map can be tokenized and passed as a spatial prior. The model refines and harmonizes the depth across multiple views rather than inventing scale from scratch.
- **Intrinsic Prior Injection:** If an iPhone or DSLR focal length is known, it can be passed directly as a prior token, locking the perspective geometry.

### 2.3 The 3D Gaussian Splatting (3DGS) Representation
Current RoomAI produces discrete, untextured or sparse point clouds (40,000 points) that appear hollow when viewed from oblique angles. HunyuanWorld-Mirror directly predicts 3D Gaussian primitives:
$$\mathcal{G}_i = \{\mathbf{\mu}_i \in \mathbb{R}^3, \mathbf{\Sigma}_i \in \mathbb{R}^{3 \times 3}, \alpha_i \in [0, 1], \mathbf{c}_i \in \mathbb{R}^k\}$$
This allows real-time, photo-consistent continuous rendering using fast WebGL Gaussian Splat viewers in the browser.

---

## 3. Deep Dive: VGGT (Meta / Oxford Baseline)

### 3.1 Architectural Anatomy (*facebook/VGGT-1B*)
VGGT (*Visual Geometry Grounded Transformer*) uses an alternating attention paradigm:
1. **Frame-wise Self-Attention:** Extracts dense spatial and semantic details within each image independently.
2. **Cross-View Cross-Attention:** Enforces multi-view parallax constraints, epipolar alignment, and global 3D scene coherence across all views simultaneously.

### 3.2 Key Outputs
- Camera extrinsic matrices $[R \mid \mathbf{t}]$ and intrinsic field of view.
- Dense 3D point maps in camera and world frames.
- Dense 3D point tracks across frames.

### 3.3 Limitations Relative to HunyuanWorld-Mirror for RoomAI
- **No Surface Normals:** Does not output surface normal maps, which are essential for RoomAI's architectural wall, ceiling, and floor boundary identification.
- **No 3DGS Head:** Outputs only 3D coordinate point maps. Photorealistic novel-view synthesis requires downstream post-optimization.
- **No Prior Prompting:** Cannot ingest external metric depth priors from Depth Anything V2.

---

## 4. Hardware Feasibility on RTX 4060 (8 GB VRAM)

### 4.1 Inference Memory Budget (Half Precision / BF16)
- **1B Weights:** $1.0 \times 10^9 \times 2\text{ bytes} \approx 2.05\text{ GB}$.
- **PyTorch / CUDA Overhead:** ~0.8 GB.
- **Activation Memory (4 views @ $512 \times 512$):** ~2.2 – 3.2 GB.
- **Total Estimated Inference VRAM:** **~5.1 – 6.1 GB**.
- **Headroom on 8.0 GB RTX 4060:** **~1.9 – 2.9 GB available**.
- **Verdict:** Feasible for local inference evaluation with view count $\le 6$ and resolution $512 \times 512$.

### 4.2 Training Memory Reality
- Full fine-tuning of a 1B parameter model with AdamW requires:
  - Weights (FP16): 2 GB
  - Gradients (FP16): 2 GB
  - Optimizer States (FP32 First/Second Moments): 8 GB
  - Activation Buffers for Backprop: 10–18 GB
  - **Total Training VRAM:** $\ge 24\text{ GB}$ to $32\text{ GB}$.
- **Verdict:** Full fine-tuning will cause an immediate Out-Of-Memory crash on the 8 GB RTX 4060.
- **Strategic Plan:**
  1. Use RTX 4060 for environment setup, architecture verification, data pipeline validation, smoke tests, and pretrained evaluation.
  2. Implement LoRA / PEFT adapters or head-only fine-tuning for lightweight local experiments.
  3. Formulate the full multi-node/cloud GPU training specification (NVIDIA A100-SXM4-80GB / H100) for full-scale fine-tuning.

---

## 5. Architectural Recommendation

1. **Adopt HunyuanWorld-Mirror as the Primary 3D Foundation Model Candidate** due to:
   - Any-Prior Prompting (direct synergy with RoomAI's Depth Anything V2 metric priors).
   - Surface normal prediction (enables geometric wall and floor extraction).
   - 3D Gaussian Splatting output (enables photorealistic continuous 3D room viewing).
2. **Benchmark against VGGT-1B as the Secondary Foundational Baseline**.
3. **Benchmark against Existing RoomAI Pipeline** to quantify multi-view consistency gains, point density improvements, and geometric error reductions.

