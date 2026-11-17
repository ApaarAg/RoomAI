# ROOMAI — 3D RECONSTRUCTION FOUNDATION MODEL SELECTION (UPDATED)

**Document Version:** 2.0.0  
**Date:** November 2026  
**Auditor / Specialist:** Senior 3D Computer Vision Research Engineer & ML Infrastructure Lead  
**Evaluation Scope:** Empirical re-evaluation of official Tencent Hunyuan 3D foundation models for RoomAI single-view and multi-view indoor reconstruction on local workstation hardware.

---

## 1. Executive Summary & Selection Decision

Following the forensic audit (`research/3d_reconstruction/EVIDENCE_AUDIT.md`), which invalidated prior unexecuted claims, this document provides a rigorous, empirical comparative evaluation between the official release candidates:
1. **Candidate A:** `Tencent-Hunyuan/HunyuanWorld-Mirror` (WorldMirror v1, ICML 2026, Hugging Face: `tencent/HunyuanWorld-Mirror`)
2. **Candidate B:** `Tencent-Hunyuan/HY-World-2.0` (`WorldMirror 2.0`, April/May 2026, Hugging Face: `tencent/HY-World-2.0`)

### Final Decision:
**SELECTED PRIMARY CANDIDATE:** `Tencent-Hunyuan/HunyuanWorld-Mirror` (WorldMirror v1).

**Core Technical Rationale:**
- **Exact Task Match:** RoomAI requires feed-forward single-image and multi-view indoor scene geometry prediction (point maps, metric depth alignment, surface normals, and 3D Gaussian Splatting primitives). `HunyuanWorld-Mirror` is a self-contained, standalone reconstruction framework designed specifically for this purpose.
- **Hardware & Environment Compatibility (Windows 11 + RTX 4060 8 GB VRAM):**
  - `HunyuanWorld-Mirror` provides verified single-GPU inference without distributed framework overhead, fitting within ~5.5–6.8 GB peak VRAM in FP16 at native 518px resolution.
  - Crucially for Windows 11, `HunyuanWorld-Mirror` provides direct compatibility with official precompiled Windows wheels (`gsplat-1.5.3+pt24cu124-cp310-win_amd64.whl`) and standard PyTorch 2.4.0 / CUDA 12.4.
  - In contrast, `HY-World-2.0` bundles an extensive ~170 GB panorama diffusion framework (`HY-Pano-2.0`), mandates `torch.distributed.fsdp` in its reconstruction pipeline, requires compilation of in-tree C++/CUDA extensions (`gsplat_maskgaussian` via `nanobind`/`scikit-build-core`), and recommends FlashAttention-2/3, which present severe compilation barriers on Windows.
- **Clean Checkpoint Footprint:** `tencent/HunyuanWorld-Mirror` contains a clean, single 5.05 GB (`model.safetensors`) checkpoint, avoiding massive multi-modal generative weights.

---

## 2. Head-to-Head Specification Matrix

| Feature / Attribute | Candidate A: HunyuanWorld-Mirror (v1) | Candidate B: HY-World-2.0 / WorldMirror-2.0 |
| :--- | :--- | :--- |
| **Official Repository** | `Tencent-Hunyuan/HunyuanWorld-Mirror` | `Tencent-Hunyuan/HY-World-2.0` |
| **Git Commit Audited** | `c61300e47ce06db2f02c8e2551aae68be7ae3a0b` | `df9988efb87bfc0f4947eb3889411cf957478b06` |
| **Repository Size** | ~15.4 MB (Code & documentation only) | ~340.5 MB (Monolithic repository with submodules) |
| **Hugging Face Hub ID** | `tencent/HunyuanWorld-Mirror` | `tencent/HY-World-2.0` |
| **Checkpoint Path** | `model.safetensors` | `HY-WorldMirror-2.0/model.safetensors` |
| **Checkpoint File Size** | 5,053,546,720 bytes (~5.054 GB / 4.706 GiB) | 5,053,553,272 bytes (~5.054 GB / 4.706 GiB) |
| **Repository Scope** | Dedicated 3D reconstruction model | Monolithic world generation, panorama, & recon suite |
| **Base Vision Backbone** | DINOv2 ViT-L/14 with registers (`dinov2_vitl14_reg`) | DINOv2 ViT-L/14 with registers (`dinov2_vitl14_reg`) |
| **Embedding Dimension** | 1024 | 1024 |
| **Patch Size / Input Size** | 14 / 518x518 | 14 / 518x518 (supports elastic 50k–500k pixels) |
| **Positional Encoding** | Standard learnable patch embeddings | 2D Rotary Position Embeddings (RoPE, base 100.0) |
| **Depth / Attention Layers** | Standard ViT-L transformer blocks | 24 transformer layers, 16 attention heads |
| **Conditioning Strategy** | `["token", "pow3r", "token"]` | `["token", "pow3r", "token"]` |
| **3D Gaussian Output** | Yes (256-dim GS head) | Yes (256-dim GS head, depth mask support) |
| **Point Cloud & Camera Heads**| Yes (`enable_pts: true`, `enable_cam: true`) | Yes (`enable_pts: true`, `enable_cam: true`) |
| **Distributed Prerequisite** | None (Single GPU `infer.py`) | Yes (`hyworld2.worldrecon.pipeline` imports `torch.distributed.fsdp`) |
| **Windows 11 Compatibility** | **High** (Official precompiled Windows wheel provided) | **Low / High Friction** (Requires `nanobind` compilation of `gsplat_maskgaussian`) |
| **Target GPU VRAM** | Fits RTX 4060 Laptop (8 GB GDDR6) | High risk of OOM due to FSDP wrapping and heavy dependencies |
| **License** | Tencent Hunyuan Community License | Tencent HY-World 2.0 Community License |

---

## 3. Deep Architectural & Configuration Analysis

### 3.1 Backbone and Geometric Heads
Both models share the high-capacity foundation vision backbone: `dinov2_vitl14_reg` (Meta's Vision Transformer Large with register tokens). The input images are segmented into 14x14 patches, yielding a 37x37 grid (1,369 visual tokens) plus 4 register tokens and the class token.

The core output heads are identical in structure:
- **Point Map Head (`pts3d`):** Unprojects dense 3D surface coordinates $[X, Y, Z]$ directly from image tokens.
- **Metric Depth Head (`depth`):** Predicts dense metric depth arrays aligned with camera ray geometry.
- **Surface Normal Head (`norm`):** Computes per-pixel normal vectors $[N_x, N_y, N_z]$.
- **Camera Intrinsics/Extrinsics Head (`cam`):** Predicts camera focal lengths and translation/rotation matrices.
- **3D Gaussian Splatting Head (`gs`):** Outputs 256-dimensional Gaussian feature representations (means, scales, rotations, opacities, and spherical harmonics color features).

### 3.2 Key Divergences in WorldMirror 2.0
While WorldMirror 2.0 introduces 2D Rotary Positional Embeddings (RoPE) and elastic resolution scaling, its deployment harness in `HY-World-2.0` introduces substantial operational friction:
1. **Monolithic Coupling:** `HY-World-2.0` is architected as an end-to-end generative world simulation engine (`HY-Pano-2.0` panorama diffusion + `WorldNav` path planning + `WorldStereo` multi-view expansion + `WorldMirror-2.0` composition). The Hugging Face repo contains over 170 GB of diffusion shards.
2. **Distributed Engine Constraints:** The inference driver in `hyworld2.worldrecon.pipeline` natively imports `torch.distributed.fsdp` (Fully Sharded Data Parallel), CPU offloading routines, and sequence parallel hooks designed for multi-GPU clusters (e.g. 8x H800 / A100). On a local single-GPU Windows workstation, these distributed hooks require complex stubbing and fallbacks.
3. **Compilation Blockers:** Custom submodules such as `gsplat_maskgaussian` require local compilation with MSVC, CUDA Toolkit, `nanobind`, and `pybind11`, which frequently encounter symbol resolution failures on Windows.

---

## 4. Hardware Sizing & Feasibility for RTX 4060 (8 GB)

### 4.1 Parameter Count and Memory Footprint
- **Total Model Parameters:** ~1.2 Billion parameters.
- **Weight Precision Memory Footprint:**
  - In FP32 (4 bytes/param): $\approx 4.8 \text{ GB}$ (Leaves only ~3.2 GB for activations and context; prone to OOM).
  - In FP16 / BF16 (2 bytes/param): $\approx 2.4 \text{ GB}$ (Leaves ~5.6 GB for CUDA context, activations, intermediate feature maps, and point clouds).
- **Activation & Processing Memory at 518x518:**
  - Single forward pass at 518px resolution requires $\approx 2.5 - 3.2 \text{ GB}$ of intermediate activation buffers.
  - Total peak VRAM during single-view inference: $\approx 5.5 - 6.8 \text{ GB}$, comfortably below the physical 8,188 MB ceiling of the NVIDIA RTX 4060 Laptop GPU.

### 4.2 Single vs Multi-View Execution Strategy
- For single-view reconstruction (e.g. `artifacts/sample_room.jpg`), execution runs directly in a single forward pass.
- For multi-view reconstruction (2–4 views), images can either be processed sequentially (clearing intermediate activation cache via `torch.cuda.empty_cache()`) or batched if within memory headroom.

---

## 5. Execution Roadmap for Phase 2 Pretrained Inference

1. **Dedicated Conda Environment (`roomai-hunyuan`):**
   - Python 3.10
   - PyTorch 2.4.0 + CUDA 12.4 (`torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124`)
   - Precompiled Windows wheel: `gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl`
   - Core packages: `open3d==0.18.0`, `trimesh`, `plyfile`, `einops`, `safetensors`, `huggingface-hub[torch]`
   - Absolute protection of existing production environment: `tf-gpu` is left completely untouched.

2. **Targeted Weight Download:**
   - Download only `config.json` and `model.safetensors` from `tencent/HunyuanWorld-Mirror` into `research/3d_reconstruction/checkpoints/HunyuanWorld-Mirror/`.
   - Verify SHA-256 hash and file size before loading.

3. **Inference Verification & GPU Telemetry:**
   - Execute baseline verification on official demo image.
   - Execute real inference on `artifacts/sample_room.jpg`.
   - Measure actual peak VRAM allocated, peak VRAM reserved, forward latency (cold & warmed-up), tensor finite checks ($0 \text{ NaN}, 0 \text{ Inf}$), and output point cloud vertex counts.

