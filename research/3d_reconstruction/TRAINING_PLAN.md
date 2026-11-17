# ROOMAI — 3D FOUNDATION MODEL TRAINING INFRASTRUCTURE & EXPERIMENT PLAN

**Date:** 2026-11-07  
**Role:** Senior 3D Computer Vision Research Engineer & ML Infrastructure Lead  
**Primary Candidate:** HunyuanWorld-Mirror (1B Transformer)  
**Baseline Model:** VGGT-1B  
**Target Hardware:**
- **Local Workstation:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) — *Designated for smoke tests, data loader validation, and lightweight LoRA adapter tests*.
- **Cloud Infrastructure Target:** $1\times$ to $4\times$ NVIDIA A100-SXM4-80GB or NVIDIA L40S (48GB) — *Designated for full foundation model fine-tuning*.

---

## 1. Hardware Feasibility & Cloud Infrastructure Specification

### 1.1 Local RTX 4060 (8 GB VRAM) Reality Check
Full fine-tuning of a 1.0 Billion parameter model with AdamW optimizer requires:
- Model Parameters (FP16): 2.05 GB
- Gradients (FP16): 2.05 GB
- AdamW Optimizer States (FP32 First and Second Moments): 8.20 GB
- Activation Tensors (Multi-view cross-attention during backward pass): 12.0 – 18.0 GB
- **Total Memory Required for Full Backpropagation:** $\ge 24.3\text{ GB}$.

> [!WARNING]
> Attempting full backpropagation on an 8 GB laptop GPU will trigger an immediate CUDA Out-Of-Memory (OOM) crash. The RTX 4060 is strictly reserved for:
> 1. Dataset pipeline and data-loader verification.
> 2. Zero-shot / pretrained inference evaluation.
> 3. Parameter-Efficient Fine-Tuning (PEFT / LoRA) with rank $r=8$ applied only to attention projection heads ($W_q, W_v$).
> 4. Geometric loss function debugging.

### 1.2 Cloud Compute Specification for Full Training
For full geometric and 3DGS fine-tuning across multi-view indoor scenes:
- **Compute Instance:** $1\times$ or $2\times$ NVIDIA A100-SXM4-80GB (or NVIDIA H100 80GB)
- **Host Memory:** 128 GB RAM
- **Storage:** 500 GB NVMe SSD (fast data streaming)
- **Environment:** PyTorch 2.4.0 + CUDA 12.4 + FlashAttention-2 + `gsplat` (pt24cu124)
- **Estimated Compute Hours:**
  - Experiment B (Smoke / Mini-fit): 4 GPU hours ($1\times$ A100)
  - Experiment C (Geometric Heads): 24 GPU hours ($2\times$ A100)
  - Experiment D (3DGS Stage): 36 GPU hours ($2\times$ A100)
  - Experiment E (Ablations): 20 GPU hours ($2\times$ A100)

---

## 2. Staged Experimental Protocol

```
┌────────────────────────────────────────────────────────────────────────┐
│ Experiment A: Zero-Shot Pretrained Baseline Evaluation (COMPLETE)     │
│ - Run off-the-shelf checkpoint on held-out test scenes. Record metrics.│
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Experiment B: Small Controlled Fine-Tuning & Pipeline Verification     │
│ - Train on 10 curated scenes (50 frames each) for 2 epochs.           │
│ - Verify data loading, loss convergence, gradient clipping, checkpoint│
│   resume, and tensorboard logging.                                     │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Experiment C: Geometric Multi-Task Fine-Tuning                         │
│ - Freeze vision transformer backbone. Train metric depth, surface     │
│   normals, and camera pose heads on harmonized indoor dataset.         │
│ - Condition with RoomAI Depth Anything V2 metric priors.               │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Experiment D: 3D Gaussian Splatting (3DGS) Representation Training     │
│ - Activate 3DGS prediction heads (Gaussian centers, scales, opacities).│
│ - Supervised photometric novel-view rendering loss via gsplat.         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Experiment E: Systematic Ablation Studies                              │
│ - E1: Pretrained vs Fine-Tuned.                                        │
│ - E2: Prior Prompting Enabled vs Disabled (Quantify Depth Anything V2)│
│ - E3: View Count Impact (2 views vs 3 views vs 6 views).               │
│ - E4: Image Resolution Impact ($384 \times 384$ vs $512 \times 512$).   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Loss Formulations & Multi-Task Objective

The joint multi-task training loss for RoomAI geometric fine-tuning balances five distinct supervision signals:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{pts}} \mathcal{L}_{\text{pts}} + \lambda_{\text{norm}} \mathcal{L}_{\text{norm}} + \lambda_{\text{cam}} \mathcal{L}_{\text{cam}} + \lambda_{\text{3dgs}} \mathcal{L}_{\text{3dgs}}$$

### 3.1 Metric Depth Loss ($\mathcal{L}_{\text{depth}}$)
Scale-invariant logarithmic depth loss combined with gradient smoothness:
$$\mathcal{L}_{\text{depth}} = \frac{1}{N}\sum_i (d_i - d_i^*)^2 - \frac{0.5}{N^2} \left(\sum_i (d_i - d_i^*)\right)^2 + \gamma \|\nabla D - \nabla D^*\|_1$$
where $d_i = \ln(D_i + 1e-4)$ and $d_i^* = \ln(D_i^* + 1e-4)$.

### 3.2 3D Point Map Loss ($\mathcal{L}_{\text{pts}}$)
Euclidean $L_1$ coordinate error between predicted 3D point map $\mathbf{P}(u, v)$ and ground truth metric points $\mathbf{P}^*(u, v)$:
$$\mathcal{L}_{\text{pts}} = \frac{1}{HW}\sum_{u, v} \|\mathbf{P}(u, v) - \mathbf{P}^*(u, v)\|_1$$

### 3.3 Surface Normal Angular Loss ($\mathcal{L}_{\text{norm}}$)
Cosine similarity loss between unit surface normal vectors:
$$\mathcal{L}_{\text{norm}} = 1 - \frac{1}{HW}\sum_{u, v} \mathbf{n}(u, v) \cdot \mathbf{n}^*(u, v)$$

### 3.4 Camera Pose Loss ($\mathcal{L}_{\text{cam}}$)
Geodesic rotation distance and translation error:
$$\mathcal{L}_{\text{cam}} = \arccos\left(\frac{\text{Tr}(\mathbf{R}^T \mathbf{R}^*) - 1}{2}\right) + \|\mathbf{t} - \mathbf{t}^*\|_2$$

### 3.5 3DGS Photometric Rendering Loss ($\mathcal{L}_{\text{3dgs}}$)
Novel view rendering loss synthesized via `gsplat` rasterizer against target view $I^*$:
$$\mathcal{L}_{\text{3dgs}} = (1 - \alpha) \|I_{\text{render}} - I^*\|_1 + \alpha (1 - \text{SSIM}(I_{\text{render}}, I^*))$$
where $\alpha = 0.2$.

---

## 4. Training Infrastructure Safeguards & MLOps Invariants

1. **Versioned Checkpoints:** Checkpoints are stored in `checkpoints/v{epoch}_{step}/`. Checkpoint naming includes timestamp and validation loss.
2. **Never Overwrite Previous Checkpoints:** The best previous checkpoint and initial pretrained baseline weights are strictly write-protected.
3. **Reproducibility:** Global random seeds (`torch.manual_seed(42)`, `np.random.seed(42)`) are enforced across PyTorch, CUDA, and data loaders.
4. **Gradient Clipping:** Max gradient norm is clamped to $1.0$ (`torch.nn.utils.clip_grad_norm_`) to prevent exploding gradients during early multi-task training.
5. **Early Stopping:** If validation loss fails to improve for 5 consecutive evaluation checkpoints, training halts automatically.
6. **Telemetry & Logging:** Real-time VRAM allocation, GPU temperature, step throughput, and component loss values are logged to TensorBoard and persistent JSON summaries.

