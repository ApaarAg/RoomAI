# Phase 9: Multi-Task Gradient Interaction and Conflict Audit

**Author:** Senior 3D Computer Vision Research Scientist & Foundation Model Optimization Engineer  
**Repository:** `F:\RoomAI`  
**Execution Environment:** Windows 11, Conda Environment `roomai-hunyuan`, PyTorch 2.11.0 + CUDA 12.8  
**Audit Artifact:** [`research/3d_reconstruction/experiments/phase9/gradient_interaction_audit.json`](file:///F:/RoomAI/research/3d_reconstruction/experiments/phase9/gradient_interaction_audit.json)  
**Date:** 2026-11-08  

---

## 1. Executive Summary

In multi-task geometric foundation models, simultaneously optimizing depth, 3D points, and surface normals can induce destructive gradient interference in shared backbone layers. In Phase 9, we conducted a rigorous mathematical gradient conflict audit across 15 distinct training batches, evaluating gradients backpropagated into both the task-specific prediction heads ($98.0\text{M}$ parameters) and the shared representation layers ($25.2\text{M}$ parameters in the final frame- and global-attention blocks of the Visual Geometry Transformer).

### Key Empirical Findings
1. **Geometric Task Orthogonality:** All pairwise cosine similarities in the shared representation space hover within $[-0.07, +0.15]$, averaging near zero. The gradients operate in largely orthogonal subspaces.
2. **Depth–Point Synergy:** Depth and 3D Point tasks demonstrate positive average alignment ($\cos(\mathbf{g}_d, \mathbf{g}_p) = +0.0417$) with a low conflict frequency of only **$33.3\%$**.
3. **Surface Normal Decoupling:** Surface normals exhibit weak negative alignment with depth ($\cos(\mathbf{g}_d, \mathbf{g}_n) = -0.0075$, $60.0\%$ conflict frequency) and point maps ($\cos(\mathbf{g}_p, \mathbf{g}_n) = +0.0116$, $53.3\%$ conflict frequency).
4. **Cancellation Ratio:** The destructive interference cancellation ratio is **$38.2\%$**, characteristic of healthy multi-task regularization rather than catastrophic task interference.
5. **PCGrad / Surgery Recommendation:** Gradient projection surgery (e.g. PCGrad) is **not necessary** for head-only fine-tuning. For backbone fine-tuning, keeping $w_{norm} \le 0.1$ effectively mitigates directional tension with depth.

---

## 2. Mathematical Formulation of Gradient Interaction

Let $\mathcal{L}_{depth}$, $\mathcal{L}_{pts}$, and $\mathcal{L}_{norm}$ denote the masked, canonical scene-normalized losses for depth, 3D point cloud, and surface normals. For shared parameter vector $\boldsymbol{\theta}_{shared} \in \mathbb{R}^P$:
$$\mathbf{g}_d = \nabla_{\boldsymbol{\theta}} \mathcal{L}_{depth}, \quad \mathbf{g}_p = \nabla_{\boldsymbol{\theta}} \mathcal{L}_{pts}, \quad \mathbf{g}_n = \nabla_{\boldsymbol{\theta}} \mathcal{L}_{norm}$$

The pairwise cosine similarity is defined as:
$$\cos(\mathbf{g}_i, \mathbf{g}_j) = \frac{\langle \mathbf{g}_i, \mathbf{g}_j \rangle}{\|\mathbf{g}_i\|_2 \|\mathbf{g}_j\|_2}$$

A **gradient conflict** occurs when $\cos(\mathbf{g}_i, \mathbf{g}_j) < 0$, signifying that descending along task $i$ increases the loss of task $j$.

The combined multi-task gradient is:
$$\mathbf{g}_{comb} = w_d \mathbf{g}_d + w_p \mathbf{g}_p + w_n \mathbf{g}_n$$
The destructive interference cancellation ratio is:
$$\mathcal{C} = 1 - \frac{\|\mathbf{g}_{comb}\|_2}{\sum_i w_i \|\mathbf{g}_i\|_2}$$

---

## 3. Empirical Audit Results

Evaluated across 15 real indoor Hypersim training scenes on the RTX 4060 GPU:

### 3.1 Pairwise Cosine Similarities in Shared Backbone Layers

| Task Pair | Mean Cosine Similarity $\cos(\mathbf{g}_i, \mathbf{g}_j)$ | Conflict Frequency ($\cos < 0$) | Interaction Regime |
| :--- | :---: | :---: | :--- |
| **Depth vs. 3D Points** | **$+0.0417$** | **$33.3\%$** | **Synergistic Alignment** |
| **Depth vs. Surface Normals** | **$-0.0075$** | **$60.0\%$** | **Weak Orthogonal Tension** |
| **3D Points vs. Surface Normals** | **$+0.0116$** | **$53.3\%$** | **Near-Orthogonal** |

### 3.2 Gradient Norms and Multi-Task Cancellation

| Gradient Quantity | Formulation | Mean L2 Norm | Percentage of Total Signal |
| :--- | :--- | :---: | :---: |
| **Depth Gradient** | $w_d \|\mathbf{g}_d\|_2$ ($w_d = 1.0$) | **$1.3221$** | $42.8\%$ |
| **Point-Map Gradient** | $w_p \|\mathbf{g}_p\|_2$ ($w_p = 0.5$) | **$0.7946$** (Raw: $1.5892$) | $25.7\%$ |
| **Surface Normal Gradient** | $w_n \|\mathbf{g}_n\|_2$ ($w_n = 0.2$) | **$0.9724$** (Raw: $4.8618$) | $31.5\%$ |
| **Linear Weighted Sum** | $\sum w_i \|\mathbf{g}_i\|_2$ | **$3.0891$** | $100.0\%$ |
| **Actual Combined Gradient** | $\|\mathbf{g}_{comb}\|_2$ | **$1.9089$** | — |
| **Cancellation Ratio** | $1 - (\|\mathbf{g}_{comb}\| / \sum w_i \|\mathbf{g}_i\|)$ | **$38.2\%$** | Healthy Regularization |

---

## 4. Architectural Conclusions for Phase 9 Fine-Tuning

1. **Point Loss Gradient Contained:** In Phase 7, unnormalized point loss accounted for $>90\%$ of total gradient magnitude. Under canonical normalization and $w_{pts}=0.5$, point gradients contribute **$25.7\%$**, in optimal balance with depth ($42.8\%$).
2. **Backbone Fine-Tuning Safe:** In shared transformer attention blocks (`frame_blocks[-1]` and `global_blocks[-1]`), task alignment is predominantly orthogonal ($\cos \approx 0$). There is no catastrophic destructive interference.
3. **Weight Calibration for Backbone Adaptation:** When unfreezing backbone blocks in Phase 9 Experiment B, setting $w_{depth} = 1.0$, $w_{pts} = 0.5$, and $w_{norm} = 0.1$ provides clean gradient flow without requiring expensive runtime gradient projection surgery.

