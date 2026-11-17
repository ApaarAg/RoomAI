# RoomAI Object Detection: Comprehensive Training Plan & Optimization Strategy

**Document Version:** 1.0.0 (Milestone 2 Formal Deliverable)  
**Date:** 2026-10-29  
**Target Architecture:** Ultralytics YOLO11 (YOLO11s Primary, YOLO11n Lightweight)  
**Target Dataset:** Open Images V7 Pilot (`roomai_open_images_pilot`, 18 Harmonized Classes, 10,000 Samples)  
**Hardware Profile:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB / 7.996 GB VRAM), 24 GB System RAM, Windows 11  
**Runtime Stack:** Python 3.13, PyTorch 2.6.0+cu124 / 2.11.0+cu128, CUDA 12.4/12.8, Ultralytics 8.4.x, `$env:USE_TF="0"`  

---

## 1. Executive Summary & Optimization Objectives

### 1.1 Problem Statement & Domain Relevance
RoomAI is an automated architectural and interior space understanding system providing three core perception modalities:
1. **Multi-Domain Real-World Object Detection** (`models/detection_model.py`, `services/detection_service.py`)
2. **Instance Segmentation for Room Envelopes** (`models/segmentation_model.py`, `checkpoints/yolo11n-seg.pt`)
3. **Monocular Relative Depth Estimation** (`models/scene3d_model.py`, `checkpoints/depth_anything_v2_small`)

The existing detection capability operates predominantly on canonical indoor furniture (`checkpoints/indoor_detector_a3_yolo11s_800px_best.pt`, 12 classes). To expand RoomAI's spatial reasoning into architectural facades, vertical circulation, residential exteriors, and parking infrastructure, a curated 10,000-image dataset from Open Images V7 was audited and converted into an 18-class harmonized dataset (`F:\RoomAI_Data\yolo_open_images_pilot\data.yaml`).

The core challenge is training high-precision, high-recall object detectors on this diverse real-world dataset under a strict **8 GB physical VRAM budget** on an NVIDIA GeForce RTX 4060 Laptop GPU, while preserving complete backward compatibility, zero regression in existing segmentation and depth estimation capabilities, and absolute safeguard of existing production checkpoints.

### 1.2 Core Success Metrics
The performance of all experimental configurations and the final deployed candidate will be evaluated across:
- **Detection Accuracy:**
  - $\text{mAP}@0.50$ (overall and per-class): Primary benchmark for localization and classification correctness at standard IoU.
  - $\text{mAP}@0.50:0.95$ (COCO primary metric): Metric for tight boundary alignment across multiple IoU thresholds ($0.50, 0.55, \dots, 0.95$).
  - Mean Precision ($P$) and Mean Recall ($R$) at optimal $F_1$-score threshold.
- **Inference Efficiency:**
  - Preprocessing, inference, and postprocessing latency per image (measured in milliseconds on GPU at batch size 1).
  - Steady throughput (Frames Per Second, $\text{FPS} = 1000 / T_{\text{total}}$).
  - Peak GPU VRAM allocated and reserved during steady-state inference.
- **Training Stability & Resource Envelope:**
  - Peak GPU memory consumption strictly $< 6.0\text{ GB}$ (leaving $\ge 2.0\text{ GB}$ safety headroom on the 8 GB GPU).
  - Loss convergence stability (smooth decline in Box CIoU loss, BCE Classification loss, and DFL Distribution Focal loss; zero NaN or Inf).
  - Zero leakage between data splits (cryptographic MD5 and FiftyOne ImageID disjointness).

### 1.3 Hardware Boundaries & 8 GB VRAM Invariants
All experiments operate under the constraints of the host workstation:
- **Physical VRAM:** 8,188 MB (7.996 GB) on NVIDIA GeForce RTX 4060 Laptop GPU.
- **Display Compositor Overhead:** Windows Desktop Window Manager (DWM) consumes 300–600 MB of VRAM for desktop rendering.
- **VRAM Upper Bound:** Any training process reserving $\ge 6,500\text{ MB}$ risks immediate CUDA Out Of Memory (OOM) due to desktop buffer allocation spikes.
- **The 6.0 GB Training Ceiling:** Peak reserved VRAM during all training operations must not exceed **6,000 MB (5.86 GB)**, guaranteeing an absolute minimum safety headroom of **2,000 MB (1.95 GB)**.
- **Multi-Process Dataloader Stability:** PyTorch multiprocessing on Windows initializes via `spawn`. To avoid host RAM exhaustion, paging thrashing, and named-pipe deadlocks, Dataloader worker count is strictly fixed at `workers=2`.
- **Global Environment Invariant:** `$env:USE_TF="0"` (or `os.environ["USE_TF"] = "0"`) must be exported in all terminal environments and Python runners to ensure HuggingFace Transformers / PyTorch depth estimation pipelines remain functional.

---

## 2. Mathematical VRAM Budget Modeling & Empirical Profiling

### 2.1 Analytical Formulation of GPU Memory Allocation
Total peak GPU VRAM ($M_{\text{peak}}$) consumed during PyTorch / Ultralytics neural network training is governed by the analytical relationship:

$$M_{\text{peak}} = M_{\text{context}} + M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}}(B, R) + M_{\text{allocator\_overhead}}$$

1. **CUDA Driver & Runtime Context ($M_{\text{context}}$):**
   Initializing CUDA 12.x and cuDNN primitives on the RTX 4060 allocates a static baseline context:
   $$M_{\text{context}} \approx 700\text{ MB} \text{ to } 750\text{ MB}$$

2. **Model Parameter Memory ($M_{\text{weights}}$):**
   In Automatic Mixed Precision (AMP), PyTorch maintains master weights in FP32 (4 bytes per parameter) and downcasts weights to FP16 (2 bytes per parameter) for the forward pass:
   $$M_{\text{weights}} = P \times (4 + 2) \text{ bytes} = 6P \text{ bytes}$$
   - For **YOLO11s** ($P = 9.46 \times 10^6$ parameters): $M_{\text{weights}} \approx 56.8\text{ MB}$
   - For **YOLO11n** ($P = 2.62 \times 10^6$ parameters): $M_{\text{weights}} \approx 15.7\text{ MB}$

3. **Gradient Memory ($M_{\text{gradients}}$):**
   Gradients are accumulated in FP32 (4 bytes per parameter):
   $$M_{\text{gradients}} = P \times 4 \text{ bytes} = 4P \text{ bytes}$$
   - For **YOLO11s**: $M_{\text{gradients}} \approx 37.8\text{ MB}$
   - For **YOLO11n**: $M_{\text{gradients}} \approx 10.5\text{ MB}$

4. **Optimizer State Memory ($M_{\text{optimizer}}$):**
   - For **SGD with Momentum** (1 momentum buffer per parameter in FP32):
     $$M_{\text{opt\_SGD}} = P \times 4 \text{ bytes} = 4P \text{ bytes} \quad (\approx 37.8\text{ MB for YOLO11s})$$
   - For **AdamW** (2 first/second moment buffers $m_t, v_t$ per parameter in FP32):
     $$M_{\text{opt\_AdamW}} = P \times 8 \text{ bytes} = 8P \text{ bytes} \quad (\approx 75.7\text{ MB for YOLO11s})$$

   **Static Memory Total ($M_{\text{static}} = M_{\text{context}} + M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}}$):**
   - YOLO11s (SGD): $720 + 56.8 + 37.8 + 37.8 = \mathbf{852.4\text{ MB}}$
   - YOLO11s (AdamW): $720 + 56.8 + 37.8 + 75.7 = \mathbf{890.3\text{ MB}}$
   - YOLO11n (SGD): $720 + 15.7 + 10.5 + 10.5 = \mathbf{756.7\text{ MB}}$

5. **Dynamic Activation Memory ($M_{\text{activations}}(B, R)$):**
   Dynamic memory is dominated by intermediate feature tensor maps retained during the forward pass to evaluate backward gradients. Activation memory scales **linearly with batch size $B$** and **quadratically with input spatial resolution $R$** ($H \times W$):
   $$M_{\text{activations}} \propto B \times R^2$$

6. **PyTorch Caching Allocator & Chunk Fragmentation ($M_{\text{allocator\_overhead}}$):**
   PyTorch manages memory via memory pooling in 2 MB and 20 MB segment blocks. The difference between *allocated memory* and *reserved memory* represents caching allocator overhead, typically adding a 12% to 18% wedge.

### 2.2 Precision Economics: AMP (FP16) vs FP32
Training with Automatic Mixed Precision (`amp=True`) dynamically scales gradients using PyTorch `torch.cuda.amp.GradScaler`:
- Forward activations and convolution tensor contractions execute in native NVIDIA 4th-Gen Tensor Core FP16.
- Activation tensor volume in memory is halved compared to FP32 ($2\text{ bytes}$ vs $4\text{ bytes}$ per element).
- Training throughput increases by $\approx 2.2\times$ on the Ada Lovelace RTX 4060 architecture.
- Numerical stability is guaranteed by maintaining master weights and loss scaling in FP32.

### 2.3 Spatial Scaling Law: Area Expansion from 640px to 800px
Increasing input resolution expands the spatial grid processed by the backbone, neck, and detection heads. The area scaling factor $\alpha$ is:
$$\alpha = \left(\frac{800}{640}\right)^2 = (1.25)^2 = 1.5625 \quad (+56.25\% \text{ pixel area})$$

Because activation volume scales directly with spatial pixel count, intermediate feature tensors consume $56.25\%$ more memory per batch.
- At $B=8$ on YOLO11s:
  - Peak allocated at 640px: $1,835.5\text{ MB}$
  - Peak allocated at 800px: $2,848.2\text{ MB}$
  - Empirical ratio: $\frac{2,848.2}{1,835.5} = \mathbf{1.5517} \approx 1.5625$ ($99.3\%$ agreement with analytical law).

### 2.4 Empirical VRAM Profiling Matrix (RTX 4060 Laptop GPU, 12 Cells)
The following 12-cell empirical profiling matrix was measured directly on the host NVIDIA RTX 4060 Laptop GPU under PyTorch CUDA with `amp=True`:

| Cell # | Architecture | Resolution ($R$) | Batch Size ($B$) | Peak Allocated | Peak Reserved | VRAM Headroom | Operational Verdict | Safety Assessment |
|---|---|---|---|---|---|---|---|---|
| **C1** | **YOLO11n** (2.62M) | 640px | 8 | 1,000.2 MB (0.98 GB) | 1,216.0 MB (1.19 GB) | 6.81 GB (85.1%) | **SAFE** | Minimal footprint, massive headroom |
| **C2** | **YOLO11n** (2.62M) | 640px | 16 | 1,982.9 MB (1.94 GB) | 2,254.0 MB (2.20 GB) | 5.79 GB (72.5%) | **SAFE** | Ideal fast fine-tuning configuration |
| **C3** | **YOLO11n** (2.62M) | 640px | 32 | 3,878.1 MB (3.79 GB) | 4,402.0 MB (4.30 GB) | 3.70 GB (46.2%) | **SAFE** | High throughput, stable gradient updates |
| **C4** | **YOLO11n** (2.62M) | 800px | 8 | 1,588.3 MB (1.55 GB) | 1,788.0 MB (1.75 GB) | 6.25 GB (78.2%) | **SAFE** | High-res lightweight baseline |
| **C5** | **YOLO11n** (2.62M) | 800px | 16 | 3,086.1 MB (3.01 GB) | 3,382.0 MB (3.30 GB) | 4.69 GB (58.7%) | **SAFE** | Strong headroom, high accuracy |
| **C6** | **YOLO11n** (2.62M) | 800px | 32 | 6,064.4 MB (5.92 GB) | 6,930.0 MB (6.77 GB) | 1.23 GB (15.4%) | **TIGHT** | Upper limit for YOLO11n on 8 GB |
| **C7** | **YOLO11s** (9.46M) | 640px | 8 | 1,835.5 MB (1.79 GB) | 2,060.0 MB (2.01 GB) | 5.98 GB (74.8%) | **SAFE** | Safe low-footprint baseline |
| **C8** | **YOLO11s** (9.46M) | 640px | 16 | 3,552.9 MB (3.47 GB) | 4,042.0 MB (3.95 GB) | **4.05 GB (50.6%)** | **SAFE** | **Primary Baseline (Exp 1 Recommended)** |
| **C9** | **YOLO11s** (9.46M) | 640px | 32 | 6,963.5 MB (6.80 GB) | 7,886.0 MB (7.70 GB) | 0.29 GB (3.7%) | **RISKY** | Dangerous on Windows (DWM buffer spikes cause OOM) |
| **C10**| **YOLO11s** (9.46M) | 800px | 8 | 2,848.2 MB (2.78 GB) | 3,166.0 MB (3.09 GB) | **4.90 GB (61.3%)** | **SAFE** | **Primary High-Res (Exp 2 Recommended with Accum=2)** |
| **C11**| **YOLO11s** (9.46M) | 800px | 16 | 5,537.4 MB (5.41 GB) | 6,252.0 MB (6.11 GB) | 1.89 GB (23.6%) | **TIGHT** | Borderline safe; vulnerable to multi-worker leaks |
| **C12**| **YOLO11s** (9.46M) | 800px | 32 | 10,918.6 MB (10.66 GB)| 12,318.0 MB (12.03 GB)| -4.03 GB (-50.4%)| **CUDA OOM** | Physically exceeds 8 GB VRAM capacity |

### 2.5 Gradient Accumulation Mechanics for 800px Training
At 800px resolution, executing batch size 16 directly (Cell C11) pushes peak reserved VRAM to $6,252\text{ MB}$, leaving only $1.89\text{ GB}$ headroom and violating the $2.0\text{ GB}$ headroom invariant.

To achieve the regularizing benefits of a 16-sample mini-batch while maintaining total memory safety:
- Set physical micro-batch size: $B_{\text{micro}} = 8$
- Set gradient accumulation steps: $N_{\text{accum}} = 2$
- Effective mini-batch size:
  $$B_{\text{effective}} = B_{\text{micro}} \times N_{\text{accum}} = 8 \times 2 = 16$$
- VRAM footprint drops from $6,252\text{ MB}$ down to **3,166 MB (3.09 GB)**, expanding safety headroom to **4.90 GB (61.3% free)**.
- Optimizer weights update once every 2 micro-batches, mathematically matching the exact gradient expectation of $B=16$.

---

## 3. Dataset Profile & Class Imbalance Analysis

### 3.1 Corpus Dimensions & Partitioning
The target dataset is derived from Open Images V7, audited in `docs/dataset_audit.md`, and stored outside the repository at `F:\RoomAI_Data\yolo_open_images_pilot`.

- **Total Corpus Size:** 10,000 images, 43,407 bounding boxes across 18 classes.
- **Split Breakdown:**
  - `Train` (80.0%): 8,000 images, 34,369 bounding boxes.
  - `Validation` (10.0%): 1,000 images, 4,605 bounding boxes.
  - `Held-Out Test` (10.0%): 1,000 images, 4,433 bounding boxes.
- **Data Integrity:** Stratified multi-label partitioning preserving tail-class frequencies across splits. Zero near-duplicate image leakage verified by MD5 hashing.

### 3.2 The 264.3:1 Long-Tailed Imbalance Challenge
The bounding box distribution across the 18 harmonized classes exhibits extreme Pareto skew:

| Class ID | Harmonized Class Name | Total Boxes | % of Dataset | Train Boxes | Val Boxes | Test Boxes | Imbalance Tier |
|---|---|---|---|---|---|---|---|
| **15** | `Tree` | 18,499 | 42.62% | 14,642 | 1,972 | 1,885 | Head (Extreme) |
| **3** | `Window` | 9,018 | 20.77% | 7,222 | 995 | 801 | Head |
| **12** | `Car` | 4,267 | 9.83% | 3,363 | 473 | 431 | Head |
| **0** | `Building` | 3,491 | 8.04% | 2,752 | 362 | 377 | Core Architecture |
| **1** | `House` | 2,233 | 5.14% | 1,770 | 228 | 235 | Core Architecture |
| **5** | `Chair` | 2,170 | 5.00% | 1,739 | 212 | 219 | Core Interior |
| **7** | `Table` | 1,533 | 3.53% | 1,228 | 158 | 147 | Core Interior |
| **2** | `Door` | 895 | 2.06% | 711 | 94 | 90 | Core Architecture |
| **16** | `Motorcycle` | 258 | 0.59% | 190 | 36 | 32 | Mid-tail (Parking) |
| **14** | `Bus` | 224 | 0.52% | 172 | 26 | 26 | Mid-tail (Parking) |
| **13** | `Truck` | 194 | 0.45% | 140 | 30 | 24 | Mid-tail (Parking) |
| **17** | `Bicycle` | 184 | 0.42% | 148 | 18 | 18 | Mid-tail (Parking) |
| **10** | `Cabinetry` | 142 | 0.33% | 92 | 26 | 24 | Tail Interior |
| **11** | `Bench` | 118 | 0.27% | 92 | 14 | 12 | Tail Outdoor |
| **9** | `Bookcase` | 106 | 0.24% | 87 | 9 | 10 | Tail Interior |
| **4** | `Stairs` | 104 | 0.24% | 76 | 13 | 15 | Tail Architecture |
| **6** | `Couch` | 96 | 0.22% | 77 | 10 | 9 | Tail Interior |
| **8** | `Bed` | 70 | 0.16% | 56 | 6 | 8 | Starved Tail |

**Maximum Imbalance Ratio:**
$$\frac{N_{\max} (\text{Tree})}{N_{\min} (\text{Bed})} = \frac{18,499}{70} = \mathbf{264.3 : 1}$$

Without targeted interventions, standard cross-entropy loss causes gradients from `Tree` and `Window` (constituting $> 63\%$ of all boxes) to dominate parameter updates, causing severe recall suppression on tail classes (`Bed`, `Couch`, `Stairs`, `Bookcase`).

### 3.3 Task Boundary Invariants
- **Detection Exclusivity:** Open Images pilot annotations are 2D axis-aligned bounding boxes ($x, y, w, h$). No segmentation masks are present.
- **Isolation from Segmentation:** As dictated by R2 and F7, detection models must not initialize or train a segmentation mask head. The segmentation service (`models/segmentation_model.py`, `checkpoints/yolo11n-seg.pt`) remains completely independent.
- **Isolation from Monocular Depth:** Depth estimation relies on Depth Anything V2 Small. The detector training pipeline does not alter depth dependencies or configuration.

---

## 4. Model Architecture & Transfer Learning Strategy

### 4.1 Model Family Selection: Ultralytics YOLO11
YOLO11 introduces enhanced multi-scale feature pyramids (C3k2 blocks and SPPF) with spatial attention mechanisms (C2PSA), delivering higher mAP per FLOP than previous YOLO architectures.

- **Primary Architecture: YOLO11s (`yolo11s.pt`)**
  - Parameters: 9.46M
  - Computational Complexity: 21.5 GFLOPs (at 640px), 33.6 GFLOPs (at 800px)
  - Characteristics: High representational capacity for fine structural architectural elements (window mullions, door frames, complex staircases).
- **Lightweight Alternative: YOLO11n (`yolo11n.pt`)**
  - Parameters: 2.62M (72.3% smaller than YOLO11s)
  - Computational Complexity: 6.5 GFLOPs (at 640px)
  - Characteristics: Exceptional inference speed (7–8 ms per image, > 130 FPS on RTX 4060; ~25 ms on CPU), serving as the high-throughput mobile/CPU deployment baseline.

### 4.2 Transfer Learning Mechanism
Pretrained models `yolo11s.pt` and `yolo11n.pt` are pretrained on MS COCO (80 classes). Fine-tuning on Open Images pilot operates as follows:
1. **Backbone & Feature Pyramid Transfer:** All convolutional backbone weights (layers 0–9) and neck feature aggregation layers (layers 10–22) are transferred directly.
2. **Detection Head Re-initialization:** The final classification convolution layers in the decoupled detection heads (layer 23) are re-initialized from 80 output logits to 18 output logits ($nc = 18$).
3. **Bounding Box Regression Warm Start:** The bounding box distribution focal loss (DFL) channels and regression heads remain fully initialized from COCO, maintaining strong localization priors from iteration 0.

### 4.3 Layer Freezing vs Full Fine-Tuning Analysis
- **Phase 1 (Epochs 0–3 Warmup):** Gradients flow through all layers with linear warmup from $lr = 0.0001$ to $0.01$. Because the new head has random classification weights, low initial learning rates prevent destructive backpropagation into pretrained backbone features.
- **Phase 2 (Epochs 4–25):** Full fine-tuning of all parameters allows intermediate feature representations to adapt from COCO natural scenes to architectural structures (e.g. distinguishing `House` vs `Building`, or identifying distant repetitive `Window` grids).

---

## 5. Controlled Experiment Matrix & Hypotheses

To systematically satisfy requirements R4, R5, and R6, four controlled experiments and one pre-flight smoke test are designed. Each experiment tests an isolated, falsifiable hypothesis.

### 5.1 Pre-Flight Smoke Test: Rapid 1-Epoch GPU Verification
- **Identifier:** `smoke_test_open_images_yolo11s`
- **Model:** `yolo11s.pt`
- **Execution Script:** `scripts/smoke_test_gpu_open_images.py`
- **Configuration:** `imgsz=640`, `batch=16`, `epochs=1`, `amp=True`, `workers=2`, `device=0`.
- **Purpose:** Assert pipeline operational integrity:
  1. CUDA RTX 4060 binding and AMP validity.
  2. Dataloader stability on 8,000 training images without deadlock.
  3. Loss finiteness (Box, Cls, DFL losses $\in (0, \infty)$).
  4. VRAM safety (peak reserved $< 6.0\text{ GB}$, headroom $> 2.0\text{ GB}$).
  5. Checkpoint delivery to `F:\RoomAI_Data\checkpoints\`.
  6. Zero mutation of existing production checkpoints in `checkpoints/` (cryptographic MD5 hash assertion).

### 5.2 Experiment 1 (Exp 1): Baseline Fine-Tuning
- **Identifier:** `exp1_yolo11s_baseline_640`
- **Model:** `yolo11s.pt`
- **Hyperparameters:**
  - Resolution: `imgsz=640`
  - Batch size: `batch=16`
  - Optimizer: `SGD` (`momentum=0.937`, `weight_decay=0.0005`)
  - Learning rate: `lr0=0.01`, `lrf=0.01` (Cosine decay)
  - Epochs: `epochs=25`, `patience=10`, `warmup_epochs=3.0`
  - Mixed Precision: `amp=True`, `workers=2`
- **Estimated Resource Footprint:** Peak reserved VRAM $\approx 4.04\text{ GB}$ ($4.05\text{ GB}$ headroom, $50.6\%$ free).
- **Hypothesis:** Fine-tuning on 8,000 domain-specific architectural samples will establish strong baseline detection performance, achieving $\ge 45.0\% \text{ mAP}@0.50$ across the 18 classes on validation data, dramatically outperforming raw zero-shot COCO weights.

### 5.3 Experiment 2 (Exp 2): High-Resolution Fine-Tuning
- **Identifier:** `exp2_yolo11s_highres_800`
- **Model:** `yolo11s.pt`
- **Hyperparameters:**
  - Resolution: `imgsz=800` (+56.25% pixel area)
  - Micro-batch size: `batch=8`
  - Gradient Accumulation: `accumulate=2` (Effective batch size = 16)
  - Optimizer: `SGD` (`lr0=0.01`, `cos_lr=True`)
  - Epochs: `epochs=25`, `patience=10`
  - Mixed Precision: `amp=True`, `workers=2`
- **Estimated Resource Footprint:** Peak reserved VRAM $\approx 3.17\text{ GB}$ ($4.90\text{ GB}$ headroom, $61.3\%$ free).
- **Hypothesis:** Expanding spatial resolution to 800px preserves structural line details, mullions, and distant objects. Exp 2 will increase validation $\text{mAP}@0.50$ on small/distant objects (`Window`, `Chair`, `Door`) by $\ge 3.0\%$ compared to Exp 1, while maintaining $> 4.8\text{ GB}$ of VRAM headroom via gradient accumulation.

### 5.4 Experiment 3 (Exp 3): Lightweight Model Comparison
- **Identifier:** `exp3_yolo11n_lightweight_640`
- **Model:** `yolo11n.pt` (2.62M parameters)
- **Hyperparameters:**
  - Resolution: `imgsz=640`
  - Batch size: `batch=16`
  - Optimizer: `SGD` (`lr0=0.01`, `cos_lr=True`)
  - Epochs: `epochs=25`, `patience=10`
  - Mixed Precision: `amp=True`, `workers=2`
- **Estimated Resource Footprint:** Peak reserved VRAM $\approx 2.25\text{ GB}$ ($5.79\text{ GB}$ headroom, $72.5\%$ free).
- **Hypothesis:** YOLO11n will provide $\approx 2.0\times$ faster GPU inference latency (~7–8 ms vs ~14–15 ms) and minimal memory footprint, while retaining within $4.0\%$ of Exp 1's validation $\text{mAP}@0.50$, serving as the primary candidate for latency-critical and CPU-fallback deployments.

### 5.5 Experiment 4 (Exp 4): Tail-Class Imbalance Remedy (Repeat Factor Sampling)
- **Identifier:** `exp4_yolo11s_rfs_imbalance_remedy`
- **Model:** Best architecture from Exps 1–2 (YOLO11s)
- **Data Intervention:** Deterministic Repeat Factor Sampling (RFS) applied to the training split:
  - Categories with $N_{\text{train}} < 100$ boxes (`Bed` [56], `Couch` [77], `Stairs` [76]): $5\times$ oversampling factor.
  - Categories with $100 \le N_{\text{train}} < 300$ boxes (`Cabinetry` [92], `Bookcase` [87], `Bench` [92], `Truck` [140], `Bus` [172], `Motorcycle` [190]): $2\times$ oversampling factor.
  - Head categories ($N_{\text{train}} > 1000$ boxes): $1\times$ (standard sampling).
  - Validation and Test splits remain **100% untouched and unrepeated**.
- **Hyperparameters:** `imgsz=640` (or `800`), `batch=16`, `lr0=0.01`, `cos_lr=True`, `epochs=25`, `patience=10`.
- **Hypothesis:** RFS guarantees tail classes participate in every gradient update iteration, boosting tail-class average recall by $\ge 15.0\%$ and improving tail-class $\text{mAP}@0.50$ (`Bed`, `Couch`, `Stairs`) without degrading head-class precision.

### 5.6 Comparative Summary Matrix of Planned Experiments

| Experiment ID | Base Model | Imgsz | Batch Size | Accum | Epochs | VRAM Reserved | Safety Headroom | Target Evaluation Focus |
|---|---|---|---|---|---|---|---|---|
| **Smoke Test** | `yolo11s.pt` | 640px | 16 | 1 | 1 | 4,030 MB | 4.06 GB (50.8%) | Pipeline integrity, AMP, zero mutation |
| **Exp 1** | `yolo11s.pt` | 640px | 16 | 1 | 25 | 4,042 MB | 4.05 GB (50.6%) | Baseline fine-tuning across 18 classes |
| **Exp 2** | `yolo11s.pt` | 800px | 8 | 2 | 25 | 3,166 MB | 4.90 GB (61.3%) | High-resolution architectural detail |
| **Exp 3** | `yolo11n.pt` | 640px | 16 | 1 | 25 | 2,254 MB | 5.79 GB (72.5%) | Latency/throughput & CPU candidate |
| **Exp 4** | `yolo11s.pt` | 640px | 16 | 1 | 25 | 4,042 MB | 4.05 GB (50.6%) | Long-tailed class recall recovery |

---

## 6. Hyperparameter Schedules, Loss Functions & Augmentations

### 6.1 Optimizer Selection: SGD vs AdamW
- **Primary Optimizer: SGD with Nesterov Momentum**
  - Settings: `momentum=0.937`, `weight_decay=0.0005`, `lr0=0.01`.
  - Rationale: SGD exhibits superior generalization and final mAP on object detection when training with mini-batches of size $\ge 16$. It requires only 1 state buffer ($4\text{ bytes/param}$) compared to AdamW ($8\text{ bytes/param}$), saving $> 38\text{ MB}$ of static memory.
- **Secondary Optimizer: AdamW (Exploration Fallback)**
  - Settings: `lr0=0.001`, `weight_decay=0.01`, `betas=(0.9, 0.999)`.
  - Used if SGD encounters initial gradient oscillation on small tail classes.

### 6.2 Learning Rate Schedules
- **Warmup Schedule:** Linear warmup over the first $3.0$ epochs (`warmup_epochs=3.0`), scaling learning rate from $lr_{\text{warmup}} = 0.0001$ to $lr0 = 0.01$. Momentum scales from $0.8$ to $0.937$.
- **Decay Policy:** Cosine annealing (`cos_lr=True`) decaying learning rate to $1\%$ of initial value ($lrf = 0.01$, final $lr = 0.0001$) over the training horizon.
- **Early Stopping:** `patience=10`. If validation $\text{mAP}@0.50:0.95$ does not improve for 10 consecutive epochs, training terminates to prevent overfitting.

### 6.3 Loss Formulations
Total loss is a weighted multi-task objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

- **Complete IoU Box Loss ($\mathcal{L}_{\text{CIoU}}$, $\lambda_{\text{box}} = 7.5$):**
  Accounts for overlap area, central point distance, and aspect ratio consistency:
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
- **Binary Cross-Entropy Class Loss ($\mathcal{L}_{\text{BCE}}$, $\lambda_{\text{cls}} = 0.5$):**
  Evaluated across all 18 classes using sigmoid focal weighting to manage multi-label background instances.
- **Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$, $\lambda_{\text{dfl}} = 1.5$):**
  Models bounding box coordinates as continuous probability distributions over discrete bins, improving localization precision around ambiguous boundaries.

### 6.4 Data Augmentation Policy
Augmentation hyperparameters are tailored for architectural and indoor geometries:
- **Horizontal Flipping:** `fliplr=0.5` (Architectural scenes exhibit horizontal bilateral symmetry).
- **Vertical Flipping:** `flipud=0.0` (**Strict Invariant:** Buildings, houses, stairs, tables, and chairs have strict gravitational polarity. Vertical flipping produces unnatural, physically impossible architectural priors).
- **Mosaic Augmentation:** `mosaic=1.0` (Stitches 4 images into one, expanding multi-scale object density and small object representations).
- **Scale Jitter:** `scale=0.5` (Random zoom between $50\%$ and $150\%$).
- **Color Space Jitter:** `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4` (Robustness to varying daylight, interior lighting, and shadows).

### 6.5 Mosaic Shutdown Protocol (`close_mosaic=10`)
During the final 10 epochs of training, 4-image mosaic augmentation is disabled (`close_mosaic=10`). The model trains exclusively on natural, uncropped images to fine-tune anchor regression to real-world aspect ratios and clean background distributions.

---

## 7. Evaluation Methodology & Anti-Leakage Protocol

### 7.1 Split Discipline: Validation vs Held-Out Test
- **Validation Split (`images/val`, 1,000 images, 4,605 boxes):**
  - Evaluated at the end of every training epoch.
  - Used for learning rate scheduling, early stopping (`patience=10`), and best checkpoint selection (`best.pt`).
  - Used for post-training confidence threshold and NMS IoU optimization.
- **Held-Out Test Split (`images/test`, 1,000 images, 4,433 boxes):**
  - **Completely isolated and frozen** during all exploration and training phases.
  - Evaluated **exactly once** on the single winning candidate model selected from validation metrics.
  - Never used for tuning confidence thresholds, NMS parameters, or model selection.

### 7.2 Validation-Only Threshold Tuning Methodology
Confidence thresholds ($T_{\text{conf}}$) and NMS IoU thresholds ($T_{\text{IoU}}$) will be swept across grid values on the validation split:
- Sweep $T_{\text{conf}} \in [0.10, 0.70]$ in steps of $0.05$.
- Sweep $T_{\text{IoU}} \in [0.45, 0.70]$ in steps of $0.05$.
- Compute the harmonic mean $F_1(T) = \frac{2 \cdot P(T) \cdot R(T)}{P(T) + R(T)}$ on validation data.
- The optimal operating point $T^*$ maximizing validation $F_1$ is locked and frozen prior to final test set evaluation.

### 7.3 Metric Deliverables Suite
All evaluations will log structured JSON and Markdown summaries containing:
- Precision ($P$), Recall ($R$), $F_1$-score
- $\text{mAP}@0.50$ (overall and per each of the 18 classes)
- $\text{mAP}@0.50:0.95$ (overall and per each of the 18 classes)
- Preprocessing, inference, postprocessing latency (ms) and steady FPS
- Peak GPU allocated and reserved memory (MB)

---

## 8. Rollback Protocol, Checkpoint Governance & Deliverables

### 8.1 Checkpoint Storage Outside Source Repository
To maintain Git repository hygiene and avoid bloated binary commit history:
- All training runs (`runs/train/*`) and model weight checkpoints are saved outside the repository in:
  ```text
  F:\RoomAI_Data\runs\train\<experiment_name>\
  F:\RoomAI_Data\checkpoints\<experiment_name>_best.pt
  ```
- `.gitignore` lines 44–63 enforce exclusions for `checkpoints/`, `runs/`, `data/`, and `*.pt`.

### 8.2 Preservation of Existing Production Checkpoints
The repository hosts 21 historical and production checkpoints in `F:\RoomAI\checkpoints\`.
- **Primary Production Detector:** `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` (mAP50 = 77.60% on canonical indoor).
- **Rollback Baseline Detector:** `checkpoints/deepfurniture_yolo11n_finetune_best.pt` (mAP50 = 73.10% on DeepFurniture).
- **Preservation Policy:**
  - Neither file shall be overwritten, renamed, or modified.
  - Pre-training and post-training MD5 hashes of all 21 files are computed and asserted identical in all training scripts.
  - If the fine-tuned model fails any validation or deployment criterion, RoomAI defaults seamlessly to `indoor_detector_a3_yolo11s_800px_best.pt`.

### 8.3 Streamlit Integration Criteria & CPU Fallback Verification
Prior to merging any new candidate model into `config/model_config.yaml` and Streamlit (`app.py`, `ui/`):
1. **Pydantic Response Schema:** The candidate model wrapped in `models/detection_model.py` must return valid `DetectionResponse` instances with valid normalized bounding box coordinates $[0.0, 1.0]$.
2. **CPU Fallback:** Must successfully instantiate and run inference on `device="cpu"` in FP32 without errors, simulating environments lacking CUDA acceleration.
3. **Zero Regressions:** Automated tests must confirm `checkpoints/yolo11n-seg.pt` (instance segmentation) and `checkpoints/depth_anything_v2_small` (monocular depth) remain 100% operational with 0 errors.

---

## 9. Exact Reproduction & CLI Commands

All training and evaluation workflows are fully reproducible via standard terminal commands under Conda `tf-gpu`:

```powershell
# Step 0: Ensure TF-GPU environment and depth safeguard
$env:USE_TF="0"

# Step 1: Execute Rapid 1-Epoch GPU Transfer Learning Smoke Test
python scripts/smoke_test_gpu_open_images.py --base-checkpoint yolo11s.pt --batch 16 --imgsz 640 --workers 2 --device 0

# Step 2: Execute Experiment 1 (Baseline Fine-Tuning YOLO11s, 640px)
python scripts/train_open_images.py --model yolo11s.pt --data F:/RoomAI_Data/yolo_open_images_pilot/data.yaml --imgsz 640 --batch 16 --epochs 25 --device 0 --name exp1_yolo11s_baseline_640

# Step 3: Execute Experiment 2 (High-Resolution YOLO11s, 800px, Accumulate=2)
python scripts/train_open_images.py --model yolo11s.pt --data F:/RoomAI_Data/yolo_open_images_pilot/data.yaml --imgsz 800 --batch 8 --accumulate 2 --epochs 25 --device 0 --name exp2_yolo11s_highres_800

# Step 4: Execute Experiment 3 (Lightweight YOLO11n Comparison)
python scripts/train_open_images.py --model yolo11n.pt --data F:/RoomAI_Data/yolo_open_images_pilot/data.yaml --imgsz 640 --batch 16 --epochs 25 --device 0 --name exp3_yolo11n_lightweight_640

# Step 5: Execute Experiment 4 (Tail-Class Imbalance Remedy RFS)
python scripts/train_open_images.py --model yolo11s.pt --data F:/RoomAI_Data/yolo_open_images_pilot/data_rfs.yaml --imgsz 640 --batch 16 --epochs 25 --device 0 --name exp4_yolo11s_rfs_imbalance_remedy

# Step 6: Single Final Evaluation on Held-Out Test Set (Best Candidate Model Only)
python scripts/evaluate_detection_model.py --checkpoint F:/RoomAI_Data/checkpoints/best_candidate.pt --data F:/RoomAI_Data/yolo_open_images_pilot/data.yaml --split test --imgsz 640 --device 0
```
