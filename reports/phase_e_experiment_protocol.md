# RoomAI — Phase E Few-Shot Fine-Tuning Experiment Protocol
**Document ID:** `reports/phase_e_experiment_protocol.md`  
**Author:** Skeptical Senior Computer Vision Research Engineer & QA Architect  
**Date:** October 31, 2026  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6), 24 GB System RAM  
**Execution Environment:** `tf-gpu` Conda Environment (PyTorch 2.11.0+cu128, CUDA 12.8)

---

## 1. Scientific Motivation & Core Hypotheses

Before allocating GPU compute to few-shot fine-tuning, a research engineer must question whether fine-tuning is actually necessary or potentially harmful.

### Hypotheses Under Test
- **$H_0$ (Foundation Sufficiency Hypothesis):** Pretrained SegFormer-B1 weights already represent optimal convergence on ADE20K. Fine-tuning on small indoor subsets will induce catastrophic forgetting or overfit to specific room layouts without providing statistically meaningful general gains ($< +1.5\%$ mIoU).
- **$H_1$ (Indoor Adaptation Hypothesis):** Constraining the loss landscape to residential indoor rooms improves classification on difficult structural boundaries (e.g., doors, thin table frames, cabinets) by $\ge +3.0\%$ IoU while preserving planar surface accuracy (walls, floors, ceilings).

---

## 2. Model Architecture & Parameter Freezing Strategy

To prevent catastrophic forgetting of general visual features, we employ **Decoder-Only Adaptation**:

```
[Input Image: 512 x 512]
         │
         ▼
┌──────────────────────────────────────────────┐
│ Mix Transformer (MiT-B1) Encoder Backbone   │  <--- FROZEN (Requires no gradients, zero weight drift)
│ 4 Hierarchical Stages (C1, C2, C3, C4)      │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ All-MLP Decoder Head                         │  <--- TRAINABLE (~0.5 M parameters)
│ Linear Projections + MLP Fusion + Classifier │
└──────────────────────────────────────────────┘
         │
         ▼
[Output Logits: 150 Classes]
```

- **Frozen Parameters:** 13.2 M parameters (MiT-B1 backbone).
- **Trainable Parameters:** 0.52 M parameters (All-MLP decoder head).
- **GPU Memory Benefit:** Eliminating encoder backpropagation reduces peak training VRAM from ~5.2 GB to **~2.1 GB**, allowing comfortable batch size 4 on the 8 GB RTX 4060.

---

## 3. Training Hyperparameters & Execution Protocol

| Hyperparameter | Value | Scientific Rationale |
|---|---|---|
| **Base Architecture** | `nvidia/segformer-b1-finetuned-ade-512-512` | Empirically proven winner in Phase C benchmark (faster and +3.75% higher mIoU than B0). |
| **Control Architecture** | `nvidia/segformer-b0-finetuned-ade-512-512` | Ultra-lightweight control baseline (3.75M total params). |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$) | Standard for vision transformer adaptation. |
| **Learning Rate** | $1 \times 10^{-4}$ | Low learning rate to ensure gentle head calibration without divergence. |
| **LR Schedule** | Cosine Annealing with 1 epoch warm-up | Smooth convergence to zero learning rate. |
| **Weight Decay** | $0.01$ | Prevents weight explosion in the linear classification layers. |
| **Batch Size** | 4 per step | Fits within 2.1 GB VRAM. |
| **Gradient Accumulation** | 2 steps (Effective Batch Size = 8) | Stabilizes gradient estimates. |
| **Precision** | `fp16` (Automatic Mixed Precision via `torch.cuda.amp`) | 2x throughput, identical numerical stability. |
| **Loss Function** | Cross-Entropy Loss with `ignore_index=255` | ADE20K background/unlabeled pixels (index 0 mapped to 255) receive zero loss. |
| **Epoch Schedule** | 1% & 5% Splits: **10 Epochs**<br>10% & 20% Splits: **5 Epochs** | Balances step count across differing split sizes (~120–300 gradient steps per experiment). |

---

## 4. Evaluation & Verification Metric Suite

Every model checkpoint will be evaluated on the **Held-Out Golden Test Set (200 Quarantined Indoor Scenes)**:

1. **Primary Metric:** Standard Dataset-Level mIoU across 10 Target Indoor Classes:
   $$\text{mIoU}_{\text{Target10}} = \frac{1}{10} \sum_{c \in \text{Target}} \frac{\sum_i \text{TP}_{i,c}}{\sum_i (\text{TP}_{i,c} + \text{FP}_{i,c} + \text{FN}_{i,c})}$$
2. **Safety Metric (Catastrophic Forgetting Check):** Planar surface stability:
   $$\Delta \text{Planar} = \text{IoU}_{\text{wall}}^{\text{FT}} + \text{IoU}_{\text{floor}}^{\text{FT}} + \text{IoU}_{\text{ceiling}}^{\text{FT}} - \left(\text{IoU}_{\text{wall}}^{\text{ZeroShot}} + \text{IoU}_{\text{floor}}^{\text{ZeroShot}} + \text{IoU}_{\text{ceiling}}^{\text{ZeroShot}}\right)$$
   *Condition:* $\Delta \text{Planar}$ must NOT drop by more than $-1.0\%$.
3. **Generalization Metric:** Dataset mIoU across ALL present classes in the test set.

---

## 5. Formal GO / NO-GO Criteria for Phase E

| Decision Criterion | Threshold for GO | Threshold for NO-GO |
|---|---|---|
| **Data Partitioning** | Quarantined 200-image Golden Test Set verified with zero train leakage. | Any leakage detected between train and test splits. |
| **Safety Rule** | User review and explicit written authorization provided. | User has not authorized training. |
| **VRAM & Thermal Budget** | Training VRAM $\le 4.0$ GB, GPU temperature $< 75^\circ\text{C}$. | Out of memory error or thermal throttling. |
| **Production Model Protection** | Production Models A3, B2, and C2 remain 100% untouched and preserved. | Any modification to A3/B2/C2 checkpoints. |

### Final Auditor Recommendation:
**AUDIT COMPLETE — STATUS: READY FOR USER DECISION.**
All methodological discrepancies in Phases A–D have been isolated, documented, and corrected. The dataset isolation standard and Phase E experiment protocols are strictly defined and reproducible. Training must not commence until the user reviews this audit and provides authorization.

---

## 6. ADE20K Pretraining Overlap Caveat

**Reference:** `reports/phase_e_split_integrity_report.json`, `reports/dataset_split_protocol.md` §6.

The base checkpoint `nvidia/segformer-b1-finetuned-ade-512-512` was pretrained on the ADE20K **training** split (20,210 images). All Phase E fine-tuning subsets are sampled from that **same training split**. Consequently:

- The fine-tuning experiment measures **continued/refreshed training on data already represented in the pretraining corpus**, not adaptation to a novel domain.
- **No claim of independent generalization** may be made on the basis of the fine-tuning split alone.
- Generalization is assessed **only** on the ADE20K **validation**-split-derived evaluation sets (Golden Test 200, validation tuning 50, benchmark 50), which the pretrained checkpoint never observed.
- Any report of fine-tuned gains must state explicitly that the training images are a subset of the original pretraining corpus.

A future genuine domain-adaptation experiment would require a fine-tuning corpus drawn from outside ADE20K (e.g., proprietary residential RGB-D scans). No such corpus exists in the workspace; this is recorded as a known limitation.

---

## 7. FastSAM Mask Refinement Guidance

**Reference:** `reports/mask_refinement_comparison.json`, `reports/phase_a_to_d_independent_audit.md` §8.3.

Phase E concerns semantic segmentation (SegFormer) and does not retrain FastSAM. However, the downstream pipeline uses FastSAM masks, so the refinement policy is fixed here:

1. **Do not automatically clip all masks to A3 bounding boxes.** Hard clipping destroys instances when the FastSAM mask lands outside the prompt box (observed in `mask_refinement_comparison.json`: multiple cases where inside-ratio = 0.0 and hard clipping yields a 0-pixel mask).
2. **Default strategy:** morphological connected-component soft refinement (largest component overlapping the box; prune disconnected islands; no hard spatial clamp). This preserves full masks while reducing fragmentation.
3. **Object-level mask IoU is uncomputable** because no manually annotated instance ground truth exists in the workspace. This limitation must be reported; no mask-accuracy metric may be invented.
4. A manual polygon-annotation effort on a small evaluation subset is a prerequisite for any future mask-accuracy claim.

---

## 8. VRAM Budget Reconciliation

**Reference:** `reports/vram_reconciliation_report.json`, `reports/phase_a_to_d_independent_audit.md` §8.4.

| Memory Layer | Value |
|---|---|
| Total static model weights (4 models) | 230.3 MB |
| Peak dynamic inference activations | 190.2 MB |
| PyTorch peak allocated | 420.5 MB (0.41 GB) |
| PyTorch peak reserved | 526.0 MB (0.51 GB) |
| nvidia-smi total GPU capacity | 8,188 MiB (8 GB) |

Static model memory (230 MB) is cleanly separable from inference-time memory (190 MB activations). The Phase E training VRAM budget (≤ 4.0 GB, per §5) is validated: the pilot peaked at 2,220 MB allocated / 3,466 MB reserved (see `reports/phase_e_pilot_results.json`), well under the 4.0 GB ceiling and the 8 GB hardware limit.

---

## 9. Pilot Prerequisites (Pre-Full-Campaign)

Before the full few-shot campaign (1%, 5%, 10%, 20% splits × B1/B0) is authorized, a single decoder-only SegFormer-B1 pilot on the 1% split must be completed and reviewed. The pilot:

- Preserves the original pretrained checkpoint (saves to `checkpoints/experimental/segformer_b1_pilot/`).
- Records exact trainable parameter count (564,374).
- Records GPU memory, training time, and validation metrics.
- Compares fine-tuned vs frozen baseline on the quarantined Golden Test Set.
- Does **not** use the Golden Test Set during training or model selection.

Pilot results are documented in `reports/phase_e_pilot_results.md`. The full campaign must not proceed without explicit user approval of the pilot outcome.

