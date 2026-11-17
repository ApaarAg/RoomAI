# RoomAI — Phase E Pilot Results & GO/NO-GO Decision
**Document ID:** `reports/phase_e_pilot_results.md`
**Author:** Skeptical Senior Computer Vision Research Engineer & QA Architect
**Date:** October 31, 2026
**Pilot Script:** `scripts/run_phase_e_pilot.py`
**Raw Metrics:** `reports/phase_e_pilot_results.json`
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6), 24 GB System RAM, Windows 11
**Environment:** `tf-gpu` Conda (Python 3.10.20, PyTorch 2.11.0+cu128, Transformers 4.49.0)

---

## 1. Pilot Configuration

| Item | Value |
|---|---|
| Base model | `nvidia/segformer-b1-finetuned-ade-512-512` |
| Adaptation strategy | Decoder-only (All-MLP head trainable; MiT-B1 encoder frozen) |
| Training split | `data/splits/train_1pct_48.json` (48 images, ADE20K **training** split) |
| Validation tuning split | `data/splits/validation_tuning_50.json` (50 images, ADE20K **validation** split) |
| Golden Test split | `data/splits/golden_test_200.json` (200 images, ADE20K **validation** split) |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Batch size | 4 |
| Precision | fp16 (AMP) |
| Epochs | 3 |
| Loss | CrossEntropyLoss(ignore_index=255) |

### 1.1 Checkpoint Preservation
- The original pretrained checkpoint is loaded **read-only** from the HuggingFace cache (`nvidia/segformer-b1-finetuned-ade-512-512`). It is never overwritten.
- The fine-tuned pilot weights are saved to a **separate** directory: `checkpoints/experimental/segformer_b1_pilot/` containing `config.json` (7,355 B), `model.safetensors` (54,889,536 B ≈ 52.3 MB), and `preprocessor_config.json` (394 B).
- Production detector checkpoints **A3** (`checkpoints/indoor_detector_a3_yolo11s_800px_best.pt`), **B2** (`checkpoints/exterior_detector_b2_yolo11s_best.pt`), and **C2** (`checkpoints/parking_detector_c2_yolo11s_pilot_best.pt`) were **not modified**.

### 1.2 Trainable Parameter Accounting

| Parameter Group | Count | % of Total |
|---|---|---|
| Total model parameters | 13,715,798 | 100.0% |
| Frozen (MiT-B1 encoder) | 13,151,424 | 95.89% |
| **Trainable (All-MLP decode head)** | **564,374** | **4.11%** |

Only 0.56 M parameters (the decoder head) receive gradients. The encoder backbone is frozen to prevent catastrophic forgetting of general visual features.

### 1.3 Scene-Independent Split Verification

Per `reports/phase_e_split_integrity_report.json`, all partitions are mutually disjoint:
- Golden Test (200) ∩ Benchmark (50) = 0
- Golden Test (200) ∩ Validation Tuning (50) = 0
- Golden Test (200) ∩ Fine-tune Train (48) = 0
- Benchmark (50) ∩ Validation Tuning (50) = 0
- Benchmark (50) ∩ Fine-tune Train (48) = 0
- Validation Tuning (50) ∩ Fine-tune Train (48) = 0

The Golden Test Set was **not** used during training or model selection (verified in `scripts/run_phase_e_pilot.py`: training loop uses only `train_1pct_48.json`; per-epoch monitoring uses only `validation_tuning_50.json`; Golden Test is evaluated exactly twice — frozen baseline before training, final measurement after training).

---

## 2. GPU Memory & Training Time

| Metric | Value |
|---|---|
| Training time | 21.57 s (0.36 min) |
| Peak training VRAM allocated | 2,220.0 MB (2.17 GB) |
| Peak training VRAM reserved | 3,466.0 MB (3.39 GB) |
| VRAM budget ceiling (per protocol §5) | 4,096 MB (4.0 GB) |
| Hardware capacity (nvidia-smi) | 8,188 MiB (8 GB) |

**Verdict:** Training VRAM (2.22 GB allocated / 3.39 GB reserved) is comfortably under the 4.0 GB budget and the 8 GB hardware limit. No OOM or thermal throttling observed.

---

## 3. Per-Epoch Training & Validation Logs

Validation monitoring used the **validation tuning set only** (50 images). The Golden Test Set was not consulted during training.

| Epoch | Train Loss | Val Loss (tuning) | Val Target-10 mIoU (tuning) |
|---|---|---|---|
| 1 | 0.4811 | 0.7240 | 0.6473 |
| 2 | 0.4215 | 0.7290 | 0.6456 |
| 3 | 0.4618 | 0.7367 | 0.6459 |

**Observation:** Train loss decreased from epoch 1→2 then rose slightly at epoch 3, while validation loss **increased monotonically** across all three epochs. This is an early signal of overfitting on the 48-image training subset — the decoder head is memorizing the tiny training set without generalizing, and validation tuning mIoU is essentially flat (0.6473 → 0.6459).

---

## 4. Frozen Baseline vs Fine-Tuned Pilot on Golden Test Set (200 images)

Both evaluations use the quarantined Golden Test Set, evaluated once before training (frozen pretrained weights) and once after training (fine-tuned decoder head).

### 4.1 Aggregate Metrics

| Metric | Frozen Baseline | Fine-Tuned Pilot | Delta |
|---|---|---|---|
| Mean loss | 0.6313 | 0.6660 | +0.0347 (worse) |
| **Target-10 mIoU** | **0.6638** | **0.6570** | **−0.0068 (−0.68 pp)** |
| All-present mIoU (88 classes) | 0.3017 | 0.2753 | −0.0264 (−2.64 pp) |
| Planar stability (wall+floor+ceiling avg) | 0.8354 | 0.8334 | −0.0020 (−0.20 pp) |

### 4.2 Per-Class IoU (Target 10 Classes)

| Class | Frozen Baseline | Fine-Tuned Pilot | Delta |
|---|---|---|---|
| wall | 0.8038 | 0.8002 | −0.0036 |
| floor | 0.8244 | 0.8216 | −0.0028 |
| ceiling | 0.8781 | 0.8785 | +0.0004 |
| windowpane | 0.6176 | 0.6089 | −0.0087 |
| door | 0.4078 | 0.3916 | −0.0162 |
| bed | 0.8418 | 0.8306 | −0.0112 |
| sofa | 0.5992 | 0.6004 | +0.0012 |
| chair | 0.5164 | 0.5094 | −0.0070 |
| table | 0.5259 | 0.5062 | −0.0197 |
| cabinet | 0.6229 | 0.6229 | 0.0000 |

### 4.3 Catastrophic-Forgetting Safety Check

Per protocol §4, the planar-surface stability delta must not drop by more than −1.0 pp.
- $\Delta\text{Planar} = -0.20$ pp → **within tolerance** (no catastrophic forgetting of walls/floors/ceilings).

However, the fine-tuned model regressed on **8 of 10** target classes and on the all-present mIoU (−2.64 pp). The single largest regression is on `table` (−1.97 pp). Only `ceiling` (+0.04 pp) and `sofa` (+0.12 pp) improved marginally.

---

## 5. Interpretation

### 5.1 Hypothesis Test Outcome

- **$H_0$ (Foundation Sufficiency):** The pretrained SegFormer-B1 already represents near-optimal convergence on ADE20K residential indoor scenes. The 1% decoder-only adaptation **did not improve** Target-10 mIoU (−0.68 pp) or all-present mIoU (−2.64 pp). This is **consistent with $H_0$** at the 1% data regime.
- **$H_1$ (Indoor Adaptation):** Required ≥ +3.0 pp IoU improvement on difficult structural boundaries. The pilot produced **negative** deltas on door (−1.62 pp), table (−1.97 pp), and windowpane (−0.87 pp). $H_1$ is **not supported** at the 1% regime.

### 5.2 Why the Pilot Regressed

1. **Pretraining overlap (see `reports/dataset_split_protocol.md` §6):** The 48 fine-tuning images are a subset of the ADE20K **training** split — the exact corpus the checkpoint was already trained on. Re-exposing the decoder to 48 of 20,210 already-seen images introduces no novel signal; it only perturbs the decoder head toward the specific layouts of those 48 scenes.
2. **Overfitting signal:** Validation tuning loss rose monotonically (0.724 → 0.729 → 0.737) while train loss fell then rose, indicating the decoder began memorizing the tiny training set.
3. **Decoder-only scope:** Only 0.56 M parameters were trainable. The encoder features are frozen and already optimal; the decoder cannot extract new representational capacity from 48 images.

### 5.3 What This Pilot Does and Does Not Show

- **Does show:** Decoder-only fine-tuning on 48 in-distribution ADE20K training images is **safe** (no catastrophic forgetting, planar stability preserved) but **not beneficial** at the 1% regime. The pretrained checkpoint is already a strong residential surface parser.
- **Does NOT show:** Independent domain adaptation. The fine-tuning images were already seen during pretraining. A genuine domain-adaptation claim would require a non-ADE20K residential corpus, which does not exist in the workspace.

---

## 6. GO / NO-GO Decision for the Full Few-Shot Campaign

### 6.1 Pre-Pilot Gate Checks (all required for GO)

| Gate | Status | Evidence |
|---|---|---|
| Golden Test Set disjoint from benchmark, validation, and fine-tuning | PASS | `reports/phase_e_split_integrity_report.json` |
| No test image used for HP/threshold/early-stopping | PASS | `scripts/run_phase_e_pilot.py` code review |
| Original pretrained checkpoint preserved | PASS | `checkpoints/experimental/segformer_b1_pilot/` separate; A3/B2/C2 untouched |
| Trainable parameters recorded | PASS | 564,374 |
| GPU memory recorded | PASS | 2,220 MB alloc / 3,466 MB reserved |
| Training time recorded | PASS | 21.57 s |
| Validation metrics recorded | PASS | per-epoch logs in §3 |
| Comparison vs frozen baseline | PASS | §4 |
| VRAM within budget (≤ 4 GB) | PASS | 2.17 GB allocated |
| Planar stability within tolerance (≥ −1.0 pp) | PASS | −0.20 pp |

### 6.2 Scientific Outcome Gate

| Criterion (protocol §5) | Threshold for GO | Pilot Result | Met? |
|---|---|---|---|
| Target-10 mIoU improvement | ≥ +1.5 pp (to reject $H_0$) | **−0.68 pp** | **NO** |
| Difficult-boundary class improvement | ≥ +3.0 pp (to support $H_1$) | door −1.62, table −1.97 | **NO** |
| All-present mIoU not degraded | ≥ 0 pp | −2.64 pp | **NO** |

### 6.3 Decision

> ## ⛔ NO-GO — Do not proceed to the full few-shot campaign without redesign.

**Rationale:**

1. **The pilot failed every scientific improvement criterion.** Fine-tuning the decoder on 48 in-distribution ADE20K training images **regressed** segmentation quality on the held-out Golden Test Set (Target-10 −0.68 pp, all-present −2.64 pp). The pretrained checkpoint is already at or above the achievable ceiling for this data regime.
2. **The pretraining-overlap limitation is structural.** Because the fine-tuning corpus is a subset of the pretraining corpus, scaling to 5%/10%/20% will only re-expose more already-seen images. There is no reason to expect a different sign of effect at higher shot counts — the encoder is frozen and the decoder has no novel signal to learn. Running the full campaign as designed would burn GPU time to confirm a negative.
3. **The validation-loss trajectory confirms overfitting**, not underfitting. More data of the same kind will not fix this; it will only delay the overfitting onset.

### 6.4 Conditions That Would Re-Enable a GO

A full few-shot campaign would only be scientifically justified if **one or more** of the following is satisfied:

1. **Novel-domain corpus:** Acquire a residential segmentation dataset **outside** ADE20K (e.g., proprietary room scans, NYUv2 RGB-D, SUN RGB-D with ADE20K-compatible labels). Fine-tune on the novel corpus and evaluate on the ADE20K-validation-derived Golden Test Set. This would measure genuine domain adaptation.
2. **Encoder fine-tuning (not decoder-only):** If the goal is to shift the feature representation, unfreeze selected encoder stages with a very low learning rate. This is higher-risk (catastrophic forgetting) and higher-VRAM, but is the only way to extract new capacity from the same images. Re-run the pilot with encoder-unfrozen before committing to a campaign.
3. **Different objective:** If the goal is class-boundary refinement (door, table, windowpane — the worst classes), consider a boundary-aware loss (e.g., boundary loss or Lovász-Softmax) rather than plain cross-entropy, and evaluate specifically on boundary IoU.

Until one of these conditions is met and a revised pilot demonstrates a positive delta, the full few-shot campaign is **not authorized**.

---

## 7. Limitations Honestly Reported

- **No independent generalization claim:** Fine-tuning images are a subset of the ADE20K training split (the pretraining corpus). See `reports/dataset_split_protocol.md` §6.
- **No object-level mask IoU:** Manually annotated instance ground truth does not exist in the workspace. Mask-accuracy claims are not made. See `reports/phase_a_to_d_independent_audit.md` §8.3.
- **No metric-depth ground truth:** Quantitative depth metrics (AbsRel, RMSE) are uncomputable. See `reports/phase_a_to_d_independent_audit.md` §4.
- **nvidia-smi process memory:** The WDDM `--query-compute-apps` query did not match the Python PID; total GPU capacity (8,188 MiB) is reported instead. PyTorch reserved memory (526 MB inference / 3,466 MB training) is the authoritative in-process figure.
