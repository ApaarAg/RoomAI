# ROOMAI — PHASE 9.1: EXPERIMENT SAMPLE TRACE & DATASET COVERAGE AUDIT

**Author:** Senior 3D Computer Vision Research Scientist & Dataset Engineering Lead  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Status:** COMPLETE & EMPIRICALLY RECONCILED  

---

## 1. EXECUTIVE SUMMARY & FORENSIC FINDING

### Core Finding: Experiment B Was NOT Multi-Dataset
A rigorous forensic audit of `research/3d_reconstruction/scripts/train_phase9_experiments.py` reveals the exact mechanism of the sample coverage discrepancy:

```python
# Line 237 in train_phase9_experiments.py:
def execute_phase9_experiment(exp_name, adaptation_mode="head_only", epochs=2, device="cuda"):
    train_dataset = ShuffledCanonicalDataset(MANIFEST_PATH, split="train", dataset_filter="hypersim")
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=g, collate_fn=collate_single)
```

1. **Hardcoded Filter:** `dataset_filter="hypersim"` was hardcoded inside `execute_phase9_experiment()`. There was no mechanism passing a multi-dataset filter.
2. **ScanNet Exclusion:** The 173 ScanNet training frames (`scene0015_00`, `scene0019_00`, `scene0019_01`) were **completely excluded** from both Experiment A and Experiment B.
3. **True Classification of Experiment B:** Experiment B was **NOT** a multi-dataset experiment. It was a **Partial Backbone Adaptation on Hypersim-only** (unfreezing `frame_blocks[-1]` and `global_blocks[-1]` with $lr = 5 \times 10^{-7}$).
4. **Identical Sampling Sequence:** Because both experiments initialized `torch.Generator().manual_seed(42)` on the identical 31 Hypersim training frames, **both experiments traversed the exact same 62 samples in the exact same sequence**.
5. **Optimizer Updates:** Both experiments performed exactly **62 optimizer updates** (31 steps per epoch $\times$ 2 epochs, batch size 1, gradient accumulation = 1).

---

## 2. SAMPLE COVERAGE INVENTORY

| Parameter | Manifest Total (Train) | Experiment A (Actual) | Experiment B (Actual) | Intended / Documented |
| :--- | :---: | :---: | :---: | :---: |
| **Hypersim Frames** | 31 | **31** (100.0%) | **31** (100.0%) | 31 |
| **ScanNet Frames** | 173 | **0** (0.0%) | **0** (0.0%) | 173 (Excluded by filter) |
| **Total Training Frames** | 204 | **31** | **31** | 204 |
| **Batch Size** | 1 | 1 | 1 | 1 |
| **Epochs** | — | 2 | 2 | 2 |
| **Optimizer Steps** | — | **62** | **62** | 62 |
| **Gradient Accumulation** | — | 1 (none) | 1 (none) | 1 |
| **Unique Scenes Traversed** | 6 | 3 (`ai_011_010`, `ai_015_007`, `ai_047_008`) | 3 (`ai_011_010`, `ai_015_007`, `ai_047_008`) | 3 in Exp A, 6 in Exp B |

---

## 3. COMPLETE STEP-BY-STEP SAMPLE TRACE (62 OPTIMIZER UPDATES)

Below is the complete, deterministic sample trace for **Experiment A** and **Experiment B** (which traversed the identical shuffled sequence due to identical seed 42):

| Experiment | Epoch | Batch | Dataset | Scene ID | Frame IDs | Optimizer Update |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Experiment A (Head-Only) | 1 | 1 | `hypersim` | `ai_047_008` | #78 | Step 1 |
| Experiment A (Head-Only) | 1 | 2 | `hypersim` | `ai_015_007` | #73 | Step 2 |
| Experiment A (Head-Only) | 1 | 3 | `hypersim` | `ai_047_008` | #95 | Step 3 |
| Experiment A (Head-Only) | 1 | 4 | `hypersim` | `ai_011_010` | #31 | Step 4 |
| Experiment A (Head-Only) | 1 | 5 | `hypersim` | `ai_011_010` | #81 | Step 5 |
| Experiment A (Head-Only) | 1 | 6 | `hypersim` | `ai_047_008` | #88 | Step 6 |
| Experiment A (Head-Only) | 1 | 7 | `hypersim` | `ai_015_007` | #56 | Step 7 |
| Experiment A (Head-Only) | 1 | 8 | `hypersim` | `ai_047_008` | #87 | Step 8 |
| Experiment A (Head-Only) | 1 | 9 | `hypersim` | `ai_047_008` | #96 | Step 9 |
| Experiment A (Head-Only) | 1 | 10 | `hypersim` | `ai_047_008` | #82 | Step 10 |
| Experiment A (Head-Only) | 1 | 11 | `hypersim` | `ai_047_008` | #97 | Step 11 |
| Experiment A (Head-Only) | 1 | 12 | `hypersim` | `ai_015_007` | #39 | Step 12 |
| Experiment A (Head-Only) | 1 | 13 | `hypersim` | `ai_011_010` | #53 | Step 13 |
| Experiment A (Head-Only) | 1 | 14 | `hypersim` | `ai_047_008` | #80 | Step 14 |
| Experiment A (Head-Only) | 1 | 15 | `hypersim` | `ai_047_008` | #81 | Step 15 |
| Experiment A (Head-Only) | 1 | 16 | `hypersim` | `ai_047_008` | #85 | Step 16 |
| Experiment A (Head-Only) | 1 | 17 | `hypersim` | `ai_015_007` | #40 | Step 17 |
| Experiment A (Head-Only) | 1 | 18 | `hypersim` | `ai_047_008` | #94 | Step 18 |
| Experiment A (Head-Only) | 1 | 19 | `hypersim` | `ai_011_010` | #79 | Step 19 |
| Experiment A (Head-Only) | 1 | 20 | `hypersim` | `ai_011_010` | #83 | Step 20 |
| Experiment A (Head-Only) | 1 | 21 | `hypersim` | `ai_011_010` | #82 | Step 21 |
| Experiment A (Head-Only) | 1 | 22 | `hypersim` | `ai_015_007` | #32 | Step 22 |
| Experiment A (Head-Only) | 1 | 23 | `hypersim` | `ai_011_010` | #89 | Step 23 |
| Experiment A (Head-Only) | 1 | 24 | `hypersim` | `ai_047_008` | #79 | Step 24 |
| Experiment A (Head-Only) | 1 | 25 | `hypersim` | `ai_015_007` | #66 | Step 25 |
| Experiment A (Head-Only) | 1 | 26 | `hypersim` | `ai_047_008` | #86 | Step 26 |
| Experiment A (Head-Only) | 1 | 27 | `hypersim` | `ai_011_010` | #37 | Step 27 |
| Experiment A (Head-Only) | 1 | 28 | `hypersim` | `ai_047_008` | #84 | Step 28 |
| Experiment A (Head-Only) | 1 | 29 | `hypersim` | `ai_047_008` | #89 | Step 29 |
| Experiment A (Head-Only) | 1 | 30 | `hypersim` | `ai_047_008` | #83 | Step 30 |
| Experiment A (Head-Only) | 1 | 31 | `hypersim` | `ai_011_010` | #66 | Step 31 |
| Experiment A (Head-Only) | 2 | 1 | `hypersim` | `ai_047_008` | #89 | Step 32 |
| Experiment A (Head-Only) | 2 | 2 | `hypersim` | `ai_011_010` | #82 | Step 33 |
| Experiment A (Head-Only) | 2 | 3 | `hypersim` | `ai_015_007` | #66 | Step 34 |
| Experiment A (Head-Only) | 2 | 4 | `hypersim` | `ai_047_008` | #82 | Step 35 |
| Experiment A (Head-Only) | 2 | 5 | `hypersim` | `ai_011_010` | #81 | Step 36 |
| Experiment A (Head-Only) | 2 | 6 | `hypersim` | `ai_047_008` | #83 | Step 37 |
| Experiment A (Head-Only) | 2 | 7 | `hypersim` | `ai_011_010` | #53 | Step 38 |
| Experiment A (Head-Only) | 2 | 8 | `hypersim` | `ai_047_008` | #79 | Step 39 |
| Experiment A (Head-Only) | 2 | 9 | `hypersim` | `ai_047_008` | #84 | Step 40 |
| Experiment A (Head-Only) | 2 | 10 | `hypersim` | `ai_015_007` | #39 | Step 41 |
| Experiment A (Head-Only) | 2 | 11 | `hypersim` | `ai_047_008` | #80 | Step 42 |
| Experiment A (Head-Only) | 2 | 12 | `hypersim` | `ai_015_007` | #73 | Step 43 |
| Experiment A (Head-Only) | 2 | 13 | `hypersim` | `ai_011_010` | #89 | Step 44 |
| Experiment A (Head-Only) | 2 | 14 | `hypersim` | `ai_011_010` | #66 | Step 45 |
| Experiment A (Head-Only) | 2 | 15 | `hypersim` | `ai_047_008` | #94 | Step 46 |
| Experiment A (Head-Only) | 2 | 16 | `hypersim` | `ai_015_007` | #32 | Step 47 |
| Experiment A (Head-Only) | 2 | 17 | `hypersim` | `ai_047_008` | #88 | Step 48 |
| Experiment A (Head-Only) | 2 | 18 | `hypersim` | `ai_047_008` | #81 | Step 49 |
| Experiment A (Head-Only) | 2 | 19 | `hypersim` | `ai_047_008` | #86 | Step 50 |
| Experiment A (Head-Only) | 2 | 20 | `hypersim` | `ai_047_008` | #85 | Step 51 |
| Experiment A (Head-Only) | 2 | 21 | `hypersim` | `ai_047_008` | #97 | Step 52 |
| Experiment A (Head-Only) | 2 | 22 | `hypersim` | `ai_011_010` | #31 | Step 53 |
| Experiment A (Head-Only) | 2 | 23 | `hypersim` | `ai_047_008` | #96 | Step 54 |
| Experiment A (Head-Only) | 2 | 24 | `hypersim` | `ai_011_010` | #79 | Step 55 |
| Experiment A (Head-Only) | 2 | 25 | `hypersim` | `ai_047_008` | #95 | Step 56 |
| Experiment A (Head-Only) | 2 | 26 | `hypersim` | `ai_047_008` | #78 | Step 57 |
| Experiment A (Head-Only) | 2 | 27 | `hypersim` | `ai_047_008` | #87 | Step 58 |
| Experiment A (Head-Only) | 2 | 28 | `hypersim` | `ai_015_007` | #40 | Step 59 |
| Experiment A (Head-Only) | 2 | 29 | `hypersim` | `ai_011_010` | #37 | Step 60 |
| Experiment A (Head-Only) | 2 | 30 | `hypersim` | `ai_011_010` | #83 | Step 61 |
| Experiment A (Head-Only) | 2 | 31 | `hypersim` | `ai_015_007` | #56 | Step 62 |

---

## 4. SCIENTIFIC RECLASSIFICATION & IMPLICATIONS

1. **Correction of Phase 9 Narrative:**
   - Phase 9's narrative text described Experiment B as evaluating multi-dataset training, inheriting the planned terminology from Phase 8.
   - The empirical code execution, however, was a controlled architectural comparison: **Head-Only Adaptation** (Exp A) vs. **Partial Backbone Adaptation** (Exp B), both trained strictly on the commercial-candidate Track A (Hypersim).
2. **Scientific Validity of the Comparison:**
   - Because both experiments trained on the identical 31 Hypersim samples with the identical seed, the comparison between Exp A and Exp B is a **pure, unconfounded ablation of backbone unfreezing**.
   - The finding remains valid: unfreezing late transformer blocks did not improve held-out validation on 31 samples, while increasing compute by $3.2\times$.
3. **ScanNet Separation Confirmed:**
   - Track B (ScanNet) was never mixed into either checkpoint's training weights. Both `phase9_exp_a_head_only_final.ckpt` and `phase9_exp_b_partial_backbone_final.ckpt` remain pure Track A (Hypersim-only) artifacts, satisfying the commercial licensing firewall.