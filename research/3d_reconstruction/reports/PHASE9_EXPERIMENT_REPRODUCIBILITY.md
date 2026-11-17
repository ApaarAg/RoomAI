# Phase 9: Forensic Audit of Phase 8 Experiment Reproducibility

**Author:** Senior 3D Computer Vision Research Scientist & Foundation Model Optimization Engineer  
**Repository:** `F:\RoomAI`  
**Execution Environment:** Windows 11, Conda Environment `roomai-hunyuan`, PyTorch 2.11.0 + CUDA 12.8  
**Date:** 2026-11-08  

---

## 1. Executive Summary & Root-Cause Verdict

In Phase 8, Experiment A (Hypersim-only) and Experiment B (intended Multi-View Geometric Supervision across all training scenes) reported identical validation metrics:
- Hypersim Val Scaled AbsRel: **$0.6320$**
- Hypersim Val Scaled RMSE: **$4.3930\text{ m}$**
- ScanNet Val Scaled AbsRel: **$0.2666$**
- ScanNet Val Scaled RMSE: **$0.6203\text{ m}$**

An independent forensic audit ([`research/3d_reconstruction/scripts/audit_phase8_experiments.py`](file:///F:/RoomAI/research/3d_reconstruction/scripts/audit_phase8_experiments.py)) was executed on the dataset ordering, batch contents, optimizer steps, and checkpoint weights.

### Root-Cause Finding
**Experiment B executed the exact same 20 Hypersim frames as Experiment A in the exact same sequence.**
The cause is a sampling implementation defect in `run_phase8_experiments.py`:
1. The training loop iterated sequentially: `sample = train_ds.get_sample(step - 1)` for steps $1 \dots 20$.
2. In `indoor_canonical_manifest.json`, the first 31 training frames are Hypersim frames (indices $0 \dots 30$).
3. ScanNet frames reside at indices $31 \dots 203$.
4. With a fixed budget of 20 steps and no random shuffling, Experiment B (`dataset_filter=None`) sampled indices $0 \dots 19$—which were $100.0\%$ identical to Experiment A (`dataset_filter="hypersim"`).
5. Experiment B **never reached a single ScanNet frame** or novel multi-view scene sequence. Both models executed identical forward and backward passes from identical initializations.

---

## 2. Empirical Verification Evidence

### 2.1 Step-by-Step Sample Trace

| Step | Experiment A (`dataset_filter="hypersim"`) | Experiment B (`dataset_filter=None`) | Identity Match |
| :---: | :--- | :--- | :---: |
| **01** | Scene `ai_015_007` (Frame 32) | Scene `ai_015_007` (Frame 32) | **MATCH (True)** |
| **02** | Scene `ai_015_007` (Frame 39) | Scene `ai_015_007` (Frame 39) | **MATCH (True)** |
| **03** | Scene `ai_015_007` (Frame 40) | Scene `ai_015_007` (Frame 40) | **MATCH (True)** |
| **04** | Scene `ai_015_007` (Frame 56) | Scene `ai_015_007` (Frame 56) | **MATCH (True)** |
| **05** | Scene `ai_015_007` (Frame 66) | Scene `ai_015_007` (Frame 66) | **MATCH (True)** |
| **06** | Scene `ai_015_007` (Frame 73) | Scene `ai_015_007` (Frame 73) | **MATCH (True)** |
| **07** | Scene `ai_011_010` (Frame 31) | Scene `ai_011_010` (Frame 31) | **MATCH (True)** |
| **08** | Scene `ai_011_010` (Frame 37) | Scene `ai_011_010` (Frame 37) | **MATCH (True)** |
| **09** | Scene `ai_011_010` (Frame 53) | Scene `ai_011_010` (Frame 53) | **MATCH (True)** |
| **10** | Scene `ai_011_010` (Frame 66) | Scene `ai_011_010` (Frame 66) | **MATCH (True)** |
| **11** | Scene `ai_011_010` (Frame 79) | Scene `ai_011_010` (Frame 79) | **MATCH (True)** |
| **12** | Scene `ai_011_010` (Frame 81) | Scene `ai_011_010` (Frame 81) | **MATCH (True)** |
| **13** | Scene `ai_011_010` (Frame 82) | Scene `ai_011_010` (Frame 82) | **MATCH (True)** |
| **14** | Scene `ai_011_010` (Frame 83) | Scene `ai_011_010` (Frame 83) | **MATCH (True)** |
| **15** | Scene `ai_011_010` (Frame 89) | Scene `ai_011_010` (Frame 89) | **MATCH (True)** |
| **16** | Scene `ai_047_008` (Frame 78) | Scene `ai_047_008` (Frame 78) | **MATCH (True)** |
| **17** | Scene `ai_047_008` (Frame 79) | Scene `ai_047_008` (Frame 79) | **MATCH (True)** |
| **18** | Scene `ai_047_008` (Frame 80) | Scene `ai_047_008` (Frame 80) | **MATCH (True)** |
| **19** | Scene `ai_047_008` (Frame 81) | Scene `ai_047_008` (Frame 81) | **MATCH (True)** |
| **20** | Scene `ai_047_008` (Frame 82) | Scene `ai_047_008` (Frame 82) | **MATCH (True)** |

**Sample Overlap Rate:** **$20 / 20$ ($100.0\%$)**

### 2.2 Direct Checkpoint Weight Comparison
A layer-by-layer tensor difference was computed across all $1,223,250,748$ parameters:
- **Sum L1 Weight Difference between Exp A and Exp B:** **$0.14168303$** (average difference per parameter: $1.15 \times 10^{-10}$, reflecting pure CUDA non-deterministic reduction noise).
- **Sum L1 Weight Delta from Baseline (`production_model.ckpt`):**
  - Experiment A: $76.1031$
  - Experiment B: $76.1028$

---

## 3. Mandatory Corrections for Phase 9

1. **Epoch-Based Shuffled DataLoader:** Implement PyTorch `DataLoader` with deterministic random seed and shuffling so that all scenes in the training set are guaranteed to be sampled.
2. **Scene-Balanced Sampling:** In multi-dataset / multi-view experiments, construct balanced batches ensuring that diverse scenes and tracks are represented uniformly rather than sequentially according to manifest order.
3. **Sufficient Step Budget:** Base training budgets on complete dataset epochs rather than an arbitrary sub-epoch step cap (e.g. 20 steps) that fails to cover the training corpus.
4. **Automated Batch Logging:** Log unique scene IDs and frame hashes per batch to permanently guarantee that multi-scene experiments actually consume the intended data distribution.

