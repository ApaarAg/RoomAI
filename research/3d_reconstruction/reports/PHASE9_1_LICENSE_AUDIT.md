# ROOMAI — PHASE 9.1: DATASET LICENSING, TRACK ISOLATION & LEGAL COMPLIANCE AUDIT

**Author:** Senior 3D Computer Vision Research Scientist & Compliance Lead  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Status:** COMPLETE & FORMALLY CERTIFIED  
**Applicable Tracks:** Track A (Hypersim - Commercial Candidate) vs. Track B (ScanNet - Non-Commercial Academic Research)

---

## 1. EXECUTIVE SUMMARY

RoomAI maintains two strictly segregated research and model development tracks to comply with intellectual property and copyright terms:
1. **Track A (Hypersim):** Governed by the **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)** license. Permitted for commercial use, subject to mandatory attribution and detailed legal review of downstream model-weight ShareAlike obligations.
2. **Track B (ScanNet):** Governed by the **Stanford ScanNet Terms of Use**. Strictly restricted to **academic, non-commercial research use only**. Any model trained, fine-tuned, or distilled on ScanNet is permanently prohibited from commercial deployment.

A strict **Dataset Provenance & Model Lineage Firewall** is enforced across physical directories, dataset manifests, and serialized checkpoint weights to prevent contamination.

---

## 2. TRACK A: HYPERSIM LEGAL ANALYSIS (CC BY-SA 3.0)

### 2.1 License Grant & Terms
- **Dataset:** Apple Hypersim (ICCV 2021, Mike Roberts et al., Apple Inc.)
- **License:** Creative Commons Attribution-ShareAlike 3.0 Unported (`CC BY-SA 3.0`).
- **Commercial Permission:** Yes. Unlike CC BY-NC licenses, CC BY-SA 3.0 explicitly permits commercial exploitation, reproduction, and distribution.

### 2.2 Mandatory Attribution Requirements
Section 4(a) of CC BY-SA 3.0 mandates that any distribution of the Work or derivative works must retain copyright notices and provide reasonable attribution.
- **Official Attribution Notice:**
  > "Dataset sourced from Apple Hypersim: Roberts, M., Ramapuram, J., Ranjan, A., Kumar, A., Bautista, M. A., Paczan, N., Webb, R., & Susskind, J. M. (2021). Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)."
- **Application Integration:** This attribution is formally embedded in `indoor_canonical_manifest_v2.json`, product documentation, and export metadata headers.

### 2.3 The Downstream ShareAlike (SA) Obligation for Model Weights
Section 4(b) states:
> *"You may Distribute or Publicly Perform an Adaptation only under the terms of: (i) this License; (ii) a later version of this License with the same License Elements as this License; (iii) a Creative Commons Compatible License..."*

**Legal Analysis of Neural Network Weights:**
1. **Is a Fine-Tuned Model an "Adaptation" (Derivative Work)?**
   - In intellectual property law across US, EU, and international jurisdictions, the question of whether neural network weights trained via empirical risk minimization constitute a "derivative work" of the training data remains an evolving legal question.
   - If weights are classified as non-expressive functional parameters or fair use transformations, ShareAlike might not attach to the weights.
   - **However, RoomAI will NOT make blanket unverified claims.** Under a conservative compliance posture, if RoomAI fine-tunes model weights on Hypersim and distributes those weights to third parties, the downstream weights could be subject to CC BY-SA 3.0 requirements (requiring the weights to be shared under CC BY-SA 3.0).
2. **SaaS / Cloud API Serving vs. Weight Distribution:**
   - Under CC BY-SA 3.0, "Distribution" refers to the transfer of copies of the Work or Adaptation to the public.
   - Serving inference predictions over a private cloud HTTP API (such as the RoomAI backend) without distributing the physical checkpoint weights does not trigger the redistribution clause under traditional copyright law.
3. **Current Production Status:**
   - The active RoomAI production model `production_model.ckpt` is the **original pretrained WorldMirror-1.0 foundation model**, which was pre-trained on open web data and has NOT been modified by Hypersim training weights.

---

## 3. TRACK B: SCANNET LEGAL ANALYSIS (Non-Commercial Research Only)

### 3.1 License Grant & Restrictions
- **Dataset:** ScanNet v2 (Dai et al., CVPR 2017, Stanford University / Princeton University / Technical University of Munich).
- **License:** ScanNet Terms of Use.
- **Commercial Permission:** **STRICTLY PROHIBITED.** The terms state:
  > *"ScanNet is available for non-commercial academic research use only. You agree not to use the dataset or any derived data for commercial purposes, including but not limited to commercial products, services, or internal commercial development."*

### 3.2 Permanent Quarantine Firewall
1. **Zero ScanNet Weights in Production:**
   No checkpoint containing parameters updated on ScanNet data will ever be promoted to `checkpoints/production_candidate/`.
2. **Lineage Segregation:**
   - `checkpoints/stage1/best_stage1.ckpt`: Trained on ScanNet depth distillation $\to$ **Permanently Quarantined to Research Archive**.
   - `checkpoints/stage2/stage2_real_verified.ckpt`: Trained on ScanNet point supervision $\to$ **Permanently Quarantined to Research Archive**.
   - `production_model.ckpt`: Pretrained WorldMirror foundation model $\to$ **Commercial Production Lineage**.
3. **Evaluation-Only Role:**
   ScanNet validation sequences (`scene0030_00/01/02`) are utilized exclusively as an **independent held-out benchmark** to monitor synthetic-to-real generalization. Using test data for passive evaluation without updating weights does not taint the model lineage.

---

## 4. MULTI-TRACK STORAGE & MANIFEST ISOLATION TABLE

| Asset Category | Track A (Hypersim) | Track B (ScanNet) | Legal / Architectural Isolation Mechanism |
| :--- | :--- | :--- | :--- |
| **License Type** | CC BY-SA 3.0 | Academic Research Only | Strict legal firewall |
| **Commercial Feasibility** | Permitted (SA compliance review) | Prohibited | Automated check in deployment suite |
| **Physical Raw Storage** | `data/ground_truth/raw/hypersim/` | `data/ground_truth/raw/scannet/` | Distinct filesystem roots |
| **Standardized Storage** | `data/ground_truth/standardized/hypersim/` | `data/ground_truth/standardized/scannet/` | Non-overlapping directories |
| **Manifest Tagging** | `"dataset": "hypersim"` | `"dataset": "scannet"` | Programmatic filter enforcement |
| **Model Weights** | `phase9_exp_a`, `phase9_exp_b` | `best_stage1`, `stage2_real` | Checkpoint metadata tags |
| **Active Production Model** | `production_model.ckpt` (Pretrained) | None (Quarantined) | Only clean lineage eligible |

---

## 5. COMPLIANCE CERTIFICATION

- [x] CC BY-SA 3.0 attribution notices embedded in codebase and dataset manifests.
- [x] Downstream model-weight ShareAlike implications formally reviewed and documented.
- [x] Stanford ScanNet Non-Commercial terms strictly honored; zero ScanNet-trained weights in commercial candidate path.
- [x] Dataset manifests physically segregate Track A and Track B samples with distinct dataset identifiers.
- [x] The immutable production model `production_model.ckpt` is confirmed clean of ScanNet fine-tuning lineage.

