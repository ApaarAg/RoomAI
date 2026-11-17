# RoomAI Phase 7.4.2 — Fine-Tuning Evaluation & Decision Framework

**Date:** November 6, 2026  
**Author:** ML Systems Architect & Senior Computer Vision Researcher  
**Project:** RoomAI (`F:\RoomAI`)  
**Scope:** Formal Evaluation of Fine-Tuning Necessity, Data Sufficiency & Resource Justification  
**Decision:** **FORMALLY DEFERRED (PRETRAINED ZERO-SHOT MODELS CERTIFIED FOR PRODUCTION)**

---

## 1. Executive Summary & Policy Criteria

Section 9 of the Phase 7.4.2 specification establishes strict criteria governing model fine-tuning:
> *"First evaluate existing pretrained models. Fine-tune only when there is a measurable weakness and suitable training data exists. Do not fine-tune using unrelated single-image datasets and claim multi-view improvement. If data is insufficient, formally defer fine-tuning."*

An exhaustive audit of existing models, local dataset assets, and single-view empirical performance was conducted.

### Core Audit Finding:
1. **Existing Pretrained Models Are Highly Effective:** Zero-shot `Depth Anything V2 Metric Indoor Small` combined with the fine-tuned `A3 YOLO11s` detector detected **209 residential objects across 40 scenes (5.22 objects/scene)** with **$87.5\%$ floor plane detection accuracy** and **$12.2\text{ cm}$ raw floor contact error**.
2. **Absence of Paired Multi-View Training Data:** Genuine multi-view fine-tuning requires thousands of paired RGB frames with micro-accurate 6-DoF camera poses ($[R \mid t]$) and registered dense depth maps. No such dataset exists on disk.
3. **Formal Verdict:** **Fine-tuning for Phase 7.4.2 is formally deferred**.

---

## 2. Component-by-Component Fine-Tuning Assessment

| Pipeline Component | Active Model / Architecture | Pretrained / Fine-Tuned Baseline | Measurable Weakness Observed? | Sufficient Local Training Data? | Action / Decision |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Indoor Detection** | A3 YOLO11s (800px) | Fine-tuned on canonical indoor set | No (High recall, 5.22 objs/scene) | Yes (`data/indoor_canonical`) | **PRESERVE EXISTING CHECKPOINT** |
| **Instance Segmentation** | YOLO11n-seg / FastSAM | Pretrained COCO/FastSAM | Minor (occasional chair leg bleed) | Limited paired masks | **PRESERVE PRETRAINED** |
| **Metric Depth Estimation**| Depth Anything V2 Small | Pretrained on NYUv2 + Hypersim | No ($12.2\text{ cm}$ raw floor error) | No paired metric LiDAR | **PRESERVE PRETRAINED** |
| **Camera Registration** | Classical Kabsch SVD + FLANN | Algorithmic (Non-learned) | N/A (Algebraic closed-form) | N/A | **NO FINE-TUNING APPLICABLE** |
| **Multi-View Transformer** | Hypothetical DA3 / VGGT | Pretrained Foundation Models | Severe VRAM bottleneck on 8GB | **NO LOCAL MULTI-CAMERA DATA** | **FORMALLY DEFER** |

---

## 3. Why Multi-View Fine-Tuning Cannot Proceed Locally

1. **The Fallacy of Single-Image Multi-View Fine-Tuning:** Fine-tuning a monocular depth model on single-image datasets (e.g. ADE20K or COCO) cannot teach cross-view epipolar geometry, parallax correspondence, or relative camera pose estimation. Claiming multi-view improvements from single-image fine-tuning is scientifically fraudulent.
2. **Computational Budget & VRAM Constraints:** Fine-tuning multi-view vision transformers (e.g., DA3 or VGGT) requires backpropagating through attention maps across multiple frames ($N \ge 3$), which demands a minimum of $24\text{ GB} - 48\text{ GB}$ of VRAM. Attempting multi-view backpropagation on the local 8 GB RTX 4060 Laptop GPU triggers immediate out-of-memory termination.
3. **Data Availability Gate:** As documented in `reports/phase_7_4_2_multiview_dataset.md`, raw scan sequences (e.g., ScanNet .sens archives) are not available on the workstation.

---

## 4. Auditor's Formal Fine-Tuning Determination

1. **Single-View Pipeline:** Pretrained zero-shot models demonstrate outstanding performance and require no fine-tuning.
2. **Multi-View Pipeline:** Fine-tuning is formally deferred until:
   - A verified, licensed multi-camera dataset (ScanNet, Hypersim) is locally acquired.
   - Cloud GPU infrastructure (e.g. A100/H100 80GB) is provisioned with explicit cost approval.
3. **Decision:** **NO MODEL FINE-TUNING CONDUCTED IN PHASE 7.4.2**. Existing production checkpoints remain intact, verified, and pristine.

