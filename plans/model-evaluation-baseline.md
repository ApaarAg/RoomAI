# RoomAI Model Evaluation — Baseline & Environment Freeze

**Date:** 2026-11-16  
**Phase:** Phase 0 — Freeze Current State & Baseline Audit  
**Target Repository:** `F:/RoomAI`  
**Git Commit:** `8f4bf662eb73518866f516710b68a99c6b03f37a`  
**Hardware & System Profile:**
- **OS:** Windows 11
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB Dedicated VRAM)
- **CUDA Driver / Runtime:** CUDA 12.1 / PyTorch 2.5.1+cu121
- **Python Environment:** Python 3.10 (Conda `tf-gpu` environment)
- **Deep Learning Frameworks:** PyTorch 2.5.1, Ultralytics YOLO 8.3+, HuggingFace Transformers

---

## 1. Scope & Objective

This document captures the frozen state of all models, checkpoints, dataset splits, configurations, and evaluation infrastructure in RoomAI prior to conducting comparative evaluation.

The goal is to determine which existing trained checkpoint is optimal for RoomAI across its core spatial perception tasks, compare trained models against appropriate pretrained baselines and historical baselines, and establish with empirical evidence whether further training is justified.

**Strict Freeze Rules:**
- NO new training runs or fine-tuning will be executed.
- NO checkpoint weights will be modified.
- NO test-set threshold tuning or data regeneration will occur.

---

## 2. Model Inventory & Checkpoint Registry

### Group A: Indoor Detection & Household Furniture
| Model Identifier | Architecture | Checkpoint Path | Resolution | Size (MB) | Classes | Status |
|---|---|---|---|---|---|---|
| **Model A3 (Production Primary)** | YOLO11s | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | 800×800 | 18.29 MB | 12 | Active Primary |
| **Model A2 (Fine-Tuned Rollback)** | YOLO11s | `checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt` | 640×640 | 18.27 MB | 12 | Rollback Candidate |
| **Model A1 (Expanded Pilot)** | YOLO11n | `checkpoints/indoor_detector_a1_yolo11n_pilot_best.pt` | 640×640 | 5.19 MB | 12 | Lightweight Pilot |
| **DeepFurniture Baseline** | YOLO11n | `checkpoints/deepfurniture_yolo11n_finetune_best.pt` | 640×640 | 5.21 MB | 11 | Historical Baseline |
| **Pretrained COCO Baseline (s)** | YOLO11s | `yolo11s.pt` | 640 / 800 | 18.28 MB | 80 (COCO) | Pretrained Reference |
| **Pretrained COCO Baseline (n)** | YOLO11n | `yolo11n.pt` | 640 / 800 | 5.20 MB | 80 (COCO) | Pretrained Reference |

### Group B: Exterior Architecture & Building Components
| Model Identifier | Architecture | Checkpoint Path | Resolution | Size (MB) | Classes | Status |
|---|---|---|---|---|---|---|
| **Model B2 (Production Default)** | YOLO11s | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | 640×640 | 18.28 MB | 10 | Production Default |
| **Model B6 (Expanded Candidate)** | YOLO11s | `checkpoints/exterior_detector_b6_expanded_best.pt` | 640×640 | 18.28 MB | 10 | Validated Candidate |
| **Model B7 (Pilot Candidate)** | YOLO11s | `checkpoints/exterior_detector_b7_pilot_best.pt` | 640×640 | 18.28 MB | 10 | Experimental Pilot |
| **Candidate C1 (Phase 22)** | YOLO11s | `checkpoints/exterior_detector_c1_phase22_best.pt` | 640×640 | 18.28 MB | 10 | Experimental |
| **Porch Specialist** | YOLO11s | `checkpoints/exterior_specialist_porch_yolo11s_best.pt` | 640×640 | 18.27 MB | 10 | Domain Specialist |

### Group C: Parking & Outdoor Property
| Model Identifier | Architecture | Checkpoint Path | Resolution | Size (MB) | Classes | Status |
|---|---|---|---|---|---|---|
| **Model C2 (Production Primary)** | YOLO11s | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | 640×640 | 18.27 MB | 8 | Production Primary |
| **Model C1 (Lightweight Pilot)** | YOLO11n | `checkpoints/parking_detector_c1_yolo11n_pilot_best.pt` | 640×640 | 5.20 MB | 8 | Lightweight Pilot |
| **Model C3 (RFS Resampled)** | YOLO11s | `checkpoints/parking_detector_c3_yolo11s_rfs_best.pt` | 640×640 | 18.27 MB | 8 | Experimental |

### 3D Vision & Reconstruction
| Model Identifier | Architecture | Checkpoint Path | Input Size | Size | Mode |
|---|---|---|---|---|---|
| **Depth Anything V2 Small** | DINOv2 ViT-S + DPT | `checkpoints/depth_anything_v2_small/` | 518×518 | 94.58 MB | Relative Inverse Depth |
| **Depth Anything V2 Metric** | DINOv2 ViT-S + DPT | `checkpoints/depth_anything_v2_metric_indoor_small/` | 518×518 | 94.58 MB | Metric Depth (meters) |
| **WorldMirror 1.0 Baseline** | Hunyuan Transformer 1.2B | `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt` | 518×518 | 2.84 GB | Foundation Baseline |
| **WorldMirror 1.0 LoRA Rank 8** | Hunyuan Transformer 1.2B | `results/final_execution/checkpoints/exp_05_expanded_lora_rank8_final.ckpt` | 518×518 | 2.45 GB | Fine-Tuned Stage 1 |

---

## 3. Dataset & Split Definitions

| Dataset | Domain | Train Images | Val Images | Test Images (Quarantined) | Total |
|---|---|---|---|---|---|
| **`indoor_canonical`** | Indoor Residential Furniture & Architecture | 2,985 | 554 | 150 | 3,689 |
| **`exterior_b6_expanded`** | Exterior Facades & Building Elements | 1,302 | 154 | 154 | 1,610 |
| **`exterior_canonical`** | Exterior Facades Baseline | 458 | 134 | 134 | 726 |
| **`parking_canonical`** | Parking Lots, Outdoor Fixtures & Vehicles | 510 | 120 | 120 | 750 |
| **`deepfurniture`** | Public Furniture Dataset | 700 | 150 | 150 | 1,000 |
| **`ADEChallengeData2016`** | Residential Scene Parsing (ADE20K) | 4,800 (Pool) | 200 | 200 (Golden Test) | 5,200 |

### Split Verification:
- All datasets feature mutually disjoint image directories: `images/train`, `images/val`, and `images/test`.
- Labels are partitioned synchronously: `labels/train`, `labels/val`, and `labels/test`.
- Cryptographic SHA-256 and perceptual pHash checks across historical audits confirm zero byte-for-byte leaks or scene-level cross-contamination between train and test splits.

---

## 4. Preprocessing & Transform Assumptions

- **YOLO Detection Models:**
  - Dynamic letterbox padding preserving aspect ratio.
  - Pixel normalization: $[0, 255] \to [0.0, 1.0]$.
  - Input resolution: 800×800 for Model A3; 640×640 for all other YOLO detectors.
  - Standard non-maximum suppression (NMS) with `iou_threshold=0.45`, `conf_threshold=0.25` or `0.35`.
- **Depth Models:**
  - Bilinear resize and central alignment to 518×518.
  - ImageNet channel normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- **Hunyuan 3D Reconstruction:**
  - Multi-view / single-view ray alignment at 518×518.
  - Metric unprojection utilizing camera intrinsics and estimated depth boundaries.

---

## 5. Known Limitations & Failure Modes Documented

1. **Extreme Class Imbalance in Exterior Domain:**
   - Abundant classes (`window`, `door`, `facade`) dominate sample volume; rare classes (`porch`, `shutter_blind`) suffer lower recall.
2. **COCO vs Domain Label Discrepancy:**
   - Standard COCO pretrained YOLO models lack specialized architectural classes (`facade`, `roof_cornice`, `porch`, `cabinet#shelf`, `curtain`). Evaluating COCO weights directly on domain-specific datasets requires zero-shot mapping or class-subset evaluation.
3. **Small-Object False Negatives:**
   - Small lamps, decorative fixtures, and thin chair legs exhibit precision dips at 640px; higher resolution (800px) mitigates this but increases latency and VRAM footprint.
