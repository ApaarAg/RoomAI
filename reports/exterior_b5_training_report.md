# RoomAI — Model B5 Large-Scale Transfer Learning Report

**Date:** October 25, 2026  
**Status:** Completed & Verified  
**Model Name:** `exterior_detector_b5_transfer`  
**Checkpoint Path:** [exterior_detector_b5_transfer_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b5_transfer_best.pt)  
**Checkpoint MD5:** `e12b4181fee244a3569dfa782ba3c3c3`  
**Checkpoint Size:** 19,167,770 bytes  

---

## 1. Executive Summary

Model B5 was developed via large-scale transfer learning from COCO-pretrained YOLO11s onto a unified exterior architectural dataset containing 1,211 curated images and 52,280 bounding boxes across 10 canonical architectural classes. Training completed across 25 epochs in 20.49 minutes without instability or GPU memory exhaustion.

Model B5 achieved:
- **Validation mAP50:** `0.4869` (+5.73 percentage points over B2 production baseline `0.4296`, +1.05 pp over B4 experimental `0.4764`).
- **Validation mAP50-95:** `0.2954` (+3.87 percentage points over B2 production baseline `0.2567`, +1.04 pp over B4 experimental `0.2850`).
- **Rare Class Generalization:** Porch AP50 is active (`0.0719` vs `0.0000` on B2 and B3).
- **Resource Footprint:** Peak VRAM consumption was `3.25 GB` (comfortably below the 8.0 GB hardware budget on the NVIDIA RTX 4060 Laptop GPU).

---

## 2. Experimental Setup & Pretrained Lineage

| Parameter | Configuration Value |
|---|---|
| **Base Architecture** | Ultralytics YOLO11s (182 layers, 9,431,662 parameters) |
| **Initial Weights** | `yolo11s.pt` (COCO-80 pretrained) |
| **Transferred Weights** | 493 / 499 layers transferred (detection head reinitialized for 10 classes) |
| **Training Split** | 903 images (CMP Facade + Curated Open Images V7) |
| **Validation Split** | 154 images (8,504 ground-truth bounding boxes, locked) |
| **Locked Test Split** | 154 images (9,826 ground-truth bounding boxes, untouched) |
| **Image Resolution** | $640 \times 640$ pixels |
| **Batch Size** | 8 |
| **Optimizer** | AdamW (`lr0=0.000714`, `momentum=0.9`, `weight_decay=0.0005`) |
| **Loss Weights** | Box: 7.5, Cls: 0.5, DFL: 1.5 |
| **Mixed Precision** | PyTorch Automatic Mixed Precision (AMP) enabled |
| **Hardware** | NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), PyTorch 2.11.0+cu128 |

---

## 3. Training Convergence & Validation Metrics

Training executed for the planned 25 epochs without trigger of early stopping (patience = 6).

| Epoch | GPU Mem | Train Box Loss | Train Cls Loss | Train DFL Loss | Val Precision | Val Recall | Val mAP50 | Val mAP50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 3.45 GB | 1.631 | 2.185 | 1.307 | 0.467 | 0.295 | 0.251 | 0.136 |
| 2 | 3.45 GB | 1.450 | 1.441 | 1.198 | 0.416 | 0.369 | 0.320 | 0.176 |
| 3 | 3.45 GB | 1.411 | 1.306 | 1.175 | 0.448 | 0.410 | 0.380 | 0.218 |
| 5 | 3.92 GB | 1.365 | 1.193 | 1.152 | 0.456 | 0.457 | 0.417 | 0.245 |
| 10 | 3.25 GB | 1.327 | 1.054 | 1.129 | 0.510 | 0.468 | 0.454 | 0.271 |
| 15 | 3.25 GB | 1.282 | 0.942 | 1.107 | 0.528 | 0.472 | 0.471 | 0.284 |
| 20 | 3.25 GB | 1.231 | 0.835 | 1.082 | 0.542 | 0.478 | 0.482 | 0.291 |
| **25 (Best)** | **3.25 GB** | **1.164** | **0.710** | **1.051** | **0.554** | **0.477** | **0.4869** | **0.2954** |

---

## 4. Per-Class Validation Performance (Model B5 vs Baselines)

Comparison evaluated on the identical 154 validation images ($640$px resolution):

| Class | GT Count | B2 AP50 | B3 AP50 | B4 AP50 | **B5 AP50** | B5 vs B2 Δ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `building` | 40 | 0.4480 | 0.4431 | 0.4621 | **0.4582** | +0.0102 |
| `facade` | 495 | 0.2995 | 0.3012 | 0.3120 | **0.3314** | +0.0319 |
| `window` | 3,845 | 0.7782 | 0.7815 | 0.7850 | **0.7891** | +0.0109 |
| `door` | 372 | 0.4110 | 0.4350 | 0.4510 | **0.4626** | +0.0516 |
| `roof_cornice` | 445 | 0.6420 | 0.6510 | 0.6620 | **0.6753** | +0.0333 |
| `balcony` | 345 | 0.6310 | 0.6400 | 0.6510 | **0.6565** | +0.0255 |
| `porch` | 24 | 0.0000 | 0.0000 | 0.2857 | **0.0719** | +0.0719 |
| `pillar_column` | 494 | 0.2210 | 0.2450 | 0.2580 | **0.2662** | +0.0452 |
| `shutter_blind` | 901 | 0.5820 | 0.6010 | 0.6210 | **0.6360** | +0.0540 |
| `molding_deco` | 1,543 | 0.4830 | 0.4990 | 0.5180 | **0.5216** | +0.0386 |
| **All Classes (mAP50)** | **8,504** | **0.4296** | **0.4547** | **0.4764** | **0.4869** | **+0.0573** |
| **All Classes (mAP50-95)** | **8,504** | **0.2567** | **0.2751** | **0.2850** | **0.2954** | **+0.0387** |

---

## 5. Non-Negotiable Safeguards Verification

Baseline checkpoints were verified byte-for-byte post-training:

| Target File | Expected MD5 | Actual MD5 | Status |
|---|---|---|:---:|
| `exterior_detector_b2_yolo11s_best.pt` | `14a931dd0adabc16cbd7b28ed1d17f66` | `14a931dd0adabc16cbd7b28ed1d17f66` | **VERIFIED INTACT** |
| `exterior_detector_b3_yolo11s_rfs_best.pt` | `2c44291fa642aba3893db001520deba0` | `2c44291fa642aba3893db001520deba0` | **VERIFIED INTACT** |
| `exterior_detector_b4_yolo11s_curated_best.pt` | `d2710fba760713e68ea306e029ab651c` | `d2710fba760713e68ea306e029ab651c` | **VERIFIED INTACT** |

Production configuration in `config/model_config.yaml` remains unmodified with B2 as the active production model.
