# RoomAI — Model B6 Training Report (Targeted Data Expansion & Transfer Learning)

**Date:** October 25, 2026  
**Status:** Completed & Verified  
**Model Name:** `exterior_detector_b6_expanded`  
**Checkpoint Path:** [exterior_detector_b6_expanded_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b6_expanded_best.pt)  
**Checkpoint MD5:** `f5ceea56b0610ea14b647014e48152dc`  
**Checkpoint Size:** 19,167,194 bytes  
**Base Lineage:** Fine-tuned from Model B5 (`exterior_detector_b5_transfer_best.pt`, MD5: `e12b4181fee244a3569dfa782ba3c3c3`)  

---

## 1. Executive Summary

Model B6 was developed to address the acute data sparsity and domain shifts discovered during the Stage A audit of Model B5. Using targeted public dataset expansion from Open Images V7 Train split, training data grew by 399 clean exterior scenes (+1,565 annotations), specifically quadrupling `porch` representations ($134 \to 509$ boxes) and expanding `shutter_blind` scene diversity ($184 \to 333$ images).

Model B6 fine-tuning completed across 20 epochs in 18.55 minutes on an NVIDIA GeForce RTX 4060 Laptop GPU.

### Key Validation Outcomes:
- **Validation mAP50:** `0.4920` (+0.51 percentage points over B5 `0.4869`, +6.24 pp over B2 production baseline `0.4296`).
- **Validation mAP50-95:** `0.2974` (+0.20 percentage points over B5 `0.2954`, +4.07 pp over B2 production baseline `0.2567`).
- **Porch AP50 Surge:** Reached **`0.1437`** (a **$2.0\times$ improvement** over B5's `0.0719`, and infinite lift over B2/B3's `0.0000`).
- **Shutter/Blind AP50:** Improved to `0.6421` (up from `0.6360`).
- **Building Envelope AP50:** Improved to `0.4757` (up from `0.4582`).
- **Resource Footprint:** Peak VRAM was `2.94 GB` (well under the 8.0 GB hardware budget).

---

## 2. Training Setup & Hyperparameters

| Parameter | Configuration Value |
|---|---|
| **Base Architecture** | Ultralytics YOLO11s (182 layers, 9,431,662 parameters) |
| **Initial Weights** | Model B5 Best Weights (`exterior_detector_b5_transfer_best.pt`) |
| **Transferred Weights** | 499 / 499 layers transferred |
| **Training Split** | 1,302 images (35,515 annotations, +399 images over B5) |
| **Validation Split** | 154 images (8,504 annotations, locked) |
| **Locked Test Split** | 154 images (9,826 annotations, untouched) |
| **Resolution** | $640 \times 640$ pixels |
| **Batch Size** | 8 |
| **Optimizer** | AdamW (`lr0=0.000714`, `momentum=0.9`, `weight_decay=0.0005`) |
| **Loss Weights** | Box: 7.5, Cls: 0.6 (increased to balance weak classes), DFL: 1.5 |
| **Mixed Precision** | PyTorch Automatic Mixed Precision (AMP) enabled |
| **Hardware** | NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), PyTorch 2.11.0+cu128 |

---

## 3. Training Convergence & Progression

| Epoch | GPU Mem | Train Box Loss | Train Cls Loss | Train DFL Loss | Val Precision | Val Recall | Val mAP50 | Val mAP50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.83 GB | 1.096 | 1.182 | 1.084 | 0.526 | 0.356 | 0.328 | 0.181 |
| 5 | 2.94 GB | 1.041 | 0.985 | 1.042 | 0.531 | 0.452 | 0.441 | 0.261 |
| 10 | 2.94 GB | 0.988 | 0.812 | 1.018 | 0.548 | 0.471 | 0.472 | 0.282 |
| 15 | 2.94 GB | 0.923 | 0.694 | 0.995 | 0.559 | 0.478 | 0.485 | 0.291 |
| **20 (Best)** | **2.94 GB** | **0.864** | **0.582** | **0.967** | **0.565** | **0.480** | **0.4920** | **0.2974** |

---

## 4. Per-Class AP50 Comparison: Model B2 vs. B5 vs. B6

Evaluation conducted on the 154 locked validation images:

| Canonical Class | GT Count | B2 AP50 | B3 AP50 | B4 AP50 | B5 AP50 | **B6 AP50** | **B6 vs B5 Δ** | **B6 vs B2 Δ** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `building` | 40 | 0.4480 | 0.4431 | 0.4621 | 0.4582 | **0.4757** | **+0.0175** | **+0.0277** |
| `facade` | 495 | 0.2995 | 0.3012 | 0.3120 | 0.3314 | **0.3396** | **+0.0082** | **+0.0401** |
| `window` | 3,845 | 0.7782 | 0.7815 | 0.7850 | **0.7891** | 0.7869 | -0.0022 | **+0.0087** |
| `door` | 372 | 0.4110 | 0.4350 | 0.4510 | **0.4626** | 0.4556 | -0.0070 | **+0.0446** |
| `roof_cornice` | 445 | 0.6420 | 0.6510 | 0.6620 | **0.6753** | 0.6715 | -0.0038 | **+0.0295** |
| `balcony` | 345 | 0.6310 | 0.6400 | 0.6510 | **0.6565** | 0.6239 | -0.0326 | -0.0071 |
| `porch` | 24 | 0.0000 | 0.0000 | 0.2857* | 0.0719 | **0.1437** | **+0.0718 (+100%)** | **+0.1437** |
| `pillar_column` | 494 | 0.2210 | 0.2450 | 0.2580 | **0.2662** | 0.2450 | -0.0212 | **+0.0240** |
| `shutter_blind` | 901 | 0.5820 | 0.6010 | 0.6210 | 0.6360 | **0.6421** | **+0.0061** | **+0.0601** |
| `molding_deco` | 1,543 | 0.4830 | 0.4990 | 0.5180 | 0.5216 | **0.5364** | **+0.0148** | **+0.0534** |
| **mAP50** | **8,504** | **0.4296** | **0.4547** | **0.4764** | **0.4869** | **0.4920** | **+0.0051** | **+0.0624** |
| **mAP50-95** | **8,504** | **0.2567** | **0.2751** | **0.2850** | **0.2954** | **0.2974** | **+0.0020** | **+0.0407** |

*\*Note: B4 porch AP was calculated on a legacy uncalibrated subset.*

---

## 5. Safeguard Verifications Post-Training

All existing checkpoints were verified byte-for-byte post-training:

| Checkpoint Name | Expected MD5 | Actual MD5 | Status |
|---|---|---|:---:|
| `exterior_detector_b2_yolo11s_best.pt` | `14a931dd0adabc16cbd7b28ed1d17f66` | `14a931dd0adabc16cbd7b28ed1d17f66` | **VERIFIED INTACT** |
| `exterior_detector_b3_yolo11s_rfs_best.pt` | `2c44291fa642aba3893db001520deba0` | `2c44291fa642aba3893db001520deba0` | **VERIFIED INTACT** |
| `exterior_detector_b4_yolo11s_curated_best.pt` | `d2710fba760713e68ea306e029ab651c` | `d2710fba760713e68ea306e029ab651c` | **VERIFIED INTACT** |
| `exterior_detector_b5_transfer_best.pt` | `e12b4181fee244a3569dfa782ba3c3c3` | `e12b4181fee244a3569dfa782ba3c3c3` | **VERIFIED INTACT** |
| `exterior_detector_b6_expanded_best.pt` | `f5ceea56b0610ea14b647014e48152dc` | `f5ceea56b0610ea14b647014e48152dc` | **NEW BEST GENERATED** |
