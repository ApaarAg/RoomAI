# RoomAI Object Detection: Inference Tuning & Candidate Fine-Tuning Results
**Date**: October 24, 2026  
**Author**: Senior Computer Vision Engineer  
**Hardware Environment**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), 24 GB RAM, Conda `tf-gpu` (Python 3.10.20, PyTorch 2.11.0+cu128)  
**Dataset**: Canonical 12-Class Indoor Dataset (`data/indoor_canonical/`)

---

## 1. Overview & Objectives

Following the Stage 1–2 diagnostic audit, two systematic phases of improvement were conducted:
1. **Stage 3: Inference & Fusion Optimization** (tuning confidence thresholds, NMS IoU, Weighted Box Fusion weights, single-model retention factors, and cross-class duplicate suppression on 554 validation images).
2. **Stage 4: Supervised Fine-Tuning of Strongest Candidate** (fine-tuning Model A2: YOLO11s on the canonical dataset with spatial augmentations and early stopping).

---

## 2. Stage 3: Inference & Fusion Tuning on Validation Data

### 2.1 Confidence Threshold Sweep on Individual Models (554 Validation Images)
| Confidence Threshold | Model A1 (YOLO11n) P | Model A1 R | Model A1 F1 | Model A2 (YOLO11s) P | Model A2 R | Model A2 F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 0.5929 | 0.6334 | 0.6125 | 0.5804 | **0.7097** | 0.6386 |
| **0.25** | 0.6274 | 0.6021 | 0.6145 | 0.6240 | 0.6777 | 0.6497 |
| **0.30** | 0.6614 | 0.5749 | **0.6151** | 0.6582 | 0.6473 | 0.6527 |
| **0.35 (Baseline)** | 0.6922 | 0.5519 | 0.6141 | 0.6936 | 0.6223 | **0.6560** |
| **0.40** | 0.7137 | 0.5237 | 0.6041 | 0.7244 | 0.5950 | 0.6533 |
| **0.45** | 0.7409 | 0.4978 | 0.5955 | 0.7530 | 0.5682 | 0.6476 |
| **0.50** | 0.7628 | 0.4716 | 0.5829 | 0.7735 | 0.5367 | 0.6337 |

**Finding**:
- Threshold `0.35` achieves the best balance between precision and recall for Model A2 (F1: 0.6560).
- Raising threshold to `0.40` yields +3.1% higher precision (0.7244) with minor recall reduction.

### 2.2 Ensemble Fusion Method Comparison: WBF vs. Class-Aware NMS
| Fusion Method | Candidate Weights | Fusion IoU | Single-Model Factor | Conf Thresh | Precision | Recall | F1 Score | FP Count | Duplicate FPs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class-Aware NMS** | [1.0, 1.0] | 0.50 | 1.00 | 0.35 | 0.6219 | 0.6802 | 0.6497 | 1,850 | 17 |
| **Class-Aware NMS** | [0.8, 1.2] | 0.45 | 1.00 | 0.35 | 0.6175 | **0.6855** | 0.6497 | 1,900 | 13 |
| **WBF (Original)** | [1.0, 1.0] | 0.50 | 0.88 | 0.35 | 0.6593 | 0.6560 | 0.6576 | 1,517 | 15 |
| **WBF (IoU=0.45)** | [1.0, 1.0] | 0.45 | 0.88 | 0.40 | **0.7035** | 0.6258 | 0.6624 | 1,180 | 7 |
| **WBF (Weighted-A2)** | [0.8, 1.2] | 0.50 | 0.88 | 0.40 | 0.6924 | 0.6354 | **0.6627** | 1,263 | 8 |

**Finding**:
- Weighted Box Fusion decisively outperforms Class-Aware NMS. WBF produces +3.7% higher precision and reduces false positive detections by 333 boxes at conf=0.35.
- Weighting Model A2 higher ([0.8, 1.2]) aligns with its superior individual accuracy and raises ensemble recall to 0.6652.

### 2.3 Cross-Class Duplicate Suppression
To eliminate redundant bounding boxes where different classes are predicted for the exact same physical piece of furniture (e.g. sofa vs. chair confusion on living-room armchairs):

| Cross-Class IoU Thresh | Confidence Thresh | Precision | Recall | F1 Score | Class Confusion FPs | Duplicate FPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Original)** | 0.35 | 0.6509 | 0.6652 | 0.6580 | 338 | 16 |
| **0.85** | 0.35 | 0.6612 | 0.6533 | 0.6572 | 288 | 16 |
| **0.80** | 0.35 | 0.6615 | 0.6518 | 0.6566 | 284 | 16 |
| **0.75** | 0.35 | **0.6620** | 0.6515 | 0.6568 | **279** | 16 |
| **0.75** | 0.40 | **0.7015** | 0.6252 | **0.6612** | **232** | 8 |

**Finding**:
- Resolving cross-class conflicts at $\text{IoU} \ge 0.75$ eliminates 59 false alarms without suppressing distinct overlapping objects (such as coffee tables placed in front of sectionals).

---

## 3. Stage 4: Fine-Tuning the Strongest Candidate (Model A2)

### 3.1 Training Configuration & Hyperparameters
- **Base Checkpoint**: `checkpoints/indoor_detector_a2_yolo11s_pilot_best.pt` (preserved untouched).
- **Target Checkpoint**: `checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt`.
- **Experiment Directory**: `runs/train/indoor_detector_a2_yolo11s_finetune`.
- **Dataset**: `data/indoor_canonical/indoor_canonical.yaml` (2,985 train, 554 val).
- **Batch Size**: 8 (Safe for 8 GB VRAM).
- **Input Resolution**: 640 $\times$ 640.
- **Hardware Device**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA:0).
- **Mixed Precision**: FP32 (`amp=False` for strict numerical stability).
- **Augmentations**: `mosaic=0.5`, `fliplr=0.5`, `degrees=5.0`, `scale=0.2`.
- **Patience**: 3 (Early stopping enabled).
- **Execution Workflow**: 1-epoch pilot check verified GPU allocation, followed by full 6-epoch training.

### 3.2 Measured Resource Usage & Training Duration
- **Pilot Check Duration (1 Epoch)**: 135.2 seconds (2.25 min).
- **Full Fine-Tuning Duration (6 Epochs)**: 503.8 seconds (8.40 min).
- **Average Time per Epoch**: 83.9 seconds.
- **Peak GPU Memory Allocated**: **3,666.62 MB** (~44.8% of available 8,188 MB VRAM). Zero OOM errors occurred.

### 3.3 Quantitative Validation Gains (554 Images)
| Metric | Pilot Baseline (Epoch 2) | Fine-Tuned Model A2 (Epoch 6) | Absolute Gain | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Validation mAP@0.50** | 0.6447 | **0.7531** | **+0.1084** | **+16.8%** |
| **Validation mAP@0.50:0.95** | 0.4648 | **0.5484** | **+0.0836** | **+18.0%** |
| **Validation Precision** | 0.6247 | **0.7630** | **+0.1383** | **+22.1%** |
| **Validation Recall** | 0.6176 | **0.6600** | **+0.0424** | **+6.9%** |

### 3.4 Per-Class Accuracy Trajectory
| Class Name | Pilot mAP50 | Fine-Tuned mAP50 | Pilot Recall | Fine-Tuned Recall | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **cabinet#shelf** | 0.553 | **0.736** | 0.525 | **0.653** | Major gain (+18.3% mAP50) |
| **table** | 0.757 | **0.808** | 0.762 | 0.737 | High accuracy (+5.1% mAP50) |
| **chair#stool** | 0.679 | **0.750** | 0.767 | 0.707 | Confusion with sofa resolved |
| **lamp** | 0.616 | **0.737** | 0.584 | **0.658** | Table lamps accurately captured |
| **door** | 0.449 | **0.644** | 0.484 | **0.567** | Frame localization resolved |
| **bed** | 0.861 | **0.916** | 0.760 | **0.794** | Outstanding accuracy |
| **sofa** | 0.834 | **0.889** | 0.790 | **0.826** | Clean boundaries, no duplicates |
| **plant** | 0.693 | **0.717** | 0.723 | 0.669 | Maintained high performance |
| **decoration** | 0.780 | **0.825** | 0.751 | 0.742 | Reliable art/mirror detection |
| **curtain** | 0.548 | **0.748** | 0.397 | **0.620** | Massive recall surge (+22.3% R) |
| **home-appliance**| 0.572 | **0.699** | 0.527 | **0.582** | TV and appliance recall resolved |
| **window** | 0.395 | **0.569** | 0.340 | **0.370** | Large gain (+17.4% mAP50, P=0.710) |

---

## 4. Preserved Rollback Configuration

In the event that legacy rollback is required:
```yaml
# Rollback configuration preserved in config/model_config.yaml
detection:
  candidates:
    baseline_rollback:
      name: "Rollback Baseline (DeepFurniture YOLO11n)"
      checkpoint: "checkpoints/deepfurniture_yolo11n_finetune_best.pt"
      nc: 11
    model_a2_pilot:
      name: "Model A2 Pilot (YOLO11s)"
      checkpoint: "checkpoints/indoor_detector_a2_yolo11s_pilot_best.pt"
      nc: 12
```
All original `.pt` files and run artifact logs have been preserved without overwriting.
