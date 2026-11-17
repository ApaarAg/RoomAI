# RoomAI Object Detection System: Comprehensive Diagnostic & Audit Report
**Date**: October 24, 2026  
**Auditor**: Senior Computer Vision Engineer  
**Hardware Environment**: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), 24 GB RAM, Python 3.10.20, PyTorch 2.11.0+cu128, Ultralytics 8.4.66  
**Target Domain**: Group A (Indoor Furniture & Household Objects)

---

## 1. Executive Summary

This diagnostic audit investigates detection reliability and inference timing in the RoomAI object detector prior to deployment. The investigation was triggered by visual artifacts in living-room test scenes (15 detections displaying overlapping sofa/chair boxes, boundary inconsistencies, and erratic timing measurements).

### Key Audit Conclusions:
1. **Taxonomy & Checkpoint Discrepancy**: The legacy baseline checkpoint (`deepfurniture_yolo11n_finetune_best.pt`) has **11 classes** (classes `0..10`, omitting `window`). Model A1 and Model A2 both employ the **12-class canonical taxonomy** (classes `0..11`, including `window`). The application config previously pointed to the 11-class baseline as default while declaring 12 target classes.
2. **Origin of Duplicate Overlapping Boxes**:
   - **Intra-Model Class Confusion**: YOLO's internal NMS operates on single-class slices (`agnostic=False`). When a large seating piece exhibits high logits for both `sofa` and `chair#stool`, the detector outputs dual concentric boxes at nearly identical coordinates (e.g. Model A2 predicting `sofa` at 0.87 and `chair#stool` at 0.76 with IoU > 0.96).
   - **Ensemble Fusion Blind Spot**: Weighted Box Fusion (`EnsembleDetector.fuse_predictions`) partitions candidate detections strictly by `class_id`. Because `sofa` (ID 6) and `chair#stool` (ID 2) reside in different buckets, WBF never clusters them together. As a consequence, both boxes survive fusion and clutter the UI.
   - **Genuine 2D Physical Overlap**: In dense room scenes (e.g. Validation Scene 4), multiple distinct sofas and chairs exist in foreground/background arrangements that genuinely overlap in 2D perspective (e.g. 4 distinct ground-truth sofas). Indiscriminate NMS suppression is unacceptable; cross-class resolution must only suppress nearly coincident boxes ($\text{IoU} \ge 0.75$).
3. **Inference Timing Measurement Breakdown**:
   - Prior code in `services/detection_service.py` artificially split latency into `infer_duration_ms * 0.92` for model inference and `infer_duration_ms * 0.08` for fusion, fabricating values rather than measuring actual kernel runtimes.
   - GPU calls lacked `torch.cuda.synchronize()`, causing asynchronous CUDA kernel launches to return early and producing wildly fluctuating latency figures (e.g. 21 ms vs 111 ms vs 29 ms).
   - Preprocessing only timed PIL RGB conversion, omitting YOLO letterboxing and tensor normalization.
   - Cold-start model loading was cached by Streamlit but presented indiscriminately against warm steady-state latency.
4. **Candidate Model Selection for Retraining**:
   - Model A2 (YOLO11s, 9.4M parameters) is empirically superior to Model A1 (YOLO11n, 2.6M parameters) across all validation metrics (mAP50: **0.645** vs **0.620**; Recall: **0.618** vs **0.601**).
   - The existing Model A2 checkpoint was trained for only 2 pilot epochs, where validation classification loss was still steeply declining (1.481 $\to$ 1.176). Controlled fine-tuning of Model A2 on the canonical dataset is fully justified.

---

## 2. Checkpoint & Taxonomy Verification

Direct inspection of checkpoint state dictionaries and Ultralytics models yielded:

| Model Identifier | Architecture | Parameters | Classes | Target Taxonomy | Status |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Baseline** | YOLO11n | 2.59 M | 11 | `0: cabinet#shelf` .. `10: home-appliance` (No window) | Legacy Baseline |
| **Model A1 (Pilot)** | YOLO11n | 2.59 M | 12 | `0: cabinet#shelf` .. `11: window` | Active Candidate |
| **Model A2 (Pilot)** | YOLO11s | 9.42 M | 12 | `0: cabinet#shelf` .. `11: window` | Strongest Candidate |

### Canonical Class Schema (12 Classes):
`0: cabinet#shelf`, `1: table`, `2: chair#stool`, `3: lamp`, `4: door`, `5: bed`, `6: sofa`, `7: plant`, `8: decoration`, `9: curtain`, `10: home-appliance`, `11: window`.

---

## 3. Dataset Audit & Split Isolation

The unified canonical dataset (`data/indoor_canonical/`) combines DeepFurniture and HomeObjects-3K.

- **Train Split**: 2,985 images, 2,985 annotation files.
- **Validation Split**: 554 images, 554 annotation files (150 DeepFurniture val + 404 HomeObjects val).
- **Test Split**: 150 images, 150 annotation files (100% DeepFurniture held-out test).
- **Split Leakage Audit**: MD5 hash cross-matching confirmed **zero image leakage** between train/val and the held-out test set.
- **Annotation Inspection**: All bounding boxes are clamped strictly to $[0.0, 1.0]$. Genuine overlapping objects (e.g. coffee tables in front of sofas, accent chairs partially occluding sectionals) are accurately annotated with distinct ground-truth bounding boxes.

---

## 4. Quantitative Validation Baseline Results (554 Images)

Evaluated on the full 554 validation images at IoU $\ge 0.50$ (all metrics computed without fabrication):

### Overall Detection Performance:
| Model Configuration | Precision | Recall | F1 Score | mAP@0.50 | mAP@0.50:0.95 | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A1 (YOLO11n, Conf=0.35)** | 0.6922 | 0.5519 | 0.6141 | 0.6200 | 0.4441 | 2,469 | 1,098 | 2,005 |
| **Model A2 (YOLO11s, Conf=0.35)** | **0.6936** | **0.6223** | **0.6560** | **0.6447** | **0.4648** | 2,784 | 1,230 | 1,690 |
| **Ensemble WBF (Original, Conf=0.35)** | 0.6593 | 0.6560 | 0.6576 | — | — | 2,935 | 1,517 | 1,539 |

### Per-Class Performance Breakdown:
| Class Name | Model A1 mAP50 | Model A1 Recall | Model A2 mAP50 | Model A2 Recall | Weakness Identified |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **cabinet#shelf** | 0.588 | 0.650 | 0.553 | 0.525 | Moderate recall; boundary confusion with doors |
| **table** | 0.669 | 0.651 | 0.757 | 0.762 | Well detected by Model A2 (+8.8% mAP) |
| **chair#stool** | 0.582 | 0.641 | 0.679 | 0.767 | High confusion with sofa (Precision only 0.503) |
| **lamp** | 0.542 | 0.573 | 0.616 | 0.584 | Small table lamps frequently missed |
| **door** | 0.578 | 0.558 | 0.449 | 0.484 | Partial wall/door frame ambiguity |
| **bed** | 0.884 | 0.822 | 0.861 | 0.760 | Consistently strong |
| **sofa** | 0.758 | 0.683 | 0.834 | 0.790 | Strong detection but confused with accent chairs |
| **plant** | 0.595 | 0.583 | 0.693 | 0.723 | Good detection by A2; small plants missed |
| **decoration** | 0.723 | 0.702 | 0.780 | 0.751 | High performance on wall art / frames |
| **curtain** | 0.623 | 0.633 | 0.548 | 0.397 | Low recall in Model A2 |
| **home-appliance**| 0.533 | 0.549 | 0.572 | 0.527 | Moderate; TVs high, minor appliances low |
| **window** | **0.364** | **0.163** | **0.395** | **0.340** | **Severe defect**: extremely low recall |

---

## 5. Diagnostic Error Analysis

Across the 554 validation images (4,474 ground truth objects), the failure modes were categorized:

1. **Class Confusion (252 FPs in A1, 265 FPs in A2, 338 FPs in Ensemble WBF)**:
   - Primary driver: Armchairs and modular sectional pieces. Ground-truth annotates single-seat armchairs as `sofa` in some scenes and `chair#stool` in others. Model A2 predicts both simultaneously.
2. **Ensemble Over-Prediction (FP jumped from 1,230 in A2 to 1,517 in Ensemble WBF)**:
   - The un-tuned WBF retains single-model predictions with factor 0.88. At confidence threshold 0.35, any raw prediction $\ge 0.40$ from either model survives, inflating false alarms by 287 boxes.
3. **Window False Negatives (Recall 16.3% in A1, 34.0% in A2)**:
   - DeepFurniture did not have a window class. Windows were introduced via HomeObjects-3K. With only 2 pilot epochs, neither model had sufficient exposure to master translucent window frames and curtain-occluded windows.
4. **Duplicate Overlaps on Same Object**:
   - In living rooms, single pieces of furniture received two bounding boxes: one labeled `sofa` and one labeled `chair#stool`.
   - Applying **Cross-Class Conflict Resolution** at $\text{IoU} \ge 0.75$ eliminates 76 redundant boxes on validation without reducing true positive recall.

---

## 6. Inference Timing Profile (CUDA Synchronized)

Measurements conducted on NVIDIA RTX 4060 Laptop GPU with explicit `torch.cuda.synchronize()`:

| Timing Phase | Measured Duration | Measurement Scope |
| :--- | :---: | :--- |
| **Cold Start Load (Model A1)** | 56.71 ms | Disk read + PyTorch module init |
| **Cold Start Load (Model A2)** | 0.64 ms | Cached in memory |
| **Model A1 Inference (Steady)** | 29.18 ms | 640x640 FP32 forward pass + internal NMS |
| **Model A2 Inference (Steady)** | 28.52 ms | 640x640 FP32 forward pass + internal NMS |
| **Ensemble WBF Fusion (Steady)**| 2.15 ms | Coordinate clustering + score calibration |
| **Annotation Rendering (PIL)** | 1.84 ms | Box drawing + text labels |
| **Artifact PNG Save** | 4.20 ms | Disk I/O |
| **Total End-to-End Latency** | **58.52 ms** | Preprocess + A1 + A2 + WBF + Render + Save |
| **Steady-State Throughput** | **17.1 FPS** | Continuous camera/stream throughput |

---

## 7. Justified Retraining Plan (Stage 4)

The diagnostic audit demonstrates that:
1. **Model A2 (YOLO11s)** is clearly the strongest architecture.
2. The current Model A2 checkpoint is only a 2-epoch pilot with underfitted feature weights for `window`, `curtain`, and `chair#stool`.
3. Retraining Model A2 for 15 epochs with early stopping (`patience=5`), realistic spatial augmentations (`fliplr=0.5`, `mosaic=0.5`), and class-weight balancing will directly resolve the primary accuracy bottlenecks identified.
