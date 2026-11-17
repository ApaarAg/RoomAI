# RoomAI Model Optimization Baseline Audit Report

**Date:** 2026-10-24  
**Author:** RoomAI Senior Computer Vision Engineer & ML Research Lead  
**Hardware Profile:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, 8,188 MiB addressable), 24 GB System RAM, Intel Core i7  
**Software Environment:** Python 3.10.20, PyTorch 2.11.0+cu128, CUDA 12.8, Ultralytics 8.4.66, HuggingFace Transformers 4.49.0  
**Test Suite Status:** 77 Passed / 0 Failed (100% pass rate in 51.77s)

---

## 1. Executive Summary & Stage Timetable

This audit establishes trustworthy, empirically verified baselines for every model architecture and task in RoomAI before commencing Phase 2 optimization experiments toward $\ge 90\%$ performance targets.

### Execution Timetable

| Stage | Task Focus | Estimated Time | Actual Time | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Checkpoint inventory, architecture audit & genuine vs. mock verification | 15 min | 14 min | **Complete** |
| **Stage 2** | Integrity guards, split isolation & test-suite pass confirmation (`pytest -q`) | 10 min | 12 min | **Complete (77 passing)** |
| **Stage 3** | Object detection baseline replication on Val and Test splits (Groups A, B, C) | 25 min | 26 min | **Complete** |
| **Stage 4** | Deep dataset audit: duplicate leakage, scene correlation, geometry errors | 20 min | 22 min | **Complete** |
| **Stage 5** | Non-detector task evaluation (Segmentation, Depth, Visual Q&A) | 15 min | 16 min | **Complete** |
| **Stage 6** | Documentation deliverables, threshold analysis & optimization experiment plan | 20 min | 18 min | **Complete** |
| **Total** | Full Phase 2 Baseline Audit | **105 min** | **108 min** | **Verified** |

---

## 2. Complete Checkpoint Inventory & Provenance Verification

Every checkpoint in `checkpoints/` was verified by direct loading with Ultralytics YOLO and PyTorch:

| Model ID | Checkpoint Rel Path | Task | Architecture | Param Count | File Size | nc | Real vs. Mock Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **indoor_a2_finetuned** | `checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt` | `detect` | YOLO11s | 9.42M | 18.27 MB | 12 | **Real GPU Inference** |
| **indoor_a1_pilot** | `checkpoints/indoor_detector_a1_yolo11n_pilot_best.pt` | `detect` | YOLO11n | 2.59M | 5.19 MB | 12 | **Real GPU Inference** |
| **deepfurniture_baseline** | `checkpoints/deepfurniture_yolo11n_finetune_best.pt` | `detect` | YOLO11n | 2.58M | 5.21 MB | 11 | **Real GPU Inference** (Rollback) |
| **exterior_b2_finetuned** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | `detect` | YOLO11s | 9.41M | 18.28 MB | 10 | **Real GPU Inference** |
| **exterior_b1_pilot** | `checkpoints/exterior_detector_b1_yolo11n_pilot_best.pt` | `detect` | YOLO11n | 2.58M | 5.20 MB | 10 | **Real GPU Inference** |
| **parking_c2_pilot** | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | `detect` | YOLO11s | 9.41M | 18.27 MB | 8 | **Real GPU Inference** |
| **parking_c1_pilot** | `checkpoints/parking_detector_c1_yolo11n_pilot_best.pt` | `detect` | YOLO11n | 2.58M | 5.20 MB | 8 | **Real GPU Inference** |
| **segmentation** | `checkpoints/yolo11n-seg.pt` | `segment` | YOLO11n-Seg | 2.89M | 5.90 MB | 80 | **Real GPU Inference** (COCO-seg) |
| **depth_estimation** | `checkpoints/depth_anything_v2_small/` | `depth` | DINOv2 ViT-S + DPT | 24.79M | 94.58 MB | 1 | **Real GPU Inference** (Affine Rel) |
| **classification** | None (`classification_best.pt` missing) | `classify` | Unimplemented | — | — | 6 | **Mock Mode Simulated** |
| **vision_language** | None (`vlm_best.pt` missing) | `vlm` | Unimplemented | — | — | — | **Mock Mode Simulated** |
| **generative_redesign** | None (`generator_best.pt` missing) | `diffuse` | Unimplemented | — | — | — | **Mock Mode Simulated** |

> [!NOTE]
> **Scientific Integrity Finding:** Only Object Detection (Groups A, B, C), Instance Segmentation, and Monocular Depth Estimation possess real model checkpoints. Classification, Vision-Language (VLM), and Generative Redesign are implemented as UI stubs with deterministic simulated responses (`is_simulated=True`).

---

## 3. Official Object Detection Baseline Replication

All models were evaluated under identical conditions: standard resolution $640 \times 640$, batch size 8, on the dedicated NVIDIA GeForce RTX 4060 Laptop GPU. Both the validation split (`val`) and the held-out test split (`test`) were evaluated.

### Group-by-Group Performance Matrix

| Task Group & Model ID | Checkpoint | Split | mAP50 | mAP50–95 | Precision | Recall | F1 Score | Latency (ms) | Steady FPS | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Group A: Indoor A2 (YOLO11s)** | `indoor_detector_a2_...` | **Val** | **0.7531** | **0.5484** | 0.7629 | 0.6604 | **0.7080** | 10.74 ms | 93.1 FPS | 440.2 MB |
| | | **Test** | **0.7464** | **0.5921** | 0.7549 | 0.6762 | **0.7134** | 9.41 ms | 106.2 FPS | 353.0 MB |
| **Group A: Indoor A1 (YOLO11n)** | `indoor_detector_a1_...` | **Val** | 0.6865 | 0.4870 | 0.7120 | 0.6210 | 0.6634 | 6.82 ms | 146.6 FPS | 351.8 MB |
| | | **Test** | 0.6642 | 0.5050 | 0.7102 | 0.6184 | 0.6611 | 6.12 ms | 163.4 FPS | 351.8 MB |
| **Group A: DeepFurn Rollback** | `deepfurniture_...` | **Val** | 0.7410 | 0.5720 | 0.7580 | 0.6920 | 0.7235 | 6.20 ms | 161.3 FPS | 351.2 MB |
| | | **Test** | 0.7311 | 0.5895 | 0.7266 | 0.6965 | 0.7112 | 5.84 ms | 171.2 FPS | 351.2 MB |
| **Group B: Exterior B2 (YOLO11s)** | `exterior_detector_b2_...` | **Val** | **0.4484** | **0.2689** | 0.5473 | 0.4268 | **0.4796** | 11.78 ms | 84.9 FPS | 459.8 MB |
| | | **Test** | **0.3961** | **0.2427** | 0.4991 | 0.3990 | **0.4435** | 10.30 ms | 97.1 FPS | 459.8 MB |
| **Group B: Exterior B1 (YOLO11n)** | `exterior_detector_b1_...` | **Val** | 0.0120 | 0.0050 | 0.0150 | 0.0070 | 0.0095 | 6.50 ms | 153.8 FPS | 351.5 MB |
| | | **Test** | 0.0080 | 0.0030 | 0.0120 | 0.0060 | 0.0080 | 6.10 ms | 163.9 FPS | 351.5 MB |
| **Group C: Parking C2 (YOLO11s)** | `parking_detector_c2_...` | **Val** | **0.3465** | **0.1950** | 0.4731 | 0.3510 | **0.4030** | 8.56 ms | 116.8 FPS | 505.5 MB |
| | | **Test** | **0.2551** | **0.1682** | 0.5382 | 0.2640 | **0.3543** | 8.27 ms | 121.0 FPS | 506.8 MB |
| **Group C: Parking C1 (YOLO11n)** | `parking_detector_c1_...` | **Val** | 0.1509 | 0.0923 | 0.6156 | 0.1459 | 0.2359 | 7.15 ms | 139.8 FPS | 352.1 MB |
| | | **Test** | 0.1466 | 0.0933 | 0.6308 | 0.1391 | 0.2280 | 4.20 ms | 238.4 FPS | 352.3 MB |

---

## 4. Per-Class Analysis & Error Diagnosis

### Group A: Indoor Furniture (Model A2, 12 Classes)
- **Top Performing Classes:**
  - `bed`: Precision 0.9114, Recall 0.8667, F1 **0.8885**, mAP50 **0.9324**
  - `decoration`: Precision 0.7837, Recall 0.7439, F1 **0.7633**, mAP50 **0.8536**
  - `lamp`: Precision 0.7709, Recall 0.7278, F1 **0.7488**, mAP50 **0.8230**
  - `sofa`: Precision 0.7371, Recall 0.6977, F1 **0.7168**, mAP50 **0.8096**
- **Lagging Classes:**
  - `window`: On validation split, mAP50 is 0.5692 (recall 0.3697). On the test split, mAP50 is 0.0 because the legacy test split inherited from DeepFurniture contained 0 labeled windows (windows were introduced from HomeObjects-3K in train/val).
  - `chair#stool`: Precision 0.6116, Recall 0.4374, F1 **0.5100**, mAP50 **0.5245**. Stools and dining chairs suffer from partial table occlusion and thin leg confusion.
  - `curtain`: Precision 0.8290, Recall 0.5242, F1 **0.6423**, mAP50 **0.6173**. Low recall due to variable aspect ratios and integration into window frames.

### Group B: Exterior Architecture (Model B2, 10 Classes)
- **Top Performing Classes:**
  - `window`: Precision 0.7592, Recall 0.7872, F1 **0.7729**, mAP50 **0.7960**
  - `roof_cornice`: Precision 0.5977, Recall 0.6558, F1 **0.6254**, mAP50 **0.6420**
  - `balcony`: Precision 0.6012, Recall 0.5720, F1 **0.5862**, mAP50 **0.5890**
- **Severely Deficient Classes:**
  - `porch`: mAP50 **0.0000** (0 precision, 0 recall). Root cause: only 6 total instances exist in the entire dataset.
  - `pillar_column`: Precision 0.3317, Recall 0.1943, F1 **0.2451**, mAP50 **0.1721**.
  - `facade`: Precision 0.4608, Recall 0.2401, F1 **0.3157**, mAP50 **0.2703**. Severe ambiguity with `building` bounding boxes.

### Group C: Parking & Grounds (Model C2, 8 Classes)
- **Top Performing Classes:**
  - `car_vehicle`: Precision 0.6430, Recall 0.4174, F1 **0.5062**, mAP50 **0.4522**
  - `motorcycle`: Precision 0.5316, Recall 0.4675, F1 **0.4975**, mAP50 **0.3863**
- **Severely Deficient Classes:**
  - `bicycle`: Recall 0.0000 on test split (mAP50 0.0657). Misses thin frame structures.
  - `tree`: Precision 0.2197, Recall 0.0286, F1 **0.0506**, mAP50 **0.0583**. Extreme false-negative rate on foliage backdrops.

---

## 5. Non-Detector Module Status & Verification

### Semantic & Instance Segmentation
- **Architecture:** Ultralytics `YOLO11n-Seg` (`yolo11n-seg.pt`).
- **Telemetry:** Latency $7.8\text{ ms}$ on RTX 4060, Peak VRAM $320\text{ MB}$.
- **Ground Truth Evaluation:** RoomAI currently lacks dense polygon/mask annotations for interior architectural surfaces (walls, floors, ceilings). While COCO categories (chair, couch, bed) are segmented live, **reporting a numerical mask mAP or Dice coefficient without architectural polygon ground truth is scientifically unverified**.

### Monocular Depth Estimation
- **Architecture:** Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`).
- **Telemetry:** Latency $64.29\text{ ms}$ (median on 640×480), Peak VRAM $244.6\text{ MB}$, Throughput $15.6\text{ FPS}$.
- **Ground Truth Evaluation:** Relative depth is affine-invariant disparity ($0 \le d \le 1$). **Metric depth error (RMSE, AbsRel) cannot be reported without RGB-D / LiDAR ground truth**.

### Visual Question Answering (VLM)
- **Status:** Unimplemented in `CustomVisionLanguageModel` (raises `ModelNotImplementedError`).
- **Benchmark Evaluation:** An 8-item labeled evaluation dataset was created (`data/vlm_benchmark/vlm_eval_dataset.json`). `MockVisionLanguageModel` achieves a 62% keyword match score via heuristic string templates, but has **0.0% visual grounding** (zero image tokens processed).

---

## 6. High-Impact Optimization Recommendations

Based on empirical error distributions, the single highest-impact experiment for each group is:

1. **Group A (Indoor): Multi-Scale Feature Alignment ($640 \to 800\text{px}$) with Class-Balanced Loss**  
   - Directly tackles the major error modes: occluded `chair#stool` instances and aspect-ratio variance in `curtain` and `window`.
2. **Group B (Exterior): Repeat Factor Sampling & Copy-Paste for Rare Architectural Classes**  
   - Resolves the 3,332:1 class imbalance that collapsed `porch` (6 instances) and `pillar_column` (172 instances) to near-zero AP.
3. **Group C: Multi-Angle Synthetic Perspective Augmentation & Pretrained YOLO11m Transfer**  
   - Overcomes small dataset size (510 train images) and resolves the recall failure on thin `bicycle` frames and distant `tree` foliage.
