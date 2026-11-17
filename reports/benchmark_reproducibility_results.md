# RoomAI — Benchmark Reproducibility & Empirical Verification Results
**Document ID:** `reports/benchmark_reproducibility_results.md`  
**Auditor:** Skeptical Senior Computer Vision Research Engineer & QA Architect  
**Audit Date:** October 31, 2026  
**Hardware Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6), 24 GB System RAM, Windows 11  
**Software Environment:** `tf-gpu` Conda Environment (Python 3.10.20, PyTorch 2.11.0+cu128, Ultralytics 8.4.66, Transformers 4.49.0)  
**Execution Script:** `scripts/verify_benchmark_integrity.py`  
**Raw Results Artifact:** `reports/benchmark_reproducibility_results.json`

---

## 1. Executive Summary: Phase C Claims vs. Independent Verification

| Metric / Dimension | Phase C Reported Claim | Independent Audit Reproduced | Discrepancy & Methodological Analysis | Classification |
|---|---|---|---|---|
| **SegFormer-B1 mIoU** | **0.5432** | **0.5432** (Identical Ad-Hoc)<br>**0.6807** (Standard 10-Class Dataset IoU)<br>**0.3600** (Standard All-88 Present Classes IoU) | The original calculation `0.5432` was strictly reproduced, but represented an unweighted sample-instance average across only 10 selected indoor classes. Standard dataset-aggregated confusion matrix calculation gives **0.6807** for the 10 target classes and **0.3600** for all 88 present classes. | **PARTIALLY VERIFIED** |
| **Full Pipeline Latency** | **142.53 ms** | **134.97 ms** (Mean)<br>**130.25 ms** (P50)<br>**184.43 ms** (P95) | Fully verified on RTX 4060 GPU with warm-up and CUDA synchronization across 50 runs. The pipeline is slightly faster than estimated (~135 ms vs 142 ms). | **VERIFIED** |
| **Peak Active VRAM** | **1.71 GB** (Theoretical Sum) | **972.8 MB (0.95 GB)** (Allocated)<br>**2,442.0 MB (2.38 GB)** (Reserved) | The prior 1.71 GB was the arithmetic sum of isolated peak allocations. When all 4 models run in a single process, peak allocated memory is under 1 GB, and total reserved memory is 2.38 GB. | **VERIFIED** |
| **Instance Mask Accuracy** | "All 12 classes successfully segmented" | **23.06%** containment ratio without clipping.<br>**Mask AP50:95 cannot be computed.** | No ground-truth polygon masks exist in ADE20K Challenge data or DeepFurniture. While masks are generated, they bleed into adjacent walls/floors unless explicitly clipped to the bounding box. | **INCORRECT / UNVERIFIED** |
| **Metric Depth Quality** | "Accurate 3D metric reconstruction" | Scale confirmed as meters ($1.6\text{ m} - 6.8\text{ m}$), but **AbsRel / RMSE cannot be computed.** | No ground-truth depth maps exist in the workspace. Qualitative meter outputs are physically plausible, but mathematical depth accuracy is unverified against a real-world sensor ground truth. | **UNVERIFIED** |

---

## 2. Audited Benchmark Image Subset Specification

The 50 evaluation scenes were verified from `archive (2)/ADEChallengeData2016/images/validation/`:

| Category | Image ID Range | Count | Primary Objects Present | Ground-Truth Mask Type |
|---|---|---|---|---|
| **Bedroom** | `ADE_val_00000001` .. `ADE_val_00000010` | 10 | Bed, wall, floor, ceiling, nightstand, lamp, wardrobe, window | 8-bit PNG semantic mask |
| **Living Room** | `ADE_val_00000011` .. `ADE_val_00000020` | 10 | Sofa, coffee table, rug, wall, floor, ceiling, television, plant | 8-bit PNG semantic mask |
| **Kitchen** | `ADE_val_00000021` .. `ADE_val_00000030` | 10 | Cabinet, countertop, refrigerator, sink, wall, floor, ceiling | 8-bit PNG semantic mask |
| **Bathroom** | `ADE_val_00000031` .. `ADE_val_00000040` | 10 | Toilet, sink, mirror, wall, floor, bathtub, cabinet | 8-bit PNG semantic mask |
| **Dining Room** | `ADE_val_00000041` .. `ADE_val_00000050` | 10 | Dining table, chairs, chandelier, wall, floor, ceiling, sideboard | 8-bit PNG semantic mask |

**Data Leakage Verification:** Zero images overlap with training datasets (`archive (2)/ADEChallengeData2016/images/training/` or `data/indoor_canonical/`).

---

## 3. Semantic Segmentation Reproducibility & Formula Discrepancy

### 3.1 Mathematical Comparison of Formulations
1. **Ad-Hoc Sample-Instance Formulation (Phase C):**
   $$\text{mIoU}_{\text{adhoc}} = \frac{1}{|\mathcal{S}|} \sum_{(i, c) \in \mathcal{S}} \frac{|\text{Pred}_{i,c} \cap \text{GT}_{i,c}|}{|\text{Pred}_{i,c} \cup \text{GT}_{i,c}|}$$
   Where $\mathcal{S} = \{(i, c) \mid c \in \text{Target10}, |\text{Pred}_{i,c} \cup \text{GT}_{i,c}| > 0\}$.
   - Result on SegFormer-B1: **0.5432** (Identically reproduced).

2. **Standard Dataset-Aggregated Confusion Matrix Formulation (Pascal VOC / ADE20K Standard):**
   $$\text{IoU}_c = \frac{\sum_{i=1}^N \text{TP}_{i,c}}{\sum_{i=1}^N (\text{TP}_{i,c} + \text{FP}_{i,c} + \text{FN}_{i,c})}$$
   - Result on Target 10 Indoor Classes: **0.6807**.
   - Result on All 88 Classes Present in the 50 Scenes: **0.3600**.

### 3.2 Per-Class IoU Breakdown (Standard Dataset Formulation)

| Class Name | ADE20K Index | Phase C Reported IoU | Standard Dataset IoU | Variance Analysis |
|---|---|---|---|---|
| **Ceiling** | `6` | 0.7247 | **0.8346** | +10.99% (Aggregating contiguous ceiling planes eliminates small-image penalties) |
| **Floor** | `4` | 0.6834 | **0.8327** | +14.93% (Large floor planes dominate pixel counts) |
| **Wall** | `1` | 0.7095 | **0.7636** | +5.41% (Consistent high performance across all 50 scenes) |
| **Bed** | `8` | 0.6596 | **0.8878** | +22.82% (High pixel count in bedrooms; massive IoU when present) |
| **Sofa** | `24` | 0.5118 | **0.7160** | +20.42% (Strong feature representation in living rooms) |
| **Cabinet** | `11` | 0.3963 | **0.6592** | +26.29% (Standard aggregation rewards kitchen cabinet volume) |
| **Chair** | `20` | 0.3796 | **0.6302** | +25.06% (Dining room chairs aggregate into solid total TP) |
| **Windowpane** | `9` | 0.4943 | **0.6135** | +11.92% (High contrast glass/frame edges) |
| **Table** | `16` | 0.3784 | **0.4751** | +9.67% (Occasional confusion with desks and countertops) |
| **Door** | `15` | 0.2340 | **0.3948** | +16.08% (Thin door frames remain the most challenging class) |

---

## 4. Instance Segmentation Empirical Findings

### 4.1 Bounding Box Detection and Mask Generation
On the 50 validation scenes:
- Model A3 detected **308 bounding boxes** across 11 of the 12 classes (zero curtains were present in these specific scenes).
- FastSAM generated **284 masks** from the 308 prompts (92.2% generation rate).
- **Mask Containment Ratio:** **23.06%**. When FastSAM's output mask is unclipped, 76.94% of the mask pixels extend outside the input bounding box.
- **Remediation Protocol:** FastSAM masks **must be spatially masked by their bounding box prompt** (`mask = mask & box_mask`) before being passed to downstream 3D or rendering modules.

### 4.2 Class Detection & Area Distribution

| Class Label | Model A3 Detections | FastSAM Masks | Mean Mask Area (Pixels) | Visual Boundary Quality |
|---|---|---|---|---|
| `bed` | 3 | 3 | 32,687.7 | **Excellent**: Clear boundaries, high IoU |
| `cabinet#shelf` | 26 | 24 | 22,586.5 | **Good**: Slight boundary bleed into back wall |
| `table` | 40 | 38 | 21,404.6 | **Good**: Table tops crisp; thin legs occasionally soft |
| `window` | 36 | 35 | 18,644.2 | **Moderate**: Confusion between curtains and window frame |
| `door` | 20 | 18 | 17,288.1 | **Moderate**: Frame boundaries blur into wall |
| `lamp` | 40 | 36 | 16,254.4 | **Poor**: Thin lamp stands leak into background |
| `chair#stool` | 43 | 41 | 11,671.0 | **Moderate**: Adjacent dining chairs occasionally merge |
| `sofa` | 22 | 22 | 8,822.5 | **Good**: Cushions and armrests well delineated |
| `plant` | 40 | 37 | 8,425.3 | **Moderate**: Leaf silhouettes approximate |
| `decoration` | 31 | 26 | 8,025.2 | **Poor**: Small wall frames merge with wall texture |
| `home-appliance` | 7 | 6 | 3,891.3 | **Good**: Crisp geometric borders |
| `curtain` | 0 | 0 | 0.0 | N/A (Not present in these 50 images) |

---

## 5. End-to-End Latency & Memory Verification

Measurements were recorded across 50 runs with `torch.cuda.synchronize()` on the NVIDIA RTX 4060 Laptop GPU:

```
[Preprocessing]  2.72 ms
      │
      ▼
[Model A3 YOLO11s]  16.85 ms
      │
      ▼
[FastSAM Mask Head]  22.29 ms
      │
      ▼
[SegFormer-B1]  28.45 ms
      │
      ▼
[Depth Anything V2 Metric]  60.32 ms
      │
      ▼
[Fusion & Postproc]  4.34 ms
      │
      ▼
[Total Pipeline E2E]  Mean = 134.97 ms | Median = 130.25 ms | P95 = 184.43 ms
```

- **Static VRAM (All 4 models resident):** 216.2 MB
- **Peak Dynamic VRAM Allocated:** 972.8 MB (0.95 GB)
- **Peak Dynamic VRAM Reserved:** 2,442.0 MB (2.38 GB)
- **Available Headroom on 8 GB GPU:** **5.62 GB** (70.2% unused)

