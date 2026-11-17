# RoomAI Phase 22 — B7 Integrity Audit & Scientific Methodology Report

**Date:** October 29, 2026  
**Auditor:** Senior Computer Vision Research Engineer & Experimental Methodology Auditor  
**Repository:** `C:\Users\apaar\Desktop\RoomAI`  
**Execution Environment:** Windows 11, Conda environment `tf-gpu` (Python 3.10.20, PyTorch 2.11.0+cu128, Ultralytics 8.4.66), NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM), 24 GB RAM.

---

## 1. Executive Summary

This audit rigorously evaluates the Model B7 pilot experiment (`exterior_detector_b7_pilot_best.pt`), audits data/loss/evaluation integrity, resolves historical discrepancies in latency and class narratives, and establishes the empirical foundation for Candidate C1 training.

### Key Audit Findings
1. **Cryptographic Invariants Upheld:** Model B7 checkpoint is authentic and unaltered, matching the authoritative SHA-256 `0B04B97E66F033D2037BC1483D4AD4F484A00213DEFEE66A7D32AD83199E2C3F` and MD5 `41433e929910eda4414ed6541bac2fe4`. Checkpoints B2 (Production), B6 (Baseline), and the Porch Specialist were cryptographically verified before and after all operations.
2. **Loss Masking Architecture Verified:** `MaskedDetectionLoss` is actively invoked via `on_pretrain_routine_end` callback on `trainer.criterion`. Unannotated classes (`mask=0.0`) produce zero BCE loss and zero backward gradient. Fully annotated images (`mask=1.0`) produce exact mathematical identity (`max abs diff = 0.0`) with standard YOLO `v8DetectionLoss`.
3. **Mosaic Interaction Discovery:** Under Ultralytics default Mosaic augmentation, 4 images are stitched into one canvas under the primary image's mask, creating cross-tile mask mismatch during early epochs. For Phase 22 Candidate C1, setting `mosaic=0.0` completely eliminates this flaw.
4. **Reconciliation of `shutter_blind`:** The Phase 20 narrative stated that `shutter_blind` was ingested from Open Images. Auditing `data/pilot_b7_dataset/pilot_manifest.json` and `verified_class_masks.json` revealed that **0 shutter_blind boxes were ingested in the 300 pilot images** (door+window candidates filled the quota first). It was masked out (`0.0`) across all 300 images.
5. **Metric Reproduction & Statistical Rigor:** Baseline B6 and B7 metrics were independently reproduced from scratch on the locked 154-scene validation split (8,504 annotations):
   - Macro-F1: `0.5069` (B6) $\rightarrow$ `0.5200` (B7) (+1.31% abs)
   - Micro-F1: `0.6337` (B6) $\rightarrow$ `0.6394` (B7) (+0.57% abs)
   - Porch F1: `0.1935` (B6) $\rightarrow$ `0.2381` (B7) (+23.0% rel)
   - Porch AP50: `0.1437` (B6) $\rightarrow$ `0.2336` (B7) (+62.6% rel)
   - **1,000-Iteration Paired Scene Bootstrap (95% CI):** Porch F1 delta 95% CI is `[-0.1189, +0.2077]`. Because $N=24$ porch annotations exist in the validation split, the CI spans zero; formal statistical significance ($p < 0.05$) cannot be claimed despite substantial point-estimate gains.
6. **Latency Discrepancy Resolved:** The previously reported 89.68 ms (B6) vs 35.85 ms (B7) latency was caused by evaluating B6 cold without a CUDA warmup loop (kernel compilation overhead averaged over 154 scenes). Under identical warmed conditions (3 dummy passes on 640x640), B6 median latency is 46.02 ms and B7 median latency is 35.12 ms.
7. **"Zero Regressions" Claim Clarified:** The claim of "zero regressions" in Phase 20 was imprecise. In point estimates, `shutter_blind` decreased by -0.0028 and `molding_deco` by -0.0045. Both are within statistical noise ($< 0.005$), but strictly speaking, point-estimate regressions were non-zero.

---

## 2. Phase A: B7 Architecture & Checkpoint Audit

### 2.1 Checkpoint Integrity Verification
All model checkpoints in `checkpoints/` were cryptographically hashed and compared against baseline records:

| Checkpoint Name | Status / Role | File Size (bytes) | MD5 Hash | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| `exterior_detector_b2_yolo11s_best.pt` | **Production Default** | 19,163,034 | `14a931dd0adabc16cbd7b28ed1d17f66` | `C599C19241B8C60390AC0C4914DE6CEF449BC1129F73B0DB7F8F04FC62A0459F` |
| `exterior_detector_b6_expanded_best.pt` | Research Baseline | 19,167,194 | `f5ceea56b0610ea14b647014e48152dc` | `009144B272998EC9447F649E8D850E0FDB8CAE5FC973C5E0CB9EEBE717675AC9` |
| `exterior_detector_b7_pilot_best.pt` | Pilot Model | 19,167,194 | `41433e929910eda4414ed6541bac2fe4` | `0B04B97E66F033D2037BC1483D4AD4F484A00213DEFEE66A7D32AD83199E2C3F` |
| `exterior_specialist_porch_yolo11s_best.pt` | Specialist Checkpoint | 19,159,130 | `4261fc1ad5f6fd9ac29c0499e2adddfd` | `943E7012090124EA726F21B37983CC8B253B35AF36F72B381EB01D18EAC2D9B2` |

All checkpoints loaded cleanly via `ultralytics.YOLO`. `exterior_detector_b7_pilot_best.pt` matches `runs/train/exterior_detector_b7_pilot/weights/best.pt` byte-for-byte.

### 2.2 MaskedDetectionLoss Wiring Verification
`utils/masked_detection_loss.py` subclasses Ultralytics `v8DetectionLoss`. The injection mechanism operates as follows:
```python
def on_pretrain_routine_end(trainer):
    trainer.criterion = MaskedDetectionLoss(trainer.model, verified_mask_lookup=verified_masks)
```
- **Execution Confirmation:** In `trainer.train()`, `trainer.criterion` is called on every training batch.
- **Mathematical Invariants:**
  1. For fully annotated images (`mask = [1.0] * 10`), `MaskedDetectionLoss` produces identical loss values to `v8DetectionLoss` (`max abs diff = 0.0`).
  2. For unannotated classes (`mask[c] = 0.0`), the binary cross-entropy tensor slice `bce_loss[:, :, c]` is multiplied by `0.0`, resulting in zero loss and zero gradient contribution:
     $$\mathcal{L}_{\text{cls}} = \frac{1}{\sum s} \sum_{i,j,c} \mathbf{M}_{i,c} \cdot \ell_{\text{BCE}}(p_{i,j,c}, y_{i,j,c})$$
  3. Bounding box regression loss ($\mathcal{L}_{\text{box}}$) and distribution focal loss ($\mathcal{L}_{\text{DFL}}$) remain unmodified, operating exclusively on positive foreground targets assigned by the TaskAlignedAssigner.

### 2.3 Image Identity & Augmentation Mask Alignment
- In YOLO batching, `batch["im_file"]` provides the list of absolute image file paths in the batch.
- Filenames are extracted via `Path(f).name` and matched against `verified_mask_lookup`.
- **Mosaic Flaw Identified:** When `mosaic > 0`, four separate images are cropped and stitched into one training sample. The batch metadata retains only the primary image's filename. Consequently, unannotated objects in the other three quadrants could receive loss masking corresponding to the primary image.
- **Fix Applied in Phase 22:** By setting `mosaic=0.0` during Candidate C1 fine-tuning, every training image is processed individually (with scale, translation, flip, and color jitter), guaranteeing strict 1:1 image-to-mask correspondence.

### 2.4 Reconciliation of `shutter_blind`
- **Reported claim in Phase 20:** "Curated 150 architectural detail / shutter scenes (shutter_blind /m/031b6r, door, window)".
- **Audit Findings:** Auditing `pilot_manifest.json` revealed:
  - Total `shutter_blind` boxes ingested: **0**.
  - Total `shutter_blind` mask value across all 300 pilot images: **0.0**.
- **Root Cause:** In `scripts/prepare_pilot_b7_dataset.py`, the selection condition `elif (has_blind or (has_door and '/m/0d4v4' in labels))` was evaluated after door/window images. The door+window combinations filled the 150-image detail quota before any images with `/m/031b6r` were reached. Consequently, B7 did not receive any additional shutter supervision, explaining why `shutter_blind` performance remained virtually unchanged (0.6462 vs 0.6434).
- **Phase 22 Remediation:** Curation prioritized explicit ingestion of 149 shutter/blind images with valid bounding boxes.

---

## 3. Phase B: Metric Reproduction & Empirical Analysis

### 3.1 Head-to-Head Baseline Validation Results (154 Scenes)
Evaluated on the locked 154 validation scenes (8,504 ground-truth annotations) at confidence threshold = 0.25, IoU threshold = 0.50:

| Metric | Model B6 (Baseline) | Model B7 (Pilot) | Absolute Delta | Relative Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.5069 | 0.5200 | +0.0131 | +2.58% |
| **Micro-F1** | 0.6337 | 0.6394 | +0.0057 | +0.90% |
| **mAP50** | 0.4921 | 0.5074 | +0.0153 | +3.11% |
| **mAP50-95** | 0.2976 | 0.3070 | +0.0094 | +3.16% |
| **Median Latency** | 46.02 ms | 35.12 ms | -10.90 ms | -23.68% |
| **p95 Latency** | 58.44 ms | 44.18 ms | -14.26 ms | -24.40% |

### 3.2 Per-Class Breakdown

| Class Name | Ground Truth Count | B6 Precision | B6 Recall | B6 F1 | B7 Precision | B7 Recall | B7 F1 | Delta F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `building` | 40 | 0.4762 | 0.5000 | 0.4878 | 0.4773 | 0.5250 | 0.5000 | **+0.0122** |
| `facade` | 495 | 0.3583 | 0.3576 | 0.3579 | 0.3644 | 0.3636 | 0.3640 | **+0.0061** |
| `window` | 3,845 | 0.7378 | 0.7722 | 0.7546 | 0.7410 | 0.7735 | 0.7569 | **+0.0023** |
| `door` | 372 | 0.4095 | 0.5538 | 0.4709 | 0.4262 | 0.5699 | 0.4879 | **+0.0170** |
| `roof_cornice` | 445 | 0.5492 | 0.7146 | 0.6211 | 0.5732 | 0.7281 | 0.6415 | **+0.0204** |
| `balcony` | 345 | 0.6545 | 0.6261 | 0.6400 | 0.7029 | 0.6377 | 0.6687 | **+0.0287** |
| `porch` | 24 | 0.4286 | 0.1250 | 0.1935 | 0.5000 | 0.1667 | 0.2381 | **+0.0446** |
| `pillar_column` | 494 | 0.3169 | 0.3381 | 0.3271 | 0.3423 | 0.3543 | 0.3482 | **+0.0211** |
| `shutter_blind` | 901 | 0.7609 | 0.5616 | 0.6462 | 0.7567 | 0.5605 | 0.6434 | -0.0028 |
| `molding_deco` | 1,543 | 0.5422 | 0.5995 | 0.5694 | 0.5348 | 0.6001 | 0.5649 | -0.0045 |

### 3.3 Statistical Significance (1,000-Iteration Paired Scene Bootstrap)
- **Macro-F1 Delta:** Mean = `+0.0136`, 95% CI: `[-0.0075, +0.0328]`.
- **Micro-F1 Delta:** Mean = `+0.0058`, 95% CI: `[-0.0007, +0.0126]`.
- **Porch F1 Delta:** Mean = `+0.0472`, 95% CI: `[-0.1189, +0.2077]`.

> **Auditor Conclusion:** Although Porch F1 improved substantially (+23.0% rel, AP50 +62.6% rel), the 95% confidence interval spans zero due to the modest sample size ($N=24$). The improvement is directionally strong and consistent, but cannot be asserted as statistically significant at $p < 0.05$. Claims of "proven statistical superiority" are rejected in favor of empirical, calibrated reporting.

---

## 4. Phase C: Controlled Dataset Expansion

### 4.1 Candidate Pool & Hygiene Gates
- **Initial local candidates:** 1,392 images across `cache_pilot_b7_candidates`, `cache_oi_b6_candidates`, and `cache_oi_b5_images`.
- **Quality Gates Applied:**
  1. Exact MD5 deduplication against B6 train, B7 pilot train, val, and retired test.
  2. 64-bit DCT perceptual hash (pHash) screening against all 154 validation and 154 retired test scenes ($d \ge 5$).
  3. Image decode and header verification (`PIL.Image.verify()`).
  4. Bounding box coordinate sanity checks ($0 < x, y, w, h \le 1.0$).
- **Results:**
  - 399 candidates were rejected as exact MD5 collisions against the existing B6 training set.
  - 260 new clean images passed all quality gates with zero collisions and zero near-duplicates.
  - Combined with the 300 existing B7 pilot images, this yields **560 curated external images** and **1,862 total training images**.
- **Provenance Manifest:** Saved to `reports/phase22_dataset_provenance.csv` with full origin, MD5, license, and per-class masking metadata.

---

## 5. Audit Conclusion & Phase Transition

Phase A and Phase B gates are **PASSED WITH CONDITIONS**:
1. Checkpoint integrity verified.
2. Masked loss mechanics verified mathematically and empirically.
3. Latency differences reconciled (warmup requirement enforced).
4. `shutter_blind` omission diagnosed and addressed.
5. In Phase 22 training, `mosaic=0.0` is strictly enforced.
6. Model B2 remains active production default. No automatic promotion permitted.

