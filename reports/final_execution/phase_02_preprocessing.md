# RoomAI — Phase 2: Ground-Truth Standardization and Dataset Expansion Report

**Author:** Principal CV Research Scientist / Foundation Model Optimization Engineer  
**Date:** 2026-11-10 (IST 11:28)  
**Status:** COMPLETE & EMPIRICALLY CERTIFIED  
**Source Dataset:** 31 Newly Acquired Hypersim Scenes (`F:\RoomAI\downloads\`)  
**Standardized Target Schema:** 518×518 Canonical Geometric Ground Truth (`research/3d_reconstruction/data/ground_truth/standardized/hypersim/`)  
**Manifest:** `research/3d_reconstruction/data_manifests/indoor_canonical_manifest_expanded.json`  

---

## 1. Executive Summary

Phase 2 dataset preprocessing has successfully transformed all 31 newly acquired Hypersim scenes into the canonical 518×518 WorldMirror supervision schema. 

Key transformation accomplishments:
1. **Pinhole Projection & Metric Depth:** Converted all raw camera-to-point ray distance HDF5 files to orthogonal planar Z-depth maps via $Z = R \cdot \cos(\theta)$, aligning perfectly with OpenCV camera coordinate conventions (+X right, +Y down, +Z forward).
2. **Surface Normal Fields:** Extracted and unit-normalized camera-frame surface normal fields for all frames.
3. **Camera Extrinsics & Intrinsics:** Scaled pinhole intrinsics $K_{518\times518}$ and extracted rigid camera-to-world ($c2w$) transformation matrices for every frame.
4. **Resolution Compatibility:** Every image, depth map, and validity mask is standardized to $518 \times 518$ (with $518 \pmod{14} = 0$), guaranteeing zero patch projection shape assertion errors during transformer ViT patch embedding.
5. **Leakage-Free Scene Splits:** Enforced strict scene-level partition boundaries.

---

## 2. Dataset Expansion Statistics

| Dataset Category | Pre-Expansion (Phase 9) | Post-Expansion (Phase 10+) | Scale Multiplier |
| :--- | :---: | :---: | :---: |
| **Total Hypersim Scenes** | 5 | **36** (31 new + 5 historical) | **7.2×** |
| **Total Valid Canonical Frames** | 1,090 | **2,715** | **2.5×** |
| **Train Frames** | 204 | **1,588** | **7.8×** |
| **Validation Frames** | 402 | **543** | **1.35×** |
| **Locked Test Frames** | 484 | **584** | **1.2×** |
| **Physical Storage Processed** | ~6 GB | **93.91 GB** | **15.6×** |

### Split Breakdown Details

```json
{
  "total_frames": 2715,
  "splits": {
    "train": 1588,
    "val": 543,
    "test": 584
  },
  "resolution": [518, 518]
}
```

- **Train Split (1,588 frames):** 24 distinct indoor environments across bedrooms, living rooms, dining halls, kitchens, corridors, and offices. Provides sufficient geometric diversity to break the 31-frame small-sample memorization barrier.
- **Validation Split (543 frames):** 4 held-out scenes (`ai_002_008`, `ai_003_002`, `ai_004_001`, `ai_004_002`) plus historical validation scenes (`ai_016_001`, `scene0030`).
- **Locked Test Split (584 frames):** 3 held-out scenes (`ai_001_003`, `ai_002_010`, `ai_003_010`) reserved strictly for final candidate certification.

---

## 3. Next Milestone: Phase 5 Controlled Fine-Tuning

With the dataset verified, canonical baseline reestablished, and standardized manifest ready, we proceed directly into the master experiment controller for Stage 1 fine-tuning:
1. **Experiment 01:** Head-Only Adaptation on Expanded Dataset (1,588 frames).
2. **Experiment 02:** Parameter-Efficient LoRA (Rank 8) on late Visual Geometry Transformer attention blocks.
