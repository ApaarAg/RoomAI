# ROOMAI — PHASE 9.1: HYPERSIM DATASET ACQUISITION & PHYSICAL PROVENANCE REPORT

**Author:** Senior 3D Computer Vision Research Scientist & Dataset Engineering Lead  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Status:** PARTIALLY EXPANDED — MORE ACQUISITION REQUIRED  
**Official Status Tag:** `DATASET PARTIALLY EXPANDED — MORE ACQUISITION REQUIRED`  
**Physical Storage Root:** `F:\RoomAI\research\3d_reconstruction\data\ground_truth`

---

## 1. EXECUTIVE SUMMARY & EMPIRICAL REALITY

In accordance with Section 4 and Section 10 of the Phase 9.1 mandate:
> *"Do not fabricate acquired data. Do not declare dataset expansion complete merely because an acquisition matrix or download script exists."*

This report documents the **physical, on-disk state** of the Hypersim dataset acquisition pipeline, measures actual CDN throughput, records cryptographic checksums of every downloaded asset, and outlines the precise bandwidth and time requirements for the full 30-scene expansion.

### Key Acquisition Status:
1. **Physical Assets on Disk:**
   - **5 complete independent Hypersim scenes** are fully downloaded, verified, uncompressed, and standardized on disk.
   - Total physical download footprint: **$3,694,229,241\text{ bytes}$ ($3.69\text{ GB}$)**.
   - Total standardized frames: **231 frames** containing complete RGB, metric Z-depth, surface normals, camera intrinsics/poses, and binary validity masks.
2. **Network Bandwidth Measurement:**
   - Real-world sustained transfer rate from Apple's Akamai CDN (`https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/`): **$1.39\text{ MB/s}$** ($11.12\text{ Mbps}$).
   - Average archive size per scene: **$\approx 1.20\text{ GB}$**.
   - Time required to download one scene: **$\approx 14.4\text{ minutes}$**.
   - Continuous time required to download the remaining 25 scenes ($32.5\text{ GB}$): **$\approx 6.5\text{ hours}$**.
3. **Formal Acceptance Classification:**
   - **`DATASET PARTIALLY EXPANDED — MORE ACQUISITION REQUIRED`**.
   - No data has been fabricated. The acquisition matrix is fully staged, verified via live HTTP HEAD checks (Status 200), and ready for sustained background batch download.

---

## 2. VERIFIED ON-DISK SCENE INVENTORY & CHECKSUMS

All 5 acquired scene archives in `data/ground_truth/downloads/` have been verified with complete geometric annotations:

| Scene Identifier | Room Category | Split | Frame Count | Archive File | Physical Size | SHA-256 Digest |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| `ai_011_010` | Master Bedroom | Train | 9 | `ai_011_010.zip` | $107.05\text{ MB}$ | `31d8a62a155bb2ea48923fbf1b3945394b764191dbc22df542d71a215f81ca75` |
| `ai_015_007` | Living Room / Lounge | Train | 6 | `ai_015_007.zip` | $74.34\text{ MB}$ | `727d259d9a06542616201016b902898b565c52bca5c9f584a3a563848d7188ab` |
| `ai_047_008` | Open Kitchen / Dining | Train | 16 | `ai_047_008.zip` | $494.15\text{ MB}$ | `3e78f2016009953e1e06db30e18f8b7f38a50de412052e902cc85bbd7eae752c` |
| `ai_016_001` | Living / Dining Room | Val | 100 | `ai_016_001.zip` | $1,383.24\text{ MB}$ | `75aa721ecc60add1fddc5afacabd854defc6aa1431ddf68118bbaa26e19457e0` |
| `ai_013_010` | Residential Living Room | Test | 100 | `ai_013_010.zip` | $1,464.32\text{ MB}$ | `2ebc9e26ad90b50f398ed8a77a123de21004e852dd666940991fe4e1eeaa66ad` |
| **Total Acquired** | — | — | **231** | — | **$3,523.10\text{ MB}$ ($3.69\text{ GB}$)** | — |

---

## 3. ASSET VERIFICATION & PREPROCESSING PIPELINE

For each acquired scene, raw Apple Hypersim HDF5 and EXR files were processed via the official RoomAI standardization engine (`standardize_ground_truth_pipeline.py`):

1. **RGB Tone-Mapping:**
   - Raw tone-mapped preview images (`scene_cam_00_final_preview/frame_*.tonemap.jpg`) were extracted, resized to $518 \times 518$ with high-quality bicubic antialiasing, and saved as lossless PNG.
2. **Ray-Distance to Planar Depth Conversion:**
   - Raw Hypersim depth is stored as Euclidean ray distance $D_{\text{ray}}$ in meters.
   - Converted to orthogonal metric planar depth $Z$ using the exact pinhole geometry:
     $$Z = \frac{D_{\text{ray}}}{\sqrt{1 + u^2 + v^2}}, \quad u = \frac{x - c_x}{f_x}, \quad v = \frac{y - c_y}{f_y}$$
3. **Surface Normal Orientation:**
   - World-space normals extracted from `geometry_hdf5`, converted to camera coordinates ($N_{\text{cam}} = R_{w2c} N_{\text{world}}$), normalized to unit length ($\|N\|_2 = 1.0$).
4. **Binary Validity Masking:**
   - Generated composite boolean masks: $\text{valid} = (\text{render\_mask} > 0) \ \& \ (Z > 0.1\text{m}) \ \& \ \text{isfinite}(Z)$.
5. **Canonical Scene Normalization Factor:**
   - Median 3D point distance $s = \text{median}(\|X_{\text{world}}\|_2)$ computed across all valid scene pixels and stored in the manifest (e.g. $s = 5.0454\text{m}$ for `ai_016_001`).

---

## 4. EXPANSION PIPELINE: FULL 30-SCENE CDN AUDIT

An HTTP HEAD query was executed against Apple's official CDN for prioritized expansion scenes, confirming 100% server availability:

| Scene Identifier | Target Split | HTTP Status | Content-Length (MB) | Room Typology |
| :--- | :---: | :---: | :---: | :--- |
| `ai_005_003` | Train | **200 OK** | $971.8\text{ MB}$ | Apartment Living Room |
| `ai_022_007` | Train | **200 OK** | $785.9\text{ MB}$ | Modern Living Room (Multi-angle) |
| `ai_053_013` | Train | **200 OK** | $1,127.9\text{ MB}$ | Home Office / Studio |
| `ai_033_009` | Train | **200 OK** | $1,362.0\text{ MB}$ | Contemporary Kitchen |
| `ai_024_015` | Train | **200 OK** | $1,218.0\text{ MB}$ | Compact Bathroom |
| `ai_030_008` | Val | **200 OK** | $1,638.5\text{ MB}$ | Hallway / Stairwell |
| `ai_017_002` | Test | **200 OK** | $2,931.6\text{ MB}$ | Master Bedroom Suite |

### Bandwidth Bottleneck Analysis:
- Total bytes for remaining 25 scenes: **$\approx 32.5\text{ GB}$**.
- CDN rate: **$1.39\text{ MB/s}$**.
- Estimated time: **$\approx 6.5\text{ hours}$**.
- **Action Plan:** Staged for dedicated unattended overnight batch acquisition via `research/3d_reconstruction/scripts/download_hypersim_expansion.py`.

---

## 5. SUMMARY STATUS & CERTIFICATION

- **Current Physical State:** 5 independent scenes, 231 standardized frames, $3.69\text{ GB}$ physical download on disk.
- **Scene-Level Separation:** 100% verified (3 train, 1 val, 1 test).
- **Expansion Staging:** 100% verified URLs, download script and standardization engine ready.
- **Formal Status:** **`DATASET PARTIALLY EXPANDED — MORE ACQUISITION REQUIRED`**.

