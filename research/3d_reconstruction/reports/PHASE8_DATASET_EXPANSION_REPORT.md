# Phase 8: Dataset Inventory, Provenance, and Canonical Scale Report

**Author:** Senior 3D Computer Vision Research Scientist & ML Systems Research Lead  
**Repository:** `F:\RoomAI`  
**Manifest:** [`indoor_canonical_manifest.json`](file:///F:/RoomAI/research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json)  
**Total Frames:** 1,090 standardized indoor frames  
**Total Distinct Scenes:** 13 verified sequences (5 Hypersim, 8 ScanNet)  
**Date:** 2026-11-08  

---

## 1. Dual-Track Data Policy & Licensing Provenance

To strictly ensure commercial readiness while enabling scientific evaluation against real sensor depth, data is divided into two distinct tracks:

| Track | Dataset | License | Usage Policy | Permitted Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Track A** | **Apple Hypersim** | **CC BY-SA 3.0** | Commercial Production Candidate | Production weights, export models, deployed backend |
| **Track B** | **ScanNet v2** | **ScanNet TOS (Non-Commercial Research)** | Research-Only Domain Generalization | Benchmark evaluation, research diagnostics |

Checkpoints trained with Track B data are strictly segregated into research-only namespaces and will **never** be promoted into RoomAI commercial production pipelines.

---

## 2. Complete Scene Inventory & Canonical Scale Factors

All 1,090 frames across all 13 scenes were processed through the canonical scene normalization pipeline. The scene-wide scale factor $s = \operatorname{nanmedian}(\|\mathbf{X}_w\|_2)$ was computed by aggregating all valid 3D points across all frames of each scene.

| Dataset | Split | Scene ID | Frame Count | Scene Scale $s$ | Bad Case Flag | Median Metric Depth | Median Norm Depth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hypersim** | **train** | `ai_011_010` | 9 | **$3.6137\text{ m}$** | `False` | $3.57\text{ m}$ | $0.99$ |
| **Hypersim** | **train** | `ai_015_007` | 6 | **$3.5608\text{ m}$** | `False` | $1.97\text{ m}$ | $0.55$ |
| **Hypersim** | **train** | `ai_047_008` | 16 | **$8.5698\text{ m}$** | `False` | $5.39\text{ m}$ | $0.63$ |
| **Hypersim** | **val** | `ai_016_001` | 100 | **$5.0454\text{ m}$** | `False` | $3.89\text{ m}$ | $0.77$ |
| **Hypersim** | **test** (Quarantine) | `ai_013_010` | 100 | **$5.6445\text{ m}$** | `False` | $4.21\text{ m}$ | $0.75$ |
| **ScanNet** | **train** | `scene0015_00` | 98 | **$6.2789\text{ m}$** | `False` | $1.64\text{ m}$ | $0.26$ |
| **ScanNet** | **train** | `scene0019_00` | 28 | **$7.2058\text{ m}$** | `False` | $1.82\text{ m}$ | $0.25$ |
| **ScanNet** | **train** | `scene0019_01` | 47 | **$5.0492\text{ m}$** | `False` | $1.71\text{ m}$ | $0.34$ |
| **ScanNet** | **val** | `scene0030_00` | 124 | **$5.3258\text{ m}$** | `False` | $1.52\text{ m}$ | $0.29$ |
| **ScanNet** | **val** | `scene0030_01` | 82 | **$6.3958\text{ m}$** | `False` | $1.68\text{ m}$ | $0.26$ |
| **ScanNet** | **val** | `scene0030_02` | 96 | **$5.9937\text{ m}$** | `False` | $1.59\text{ m}$ | $0.27$ |
| **ScanNet** | **test** (Quarantine) | `scene0011_00` | 178 | **$5.2545\text{ m}$** | `False` | $1.47\text{ m}$ | $0.28$ |
| **ScanNet** | **test** (Quarantine) | `scene0011_01` | 206 | **$7.1754\text{ m}$** | `False` | $1.58\text{ m}$ | $0.22$ |

---

## 3. Dataset Split Structure & Quarantine Safeguards

### 3.1 Split Proportions
- **Training Set:** 204 frames ($18.7\%$) across 6 scene sequences (3 Hypersim + 3 ScanNet).
- **Validation Set:** 402 frames ($36.9\%$) across 4 scene sequences (1 Hypersim + 3 ScanNet). Used exclusively for checkpoint tuning and selection.
- **Test Set:** 484 frames ($44.4\%$) across 3 scene sequences (1 Hypersim + 2 ScanNet). Strictly quarantined and untouched during checkpoint selection.

### 3.2 Scene-Disjoint Principle
No scene ID appears in more than one split:
$$\mathcal{S}_{train} \cap \mathcal{S}_{val} = \emptyset, \quad \mathcal{S}_{train} \cap \mathcal{S}_{test} = \emptyset, \quad \mathcal{S}_{val} \cap \mathcal{S}_{test} = \emptyset$$
Furthermore, sequence variants (e.g. `scene0030_00`, `scene0030_01`, `scene0030_02`) are grouped strictly into the **same** split (`val`) to guarantee zero leakage between train and val.

---

## 4. Verification and Integrity Check

1. **Resolution:** Uniformly standardized to $518 \times 518$ square resolution matching ViT patch grid ($37 \times 37 = 1,369$ tokens at patch size 14).
2. **Data Modalities per Frame:**
   - `rgb`: 3-channel PNG in $[0, 255]$
   - `depth`: Single-channel float32 `.npy` in physical meters
   - `normal`: 3-channel float32 `.npy` unit surface normals
   - `mask`: Binary PNG validity mask ($1 = \text{valid pixel}$, $0 = \text{invalid}$)
   - `cam`: Compressed `.npz` containing $3 \times 3$ intrinsics and $4 \times 4$ camera-to-world pose
3. **Physical Scale Distribution:** Scene scale factors range from $3.56\text{ m}$ to $8.57\text{ m}$ with zero NaN or infinite values. Every scene has verified, finite geometry.

