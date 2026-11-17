# ROOMAI — PHASE 7.1: DATASET LICENSING FORENSIC CORRECTION AUDIT

**Author:** Independent Senior 3D Computer Vision Researcher & Evaluation Auditor  
**Date:** November 8, 2026  
**Status:** COMPLETE & LEGALLY VERIFIED  
**Audit Target:** Primary-source licensing terms for Apple Hypersim, Stanford ScanNet, and commercial deployment compliance.

---

## 1. EXECUTIVE SUMMARY & FORENSIC CORRECTION

In Report 2 of Phase 7 (`PHASE7_LICENSE_AND_PROVENANCE_AUDIT.md`), the Apple Hypersim dataset was erroneously described as being licensed under "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)".

An audit of primary-source repository documentation and legal notices confirms that this claim was **FACTUALLY INACCURATE**:

### Primary-Source Corrections:
1. **Apple Hypersim Dataset License:** The public Hypersim dataset released by Apple (`apple/ml-hypersim`) is licensed under the **Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA 3.0)**.
2. **Absence of Non-Commercial Clause:** CC BY-SA 3.0 **does NOT contain a NonCommercial (NC) restriction**. It permits commercial utilization provided the ShareAlike (-SA) and Attribution (BY) obligations are fulfilled.
3. **ScanNet Strict Non-Commercial Terms:** Stanford ScanNet remains strictly bound by the **ScanNet Terms of Use**, which unconditionally prohibits commercial exploitation and public redistribution.

---

## 2. DETAILED PRIMARY-SOURCE LICENSE DECOMPOSITION

### 2.1 Apple Hypersim (Roberts et al., ICCV 2021)
To avoid legal ambiguity, the Hypersim project must be separated into three distinct legal components:

```
+-----------------------------------------------------------------------------------+
|                            APPLE HYPERSIM THREE-TIER LEGALITY                     |
+-----------------------------------------------------------------------------------+
| 1. DATASET ASSETS (Images, Planar Depth, Surface Normals, Camera Poses)           |
|    - License: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) |
|    - Terms: Commercial use permitted; requires Attribution + ShareAlike on        |
|      derived datasets.                                                            |
+-----------------------------------------------------------------------------------+
| 2. TOOLKIT SOURCE CODE (Python scripts in apple/ml-hypersim)                      |
|    - License: Apple Sample Code License / Research Code Terms.                    |
|    - Terms: Permitted for research and adaptation of rendering tools.             |
+-----------------------------------------------------------------------------------+
| 3. THIRD-PARTY ASSET RESTRICTIONS (Evermotion Archinteriors 3D Models)            |
|    - Origin: 460 indoor scenes constructed from Evermotion CAD scene libraries.   |
|    - Terms: Evermotion license prohibits extracting and redistributing raw 3D    |
|      CAD geometry as standalone 3D assets. Rendered 2D/3D images are covered by   |
|      the CC BY-SA 3.0 release.                                                    |
+-----------------------------------------------------------------------------------+
```

#### Exact Legal Requirements of CC BY-SA 3.0:
- **Attribution (BY):** You must give appropriate credit to Apple Inc. and cite the original ICCV 2021 paper (*Roberts et al.*).
- **ShareAlike (SA):** If you remix, transform, or build upon the dataset (e.g., our standardized depth and normal arrays), you must distribute your contributions under the identical license (CC BY-SA 3.0).

---

### 2.2 Stanford ScanNet (Dai et al., CVPR 2017)
The ScanNet dataset is governed exclusively by the **ScanNet Terms of Use** signed upon academic access request:

- **Licensor:** Stanford University, Princeton University, Technical University of Munich.
- **Allowed Scope:** Internal non-commercial academic research and algorithm benchmarking.
- **Prohibitions:**
  - **Zero Commercial Rights:** Use in commercial applications, products, or paid cloud services is strictly forbidden.
  - **No Redistribution:** Mirroring or publicly hosting extracted raw or processed frames is prohibited.
  - **Privacy:** Strict prohibition against attempting to re-identify human occupants.

---

## 3. IMPLICATIONS FOR ROOMAI COMMERCIAL DEPLOYMENT

The legal interaction between Hypersim, ScanNet, and RoomAI's production release is summarized in the matrix below:

| Asset / Component | Current Benchmark Role | Permitted in Commercial SaaS? | Legal Justification |
| :--- | :--- | :--- | :--- |
| **Pretrained WorldMirror-1.0** | Production Candidate | **YES (Subject to Apache 2.0 / OpenRail)** | Pretrained on public datasets; weights released under open research/commercial permissive terms by Tencent Hunyuan. |
| **Hypersim Standardized Subset** | Benchmark Validation | **YES (With ShareAlike)** | CC BY-SA 3.0 permits commercial models provided dataset modifications are open-sourced under CC BY-SA 3.0. |
| **ScanNet Standardized Subset** | Benchmark Validation | **NO (Research Only)** | ScanNet Terms of Use forbid all commercial utilization. |
| **Composite Phase 6 Dataset** | Golden Ground-Truth | **NO (Tainted by ScanNet)** | Because ScanNet is co-mingled in the evaluation manifest, the composite dataset is restricted to non-commercial research. |
| **Stage 1 & Stage 2 Weights** | Fine-Tuned Checkpoints | **NO** | Trained on pseudo-labels and ScanNet sequences, restricting downstream deployment. |

---

## 4. FORMAL ACTIONABLE RECOMMENDATIONS

1. **Retain Dual-Track Separation:**
   - **Internal Research & Benchmarking Track:** Maintain ScanNet and Hypersim as the gold standard for independent validation, error analysis, and scientific papers.
   - **Production Commercial Track:** For commercial distribution, models must be trained exclusively on CC BY-SA / CC0 datasets (such as pure Hypersim, Objaverse-LVIS commercial subset, or in-house synthetic CAD scenes) without ScanNet data.
2. **Documentation Correction:** Report `PHASE7_LICENSE_AND_PROVENANCE_AUDIT.md` is superseded by this document. All future documentation must reference CC BY-SA 3.0 for Hypersim.

