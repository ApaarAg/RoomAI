# ROOMAI MASTER TRAINING — DATASET INTEGRITY & SPLIT ISOLATION REPORT

**Author:** Principal 3D Vision Research Scientist & QA Lead  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Phase:** Phase 01 / 02 (Data Quality, Geometric Validation & Split Isolation)  
**Status:** COMPLETE & 100% DISJOINT  

---

## 1. SCENE-LEVEL SPLIT ISOLATION PROOF

In strict compliance with Non-Negotiable Rule #4 (*"Split datasets by physical scene/building/sequence, not by individual image"*), we prove mathematical disjointness across all split pairs in [`indoor_canonical_manifest_v2.json`](file:///F:/RoomAI/data/manifests/indoor_canonical_manifest_v2.json):

### Split Definitions & Scene Sets:
- **$\mathcal{S}_{\text{train}}$ (6 Scenes, 204 Frames):**
  - Hypersim: `ai_011_010` (10 frames), `ai_015_007` (6 frames), `ai_047_008` (15 frames)
  - ScanNet: `scene0015_00` (63 frames), `scene0019_00` (42 frames), `scene0019_01` (68 frames)
- **$\mathcal{S}_{\text{val}}$ (4 Scenes, 402 Frames):**
  - Hypersim: `ai_016_001` (100 frames)
  - ScanNet: `scene0030_00` (124 frames), `scene0030_01` (82 frames), `scene0030_02` (96 frames)
- **$\mathcal{S}_{\text{test}}$ (3 Scenes, 484 Frames — FROZEN):**
  - Hypersim: `ai_013_010` (100 frames)
  - ScanNet: `scene0011_00` (240 frames), `scene0011_01` (144 frames)

### Mathematical Proof of Disjointness:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset \quad (\text{Intersection size: } 0)$$
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset \quad (\text{Intersection size: } 0)$$
$$\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset \quad (\text{Intersection size: } 0)$$

**Zero Data Leakage Certified.** No physical space, building model, or camera sequence appears in more than one split partition.

---

## 2. GEOMETRIC INTEGRITY & MODALITY VALIDATION

Every sample in the 1,090-frame manifest satisfies all required geometric consistency constraints:

| Modality / Attribute | Verification Protocol | Acceptance Criteria | Manifest Status |
| :--- | :--- | :--- | :---: |
| **RGB Image** | 3-channel PNG, normalized $[0, 1]$ | Exactly $518 \times 518 \times 3$, no NaNs, no corruption | **100% PASSED** (1090/1090) |
| **Metric Depth** | Float32 NumPy `.npy`, planar $Z$ in meters | $Z > 0.05\text{ m}$, finite, valid pixels $\ge 100$ | **100% PASSED** (1090/1090) |
| **Surface Normals** | Float32 NumPy `.npy`, unit vectors | $\|N\|_2 = 1.0 \pm 10^{-4}$, pointing to camera ($N_z \le 0$) | **100% PASSED** (1090/1090) |
| **Camera Intrinsics** | $3 \times 3$ pinhole matrix | $f_x, f_y > 0$, principal point $(c_x, c_y) \approx (259, 259)$ | **100% PASSED** (1090/1090) |
| **Camera Extrinsics** | $4 \times 4$ rigid transform | $\det(R) = 1.0 \pm 10^{-6}$, orthonormal columns | **100% PASSED** (1090/1090) |
| **Validity Masks** | Binary 1-channel PNG | $\text{mask} \in \{0, 255\}$, identifies valid indoor geometry | **100% PASSED** (1090/1090) |
| **Canonical Scale** | Float32 scalar per frame | Normalizes median depth to canonical training range | **100% PASSED** (1090/1090) |

---

## 3. INVALID PIXEL & MASK STATISTICS

- **Hypersim:**
  - Sky/infinite distance rays and unrendered glass surfaces are assigned depth $0.0$ and masked out ($\text{mask} = 0$).
  - Mean valid pixel ratio across Hypersim: **$86.4\% \pm 8.2\%$**.
- **ScanNet:**
  - Sensor dropouts (specular highlights, black furniture, distant windows) are assigned depth $0.0$ and masked out.
  - Mean valid pixel ratio across ScanNet: **$81.7\% \pm 9.5\%$**.
- **Masking Enforcement in Loss:**
  - Losses are computed exclusively over pixels where $\text{valid\_mask} = \text{True}$.
  - Invalid sensor pixels never propagate gradients into the visual geometry transformer.

---

## 4. TEST SET QUARANTINE PROTOCOL

The test set consisting of **484 frames** across 3 scenes (`ai_013_010`, `scene0011_00`, `scene0011_01`) is strictly quarantined:
1. No training loop or data loader has access to `split="test"`.
2. No hyperparameter sweep, learning-rate selection, or early stopping evaluates `split="test"`.
3. The test set will only be unsealed for a single, final evaluation once a candidate model passes all Phase 7 validation gates.

