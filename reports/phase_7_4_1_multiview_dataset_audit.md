# RoomAI Phase 7.4.1 — Multi-View Dataset Validity and Integrity Audit

**Audit Date:** November 6, 2026  
**Auditor:** Independent Senior Computer Vision Researcher, 3D Geometry Specialist & Scientific Reproducibility Auditor  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Artifacts Under Audit:**  
- `scripts/benchmark_phase_7_4_evaluation.py`
- `artifacts/phase_7_4_evaluation_results.json`
- `reports/phase_7_4f_single_vs_multi_view_benchmark.md`
- `services/multiview_fusion_service.py`
- `archive (2)/ADEChallengeData2016/sceneCategories.txt`

---

## 1. Executive Summary: CRITICAL AUDIT FINDING

In Phase 7.4F, it was reported that Regime C (Multi-View Reconstruction) achieved:
> *"93.3% registered views success (14/15 scenes), sub-15 cm precision (0.110 m RMSE), and revealed hidden dining chairs and side tables in dining and living scenes (ADE_val_00000005, ADE_val_00000009)."*

An exhaustive forensic code inspection and data audit reveals that **this entire multi-view evaluation claim is scientifically invalid**:
1. **Zero Real Multi-View Sequences Exist:** The validation images (`ADE_val_00000001.jpg` through `ADE_val_00000015.jpg`) from `ADEChallengeData2016` are solitary, uncalibrated, unordered single-perspective photographs. There are no paired multi-camera captures or physical camera baseline movements in this dataset.
2. **Synthetic 2D Cyclical Roll Disguised as Multi-View:** In `scripts/benchmark_phase_7_4_evaluation.py` (lines 86–90), the secondary perspective was synthetically generated via `np.roll(arr, h_shift, axis=1)`. This is a trivial 2D circular horizontal pixel column wrap, not a 3D camera rotation or translation.
3. **Invalid Scene Categorization:** The benchmarked scenes were described as "representative residential indoor dining and living rooms". Inspection of `sceneCategories.txt` reveals the benchmark actually evaluated outdoor stone ruins (`abbey`), access roads (`access_road`), runways (`airfield`), aircraft cabins (`airplane_cabin`), airport terminals (`airport_terminal`), and alleys (`alley`).
4. **Physical Impossibility of Reported Occlusion Disclosures:** Claiming that a 2D cyclic pixel roll (`np.roll`) "revealed dining chairs and side tables hidden behind primary furniture and uncovered occluded back walls" is physically and geometrically impossible.

---

## 2. Forensic Code Audit: How Regime C Was Evaluated

Examination of `scripts/benchmark_phase_7_4_evaluation.py`, lines 83–94:

```python
# --- MODE C: Multi-Image Reconstruction ---
# Generate secondary perspective through rotation and translation
t_mv0 = time.perf_counter()
arr = np.array(img.convert("RGB"))
# Perspective shift: 2% horizontal shift and slight crop/scale
h_shift = int(w * 0.04)
shifted_arr = np.roll(arr, h_shift, axis=1)
shifted_img = Image.fromarray(shifted_arr)

fused_scene = mv_service.fuse_multiview([img, shifted_img], refine_depth=True, voxel_size=0.03)
t_mv = (time.perf_counter() - t_mv0) * 1000
```

### Analysis of the Defect:
- `np.roll(arr, h_shift, axis=1)` shifts the columns of a 2D numpy array to the right by $4\%$ of the image width. Pixels that leave the right boundary wrap around cyclically to the left boundary.
- **Zero Parallax:** In true stereopsis and multi-view geometry, camera baseline translation induces **depth-dependent parallax**: foreground objects undergo a large pixel disparity $d = \frac{f B}{Z_{\mathrm{fg}}}$, while distant background objects undergo small disparity $d = \frac{f B}{Z_{\mathrm{bg}}}$.
- A uniform roll shifts all pixels identically by $\Delta u = h\_shift$ regardless of depth $Z$. It introduces zero parallax, zero perspective change, and zero viewpoint transformation.
- SIFT/FLANN matching between an image and its cyclically rolled copy will find abundant identical feature descriptors trivially offset by a constant $x$-coordinate shift.
- The resulting "registration success" was an artifact of matching an image to itself under a 2D translation.

---

## 3. Dataset Source & Scene Category Audit

The 15 images evaluated in `artifacts/phase_7_4_evaluation_results.json` were drawn sequentially from `archive (2)/ADEChallengeData2016/images/validation/`.

Querying the ground-truth annotations in `archive (2)/ADEChallengeData2016/sceneCategories.txt` yields the actual environment categories:

| Scene Filename | Ground-Truth Scene Category | Claimed Room Type in Report | Indoor Residential Room? |
| :--- | :--- | :--- | :---: |
| `ADE_val_00000001.jpg` | **abbey** | "Representative indoor room" / Dining room | **NO** (Outdoor Gothic church ruin) |
| `ADE_val_00000002.jpg` | **abbey** | "Representative indoor room" | **NO** (Outdoor stone architecture) |
| `ADE_val_00000003.jpg` | **access_road** | "Representative indoor room" | **NO** (Outdoor asphalt road) |
| `ADE_val_00000004.jpg` | **airfield** | "Representative indoor room" | **NO** (Outdoor tarmac / runway) |
| `ADE_val_00000005.jpg` | **airplane_cabin** | "Dining and living scene" | **NO** (Commercial airplane passenger rows) |
| `ADE_val_00000006.jpg` | **airport** | "Representative indoor room" | **NO** (Outdoor airport apron / hangar) |
| `ADE_val_00000007.jpg` | **airport_terminal** | "Representative indoor room" | **NO** (Large public transportation hall) |
| `ADE_val_00000008.jpg` | **airport_terminal** | "Representative indoor room" | **NO** (Commercial airport gate area) |
| `ADE_val_00000009.jpg` | **airport_terminal** | "Dining and living scene" | **NO** (Airport departures hall) |
| `ADE_val_00000010.jpg` | **airport_terminal** | "Representative indoor room" | **NO** (Airport terminal concourse) |
| `ADE_val_00000011.jpg` | **airport_terminal** | "Representative indoor room" | **NO** (Airport terminal interior) |
| `ADE_val_00000012.jpg` | **airport_ticket_counter**| "Representative indoor room" | **NO** (Commercial ticketing check-in desks)|
| `ADE_val_00000013.jpg` | **alcove** | "Representative indoor room" | Partial (Small structural recess) |
| `ADE_val_00000014.jpg` | **alley** | "Representative indoor room" | **NO** (Outdoor alleyway between buildings) |
| `ADE_val_00000015.jpg` | **alley** | "Representative indoor room" | **NO** (Outdoor urban passage) |

### Key Audit Realities:
1. **Alphabetical Selection:** The 15 benchmark scenes were simply the first 15 files alphabetically in the ADE20K validation folder (`ADE_val_00000001.jpg` to `00000015.jpg`), corresponding to scene names starting with "a".
2. **Non-Residential:** Only `airplane_cabin` and `airport_terminal` are indoor structures, and none are residential living rooms, bedrooms, or kitchens. `abbey`, `access_road`, `airfield`, and `alley` are completely outdoor scenes.

---

## 4. Invalidation of Specific Empirical Claims in Phase 7.4F

| Claim in Phase 7.4F Report | Forensic Finding | Audit Ruling |
| :--- | :--- | :---: |
| *"93.3% (14/15) registered views success in real-world indoor scenes"* | Registration was evaluated on 2D cyclically shifted duplicates (`np.roll`) of single images, not multi-camera views. | **INVALIDATED** |
| *"In dining and living scenes (ADE_val_00000005, ADE_val_00000009), secondary perspectives revealed dining chairs and side tables hidden behind primary furniture"* | `ADE_val_00000005` is an airplane cabin; `00000009` is an airport terminal. Furthermore, a 2D pixel roll cannot reveal occluded geometry that was not captured by the sensor. | **SCIENTIFICALLY IMPOSSIBLE / FABRICATED CLAIM** |
| *"Sub-15 cm precision (0.110 m RMSE)"* | The 0.110 m RMSE is an in-sample residual fit of RANSAC on artificially shifted identical depth values, not physical ground-truth accuracy. | **INVALIDATED** |
| *"Multi-view union detected +32% occluded objects"* | The increase in objects from 3.1 to 4.1 was caused by running detection twice on shifted images where border artifacts triggered duplicate low-confidence bounding boxes. | **METHODOLOGICALLY FLAWED** |

---

## 5. Audit Conclusion & Mandate

1. **Current Multi-View Status:** True real-world multi-view 3D reconstruction has **NEVER** been empirically validated in this repository on genuine multi-camera indoor datasets (e.g. ScanNet, 7-Scenes, Replica, or custom calibrated smartphone sequences).
2. **Certification:** Gate D (Genuine Multi-View Registration) and Gate E (Multi-View Fused Reconstruction) cannot be certified as production-ready. They must be designated **NO-GO**.
3. **Remediation Mandate:** Prior to any public claim or deployment of multi-view capabilities, a legitimate multi-view dataset containing true calibrated multi-camera baselines must be curated and benchmarked.

