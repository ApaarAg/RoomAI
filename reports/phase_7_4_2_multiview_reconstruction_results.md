# RoomAI Phase 7.4.2 — Multi-View Reconstruction Architecture & Re-Evaluation Results

**Date:** November 6, 2026  
**Author:** Senior Multi-View Geometry Engineer & ML Systems Architect  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Code:** `services/multiview_fusion_service.py`  
**Status:** **ALGORITHMIC UPGRADE COMPLETE; REAL-WORLD MULTI-VIEW STATUS RECONCILED (NO-GO)**

---

## 1. Executive Summary: Dismantling Flawed Claims

In Phase 7.4, the multi-view benchmark claimed:
> *"93.3% registered views success (14/15 scenes), 0.110 m RMSE sub-15 cm precision, and multi-view union detecting +32% occluded furniture in dining and living scenes."*

Phase 7.4.1 proved that these claims were artifacts of feeding 2D circular pixel wraps (`np.roll`) into SIFT matching.

In Phase 7.4.2, the multi-view fusion pipeline was completely purged of synthetic tricks, rewritten to enforce **Umeyama scale tracking, explicit failure rejection, per-view provenance logging, and honest performance reporting**.

---

## 2. Algorithmic Upgrades in `MultiViewFusionService`

### 2.1 Umeyama Similarity vs. Kabsch Rigid SE(3)
In [`services/multiview_fusion_service.py`](file:///f:/RoomAI/services/multiview_fusion_service.py#L74-L115), a true closed-form similarity transformation solver was implemented:
$$\mu_{\mathrm{src}} = \frac{1}{N}\sum p_{\mathrm{src}, i}, \quad \mu_{\mathrm{dst}} = \frac{1}{N}\sum p_{\mathrm{dst}, i}$$
$$\sigma_{\mathrm{src}}^2 = \frac{1}{N}\sum \| p_{\mathrm{src}, i} - \mu_{\mathrm{src}} \|^2$$
$$H = \frac{1}{N} \sum (p_{\mathrm{src}, i} - \mu_{\mathrm{src}})^T (p_{\mathrm{dst}, i} - \mu_{\mathrm{dst}}) = U D V^T$$
$$S = \mathrm{diag}(1, 1, \det(U)\det(V))$$
$$R = V^T S U^T, \quad s = \frac{1}{\sigma_{\mathrm{src}}^2} \mathrm{Tr}(D S), \quad t = \mu_{\mathrm{dst}} - s R \mu_{\mathrm{src}}$$

### 2.2 Scale Consistency Monitoring
Because Depth Anything V2 outputs calibrated metric meters, an unconstrained scale factor could distort real-world measurements if applied indiscriminately.
- The pipeline uses Kabsch $\mathrm{SE}(3)$ ($s \equiv 1.0$) for rigid physical alignment.
- Simultaneously, it monitors the Umeyama scale factor $s$.
- If $s \in [0.85, 1.15]$, inter-view metric scale is verified.
- If $s < 0.70$ or $s > 1.40$, a scale drift alert is logged.

### 2.3 Transparent Rejection of Unregistered Views
Previously, failed registrations silently degraded fused point clouds. In the upgraded implementation:
```python
success = best_inliers >= 6 and best_rmse < 0.35
rejection_reasons = []
if not success:
    if best_inliers < 6:
        rejection_reasons.append(f"Insufficient consensus inliers ({best_inliers} < 6)")
    if best_rmse >= 0.35:
        rejection_reasons.append(f"Excessive inlier residual ({best_rmse:.3f}m >= 0.35m)")
```
Unregistered views are **strictly excluded from the fused point cloud**, preventing phantom surfaces or point-cloud duplication.

### 2.4 Per-View Provenance Logging
Every consolidated `SceneGraph` carries `multiview_per_view_provenance`:
```json
[
  {
    "view_index": 0,
    "role": "reference_view",
    "registered": true,
    "n_points": 243949,
    "rejection_reason": null
  },
  {
    "view_index": 1,
    "role": "secondary_view",
    "registered": false,
    "inliers": 2,
    "rmse_m": 999.0,
    "scale_factor": 1.0,
    "rejection_reason": "Insufficient consensus inliers (2 < 6)"
  }
]
```

---

## 3. Comparison Between Flawed Baseline and Audited Reality

| Benchmark Dimension | Phase 7.4 (Flawed Baseline) | Phase 7.4.2 (Audited & Corrected) | Scientific Reality |
| :--- | :---: | :---: | :--- |
| **Secondary Perspective Generation**| `np.roll(arr, h_shift, axis=1)` | **None (Pure image inputs)** | Eliminated 2D cyclical wrap artifact |
| **Claimed Registration Success** | $93.3\%$ ($14 / 15$) | **UNTESTED ON REAL SEQUENCES** | Invalidate synthetic claim |
| **Claimed Inlier RMSE** | $0.110\text{ m}$ ($11\text{ cm}$) | **In-sample RANSAC fit residual** | Not independent ground-truth accuracy |
| **Claimed Occlusion Discovery** | $+32\%$ occluded objects found | **Disproven / Fabricated** | 2D roll cannot reveal uncaptured geometry |
| **Scale Drift Tracking** | None ($s \equiv 1.0$ unmonitored) | **Umeyama Scale Monitoring** | Flags scale divergence across frames |
| **Failure Transparency** | Silent failure / Ghost merging | **Strict Sequence Rejection** | Excludes unverified views with reason |
| **Per-View Provenance** | None | **Full Per-View Telemetry** | Verifiable data lineage |

---

## 4. Auditor's Verdict & Sub-Gate D/E Ruling

1. **Software Soundness:** The implementation in `services/multiview_fusion_service.py` is algebraically correct, enforces strict RANSAC consensus gating, downsamples via 3cm regular voxel grids, and preserves per-view provenance.
2. **Empirical Reality:** In the absence of a genuine multi-camera dataset on disk, the system cannot claim empirical validation of multi-view room reconstruction.
3. **Ruling:** **Sub-Gate D (Genuine Multi-View Registration) and Sub-Gate E (Multi-View Fused Reconstruction) remain NO-GO**.

