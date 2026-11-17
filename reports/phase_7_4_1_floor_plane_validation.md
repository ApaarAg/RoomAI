# RoomAI Phase 7.4.1 — Floor Plane Estimation & Geometric Snapping Audit

**Audit Date:** November 6, 2026  
**Auditor:** Independent Senior Computer Vision Researcher & 3D Geometry Specialist  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Code Under Audit:**  
- `services/room_layout_service.py` (`fit_plane_ransac`, `estimate_layout`)  
- `services/scene_fusion_service.py` (`_align_furniture_to_floor`)  
- `reports/phase_7_4d_furniture_alignment_validation.md`  
- `reports/phase_7_4f_single_vs_multi_view_benchmark.md`

---

## 1. Executive Summary: Raw Output vs. Heuristic Post-Correction

In Phase 7.4D and Phase 7.4F, it was reported that:
> *"Furniture Floor Contact Error was reduced by 98.7% from 1.463 m in the legacy baseline to an average of 0.019 m (1.9 cm) in Regime B/C, establishing physical grounding."*

The auditor examined whether this $1.9\text{ cm}$ floor contact precision represents an intrinsic advance in depth estimation accuracy or the result of a deterministic post-hoc constraint.

### Audit Finding:
1. **Algorithmic Snapping Reality:** The $0.019\text{ m}$ contact error is enforced by an explicit post-processing rule (`_align_furniture_to_floor`) that forcibly shifts the vertical centroid $Y_{\mathrm{centroid}}$ of any floor-standing object to $Y_{\mathrm{floor}} - \frac{H}{2}$ whenever the unprojected bottom deviates by more than $0.02\text{ m}$.
2. **Raw Model Performance:** Raw monocular back-projection without snapping exhibits bottom-to-floor elevation discrepancies of **$0.12\text{ m}$ to $0.48\text{ m}$** due to monocular depth noise and bounding-box edge bleed.
3. **Engineering Assessment:** While post-hoc geometric snapping is an entirely valid, industry-standard CAD/AR heuristic (ensuring furniture does not visually float in 3D viewers), presenting the resulting $0.019\text{ m}$ metric as proof that the neural depth model accurately predicts physical floor contact is scientifically conflated.

---

## 2. Floor Plane Detection Mechanism in `RoomLayoutService`

Inspection of `services/room_layout_service.py`, lines 130–152:

```python
# In camera coordinates, floor points typically occupy lower vertical quadrant (+Y)
y_threshold = float(np.percentile(xyz_arr[:, 1], 60))
floor_candidate_points = xyz_arr[xyz_arr[:, 1] >= y_threshold]

normal, d_val, inliers = fit_plane_ransac(floor_candidate_points, distance_threshold=0.08)

if normal is not None and abs(normal[1]) >= 0.70:
    # Floor normal must point predominantly along the vertical axis (|n_y| >= 0.70)
    camera_height = abs(d_val)
    floor_elevation = float(np.percentile(floor_candidate_points[:, 1], 90))
    floor_detected = True
```

### Theoretical Strengths:
1. **Vertical Constraint Verification:** Enforcing $|n_y| \ge 0.70$ prevents vertical walls from being erroneously classified as floors. The plane normal is verified to be primarily vertical.
2. **Camera Coordinate System Alignment:** In standard computer vision conventions ($+Z$ forward, $+X$ right, $+Y$ down), the floor plane lies at positive $Y$. Filtering points with $Y \ge \mathrm{P}_{60}$ effectively restricts RANSAC candidates to the lower portion of the field of view.
3. **Robust RANSAC Fitting:** Planar RANSAC with distance threshold $\tau = 0.08\text{ m}$ rejects scattered rug textures, small footstools, and floor clutter.

### Limitations in Non-Standard Scenarios:
1. **Camera Pitch Ambiguity:** When the camera is tilted sharply downwards or upwards, $Y \ge \mathrm{P}_{60}$ is no longer parallel to the physical floor. The algorithm currently assumes near-horizontal camera orientation.
2. **Missing Semantic Mask Coupling:** The candidate point selection relies solely on the geometric coordinate percentile ($Y \ge \mathrm{P}_{60}$) rather than intersecting with semantic segmentation masks (e.g. `category == "floor"` from SegFormer or YOLO-seg). If an image has large low-lying furniture (e.g. a low coffee table or mattress), points from the furniture surface can pollute the floor candidate pool.

---

## 3. Forensic Code Audit of Floor Snapping

Examination of `services/scene_fusion_service.py`, lines 336–358:

```python
def _align_furniture_to_floor(self, objects: List[SceneObject], floor_elevation_m: float):
    """Physically grounds floor-standing furniture pieces onto the estimated floor plane."""
    floor_standing_categories = {
        "table", "chair", "stool", "sofa", "couch", "bed", 
        "cabinet", "shelf", "desk", "nightstand", "tv_stand"
    }

    for obj in objects:
        cat = (obj.semantic_class or "").lower()
        if any(f_cat in cat for f_cat in floor_standing_categories):
            if obj.geometry and obj.geometry.centroid and obj.geometry.extent:
                cx, cy, cz = obj.geometry.centroid
                w, h, d = obj.geometry.extent
                current_bottom = cy + (h / 2.0)

                # If bottom deviates from floor plane, snap it to rest directly on the floor
                if abs(current_bottom - floor_elevation_m) > 0.02:
                    grounded_cy = floor_elevation_m - (h / 2.0)
                    obj.geometry.centroid = (cx, round(grounded_cy, 3), cz)
```

### Proof of Circular Metric Reporting:
In `scripts/benchmark_phase_7_4_evaluation.py`, lines 68–76:
```python
bottom_y = cy + (ph / 2.0)
delta_floor = abs(bottom_y - b_floor_elev) if floor_detected else 0.0
```
- When `_align_furniture_to_floor` executes, any error $> 0.02\text{ m}$ triggers `grounded_cy = floor_elevation_m - (h / 2.0)`.
- Recomputing `bottom_y = grounded_cy + (h / 2.0)` yields `floor_elevation_m` exactly.
- Therefore:
  $$\Delta_{\mathrm{floor}} = |\mathrm{bottom\_y} - \mathrm{b\_floor\_elev}| \le 0.02\text{ m}$$
- The benchmark report cited this identical $0.019\text{ m}$ average delta as empirical validation that the 3D pipeline had mastered physical grounding.
- **Auditor's Ruling:** The $0.019\text{ m}$ error is a mathematically guaranteed property of the clamping threshold ($0.02\text{ m}$), not a measurement of neural network accuracy.

---

## 4. Auditor's Verdict & Engineering Evaluation

| Component | Technical Implementation | Practical Utility | Scientific Truthfulness |
| :--- | :--- | :--- | :--- |
| **RANSAC Floor Estimation** | Geometric RANSAC on lower points | High in standard residential rooms | Sound when camera pitch is moderate |
| **Floor Snapping Heuristic** | Clamps object bottom to floor plane | Critical for professional AR/3D visualization | Valid heuristic, but must not be reported as neural depth accuracy |
| **Reported 0.019m Error** | Evaluation of clamped coordinates | N/A (Tautological) | Conflated; must be documented as "clamped constraint" |

### Production Verdict:
The floor plane detection and snapping constraint are **functionally sound and beneficial for user experience**—they eliminate floating furniture in the 3D viewer. However, scientific documentation must honestly clarify that grounding is achieved via geometric constraint enforcement rather than raw monocular depth regression.

