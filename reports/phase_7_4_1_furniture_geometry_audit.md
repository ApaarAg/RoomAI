# RoomAI Phase 7.4.1 — Furniture 3D Geometry & Physical Dimension Audit

**Audit Date:** November 6, 2026  
**Auditor:** Independent Senior Computer Vision Researcher & 3D Geometry Specialist  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Code & Benchmark Under Audit:**  
- `services/scene_fusion_service.py` (`_associate_depth_to_objects`, `_align_furniture_to_floor`)  
- `artifacts/phase_7_4_evaluation_results.json` (`scene_index: 1`, `ADE_val_00000001.jpg`)  
- `reports/phase_7_4d_furniture_alignment_validation.md`

---

## 1. Executive Summary: The "2.48-Meter-Tall Table" Anomaly

In `artifacts/phase_7_4_evaluation_results.json` and Phase 7.4D validation reports, the first benchmark scene (`ADE_val_00000001.jpg`) reported the following estimated 3D bounding box for an indoor `table`:
- **Centroid:** $[X = -0.18\text{ m},\; Y = 0.16\text{ m},\; Z = 8.57\text{ m}]$
- **Physical Extent:** $[\mathrm{Width} = 2.54\text{ m},\; \mathrm{Height} = 2.48\text{ m},\; \mathrm{Depth} = 1.78\text{ m}]$

### The Obvious Physical Inconsistency:
Standard residential dining and coffee tables exhibit heights between **$0.40\text{ m}$ and $0.78\text{ m}$**. A table that is **$2.48\text{ m}$ tall** (over 8 feet and 1 inch high) is physically impossible in any domestic residential setting—it is taller than standard room doorways ($2.03\text{ m}$) and nearly reaches a standard ceiling ($2.50\text{ m}$).

The auditor conducted a deep forensic investigation into the exact root causes of this anomaly.

---

## 2. Root Cause Investigation & Forensic Breakdown

### Root Cause 1: Evaluating an Exterior Stone Ruin as a Domestic Room
As discovered in the Dataset Audit (`phase_7_4_1_multiview_dataset_audit.md`), `ADE_val_00000001.jpg` is **not** an indoor living room or dining room. It belongs to the ADE20K category **`abbey`**—an outdoor stone monastery ruin.
- The 2D indoor detector (`indoor_detector_a3_yolo11s_800px_best.pt`), forced to run on an outdoor stone ruin, generated a low-confidence false positive ($\mathrm{confidence} = 0.36$) classifying a massive outdoor stone archway / carved altar structure as a `table`.

### Root Cause 2: Extreme Depth Distance ($Z = 8.57\text{ m}$)
In true residential rooms, furniture is situated within $1.5\text{ m} - 4.5\text{ m}$ of the camera.
- In `ADE_val_00000001.jpg`, the detected stone ruin was located at an extreme distance of **$Z = 8.57\text{ m}$**.

### Root Cause 3: Pinhole Unprojection with Hardcoded Intrinsics
Examination of `services/scene_fusion_service.py`, lines 401–410:
```python
z = median_d
x = (bbox_center_x - cx) * z / fx
y = (bbox_center_y - cy) * z / fy

# Estimate physical 3D extents in meters:
phys_w = float(max(0.2, (x2 - x1) * z / fx))
phys_h = float(max(0.2, (y2 - y1) * z / fy))
depth_span = float(np.percentile(fg_depths, 90) - np.percentile(fg_depths, 10)) if len(fg_depths) > 5 else 0.5
phys_d = float(max(0.3, min(max(depth_span, phys_w * 0.7), 2.2)))
```

At resolution $683 \times 512$:
- Intrinsics default to: $f_x = 1000.0,\; f_y = 1000.0$.
- The 2D bounding box detected on the abbey ruin spanned:
  $$\Delta x_{\mathrm{px}} = (x_2 - x_1) \approx 296\text{ px}$$
  $$\Delta y_{\mathrm{px}} = (y_2 - y_1) \approx 290\text{ px}$$
- Substituting these exact values into the unprojection equations:
  $$\mathrm{phys\_w} = \frac{296 \times 8.57}{1000.0} = 2.5367\text{ m} \approx \mathbf{2.54\text{ m}}$$
  $$\mathrm{phys\_h} = \frac{290 \times 8.57}{1000.0} = 2.4853\text{ m} \approx \mathbf{2.48\text{ m}}$$
  $$\mathrm{phys\_d} = \max(0.3, \min(2.54 \times 0.7, 2.2)) = \min(1.778, 2.2) \approx \mathbf{1.78\text{ m}}$$

**The mathematical deduction is irrefutable:** The $2.54\text{ m} \times 2.48\text{ m} \times 1.78\text{ m}$ extent was a direct, linear consequence of multiplying a 290-pixel bounding box by an $8.57\text{ m}$ distance under an assumed $f = 1000\text{ px}$ focal length.

---

## 3. Evaluation of the Depth Clustering Mechanism

In Phase 7.4, a foreground depth clustering algorithm was introduced to prevent bounding boxes from being pulled back into the rear wall:

```python
# Foreground object depth: isolate front surface cluster to avoid background wall pull
foreground_cutoff = float(np.percentile(valid_depths, 40))
fg_depths = valid_depths[valid_depths <= foreground_cutoff + 0.35]
if len(fg_depths) < 5:
    fg_depths = valid_depths

mean_d = float(np.mean(fg_depths))
median_d = float(np.median(fg_depths))
```

### Strengths:
1. **Wall-Pull Prevention:** In 2D bounding boxes that contain both the foreground chair/sofa and background drywall, taking the mean of all depths systematically biases the object centroid several meters backwards into the wall. Isolating depths $\le \mathrm{P}_{40} + 0.35\text{ m}$ successfully locks the centroid to the object's visible front face.
2. **Elimination of Legacy 0.45m Hardcode:** The legacy baseline had an egregious bug where object $Z$ was bound to a normalized float ($Z \approx 0.45\text{ m}$) regardless of scene geometry. The depth clustering mechanism completely resolved this bug.

### Geometric Limitations:
1. **Uncalibrated FOV Coupling:** Because $W = \Delta u \cdot \frac{Z}{f}$, any inaccuracy in the assumed focal length $f$ scales physical dimensions proportionately. If a user uploads a smartphone wide-angle photo with $f \approx 500\text{ px}$, but RoomAI assumes $f = 1000\text{ px}$, all furniture dimensions will be calculated at **$50\%$ of their true physical size**!
2. **Missing Semantic Category Prior:** The pipeline currently permits any physical height from $0.20\text{ m}$ to infinity. It lacks sanity bounds informed by semantic priors (e.g., capping dining tables to $H \le 1.1\text{ m}$, chairs to $H \le 1.3\text{ m}$, beds to $H \le 1.2\text{ m}$).

---

## 4. Auditor's Recommendations for Furniture Geometry

1. **Semantic Dimension Sanity Bounds:**  
   Incorporate physical sanity envelopes based on object semantics:
   - `table`: Height $0.35\text{ m} - 1.10\text{ m}$; Depth $0.40\text{ m} - 2.50\text{ m}$
   - `chair#stool`: Height $0.40\text{ m} - 1.30\text{ m}$; Width $0.35\text{ m} - 0.90\text{ m}$
   - `sofa`: Height $0.60\text{ m} - 1.10\text{ m}$; Depth $0.70\text{ m} - 1.40\text{ m}$
   - `bed`: Height $0.40\text{ m} - 1.20\text{ m}$; Length $1.40\text{ m} - 2.30\text{ m}$
   When estimated dimensions exceed these physical envelopes by $> 50\%$, raise an uncertainty flag in the UI informing the user that the camera focal length is uncalibrated.
2. **Interactive Focal Length / FOV Slider in UI:**  
   Allow users to specify camera type (e.g. Standard Phone $65^\circ$, Ultra-Wide $90^\circ$, DSLR $50^\circ$) to correct $f_x, f_y$ dynamically.

---

## 5. Auditor's Gate B Verdict

**Sub-Gate B (Furniture 3D Bounding-Box Estimation): CONDITIONAL GO**  
The underlying foreground clustering and unprojection math are functionally implemented and resolve the legacy floating bug. However, the system must not present dimension numbers as definitive ground truth without warning users of uncalibrated optical scale factors.

