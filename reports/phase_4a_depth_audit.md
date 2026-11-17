# RoomAI — Phase 4A: Depth Pipeline Audit Report

**Date:** 2026-10-31  
**Author:** Principal Computer Vision Research Engineer & 3D Vision Specialist  
**Scope:** Complete audit of the current monocular depth estimation pipeline in RoomAI.

---

## 1. Current Checkpoint Configuration

| Parameter | Value | Source |
| :--- | :--- | :--- |
| **Active Checkpoint** | `checkpoints/depth_anything_v2_small` (local) or `depth-anything/Depth-Anything-V2-Small-hf` (HuggingFace) | `config/model_config.yaml` |
| **Model Architecture** | DINOv2 ViT-Small (patch size 14) + DPT Head | `models/scene3d_model.py` |
| **Parameter Count** | 24.8M | Benchmark report |
| **License** | Apache-2.0 | HuggingFace model card |

---

## 2. Depth Type: Relative Inverse Depth

**Current output is strictly RELATIVE inverse depth**, not metric physical distance.

- **Output range**: [0.0, 1.0] (min-max normalized per image)
- **Semantics**: Higher values = closer to camera (foreground), Lower values = farther (background)
- **Calibration**: Affine-invariant; no physical units (meters, feet)
- **Documented in UI**: `render_depth_disclaimer()` in `ui/components.py`

---

## 3. Metric Indoor Checkpoint Status

**NOT INTEGRATED.** The configuration explicitly states:
```yaml
relative_depth_only: true
```

The Metric Indoor checkpoint (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`) exists in HuggingFace but is not loaded, configured, or exposed in the UI.

---

## 4. Image Preprocessing & Postprocessing

### Preprocessing (lines 151-154 in `scene3d_model.py`):
```python
inputs = self.processor(images=img_rgb, return_tensors="pt")
```
- Uses HuggingFace `AutoImageProcessor` which:
  - Resizes to 518×518 (native ViT-S patch division: 37×37 patches of 14×14)
  - Normalizes with ImageNet mean/std
  - Maintains aspect ratio via padding

### Postprocessing (lines 174-182):
```python
pred_aligned = torch.nn.functional.interpolate(
    pred_depth.unsqueeze(1),
    size=(orig_h, orig_w),
    mode="bicubic",
    align_corners=False,
).squeeze()
```
- **Bicubic interpolation** from 518×518 back to original image resolution
- Preserves original aspect ratio
- Output shape: `(orig_h, orig_w)`

### Normalization (lines 187-195):
```python
raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=1.0, neginf=0.0)
d_min, d_max = raw_depth.min(), raw_depth.max()
norm_depth = (raw_depth - d_min) / (d_max - d_min) if d_max > d_min else zeros
```
- **Arbitrary per-image min-max normalization** destroys metric scale
- All output forced to [0, 1] range
- NaN/Inf guarded

---

## 5. Depth Visualization Issues

**Function:** `create_depth_heatmap()` in `services/image_service.py` (lines 110-125)

### Current Implementation:
```python
norm = depth_array.astype(np.float32)
min_val, max_val = norm.min(), norm.max()
norm = (norm - min_val) / (max_val - min_val)  # Linear min-max

# Custom plasma-like colormap
r = 2.0 * norm
g = 2.0 * (1.0 - |norm - 0.5|)
b = 2.0 * (1.0 - norm)
```

### Identified Problems:
1. **Linear min-max normalization** compresses mid-range indoor scenes (typically narrow depth distributions) into flat color bands
2. **No percentile clipping** - outliers compress the bulk of the scene
3. **Custom colormap** (not standard Turbo/Inferno/Viridis) reduces interpretability
4. **No invalid pixel visualization** - NaN/zero regions blend into background
5. **Near/far interpretation**: Higher normalized value = brighter/yellow (near), Lower = dark purple (far) — correct for inverse depth, but confusing without legend

---

## 6. Invalid Value Handling

| Stage | Handling |
| :--- | :--- |
| Model output | `np.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)` — line 185 |
| Visualization | None — zero values mapped to dark purple (far background) |
| UI disclaimer | `render_depth_disclaimer()` — correctly states relative non-metric |

---

## 7. UI Integration Gaps

1. **Single visualization mode** - only plasma heatmap available
2. **No grayscale depth map** - raw values not inspectable
3. **No metric/relative toggle** - Metric Indoor model not exposed
4. **No point cloud or 3D export** - `generate_point_cloud: false` in config
5. **No camera intrinsics configuration** - prevents 3D unprojection

---

## 8. Summary of Required Changes for Phase 4

| Phase | Change Required |
| :--- | :--- |
| **4B** | Add Metric Indoor model loading, preserve meter units, add evaluation metrics |
| **4C** | Add grayscale depth, standard Turbo colormap, percentile clipping, invalid pixel masking |
| **4D** | Add camera intrinsics configuration schema (focal length, principal point) |
| **4E** | Implement depth unprojection to 3D point cloud (PLY export) |
| **4F** | Fuse semantic/instance masks with depth for scene representation |
| **4G** | Add Streamlit-compatible 3D viewer (e.g., `st.pydeck_chart` or `plotly`) |
| **4H** | Benchmark on 30+ images, document single-image limitations |