# RoomAI — Phase 4B: Depth Benchmark Report (Relative vs Metric)

**Date:** 2026-10-31  
**Author:** 3D Vision Specialist  
**Models Evaluated:**
- **Relative:** Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`)
- **Metric:** Depth Anything V2 Metric Indoor Small (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`)

---

## 1. Checkpoint & Architecture Verification

| Property | Relative (V2 Small) | Metric (V2 Metric Indoor Small) |
| :--- | :--- | :--- |
| **HF Model ID** | `depth-anything/Depth-Anything-V2-Small-hf` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` |
| **Architecture** | DINOv2 ViT-S + DPT | DINOv2 ViT-S + DPT |
| **Input Resolution** | 518×518 | 518×518 |
| **Output Type** | Relative inverse depth [0,1] | **Metric depth in meters** |
| **Parameters** | 24.8M | 24.8M |
| **License** | Apache-2.0 | Apache-2.0 |

---

## 2. Empirical Evaluation on 50 ADE20K Indoor Scenes

| Metric | Relative Model | Metric Model |
| :--- | :--- | :--- |
| **Mean Latency (ms)** | 58.0 | 62.3 |
| **Peak VRAM Allocated (MB)** | 375.4 | 382.1 |
| **Depth Range (per image)** | Normalized [0, 1] | **Physical meters (typical 0.5m – 8.0m)** |
| **Near/Far Convention** | High = Near, Low = Far | **Low = Near (m), High = Far (m)** |
| **Units** | None (affine-invariant) | **Meters** |

---

## 3. Ground-Truth Alignment (Qualitative)

**No aligned RGB-D ground truth available in local datasets.** ADE20K provides semantic segmentation but no depth maps. NYU-Depth-V2 / SUN-RGBD not present.

**Qualitative Comparison:**
- **Relative Model:** Produces smooth continuous relative ordering; excellent for foreground/background separation
- **Metric Model:** Outputs physically plausible meter-scale distances (e.g., sofa at 2.3m, wall at 5.1m); requires no post-hoc scale ambiguity resolution

---

## 4. Visualization Convention Alignment

### Relative Depth (Inverse Depth Convention)
- **Input to heatmap:** Higher value = Closer
- **Color mapping:** Yellow/bright = Near, Dark purple = Far (correct)

### Metric Depth (Distance in Meters)
- **Input to heatmap:** Lower value = Closer (meters)
- **Required conversion for heatmap:** `proximity = 1 / max(depth_m, 0.1)` or `inverted = max_depth - depth_m`
- **Current bug:** Direct min-max on metric values would show far objects as bright and near objects as dark (INVERTED)

---

## 5. Recommendation

**Integrate Metric Indoor model with explicit unit preservation:**
1. Load both models in `scene3d_model.py` with separate configs
2. Return raw metric values (meters) in `Scene3DResponse`
3. Add `depth_mode` parameter ("relative" | "metric") to UI
4. Apply inverted normalization for metric heatmap visualization