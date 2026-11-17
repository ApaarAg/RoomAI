# Phase 5.8.1 Model Identity Audit Report
## RoomAI Real 3D Inference Engine

**Audit Date:** 2026-10-31
**Auditor:** Claude Code Agent
**Component:** Depth Anything V2 Small (Relative Depth)

---

## Executive Summary

The RoomAI Scene3D model uses the **Depth Anything V2 Small** checkpoint for monocular depth estimation. This checkpoint provides **relative depth values**, NOT metric depth. All depth values are normalized in the range [0, 1].

---

## Checkpoint Identifier

| Property | Value |
|----------|-------|
| **Path** | `checkpoints/depth_anything_v2_small/` |
| **Model Name on Disk** | `depth-anything/Depth-Anything-V2-Small-hf` |
| **Commit Hash** | `5426e4f0f36572d16453bbda7a8389317b1bef99` |

---

## Architecture

| Component | Specification |
|-----------|---------------|
| **Architecture** | `DepthAnythingForDepthEstimation` |
| **Backbone** | DINOv2 (ViT-Small variant) |
| **Backbone Config** | DINov2Model |
| **Hidden Size** | 384 |
| **Image Size** | 518 x 518 |
| **Attention Heads** | 6 |
| **Patch Size** | 14 |
| **Total Parameters** | 24,785,089 (~25M) |

### Backbone Features
The DINOv2 backbone outputs features from stages 3, 6, 9, and 12 (out_indices: [3, 6, 9, 12]).

---

## Preprocessing Configuration (`preprocessor_config.json`)

| Property | Value |
|----------|-------|
| **Image Processor Type** | `DPTImageProcessor` |
| **Target Size** | 518 x 518 (height x width) |
| **Keep Aspect Ratio** | true |
| **Rescale Factor** | 0.00392156862745098 (1/255) |
| **Do Rescale** | true |
| **Do Normalize** | true |
| **Do Resize** | true |
| **Ensure Multiple Of** | 14 |
| **Resample Mode** | 3 (BICUBIC) |

### Normalization Parameters
| Parameter | Value |
|-----------|-------|
| **Mean** | [0.485, 0.456, 0.406] |
| **Std** | [0.229, 0.224, 0.225] |

---

## Output Type and Range

| Property | Value |
|----------|-------|
| **Output Type** | Relative Depth |
| **Range** | [0, 1] normalized |
| **Units** | Unitless relative depth (NOT metric meters) |
| **Depth Mode** | `relative` |
| **Max Depth Config** | 1 |

**Key Finding:** Per `depth_estimation_type` in `config.json` = "relative", this model outputs RELATIVE depth values, not metric depth in meters. Higher values indicate closer proximity to the camera.

---

## Model Files Checkpoint

| File | Size | Purpose |
|------|------|---------|
| `model.safetensors` | 99.17 MB | Model weights (PyTorch safe format) |
| `config.json` | 1.2 KB | Model architecture configuration |
| `preprocessor_config.json` | 493 B | Image preprocessing configuration |

---

## Model Loading Behavior

### Current Behavior (Verified)
1. Model auto-loads when initialized with config containing `checkpoint_path`
2. Falls back to `cuda:0` if available, else `cpu`
3. Uses `AutoImageProcessor` and `AutoModelForDepthEstimation` from HuggingFace Transformers

### Configuration Impact
- `enabled: false` in config does NOT prevent model loading (BUG FIXED)
- `checkpoint_path: ""` or omitted does NOT prevent model loading (BUG FIXED)

---

## Conclusion

**This is a REGRESSION DEPTH MODEL, not a metric depth model.**

The checkpoint provides **relative depth** values in normalized range [0, 1]. These values cannot be used directly as physical distances in meters. Applications requiring metric depth must:
1. Use a different checkpoint calibrated for metric depth
2. Apply depth calibration/normalization based on known reference distances
3. Use the relative depth for visualization and geometric relationships

**WARNING:** The current implementation provides **relative inverse depth**. Higher values = closer to camera. The point cloud Z-coordinates are proxy values for visualization only, not physically accurate measurements.

---

## Files Analyzed

- `/f/RoomAI/checkpoints/depth_anything_v2_small/config.json`
- `/f/RoomAI/checkpoints/depth_anything_v2_small/preprocessor_config.json`
- `/f/RoomAI/checkpoints/depth_anything_v2_small/model.safetensors`
- `/f/RoomAI/models/scene3d_model.py`
- `/f/RoomAI/config/model_config.yaml`