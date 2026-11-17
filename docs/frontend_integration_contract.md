# RoomAI — Frontend Integration Contract & API Specification

**Version:** 1.0.0-RC  
**Date:** October 29, 2026  
**Status:** FROZEN RELEASE SPECIFICATION  
**Target Clients:** Streamlit Web UI, React/Vite Dashboard, REST/gRPC Gateway Clients

---

## 1. Overview & Architectural Principles

This document specifies the exact contract between the RoomAI Python backend inference engine and frontend presentation layers.

### Guiding Principles
1. **Zero Silent Fallback:** In production mode (`mock_mode: false`), services execute real neural networks on GPU (`cuda:0`). If a checkpoint or runtime dependency fails, the service raises an explicit error (`FileNotFoundError` or `ModelNotImplementedError`). It **never** silently substitutes mock or random data.
2. **Explicit Simulation Flagging:** Every response schema contains an `is_simulated: bool` property. Frontends can unambiguously distinguish between real model inference (`false`) and synthetic mock mode (`true`).
3. **Normalized Coordinates:** All spatial bounding boxes are normalized to floating-point coordinates $[0.0, 1.0]$. The backend provides explicit conversion helpers for integer pixel coordinates.
4. **Isolated Task Domains:** Computer vision models are specialized by domain:
   * **Indoor:** Model A3 (12 classes, 800px input resolution)
   * **Exterior:** Model B2 (10 classes, 640px input resolution) [ACTIVE PRODUCTION DEFAULT]
   * **Parking:** Model C2 (8 classes, 640px input resolution)
   * **Depth:** Depth Anything V2 Small (relative inverse depth)
   * **Segmentation:** YOLO11n-seg (COCO 80 categories)

---

## 2. Image Input Requirements

All image processing endpoints and services accept images adhering to the following constraints:

| Parameter | Supported Range / Values | Enforcement Action / Default |
| :--- | :--- | :--- |
| **File Formats** | JPEG (`.jpg`, `.jpeg`), PNG (`.png`), WebP (`.webp`) | Rejected with `ValueError` if unsupported extension. |
| **Maximum File Size** | $\le 15$ Megabytes (MB) | Configured in `config/app.yaml`. Rejected if exceeded. |
| **Maximum Resolution** | $\le 4096 \times 4096$ pixels | Automatically clamped/downsampled to avoid GPU OOM. |
| **Recommended Resolution** | $800 \times 800$ to $1920 \times 1080$ pixels | Optimal throughput and accuracy trade-off. |
| **Color Space / Order** | RGB (3-channel, 8-bit unsigned integer) | Backend normalizes all formats (RGBA/Grayscale $\to$ RGB). |

---

## 3. Spatial Coordinate Conventions

### 3.1 Bounding Box Schema (`BoundingBox`)

Bounding boxes are represented as normalized coordinates with top-left origin $(0, 0)$:
* `x_min`: Normalized horizontal coordinate of the left edge ($0.0 \le x_{\text{min}} \le 1.0$)
* `y_min`: Normalized vertical coordinate of the top edge ($0.0 \le y_{\text{min}} \le 1.0$)
* `x_max`: Normalized horizontal coordinate of the right edge ($x_{\text{min}} \le x_{\text{max}} \le 1.0$)
* `y_max`: Normalized vertical coordinate of the bottom edge ($y_{\text{min}} \le y_{\text{max}} \le 1.0$)

```json
{
  "x_min": 0.1245,
  "y_min": 0.3421,
  "x_max": 0.4589,
  "y_max": 0.8912
}
```

### 3.2 Pixel Coordinate Conversion Formula

To render bounding boxes onto a display canvas of width $W$ and height $H$:
$$\text{left} = \lfloor x_{\text{min}} \times W \rfloor, \quad \text{top} = \lfloor y_{\text{min}} \times H \rfloor$$
$$\text{right} = \lfloor x_{\text{max}} \times W \rfloor, \quad \text{bottom} = \lfloor y_{\text{max}} \times H \rfloor$$

---

## 4. Class Taxonomies by Domain

### 4.1 Indoor Furniture & Household Objects (Model A3 — 12 Classes)

```json
{
  "0": "cabinet#shelf",
  "1": "table",
  "2": "chair#stool",
  "3": "lamp",
  "4": "door",
  "5": "bed",
  "6": "sofa",
  "7": "plant",
  "8": "decoration",
  "9": "curtain",
  "10": "home-appliance",
  "11": "window"
}
```

### 4.2 Exterior & Building Architecture (Model B2 Production — 10 Classes)

```json
{
  "0": "building",
  "1": "facade",
  "2": "window",
  "3": "door",
  "4": "roof_cornice",
  "5": "balcony",
  "6": "porch",
  "7": "pillar_column",
  "8": "shutter_blind",
  "9": "molding_deco"
}
```

### 4.3 Parking & Outdoor Grounds (Model C2 Production — 8 Classes)

```json
{
  "0": "car_vehicle",
  "1": "motorcycle",
  "2": "bicycle",
  "3": "outdoor_seating",
  "4": "outdoor_table",
  "5": "plant_bush",
  "6": "tree",
  "7": "outdoor_fixture"
}
```

---

## 5. Service Schemas & Data Contracts

### 5.1 Object Detection Service (`DetectionResponse`)

Executed via `services.detection_service.DetectionService.predict(image, confidence_threshold)`.

```json
{
  "detections": [
    {
      "class_id": 6,
      "label": "sofa",
      "confidence": 0.8842,
      "box": {
        "x_min": 0.152,
        "y_min": 0.451,
        "x_max": 0.724,
        "y_max": 0.892
      },
      "model_provenance": "indoor_a3_800px"
    }
  ],
  "total_objects": 1,
  "is_simulated": false,
  "model_name": "Model A3 (Higher-Resolution YOLO11s 800px)",
  "checkpoint": "indoor_detector_a3_yolo11s_800px_best.pt",
  "inference_time_ms": 349.81,
  "device": "cuda:0"
}
```

### 5.2 Scene3D Depth Estimation (`Scene3DResponse`)

Executed via `services.scene3d_service.Scene3DService.estimate_depth(image)`.

```json
{
  "status": "success",
  "depth_available": true,
  "point_cloud_available": false,
  "is_simulated": false,
  "message": "Depth estimation completed",
  "model_name": "Depth Anything V2 Small",
  "inference_time_ms": 176.32,
  "device": "cuda:0",
  "peak_memory_mb": 68.01,
  "depth_min": 0.0,
  "depth_max": 1.0,
  "relative_depth_only": true,
  "disclaimer": "Relative depth map output: higher values indicate proximity to camera. Not metric distance in meters; uncalibrated for physical measurement."
}
```

> **Frontend Rendering Rule:** The depth map is returned as a 2D floating point array $[0.0, 1.0]$ or an 8-bit colormapped heatmap (`uint8` RGB via `cv2.applyColorMap(..., cv2.COLORMAP_INFERNO)`). High values (1.0, yellow/white) indicate foreground surfaces closest to the camera; low values (0.0, dark purple/black) indicate background surfaces.

### 5.3 Semantic Instance Segmentation (`SegmentationResponse`)

Executed via `services.segmentation_service.SegmentationService.segment(image)`.

```json
{
  "masks": [
    {
      "category": "chair",
      "coverage_percentage": 14.25,
      "mask_shape": [768, 1024],
      "color_hex": "#3B82F6"
    }
  ],
  "categories_found": ["chair"],
  "is_simulated": false,
  "model_name": "YOLO11n-seg Pretrained"
}
```

### 5.4 Style Classification (`ClassificationResponse`)

Executed via `services.classification_service.ClassificationService.classify(image)`.

```json
{
  "predicted_class": "Scandinavian",
  "confidence": 0.892,
  "top_classes": [
    {"class_name": "Scandinavian", "score": 0.892},
    {"class_name": "Minimalist", "score": 0.071},
    {"class_name": "Modern", "score": 0.024}
  ],
  "is_simulated": false,
  "model_name": "StyleClassifier-v1"
}
```

---

## 6. Error Handling & Status Codes

When deploying as an HTTP/REST microservice or via the Streamlit backend, the following standards are enforced:

| HTTP Code | Condition | Error Code / Exception | Client Action |
| :---: | :--- | :--- | :--- |
| `200 OK` | Inference succeeded | None (`success: true`) | Render bounding boxes / depth maps. |
| `400 Bad Request` | Corrupted image or invalid format | `INVALID_IMAGE_PAYLOAD` | Prompt user to upload a valid JPEG/PNG/WebP. |
| `413 Payload Too Large` | Image exceeds 15 MB or $4096\times 4096$ | `IMAGE_TOO_LARGE` | Downsample image on client prior to upload. |
| `422 Unprocessable` | Bounding box coordinates out of bounds | `VALIDATION_ERROR` | Reject corrupted coordinates. |
| `500 Server Error` | Model weights missing | `FileNotFoundError` | Verify `checkpoints/` directory on host. |
| `503 Service Unavailable` | CUDA Out of Memory | `CUDA_OUT_OF_MEMORY` | Clear VRAM cache; retry on CPU fallback. |

### Health Check Endpoint
* **Path:** `/_stcore/health` (Streamlit Core Health)
* **Expected Response:** HTTP 200 OK with body `"ok"`
* **Expected Latency:** $< 25$ ms once server process is initialized.

---

## 7. Recommended Client Timeouts & UI Loading States

| Service Call | Expected Latency (GPU) | Recommended Client Timeout | Suggested UI Feedback |
| :--- | :---: | :---: | :--- |
| **Object Detection (B2 / A3)** | 250 – 400 ms | 3,000 ms | "Detecting architectural & interior objects..." |
| **Depth Estimation (Scene3D)** | 150 – 250 ms | 2,500 ms | "Estimating relative depth geometry..." |
| **Full Scene Analysis (Pipeline)** | 450 – 750 ms | 5,000 ms | "Running complete scene intelligence..." |
| **Generative Restyle (Stub/API)** | 2,000 – 8,000 ms | 15,000 ms | "Synthesizing design concept..." |

---

## 8. Frontend Integration Code Example (Python Client)

```python
from PIL import Image
from services.detection_service import DetectionService
from services.scene3d_service import Scene3DService

# Initialize services
detection_service = DetectionService()
scene3d_service = Scene3DService()

# Load image
image = Image.open("tests/fixtures/sample_room.jpg").convert("RGB")

# Run real detection (B2 exterior or A3 indoor depending on config)
det_response = detection_service.predict(image, confidence_threshold=0.30)
assert not det_response.is_simulated, "Production must not return simulated detections"

# Render bounding boxes onto client canvas
for item in det_response.detections:
    left, top, right, bottom = item.box.to_pixel_coords(image.width, image.height)
    print(f"[{item.label}] ({item.confidence:.2%}) -> ({left}, {top}, {right}, {bottom})")

# Run relative depth
depth_response = scene3d_service.estimate_depth(image)
print(f"Depth complete in {depth_response.inference_time_ms:.1f}ms on {depth_response.device}")
```

