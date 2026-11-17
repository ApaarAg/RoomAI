# ROOMAI — FRONTEND & CLIENT API INTEGRATION CONTRACT

**Document Version:** 1.4.0  
**Status:** Approved for Production Client Integration  
**Base URL:** `http://localhost:8000` (or configured API gateway)  
**Interactive OpenAPI Documentation:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)  
**OpenAPI Specification JSON:** `http://localhost:8000/api/v1/openapi.json`  

---

## 1. Architectural Overview & Workflow Lifecycle

All client applications (Streamlit, Web, Mobile, CLI) communicate with RoomAI via this versioned REST API. Neural network inference and heavy 3D geometric computations run asynchronously in background workers to prevent blocking client requests.

```
Client App                                                                  RoomAI API Server
    |                                                                               |
    |---- 1. POST /api/v1/uploads (multipart/form-data) --------------------------->|
    |<--- 201 Created {upload_id, width, height, sha256} ---------------------------|
    |                                                                               |
    |---- 2. POST /api/v1/jobs/reconstruct {upload_ids, model_id} ----------------->|
    |<--- 202 Accepted {job_id, status: 'QUEUED', poll_url} ------------------------|
    |                                                                               |
    |---- 3. Loop: GET /api/v1/jobs/{job_id} -------------------------------------->|
    |<--- 200 OK {status: 'RECONSTRUCTING', progress: 0.50, stage: '...'} ----------|
    |                                                                               |
    |     (Upon status == 'COMPLETED')                                              |
    |---- 4. GET /api/v1/jobs/{job_id}/results ------------------------------------>|
    |<--- 200 OK {scene_graph, objects, room_layout, artifacts} --------------------|
    |                                                                               |
    |---- 5. GET /api/v1/jobs/{job_id}/exports/glb -------------------------------->|
    |<--- 200 OK (binary glTF 2.0 file stream) -------------------------------------|
```

---

## 2. Global Conventions & Standards

### 2.1 Coordinate System & Geometric Standards
All 3D geometric data returned in `scene_graph` and exported files conforms to the standard **OpenCV Pinhole Camera Coordinate Frame**:
- **Units:** Physical meters ($m$).
- **Origin $(0, 0, 0)$:** Optical center of reference camera (View 0).
- **$+X$ Axis:** Points horizontally to the right.
- **$+Y$ Axis:** Points vertically downward toward the floor.
- **$+Z$ Axis:** Points forward along the optical camera axis into the scene.
- **Gravity Direction:** $\mathbf{g} = [0, 1, 0]$ (downward along $+Y$).
- **Floor Plane Normal:** $\mathbf{n}_{\text{floor}} \approx [0, -1, 0]$ (pointing upward).

### 2.2 Standard Error Response Envelope
Every error response returns HTTP status $\ge 400$ with a standardized JSON body:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Model 'roomai_composite_v1' supports a maximum of 10 views; received 12.",
    "remediation": "Reduce the number of uploaded images to 10 or fewer."
  },
  "timestamp": 1794077200.123
}
```

---

## 3. Endpoints Specification

### 3.1 Diagnostics & Readiness

#### `GET /health`
Liveness probe verifying that the API service is running.
- **Response `200 OK`:**
  ```json
  {
    "status": "healthy",
    "version": "1.4.0",
    "service": "RoomAI Backend API",
    "timestamp": 1794077200.0
  }
  ```

#### `GET /ready`
Readiness probe verifying GPU availability, VRAM headroom, and model checkpoints.
- **Response `200 OK`:**
  ```json
  {
    "status": "ready",
    "cuda_available": true,
    "device_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "free_vram_gb": 4.85,
    "total_vram_gb": 8.0,
    "checkpoints_ready": true,
    "models_ready": ["roomai_composite_v1"]
  }
  ```

---

### 3.2 Image Ingestion & Uploads

#### `POST /api/v1/uploads`
Uploads and validates an indoor photograph.
- **Content-Type:** `multipart/form-data`
- **Form Field:** `file` (Binary image file)
- **Validation Rules:**
  - File extension in `{'.jpg', '.jpeg', '.png', '.webp'}`.
  - File size $\le 20\text{ MB}$.
  - Resolution $\le 8192\text{px}$ and total pixels $\le 100,000,000$.
- **Response `201 Created`:**
  ```json
  {
    "status": "success",
    "upload_id": "upl_a51d932cc9a0d3c3",
    "filename": "upl_a51d932cc9a0d3c3.jpg",
    "original_filename": "living_room.jpg",
    "width": 800,
    "height": 600,
    "format": "JPEG",
    "size_bytes": 13354,
    "sha256": "a51d932cc9a0d3c3..."
  }
  ```

#### `GET /api/v1/uploads/{upload_id}`
Retrieves metadata for an existing upload.
- **Response `200 OK`:** Same schema as `POST /api/v1/uploads`.
- **Response `404 Not Found`:** If upload ID does not exist.

---

### 3.3 Reconstruction Job Management

#### `POST /api/v1/jobs/reconstruct`
Enqueues a 3D reconstruction job.
- **Request Body (`application/json`):**
  ```json
  {
    "upload_ids": ["upl_a51d932cc9a0d3c3"],
    "model_id": "roomai_composite_v1",
    "refine_depth": true,
    "downsample_points": 40000
  }
  ```
- **Response `202 Accepted`:**
  ```json
  {
    "status": "accepted",
    "job_id": "job_1794076968_2697c85c",
    "job_status": "QUEUED",
    "model_id": "roomai_composite_v1",
    "view_count": 1,
    "poll_url": "/api/v1/jobs/job_1794076968_2697c85c"
  }
  ```

#### `GET /api/v1/jobs/{job_id}`
Polls job state, progress fraction, and execution stage.
- **State Machine Values:**
  - `QUEUED`: Waiting in background worker queue.
  - `VALIDATING`: Verifying input dimensions and hardware readiness.
  - `PROCESSING`: Acquiring GPU concurrency lock and loading weights.
  - `RECONSTRUCTING`: Running neural network forward passes.
  - `FUSING`: Cross-view alignment, floor plane RANSAC, bounding box grounding.
  - `EXPORTING`: Serializing and validating `.glb` and `.ply` binaries.
  - `COMPLETED`: Finished successfully; results and downloads available.
  - `FAILED`: Terminated with an error (see `error` field).
  - `CANCELLED`: Cancelled by user.
- **Response `200 OK`:**
  ```json
  {
    "job_id": "job_1794076968_2697c85c",
    "status": "COMPLETED",
    "stage": "3D reconstruction and export completed successfully",
    "progress": 1.0,
    "model_id": "roomai_composite_v1",
    "view_count": 1,
    "timing": {
      "created_at": 1794076968.0,
      "started_at": 1794076968.2,
      "completed_at": 1794076979.1,
      "elapsed_ms": 10903.06
    },
    "error": null,
    "warnings": [],
    "created_at": 1794076968.0,
    "updated_at": 1794076979.1
  }
  ```

#### `GET /api/v1/jobs/{job_id}/results`
Retrieves structured scene graph, detected 3D furniture objects, and floor layout.
- **Response `200 OK`:**
  ```json
  {
    "job_id": "job_1794076968_2697c85c",
    "status": "COMPLETED",
    "is_completed": true,
    "model_id": "roomai_composite_v1",
    "results_summary": {
      "object_count": 2,
      "surface_count": 0,
      "valid_points": 480000,
      "depth_range_m": [0.42, 6.85],
      "room_layout": {
        "floor_plane_detected": true,
        "floor_elevation_m": 1.45,
        "room_width_m": 4.5,
        "room_length_m": 5.2,
        "room_height_m": 2.7
      },
      "objects": [
        {
          "object_id": "chair_001",
          "semantic_class": "chair",
          "confidence": 0.88,
          "extent_m": [0.65, 0.92, 0.70],
          "centroid_m": [-0.35, 0.99, 2.80],
          "is_reliable": true
        }
      ]
    },
    "artifacts": {
      "glb": {
        "path": "F:\\RoomAI\\artifacts\\jobs\\...\\scene_reconstruction.glb",
        "size_bytes": 482144,
        "is_valid": true,
        "mesh_count": 3
      },
      "ply": {
        "path": "F:\\RoomAI\\artifacts\\jobs\\...\\scene_reconstruction.ply",
        "size_bytes": 6761591,
        "is_valid": true,
        "vertex_count": 480000
      }
    }
  }
  ```

#### `GET /api/v1/jobs/{job_id}/exports/{format}`
Streams verified 3D binary assets for client download.
- **Path Parameters:**
  - `format`: `glb` (Binary glTF 2.0) or `ply` (ASCII Polygon File Format).
- **Response `200 OK`:**
  - `Content-Type: model/gltf-binary` (for GLB) or `application/octet-stream` (for PLY).
  - `Content-Disposition: attachment; filename="roomai_job_..._glb.glb"`
- **Response `409 Conflict`:** If requested before job completion.

#### `POST /api/v1/jobs/{job_id}/cancel`
Requests graceful cancellation of a job.
- **Response `200 OK`:**
  ```json
  {
    "job_id": "job_1794076968_2697c85c",
    "status": "CANCELLED",
    "cancelled": true,
    "message": "Cancellation requested for job 'job_1794076968_2697c85c'."
  }
  ```

---

### 3.4 Model Discovery

#### `GET /api/v1/models`
Discovers installed 3D reconstruction models and their capabilities.
- **Response `200 OK`:**
  ```json
  {
    "status": "success",
    "default_model_id": "roomai_composite_v1",
    "models": [
      {
        "model_id": "roomai_composite_v1",
        "name": "RoomAI Composite Geometry Pipeline",
        "version": "1.4.0",
        "status": "PRODUCTION_READY",
        "supports_single_view": true,
        "supports_multi_view": true,
        "min_views": 1,
        "max_views": 10,
        "supports_metric_depth": true,
        "supports_surface_normals": false,
        "supports_3dgs": false,
        "supports_mesh_glb": true,
        "supports_point_cloud_ply": true,
        "device": "cuda",
        "memory_required_mb": 1500,
        "description": "Production-verified geometric reconstruction fusing YOLO11s A3..."
      },
      {
        "model_id": "hunyuan_world_mirror_1b",
        "name": "HunyuanWorld-Mirror (1B Foundation Model)",
        "version": "1.0.0-staged",
        "status": "STAGED",
        "supports_single_view": true,
        "supports_multi_view": true,
        "min_views": 1,
        "max_views": 6,
        "supports_metric_depth": true,
        "supports_surface_normals": true,
        "supports_3dgs": true,
        "supports_mesh_glb": true,
        "supports_point_cloud_ply": true,
        "device": "cuda",
        "memory_required_mb": 4800,
        "description": "Next-generation 1.0B feed-forward 3D world reconstruction transformer..."
      }
    ]
  }
  ```

---

### 3.5 Experimental & Offline Endpoints

#### `POST /api/v1/experimental/vqa`
#### `POST /api/v1/experimental/redesign`
- **Response `501 Not Implemented`:**
  ```json
  {
    "status": "error",
    "error": {
      "code": "HTTP_501",
      "message": "Multimodal vision-language model weights (e.g. Qwen2-VL) are not installed locally.",
      "remediation": "This feature is disabled in production until weights and GPU quota are allocated."
    },
    "timestamp": 1794077200.0
  }
  ```

