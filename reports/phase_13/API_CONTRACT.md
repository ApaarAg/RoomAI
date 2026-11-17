# RoomAI API Contract — REST API Specification

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Currently: **None** (add via middleware for production)

## Content Types
- Request: `application/json`, `multipart/form-data` (uploads)
- Response: `application/json`, `model/gltf-binary`, `application/octet-stream`

## Endpoints

---

### 1. Health & Diagnostics

#### GET `/health`
**Liveness probe**

**Response 200**:
```json
{
  "status": "healthy",
  "version": "1.4.0",
  "service": "RoomAI Backend API",
  "timestamp": 1731234567.89
}
```

#### GET `/ready`
**Readiness probe — checks GPU, models, checkpoints**

**Response 200** (ready):
```json
{
  "status": "ready",
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 4060",
  "free_vram_gb": 3.2,
  "total_vram_gb": 8.0,
  "checkpoints_ready": true,
  "models_ready": ["worldmirror1_baseline", "worldmirror1_lora_r8"]
}
```

**Response 200** (degraded):
```json
{
  "status": "degraded",
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 4060",
  "free_vram_gb": 0.5,
  "total_vram_gb": 8.0,
  "checkpoints_ready": false,
  "models_ready": []
}
```

#### GET `/metrics`
**Prometheus metrics endpoint** (text/plain)

---

### 2. Uploads

#### POST `/uploads`
**Upload image**

**Request**: `multipart/form-data`
- `file`: Image file (JPEG, PNG, WebP, max 20MB, max 8192px, max 100MP)

**Response 201**:
```json
{
  "status": "success",
  "upload_id": "upl_a1b2c3d4e5f6",
  "filename": "upl_a1b2c3d4e5f6.jpg",
  "original_filename": "room_photo.jpg",
  "width": 1920,
  "height": 1080,
  "format": "JPEG",
  "size_bytes": 1234567,
  "sha256": "a1b2c3d4e5f6..."
}
```

**Errors**:
- 400: Invalid file, size, format, dimensions
- 500: Storage failure

#### GET `/uploads/{upload_id}`
**Get upload metadata**

**Response 200**: Same as POST response

**Errors**:
- 404: Upload not found

---

### 3. Models

#### GET `/models`
**List all registered models**

**Response 200**:
```json
{
  "status": "success",
  "default_model_id": "roomai_composite_v1",
  "models": [
    {
      "model_id": "worldmirror1_baseline",
      "name": "WorldMirror-1.0 Baseline",
      "version": "1.0.0-pretrained",
      "status": "REFERENCE",
      "supports_single_view": true,
      "supports_multi_view": true,
      "min_views": 1,
      "max_views": 6,
      "supports_metric_depth": true,
      "supports_surface_normals": true,
      "supports_3dgs": false,
      "supports_mesh_glb": true,
      "supports_point_cloud_ply": true,
      "device": "cuda",
      "memory_required_mb": 4800,
      "description": "Baseline and rollback foundation model"
    },
    {
      "model_id": "worldmirror1_lora_r8",
      "name": "WorldMirror-1.0 Stage 1",
      "version": "1.0.0-stage1-r8",
      "status": "VALIDATION_CANDIDATE",
      "supports_single_view": true,
      "supports_multi_view": true,
      "min_views": 1,
      "max_views": 6,
      "supports_metric_depth": true,
      "supports_surface_normals": true,
      "supports_3dgs": false,
      "supports_mesh_glb": true,
      "supports_point_cloud_ply": true,
      "device": "cuda",
      "memory_required_mb": 4800,
      "description": "Best reported Stage 1 candidate"
    }
  ]
}
```

#### GET `/models/{model_id}`
**Get model details and status**

**Response 200**:
```json
{
  "model_id": "worldmirror1_lora_r8",
  "name": "WorldMirror-1.0 Stage 1",
  "version": "1.0.0-stage1-r8",
  "status": "VALIDATION_CANDIDATE",
  "available": true,
  "checkpoint_verified": true,
  "checkpoint_path": "/path/to/best_stage1.ckpt",
  "capabilities": {...},
  "parameter_count": {...}
}
```

#### GET `/models/{model_id}/verify`
**Verify checkpoint integrity**

**Response 200**:
```json
{
  "model_id": "worldmirror1_lora_r8",
  "verified": true,
  "reason": null
}
```

---

### 4. Reconstruction Jobs

#### POST `/jobs/reconstruct`
**Submit 3D reconstruction job**

**Request**:
```json
{
  "upload_ids": ["upl_a1b2c3d4e5f6", "upl_f6e5d4c3b2a1"],
  "model_id": "worldmirror1_lora_r8",
  "refine_depth": true,
  "downsample_points": 40000
}
```

**Constraints**:
- `upload_ids`: 1-10 IDs (model-dependent min/max)
- `model_id`: Must be registered and available
- `downsample_points`: Max points in exported PLY/GLB

**Response 202**:
```json
{
  "status": "accepted",
  "job_id": "job_1731234567_a1b2c3d4",
  "job_status": "QUEUED",
  "model_id": "worldmirror1_lora_r8",
  "view_count": 2,
  "poll_url": "/api/v1/jobs/job_1731234567_a1b2c3d4"
}
```

**Errors**:
- 400: Invalid upload IDs, view count exceeds model limits
- 404: Upload not found
- 503: Model unavailable

#### GET `/jobs/{job_id}`
**Poll job status**

**Response 200**:
```json
{
  "job_id": "job_1731234567_a1b2c3d4",
  "status": "INFERENCE",
  "stage": "Running model inference",
  "progress": 0.45,
  "model_id": "worldmirror1_lora_r8",
  "view_count": 2,
  "timing": {
    "created_at": 1731234567.0,
    "started_at": 1731234568.5,
    "completed_at": null,
    "elapsed_ms": 12500.0
  },
  "error": null,
  "warnings": [],
  "created_at": 1731234567.0,
  "updated_at": 1731234580.0
}
```

**Job States**: `QUEUED`, `VALIDATING`, `INFERENCE`, `GEOMETRY_POSTPROCESSING`, `EXPORTING`, `COMPLETED`, `FAILED`, `CANCELLED`

#### GET `/jobs/{job_id}/results`
**Get full results (after COMPLETED)**

**Response 200**:
```json
{
  "job_id": "job_1731234567_a1b2c3d4",
  "status": "COMPLETED",
  "is_completed": true,
  "model_id": "worldmirror1_lora_r8",
  "timing": {
    "created_at": 1731234567.0,
    "started_at": 1731234568.5,
    "completed_at": 1731234595.2,
    "elapsed_ms": 26700.0
  },
  "results_summary": {
    "object_count": 0,
    "surface_count": 0,
    "valid_points": 45231,
    "depth_range_m": [0.52, 12.3],
    "room_layout": {...},
    "objects": []
  },
  "artifacts": {
    "glb": {"path": "/artifacts/jobs/job_.../scene_reconstruction.glb", "size_bytes": 2048576},
    "ply": {"path": "/artifacts/jobs/job_.../scene_reconstruction.ply", "size_bytes": 5242880}
  },
  "scene_graph": {...}
}
```

#### GET `/jobs/{job_id}/exports/{fmt}`
**Download export file**

**Parameters**: `fmt` = `glb` | `ply`

**Response**: Binary file download
- `glb`: `model/gltf-binary`
- `ply`: `application/octet-stream`

**Errors**:
- 400: Invalid format
- 404: Job/export not found
- 409: Job not COMPLETED

#### POST `/jobs/{job_id}/cancel`
**Cancel job**

**Response 200**:
```json
{
  "job_id": "job_1731234567_a1b2c3d4",
  "status": "CANCELLED",
  "cancelled": true,
  "message": "Cancellation requested for job 'job_1731234567_a1b2c3d4'."
}
```

**Errors**:
- 404: Job not found
- 409: Job already terminal (COMPLETED/FAILED/CANCELLED)

#### GET `/jobs`
**List jobs**

**Query**: `limit` (1-100, default 20), `offset`, `status`

**Response 200**: Array of job summaries

---

### 5. Exports

#### GET `/exports/{job_id}`
**List available exports for job**

**Response 200**:
```json
{
  "job_id": "job_1731234567_a1b2c3d4",
  "status": "COMPLETED",
  "exports": {
    "glb": {
      "format": "glb",
      "path": "/artifacts/jobs/.../scene_reconstruction.glb",
      "size_bytes": 2048576,
      "download_url": "/api/v1/exports/job_1731234567_a1b2c3d4/glb"
    },
    "ply": {...}
  }
}
```

#### GET `/exports/{job_id}/{fmt}`
**Download export** (same as `/jobs/{job_id}/exports/{fmt}`)

#### GET `/exports/{job_id}/scene-graph`
**Get scene graph JSON**

---

### 6. Experimental (Hunyuan)

#### POST `/experimental/hunyuan/reconstruct`
**Strict Hunyuan reconstruction (no fallback)**

Same request/response as `/jobs/reconstruct` but:
- `model_id` must be Hunyuan model
- HTTP 503 if unavailable (no fallback)
- `strict_no_fallback: true` in options

#### POST `/experimental/hunyuan/verify/{model_id}`
**Verify Hunyuan model checkpoint**

---

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `NOT_FOUND` | 404 | Resource not found |
| `SERVICE_UNAVAILABLE` | 503 | Model/service unavailable |
| `GPU_ERROR` | 500 | CUDA OOM or driver error |
| `MODEL_ERROR` | 500 | Model loading/inference failed |
| `JOB_ERROR` | 500 | Job orchestration error |
| `JOB_CANCELLED` | 499 | Job cancelled by user |
| `EXPORT_ERROR` | 500 | Export validation failed |
| `PREPROCESSING_ERROR` | 400 | Image preprocessing failed |
| `INTERNAL_SERVER_ERROR` | 500 | Unhandled exception |

## Rate Limits
Not implemented (add via middleware for production)

## Versioning
- URL version: `/api/v1/`
- Model versions in registry (semantic)
- Breaking changes → `/api/v2/`

## CORS
Configured via `CORS_ORIGINS` env var (default: `*`)