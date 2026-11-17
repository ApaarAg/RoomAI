# RoomAI Professional Backend Architecture

## Overview

This document describes the production-ready backend architecture for RoomAI, a 3D spatial understanding and scene reconstruction platform. The backend is built as a modular, versioned, and observable FastAPI application with clean separation of concerns.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│  /api/v1/uploads  /api/v1/jobs  /api/v1/models  /api/v1/exports │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Application Services                          │
│  ReconstructionService  │  ModelRegistryService  │  JobService  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Domain Services                             │
│  InferenceEngine  │  PreprocessingPipeline  │  GeometryService │
│  ExportService    │  ArtifactStore        │  MetricsCollector  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Infrastructure                              │
│  GPUWorker (Job Queue)  │  JobStore (SQLite)  │  Model Registry │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Registry Service (`app/services/model_registry/registry.py`)

Immutable model registry loaded from `models/registry/model_registry.yaml`. Provides:
- Model discovery and capability queries
- Checkpoint integrity verification (SHA-256)
- Hardware availability checking
- Alias resolution (`stage1_reference`, `stage1_candidate`, `stage2_experimental`)

**Registered Models:**
| Model ID | Name | Status | Checkpoint |
|----------|------|--------|------------|
| `worldmirror1_baseline` | WorldMirror-1.0 Baseline | REFERENCE | `production_model.ckpt` |
| `worldmirror1_lora_r8` | WorldMirror-1.0 Stage 1 | VALIDATION_CANDIDATE | `best_stage1.ckpt` |
| `worldmirror1_stage2` | WorldMirror-1.0 Stage 2 | EXPERIMENTAL | `stage2_real_verified.ckpt` |

### 2. Inference Engine (`app/services/inference/engine.py`)

Unified inference interface with:
- Single-model-in-memory policy (RTX 4060 8GB constraint)
- Automatic GPU memory management (eviction, OOM handling)
- Support for HunyuanWorld-Mirror and composite models
- Structured inference logging with VRAM tracking

### 3. Preprocessing Pipeline (`app/services/preprocessing/pipeline.py`)

Reproducible preprocessing with versioned configuration:
- File validation (size, format, dimensions, decompression bomb protection)
- EXIF orientation correction
- Square crop + patch-14 aligned resize (518×518)
- ImageNet normalization
- SHA-256 content hashing
- Full configuration serialization for reproducibility

### 4. Geometry Service (`app/services/geometry/postprocessing.py`)

Post-processing of raw model predictions:
- Coordinate transformation (Model ↔ RoomAI: `diag(1, -1, -1)`)
- Depth range filtering (0.1m – 25m)
- Confidence thresholding
- Voxel grid downsampling
- Duplicate point removal
- Normal validation and orientation correction
- Multi-view consolidation

### 5. Reconstruction Pipeline (`app/services/reconstruction/pipeline.py`)

End-to-end orchestration:
1. **Preprocessing** → 2. **Inference** → 3. **Geometry** → 4. **Export**
- Progress callbacks for real-time job status
- Per-stage timing and metrics
- Cooperative cancellation support
- Structured error handling

### 6. Export Service (`app/services/export/exporter.py`)

Validated 3D asset export:
- **PLY**: ASCII point cloud with RGB + normals
- **GLB**: Binary glTF 2.0 with point cloud + bounding box wireframes
- Built-in format validation (header magic, structure)

### 7. Artifact Store (`app/storage/artifacts.py`)

Filesystem-based storage with S3-compatible interface:
- Content-addressable uploads (SHA-256 deduplication)
- Job-scoped artifact directories
- Metadata index (JSON) with validity tracking
- Cleanup policies

### 8. Job Store (`app/services/job_store.py`)

SQLite + SQLAlchemy persistent job storage:
- Full job lifecycle (QUEUED → VALIDATING → PROCESSING → COMPLETED/FAILED/CANCELLED)
- WAL mode for concurrency
- Automatic recovery of interrupted jobs on restart
- Structured timing, results, and artifact tracking

### 9. GPU Worker (`app/workers/gpu_worker.py`)

Single-threaded background job executor:
- Sequential processing (concurrency=1 for 8GB VRAM)
- Cooperative cancellation
- GPU memory cleanup between jobs
- Pause/resume capability

### 10. Metrics Collector (`app/monitoring/metrics.py`)

Prometheus-compatible metrics:
- Job counters (started, completed, failed, cancelled)
- Latency histograms (API, inference, export, job duration)
- GPU/CPU memory tracking
- System metrics (CPU, disk, VRAM)
- `/metrics` endpoint for scraping

## API Contract

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/uploads` | Upload image |
| GET | `/api/v1/uploads/{id}` | Get upload metadata |
| POST | `/api/v1/jobs/reconstruct` | Submit reconstruction job |
| GET | `/api/v1/jobs/{id}` | Get job status |
| GET | `/api/v1/jobs/{id}/results` | Get job results |
| GET | `/api/v1/jobs/{id}/exports/{fmt}` | Download export |
| POST | `/api/v1/jobs/{id}/cancel` | Cancel job |
| GET | `/api/v1/jobs` | List jobs |
| GET | `/api/v1/models` | List models |
| GET | `/api/v1/models/{id}` | Model details |
| GET | `/api/v1/models/{id}/verify` | Verify checkpoint |
| GET | `/api/v1/exports/{job_id}` | List exports |
| GET | `/api/v1/exports/{job_id}/{fmt}` | Download export |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe |
| GET | `/metrics` | Prometheus metrics |

### Job States

```
QUEUED → VALIDATING → INFERENCE → GEOMETRY_POSTPROCESSING → EXPORTING → COMPLETED
                              ↘ FAILED
                              ↘ CANCELLED
```

### Error Format

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human readable message",
    "remediation": "Suggested action"
  },
  "timestamp": 1234567890.123
}
```

## HunyuanWorld-Mirror Integration

### Model Loading
- Architecture: `WorldMirror` (ViT-L/14 DINOv2 backbone + DPT heads)
- Checkpoints: `production_model.ckpt` (ref), `best_stage1.ckpt` (Stage 1), `stage2_real_verified.ckpt` (Stage 2)
- Precision: bfloat16 on CUDA, float32 on CPU
- Auto-detection of Stage 2 (GS head enabled)

### Preprocessing Contract
- Input: 518×518 square crop (patch-14 aligned)
- Normalization: ImageNet mean/std
- Coordinate transform: `C = diag(1, -1, -1)` (OpenGL → OpenCV)

### Outputs
- **Depth**: Metric depth in meters (1, 518, 518)
- **Points**: 3D points in RoomAI frame (518, 518, 3)
- **Normals**: Surface normals in RoomAI frame (518, 518, 3)
- **Camera**: C2W poses (OpenCV convention)

### Strict No-Fallback Mode
Experimental endpoint `/api/v1/experimental/hunyuan/reconstruct` enforces:
- Explicit model selection (no default fallback)
- Checkpoint verification before execution
- HTTP 503 if model unavailable (no silent fallback to composite)

## Reproducibility Guarantees

1. **Model Registry**: Immutable YAML with SHA-256, param counts, commit hashes
2. **Preprocessing Config**: Versioned, serialized with every job
3. **Dataset Manifests**: SHA-256 hashed manifests referenced in model specs
4. **Deterministic Exports**: Fixed GLB/PLY generation (no randomness)
5. **Job Metadata**: Full traceability (input → model → preprocessing → output)

## Deployment

### Requirements
- Python 3.10+
- PyTorch 2.6+ with CUDA 12.4
- RTX 4060 8GB (minimum) / A100 80GB (recommended for training)
- SQLite (dev) / PostgreSQL (prod)

### Startup
```bash
# Development
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Environment Variables
```bash
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@host/db
API_HOST=0.0.0.0
API_PORT=8000
MAX_CONCURRENT_JOBS=1
GPU_MEMORY_FRACTION=0.85
LOG_LEVEL=INFO
```

## Testing

```bash
# Unit tests
pytest tests/ -v

# Real Hunyuan inference (requires GPU + checkpoints)
pytest tests/test_e2e_real_hunyuan_production.py -v -s

# API integration
pytest tests/test_e2e_api_workflow.py -v
```

## Monitoring

- **Health**: `GET /health` (liveness)
- **Readiness**: `GET /ready` (model + GPU check)
- **Metrics**: `GET /metrics` (Prometheus)
- **Structured Logs**: JSON with trace_id, job_id, model_id correlation

## Security

- Input validation (size, format, dimensions)
- Decompression bomb protection (100MP limit)
- Path traversal prevention
- CORS configurable via env
- No authentication in current version (add via middleware)