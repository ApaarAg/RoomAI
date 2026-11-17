# ROOMAI — PRODUCTION APPLICATION ARCHITECTURE

**Document Version:** 1.0.0  
**Status:** Approved for Implementation  
**Author:** Staff Software Engineer, Full-Stack Architect & Product Engineer  
**System Target:** Production-Grade Indoor Scene Understanding & 3D Spatial Studio  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), Windows 11  

---

## 1. Architectural Vision & Core Principles

RoomAI is structured around four uncompromising architectural principles:
1. **Decoupled API Boundaries:** The presentation layer (Streamlit UI, Web clients, Headless CLI) never calls neural network inference code directly; it communicates exclusively through a versioned, asynchronous HTTP REST API contract (`/api/v1/`).
2. **Pluggable Model Adapters:** Model inference is strictly abstracted behind a unified `ReconstructionModelAdapter` interface. The application core does not care whether geometry is produced by RoomAI's existing composite pipeline (YOLO11s + Depth Anything V2) or the next-generation foundation model (HunyuanWorld-Mirror).
3. **Resilient Asynchronous Job Processing:** Long-running 3D reconstruction is never executed in the synchronous request-response lifecycle. Jobs are queued, persisted in a lightweight SQLite database, executed in background workers, and monitored via explicit state transitions.
4. **Strict GPU Resource Safety:** Model concurrency is governed by a strict GPU concurrency gate (`GPUConcurrencyGate`). Inference requests are serialized to prevent VRAM thrashing on 8 GB GPUs, with automatic OOM isolation and recovery.

---

## 2. High-Level Component Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                         PRESENTATION LAYER                                         |
|  - Streamlit Production UI (app.py, ui/)                                                           |
|  - Web / Mobile / CLI Clients (Consuming REST API)                                                 |
|  - Interactive 3D WebGL Viewer (Plotly 6.8.0 / Three.js glTF Container)                            |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  | HTTP / JSON (REST API Contract)
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    API & ROUTING LAYER (FastAPI)                                   |
|  - Middleware: CORS, Request Tracing, Error Handling, Decompression Bomb DOS Guard                 |
|  - Routers (/api/v1/):                                                                             |
|      * /health, /ready               -> System & Hardware Diagnostics                              |
|      * /uploads                      -> Validated Image Ingestion & Storage                        |
|      * /jobs/reconstruct             -> Asynchronous Reconstruction Job Submission                 |
|      * /jobs/{job_id}                -> Job Status & Progress Polling                              |
|      * /jobs/{job_id}/results        -> Full Structured Scene Data (Detections, Depth, 3D Graph)   |
|      * /jobs/{job_id}/exports/{fmt}  -> Validated 3D Asset Streaming (GLB / PLY)                   |
|      * /models                       -> Active Model Registry & Capability Discovery               |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                       APPLICATION SERVICES                                         |
|  +---------------------+   +---------------------+   +---------------------+   +-----------------+ |
|  |     JobService      |   |    UploadService    |   |ReconstructionService|   |  ExportService  | |
|  | (Lifecycle & State) |   | (Image Validation)  |   | (Model Orchestration|   | (GLB/PLY Stream)| |
|  +---------------------+   +---------------------+   +---------------------+   +-----------------+ |
+----------------------------------------------------------------------------------------------------+
              |                                                   |
              | Enqueue / Status Update                           | Execution Pipeline
              v                                                   v
+------------------------------------+             +------------------------------------------------+
|       PERSISTENT JOB STORE         |             |              GPU CONCURRENCY GATE              |
|        (SQLite: jobs.db)           |             |            (utils/gpu_concurrency.py)          |
|  - Durable Job Records             |             |  - Binary Semaphore (Max Concurrent Ops = 1)   |
|  - States: QUEUED -> COMPLETED     |             |  - VRAM Pre-flight Inspection (>= 1,500 MB)    |
|  - Survives Process Restarts       |             |  - Exception Recovery & Memory Cache Flushes   |
+------------------------------------+             +------------------------------------------------+
                                                                  |
                                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    MODEL ADAPTER INTERFACE LAYER                                   |
|  - Abstract Base Class: ReconstructionModelAdapter                                                 |
|                                                                                                    |
|  +----------------------------------------------+  +---------------------------------------------+ |
|  |         RoomAICompositeAdapter (Active)      |  |     HunyuanWorldMirrorAdapter (Staged)      | |
|  |  * YOLO11s A3 Indoor Furniture Detector     |  |  * 1.0B Parameter Feed-Forward Transformer  | |
|  |  * YOLO11n-Seg Instance Segmenter            |  |  * Any-Prior Depth Prompt Conditioning      | |
|  |  * Depth Anything V2 Metric Indoor Small     |  |  * Dense Surface Normal Prediction          | |
|  |  * RoomLayoutService (RANSAC Floor & Normal) |  |  * 3D Gaussian Splatting (3DGS) Head        | |
|  |  * SceneFusionService (Metric 3D Unproject)  |  |  * Native Multi-View Cross-Attention        | |
|  +----------------------------------------------+  +---------------------------------------------+ |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                      STORAGE & ASSET LAYER                                         |
|  - Scoped Artifact Storage: artifacts/jobs/{job_id}/                                               |
|  - File-based Quota Enforcement & LRU Cleanup (ArtifactManager)                                    |
|  - Verified 3D Serialization: Binary glTF 2.0 (GLB) & ASCII PLY (utils/export_3d.py)               |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Component Responsibilities & Boundaries

### 3.1 API & Routing Layer (`api/`)
- **Framework:** FastAPI with Pydantic v2 data validation schemas.
- **Responsibilities:**
  - Enforce payload size limits (max 20 MB per file, max 100M total image pixels).
  - Sanitize filenames to prevent path traversal attacks.
  - Return typed JSON error envelopes with machine-readable error codes and human-readable remediation hints.
  - Expose self-documenting OpenAPI documentation at `/docs` and `/redoc`.

### 3.2 Job Processing & Lifecycle Management (`services/job_service.py`)
- **Durable Persistence:** Backed by SQLite database (`artifacts/jobs.db`).
- **State Machine:**
  ```
  [SUBMITTED]
       |
       v
    QUEUED  --------> CANCELLED
       |
       v
   VALIDATING
       |
       v
   PROCESSING
       |
       v
  RECONSTRUCTING
       |
       v
     FUSING
       |
       v
   EXPORTING
       |
       v
   COMPLETED  (or FAILED on unhandled error)
  ```
- **Execution Engine:** Background worker thread pool processing queued jobs FIFO with atomic database state updates.

### 3.3 Model Adapter Layer (`models/adapters/`)
- **Design Pattern:** Strategy Pattern with factory registry.
- **Contract Definition:**
  ```python
  class ReconstructionModelAdapter(ABC):
      @abstractmethod
      def initialize(self) -> None: ...
      @abstractmethod
      def get_capabilities(self) -> ModelCapabilities: ...
      @abstractmethod
      def is_available(self) -> Tuple[bool, Optional[str]]: ...
      @abstractmethod
      def validate_inputs(self, images: List[Image.Image]) -> Tuple[bool, Optional[str]]: ...
      @abstractmethod
      def reconstruct(self, images: List[Image.Image], options: Dict[str, Any]) -> SceneGraph: ...
      @abstractmethod
      def cleanup(self) -> None: ...
  ```
- **Capability Declaration:** Every adapter must explicitly declare:
  - `supports_single_view`: bool
  - `supports_multi_view`: bool
  - `max_views`: int
  - `supports_metric_depth`: bool
  - `supports_surface_normals`: bool
  - `supports_3dgs`: bool
  - `memory_required_mb`: int

### 3.4 GPU Execution & Concurrency Strategy
- **Resource Bottleneck:** RTX 4060 Laptop GPU with 8,187.5 MB VRAM.
- **Concurrency Invariant:** No two GPU inference operations may execute simultaneously (`max_concurrent_ops = 1`).
- **Pre-Flight Memory Guard:** Before launching model forward passes, `torch.cuda.mem_get_info()` verifies that free VRAM exceeds the model's declared threshold (e.g. 1,500 MB for Depth Anything V2; 5,000 MB for HunyuanWorld-Mirror).
- **OOM Circuit Breaker:**
  ```python
  try:
      with GPU_GATE.acquire():
          output = model.forward(tensor)
  except torch.cuda.OutOfMemoryError as oom:
      torch.cuda.empty_cache()
      logger.critical(f"CUDA OOM in job {job_id}: {oom}")
      job_service.fail_job(job_id, error="CUDA_OUT_OF_MEMORY", remediation="Reduce image resolution or view count.")
  ```

### 3.5 Asset Storage & Retention Strategy
- **Session Scoping:** All uploaded images, intermediate representations, and exported 3D files are strictly isolated under `artifacts/jobs/{job_id}/`.
- **Deduplication:** Uploaded images are hashed (SHA-256) to avoid redundant storage.
- **Retention Lifecycle:** Completed jobs and associated 3D binaries are retained for 24 hours. A background retention routine evicts oldest files when cumulative artifact storage exceeds 500 MB.

---

## 4. End-to-End Data Flow

```
1. Client POSTs Image(s) to /api/v1/uploads
   -> Server validates format, dimensions, DOS safety
   -> Writes to artifacts/uploads/{hash}.jpg
   -> Returns upload_id, dimensions, file size

2. Client POSTs to /api/v1/jobs/reconstruct with upload_ids, model_id, options
   -> Server validates view count against selected model capabilities
   -> Inserts row into jobs.db (status='QUEUED', progress=0.0)
   -> Enqueues task into JobWorkerQueue
   -> Returns job_id immediately (HTTP 202 Accepted)

3. Background Worker dequeues job
   -> Updates status='VALIDATING', progress=0.10
   -> Loads image bytes from disk into PIL Images
   -> Updates status='PROCESSING', progress=0.25
   -> Acquires GPU_GATE lock
   -> Executes adapter.reconstruct(images)
      * Sub-step: Object Detection (YOLO11s) -> progress=0.45
      * Sub-step: Metric Depth Estimation -> progress=0.65
      * Sub-step: 3D Point Cloud Unprojection -> progress=0.80
      * Sub-step: Floor Plane RANSAC Fitting -> progress=0.88
   -> Releases GPU_GATE lock
   -> Updates status='EXPORTING', progress=0.92
   -> Serializes SceneGraph to .glb and .ply in artifacts/jobs/{job_id}/
   -> Validates exported bytes with struct-level parsers
   -> Updates status='COMPLETED', progress=1.00, sets completed_at

4. Client polls GET /api/v1/jobs/{job_id}
   -> Receives status, progress, stage, execution time

5. Upon status='COMPLETED', Client calls:
   -> GET /api/v1/jobs/{job_id}/results -> Visualizes 3D point cloud & objects in viewer
   -> GET /api/v1/jobs/{job_id}/exports/glb -> Downloads verified 3D GLB model
   -> GET /api/v1/jobs/{job_id}/exports/ply -> Downloads verified ASCII PLY cloud
```

---

## 5. Security & Robustness Invariants

1. **Path Traversal Protection:** All file operations sanitize incoming file identifiers via `os.path.basename` and resolve canonical absolute paths within `artifacts/`. Any access outside `artifacts/` returns HTTP 403 Forbidden.
2. **Decompression Bomb Protection:** Pillow's maximum pixel threshold is clamped to 100,000,000 pixels (`Image.MAX_IMAGE_PIXELS = 100_000_000`). Images exceeding this bound are rejected at upload validation time.
3. **CORS Configuration:** Configured with explicit origins (defaults to `localhost:8501`, `localhost:3000`, `127.0.0.1:*`).
4. **Secret Isolation:** No secrets or credentials are hardcoded. Environment settings are governed by `.env` loaded via `pydantic-settings`.
5. **No Fake AI:** Features lacking local weights (VQA, Generative Redesign, Room Classification) are formally advertised as `UNAVAILABLE` in the capabilities schema. Calling them in real mode returns HTTP 501 Not Implemented with honest remediation guidance.

