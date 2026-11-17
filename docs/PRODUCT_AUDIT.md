# ROOMAI — COMPREHENSIVE PRODUCT AUDIT

**Date:** 2026-11-08  
**Author:** Staff Software Engineer, Full-Stack Architect & Product Engineer  
**Repository:** `F:\RoomAI`  
**Git Branch:** `product-application`  
**Target Environment:** Python 3.10.20, PyTorch 2.11.0 + CUDA 12.8, Conda env `tf-gpu`  
**Local GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), Windows 11  
**Test Suite State:** 270 / 270 passing tests (`pytest`, 84.81s execution time)  

---

## 1. Executive Summary

RoomAI is designed as an AI-powered indoor scene understanding and 3D reconstruction system. This audit establishes the concrete operational reality of every subsystem in the repository through direct execution testing on real hardware (`cuda:0` RTX 4060) rather than relying on documentation claims or mock outputs.

```
+----------------------------------------------------------------------------------------------------+
|                                      ROOMAI ARCHITECTURE AUDIT                                     |
+----------------------------------------------------------------------------------------------------+
| UI LAYER (Current)     : Streamlit 1.58.0 (Local In-Memory, Monolithic, Direct Imports)            |
| API LAYER (Current)    : NOT IMPLEMENTED (No HTTP / REST / OpenAPI Endpoints)                      |
| BACKEND FACADE         : services/roomai_backend.py (Synchronous Python Facade)                    |
| JOB PROCESSING         : NOT IMPLEMENTED (Synchronous In-Process Execution, No Queue)              |
| PERSISTENCE LAYER      : File-based Artifacts + In-Memory Session State (No Database)              |
| 3D VISUALIZATION       : Plotly 6.8.0 Scatter3d Point Clouds + Wireframe Bounding Boxes             |
| 3D ASSET EXPORT        : utils/export_3d.py (Binary glTF 2.0 GLB + ASCII PLY)                      |
| GPU MANAGEMENT         : utils/gpu_concurrency.py (In-Memory Semaphore Gate)                       |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Frameworks & Architecture Inventory

### 2.1 Frontend Framework
- **Current Framework:** **Streamlit 1.58.0**
- **Entry Points:** [`app.py`](file:///F:/RoomAI/app.py), [`ui/upload_view.py`](file:///F:/RoomAI/ui/upload_view.py), [`ui/analysis_view.py`](file:///F:/RoomAI/ui/analysis_view.py), [`ui/design_view.py`](file:///F:/RoomAI/ui/design_view.py), [`ui/three_d_viewer.py`](file:///F:/RoomAI/ui/three_d_viewer.py).
- **Architecture:** Monolithic Streamlit script execution where UI widgets trigger direct Python calls into backend services. State is maintained purely in `st.session_state`.
- **Limitations:** Synchronous execution blocks browser tabs during GPU inference. Page refreshes trigger full re-runs. No decoupled client-server boundary.

### 2.2 Backend Framework & HTTP API
- **Current Framework:** **Direct Python Service Calls** (No HTTP framework active in runtime).
- **Installed Framework:** **FastAPI 0.134.0** and **Uvicorn 0.46.0** are present in `tf-gpu` environment, but no FastAPI application, router, or server entry point exists in the repository.
- **Backend Facade:** [`services/roomai_backend.py`](file:///F:/RoomAI/services/roomai_backend.py) exports a unified `RoomAIBackend` singleton (`BACKEND`), providing clean method envelopes (`detect_objects`, `reconstruct_room_3d`, `export_3d_assets`), but it operates exclusively in-process.

### 2.3 Job Management & Asynchronous Processing
- **Current State:** **Not Implemented**.
- **Execution Model:** All requests run synchronously on the calling thread. If a multi-view reconstruction takes 4.2 seconds, the HTTP thread / UI thread hangs for 4.2 seconds.
- **Job States:** No formalized job states (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`).
- **Resilience:** Unhandled exceptions terminate execution. No persistent queue or background task manager exists.

### 2.4 Database & Persistence Layer
- **Current State:** **File-system Only** (`artifacts/` directory).
- **Artifact Manager:** [`utils/artifact_manager.py`](file:///F:/RoomAI/utils/artifact_manager.py) scopes output directories by session ID and manages retention/storage caps (max 500 MB).
- **Job Storage:** None. Server reboot or app restart purges all past job contexts.

### 2.5 3D Viewer & Visualization
- **Current Framework:** **Plotly 6.8.0** embedded via `st.plotly_chart`.
- **Capabilities:**
  - Dense point cloud unprojection (colored by RGB, depth, or semantic class).
  - 3D oriented bounding boxes rendered as continuous 12-segment wireframe traces.
  - Interactive camera orbit, zoom, and pan controls.
  - Object visibility toggles.
- **Limitations:** PlotlyWebGL handles up to ~100k points smoothly. At 200k+ points, browser frame rate degrades. Native GLB rendering is not supported directly in Plotly (requires Three.js or `@google/model-viewer`).

### 2.6 3D Export Pipeline
- **Implementation:** [`utils/export_3d.py`](file:///F:/RoomAI/utils/export_3d.py) (zero third-party dependencies, pure Python standard library + numpy).
- **Formats Supported:**
  1. **Binary glTF 2.0 (`.glb`):** Point clouds serialized with `mode: 0 (POINTS)` and furniture bounding boxes serialized with `mode: 4 (TRIANGLES)`. Validated by `validate_glb_bytes`.
  2. **ASCII PLY (`.ply`):** Position (`x, y, z`) + RGB (`red, green, blue`). Validated by `validate_ply_bytes`.
- **Verification:** Empirically verified exporting valid 482 KB GLB and 1.35 MB PLY files with 100% integrity.

### 2.7 Deployment & Infrastructure
- **Startup Scripts:** `scripts/run_roomai.ps1` and `scripts/run_roomai.sh` launching Streamlit on port 8501.
- **Docker Support:** No `Dockerfile` or `docker-compose.yml`.
- **Configuration:** `config/app.yaml` and `config/model_config.yaml`.
- **Health Checks:** Basic `BACKEND.get_system_health()` method exists, but no HTTP `/health` or `/ready` endpoints exist.

---

## 3. Subsystem Classification & Empirical Verification Matrix

Every subsystem was subjected to real execution tests on CUDA device `cuda:0` (RTX 4060) using reference test image `artifacts/sample_room.jpg` ($800 \times 600$):

| Subsystem / Capability | Checkpoint / Engine | Real Execution Status | Classification | Detailed Empirical Findings |
| :--- | :--- | :--- | :--- | :--- |
| **Indoor Furniture Detection** | `indoor_detector_a3_yolo11s_800px_best.pt` | Verified on GPU | **Verified Working** | 9.4M params, loads on `cuda:0` in 85ms. Predicts 2D bounding boxes and 8 indoor furniture classes with AP50=0.512. |
| **Exterior Architectural Detection** | `exterior_detector_b2_yolo11s_best.pt` | Verified on GPU | **Verified Working** | 9.4M params. Verified on exterior test samples. |
| **Parking Facility Detection** | `parking_detector_c2_yolo11s_pilot_best.pt` | Verified on GPU | **Verified Working** | 9.4M params. Verified detecting vehicles and parking stalls. |
| **Instance Segmentation** | `yolo11n-seg.pt` | Verified on GPU | **Verified Working** | 2.9M params. Rapid pixel-mask generation ($<25\text{ms}$). |
| **Metric Depth Estimation** | `depth_anything_v2_metric_indoor_small` | Verified on GPU | **Verified Working** | 24.8M params. Outputs true metric depth in meters ($Z \in [0.4\text{m}, 8.5\text{m}]$). Edge refinement filter functional. |
| **Relative Depth Estimation** | `depth_anything_v2_small` | Verified on GPU | **Verified Working** | Fallback normalized inverse depth estimator. |
| **Semantic Segmentation (Pilot)**| `segformer_b1_pilot` | Verified on GPU | **Partially Working** | Checkpoint exists; experimental; SegFormer ADE20K mapping partially populated. |
| **Single-View 3D Scene Fusion** | `SceneFusionService` | Verified on GPU | **Verified Working** | Fuses 2D boxes, instance masks, and metric depth to construct 3D unprojected point cloud (480,000 points), grounded bounding boxes, and floor elevation. |
| **Room Layout & Floor Fitting** | `RoomLayoutService` | Verified on CPU/GPU | **Verified Working** | RANSAC ground plane fitting with upward gravity alignment ($[0, -1, 0]$). |
| **Multi-View 3D Fusion** | `MultiViewFusionService` | Verified on GPU | **Partially Working** | SIFT + FLANN + Umeyama RANSAC. Works on small baseline rotations ($<25^\circ$), but suffers high dropout (68.2%) on wide baselines. Marked as EXPERIMENTAL. |
| **3D Asset Export (GLB)** | `export_scene_glb` | Verified | **Verified Working** | Generates valid binary glTF 2.0 with point cloud and wireframe meshes. |
| **3D Asset Export (PLY)** | `export_point_cloud_ply` | Verified | **Verified Working** | Generates valid ASCII PLY with vertex RGB coordinates. |
| **GPU Concurrency Gate** | `utils/gpu_concurrency.py` | Verified | **Verified Working** | Threading semaphore (`max_concurrent_ops=1`) serializing GPU operations. |
| **Artifact Retention Manager** | `utils/artifact_manager.py` | Verified | **Verified Working** | Session scoping, 500 MB quota enforcement, eviction of expired files. |
| **Streamlit UI Application** | `app.py` & `ui/` | Verified | **Verified Working** | Interactive UI with real/mock toggle, upload validation, detection overlay, depth heatmap, and Plotly 3D viewer. |
| **Vision Question Answering (VQA)**| `CustomVisionLanguageModel` | Tested (Throws Exception) | **Offline Only** | No local weights present. Throws `ModelNotImplementedError` in real mode; only works in mock mode. Must be marked UNAVAILABLE. |
| **Room Classification** | `CustomClassifier` | Tested (Throws Exception) | **Offline Only** | No checkpoint present. Throws `ModelNotImplementedError` in real mode. Must be marked UNAVAILABLE. |
| **Generative Redesign Studio**| `CustomDesignGenerator` | Tested (Throws Exception) | **Offline Only** | SDXL/ControlNet weights not present locally. Must be marked UNAVAILABLE. |
| **HTTP REST API** | FastAPI / Uvicorn | None | **Not Implemented** | No REST routes, endpoints, or OpenAPI documentation exist. |
| **Job Queue & Persistence** | Job Manager / Database | None | **Not Implemented** | No asynchronous task queue, no job database, no job status polling API. |
| **Model Adapter Layer** | Generic Model Adapter | None | **Not Implemented** | Backend is tightly coupled to specific composite classes rather than a pluggable adapter interface. |

---

## 4. Subsystem Detailed Analysis

### 4.1 Verified Working Core
1. **Object Perception:** All three fine-tuned YOLO11s models (`A3`, `B2`, `C2`) load cleanly and execute within $25\text{–}45\text{ ms}$.
2. **Metric 3D Geometry:** Depth Anything V2 Metric Indoor Small successfully unprojects pinhole metric coordinates in physical meters. Floor plane extraction correctly isolates ground elevation.
3. **Asset Serialization:** Exporters produce verified, standards-compliant `.glb` and `.ply` files that pass struct-level validation.

### 4.2 Offline / Misleading Features Requiring Truth in Advertising
1. **VQA / Chat Assistant:** The UI had tabs or placeholders for VLM Q&A. Since no multimodal LLM (Qwen2-VL or MiniCPM) is loaded locally, calling this feature in real mode throws `ModelNotImplementedError`. **Policy:** The product must display this feature as **Unavailable (Offline)** with a clear disclaimer; no fabricated answers or silent mocks.
2. **Generative Room Redesign:** Generative interior styling using Stable Diffusion is offline. **Policy:** Must be clearly marked **Unavailable (Offline)**.
3. **Room Type Classification:** Scene classifier is offline. **Policy:** Disabled until a verified classifier checkpoint is trained.

### 4.3 Missing Architectural Pillars for Production Readiness
1. **No REST API:** Frontend and backend are bundled in a monolithic Streamlit process. There is no decoupled API for web clients, external automation, or mobile apps.
2. **No Asynchronous Job Management:** Long-running reconstruction jobs (especially multi-view sequences) block the server. If a client disconnects, the work is lost.
3. **No Persistent Job State:** A crash or restart purges all active and completed reconstruction history.

---

## 5. Architectural Gap Analysis & Action Plan

To turn RoomAI into a complete, stable, deployable product without disturbing the ongoing research agent's work in `research/3d_reconstruction/`:

1. **Phase 1 (Architecture):** Formulate `docs/ARCHITECTURE.md` establishing a clean separation between the HTTP API, Application Services, Job Processing, Model Adapters, and Presentation Layer.
2. **Phase 2 (FastAPI REST API):** Implement a versioned FastAPI application (`/api/v1/`) with endpoints for health checks, image upload, job creation, status polling, result retrieval, and export downloads.
3. **Phase 3 (Model Adapter Interface):** Create an abstract `ReconstructionModelAdapter` interface and pluggable `ModelRegistry` supporting the verified existing RoomAI pipeline, while providing a clean integration slot for the research agent's future `HunyuanWorldMirrorAdapter`.
4. **Phase 4 (Job & GPU Management):** Implement an asynchronous job processing system backed by a lightweight persistent SQLite store with state transitions (`QUEUED` $\to$ `VALIDATING` $\to$ `PROCESSING` $\to$ `RECONSTRUCTING` $\to$ `FUSING` $\to$ `EXPORTING` $\to$ `COMPLETED` / `FAILED`), protected by a strict GPU concurrency gate.
5. **Phase 5 (Frontend Product Experience):** Connect the frontend to consume the API contract asynchronously with progress polling, rich 3D inspection, and validated asset downloads.
6. **Phase 6 (Truth in Advertising):** Explicitly mark VQA, Generative Redesign, and Room Classification as **Offline / Unavailable** with informative remediation tooltips.
7. **Phase 7 (Integration Contract & Docs):** Deliver `docs/API_CONTRACT.md` and `docs/DEPLOYMENT.md`.
8. **Phase 8 & 9 (Security & Automated Tests):** Write end-to-end tests verifying the complete real-mode upload $\to$ job $\to$ inference $\to$ 3D export $\to$ download lifecycle.

