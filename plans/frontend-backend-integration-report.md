# RoomAI — Final Integration Audit & Rigorous Verification Report

**Date**: November 16, 2026  
**Auditor**: Senior ML & Backend Infrastructure Reviewer  
**Status**: Pre-Staging Audited, Hardened & Empirically Verified  

---

## 1. Audit of the 2D → 3D Pipeline

### Exact Determination: Option C (Provisional 3D Ray Projection with Assumed Depth)
A rigorous inspection of `web/lib/spatial/coordinates.ts`, `LiveInferenceExhibit.tsx`, and `DepthExhibitCanvas.tsx` confirms:
- **Model A3 is strictly a 2D object detector** (`YOLO11s` at 800px) returning 2D bounding boxes $[x_1, y_1, x_2, y_2]$ normalized to $[0, 1]$.
- The pipeline does **NOT** measure depth from sensor data, stereo vision, or metric monocular depth prediction during object detection.
- Instead, the pipeline computes the 2D bounding box center in Normalized Device Coordinates (NDC) and unprojects it along the optical camera ray assuming a pinhole camera model ($\text{FOV} = 34^\circ$) and a **provisional nominal scene depth plane** ($Z = 3.0\text{m}$).
- **Architectural Clarification**: The result is formally documented and typed as `ProvisionalSpatialObject3D` with explicit flag `depthIsAssumed: true`. The UI has been updated to state:  
  `"PROVISIONAL 3D CAMERA-RAY PROJECTION (ASSUMED SCENE DEPTH Z = 3.0m)"`  
  with note: `"Pinhole optical unprojection along camera rays (FOV 34°). Metric depth is provisionally assumed at 3.0m; not sensor measurement."`

---

## 2. Audit of Cache Correctness & Separation

### Cache Key Specification
In `api/routers/detect.py`, the cache key incorporates every parameter that influences inference results:
$$\text{Key} = \Big(\text{sha256\_digest}(\text{image\_bytes}), \; \text{round}(\text{conf\_thresh}, 4), \; \text{model\_name}, \; \text{imgsz}\Big)$$

### Mathematical Proof & Test Evidence
Automated integration tests (`tests/test_api_detect.py::test_cache_separation_across_different_thresholds` and `test_cache_invalidation_and_model_version_isolation`) verify:
1. **Threshold Disjointness**: Submitting the same image with $\text{conf}=0.15$ followed by $\text{conf}=0.85$ results in a cache miss on the second call (`cached: false`). The high-threshold query is executed fresh and is never erroneously served from the low-threshold cache.
2. **Repeated Query Hit**: Repeating $\text{conf}=0.15$ returns `cached: true` with sub-millisecond execution. Repeating $\text{conf}=0.85$ returns `cached: true`.
3. **Model Version Separation**: Changes to `model_name` or `imgsz` partition the cache entries.
4. **Cache Invalidation**: Calling `GLOBAL_MODEL_A3_ENGINE.clear_cache()` flushes all entries, guaranteeing that subsequent requests trigger fresh inferences.

---

## 3. Audit of Latency Benchmark Methodology

### Precise Methodology
- **Execution Mode**: Sequential HTTP requests executed via Starlette ASGI `TestClient` directly into the FastAPI application pipeline without external proxy hops.
- **Hardware Synchronization**: `torch.cuda.synchronize()` is enforced immediately before and after inference, ensuring that measurements capture true GPU kernel execution rather than asynchronous queue submission.
- **Sample Count**: Exactly **20 warm requests per configuration** across **4 distinct configurations** = **80 total warm runs**, preceded by 1 server boot pre-warm pass and 1 pre-benchmark pass per configuration.
- **Payload Diversity**: Each uncached run uses a uniquely seeded synthetic room image to prevent LRU cache hits.

### Empirical Latency Table (80 Warm Runs on NVIDIA GeForce RTX 4060 Laptop GPU)

| Test Configuration | Payload Size | Mode | Total E2E p50 | Total E2E p95 | Total E2E Mean (Min – Max) | GPU Forward p50 | GPU Forward p95 | Server Total p50 | Server Total p95 | Preprocess Mean | Postprocess Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small (640x480)** | 7.3 KB | Fresh Uncached | **18.65 ms** | **23.02 ms** | 19.29 ms (17.42 – 23.78 ms) | 15.36 ms | 18.60 ms | 16.76 ms | 20.92 ms | 0.77 ms | 0.79 ms |
| **Medium (1280x720)** | 18.3 KB | Fresh Uncached | **23.55 ms** | **26.87 ms** | 24.20 ms (21.63 – 31.07 ms) | 18.98 ms | 21.94 ms | 21.62 ms | 24.99 ms | 2.03 ms | 0.60 ms |
| **Large (1920x1080)** | 39.5 KB | Fresh Uncached | **32.40 ms** | **35.15 ms** | 32.21 ms (29.47 – 35.87 ms) | 25.88 ms | 28.22 ms | 30.25 ms | 32.90 ms | 3.73 ms | 0.45 ms |
| **Cached Query** | 7.3 KB | LRU Hit | **1.93 ms** | **2.16 ms** | 1.94 ms (1.77 – 2.18 ms) | 0.00 ms | 0.00 ms | 0.65 ms | 0.79 ms | 0.66 ms | 0.03 ms |

### Client-Side Latency Scope
DOM mutation and SVG overlay update time in the browser is measured at **1.8 – 3.2 ms** via `performance.now()`. This represents **DOM mutation and layout scheduling latency**, not total browser frame compositing time (which depends on display refresh vsync intervals of 16.6ms / 60Hz).

---

## 4. Audit of Concurrency, VRAM & GPU Lock Policy

### Concurrency Policy Determination
**Serialized GPU execution is the current recommended operational policy based on the observed benchmark: it prevents concurrent GPU execution and keeps VRAM stable, while higher concurrency primarily increases queueing latency without producing proportional throughput gains.**

### Investigation of Prior Concurrency Discrepancy (C=1 at ~10.91 req/s vs C=2 at ~41.77 req/s)
In initial preliminary testing, an apparent discrepancy was observed where C=1 reported ~10.91 req/s while C=2 reported ~41.77 req/s. A hostile audit of `scripts/benchmark_concurrency.py` isolated three methodological causes:
1. **Accidental In-Memory Cache Hits**: The initial script computed seeds as `(r * 100 + i + 1)`. When C=1 completed rounds 0..4 for worker 0 (`i=0`), seeds `[1, 101, 201, 301, 401]` were entered into the LRU cache. When C=2 subsequently executed, worker 0 requested the exact same seeds `[1, 101, 201, 301, 401]`. Exactly 50% of the C=2 batch hit the in-memory cache, returning in ~2 ms with 0.0 ms GPU inference time, artificially inflating throughput and deflating latency.
2. **Cold JIT / Route Initialization Overhead in C=1**: In the initial run, only 5 requests were sent for C=1 without a prior client HTTP pipeline warmup. Request #1 bore the full cold ASGI routing, PyTorch CUDA kernel JIT, and model graph initialization of 216.3 ms. Averaged across only 5 requests, this single outlier severely dragged down C=1 throughput to 10.91 req/s.
3. **Synchronous CPU Image Generation inside Timed Loop**: In the initial script, PIL JPEG compression occurred synchronously inside the timed duration loop, adding CPU time to the measured batch duration.

### Clean, Cache-Isolated Concurrency Benchmark (75 Requests, Clean Caches, Pre-Warmed Pipeline)
Conducted via the corrected `scripts/benchmark_concurrency.py` across $C \in \{1, 2, 4, 8\}$ (5 rounds each, unique non-overlapping seeds, pre-generated payloads, pre-warmed client):

| Concurrency Level | Success / Total | Throughput | E2E Latency p50 | E2E Latency p95 | Queue Wait p50 | Queue Wait p95 | GPU Forward Mean | Peak Allocated Tensor VRAM | PyTorch Reserved VRAM | Total Device GPU Memory (`nvidia-smi`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Worker** | 5 / 5 (0 fail) | **25.15 req/s** | 40.35 ms | 45.35 ms | 0.00 ms | 0.01 ms | 31.50 ms | 107.13 MB | 146.00 MB | 1005.00 MB |
| **2 Concurrent** | 10 / 10 (0 fail) | **29.88 req/s** | 53.04 ms | 70.26 ms | 10.91 ms | 31.23 ms | 27.44 ms | 107.13 MB | 146.00 MB | 1005.00 MB |
| **4 Concurrent** | 20 / 20 (0 fail) | **29.40 req/s** | 86.52 ms | 135.59 ms | 41.35 ms | 88.36 ms | 28.51 ms | 107.13 MB | 146.00 MB | 1005.00 MB |
| **8 Concurrent** | 40 / 40 (0 fail) | **30.07 req/s** | 159.24 ms | 261.57 ms | 100.14 ms | 202.62 ms | 28.72 ms | 107.13 MB | 146.00 MB | 1005.00 MB |

### Rigorous Interpretation of VRAM Metrics
The metric previously described casually as "107.13 MB Peak VRAM" was audited. It strictly measures **active PyTorch tensor memory allocations**. It must never be termed "total VRAM usage". The exact memory composition on the RTX 4060 8GB is:
- **`torch.cuda.memory_allocated()`**: **82.04 MB** (active memory currently occupied by PyTorch model weights and working tensors).
- **`torch.cuda.max_memory_allocated()`**: **107.13 MB** (the true peak tensor allocation during the forward pass).
- **`torch.cuda.memory_reserved()` / `max_memory_reserved()`**: **146.00 MB** (the memory blocks requested from and managed by the PyTorch caching allocator from the CUDA driver).
- **CUDA Runtime & Driver Context Overhead**: ~400–600 MB (CUDA context, cuBLAS/cuDNN runtime kernels, and shared libraries).
- **Total Process / Device Memory (`nvidia-smi`)**: **1005.00 MB** (total GPU memory allocated across all OS display processes and the Python runtime).

Because `asyncio.Lock` serializes forward execution, tensor memory remains bounded at 107.13 MB peak allocation regardless of concurrency.

---

## 5. Verification of Pre-Staging Controls

| Control Criterion | Implementation Detail | Verification Status |
| :--- | :--- | :--- |
| **Configurable CORS** | Controlled by `ROOMAI_CORS_ORIGINS` env var (comma-separated origins). Defaults to development fallback (`*`) when unset. | Verified in `api/main.py`. |
| **NEXT_PUBLIC_API_URL** | Configurable via `.env.local` or deployment platform; falls back to `http://localhost:8000`. | Verified in `web/lib/api/client.ts`. |
| **20 MB Application Upload Limit** | Enforced in `api/routers/detect.py`; streams exceeding 20 MB return HTTP 413 `FILE_TOO_LARGE`. | Verified via `test_file_too_large_returns_413`. |
| **Reverse Proxy 25 MB Limit** | Nginx / Traefik / Caddy must configure `client_max_body_size 25M` to avoid premature client disconnects before reaching FastAPI. | Documented as staging prerequisite. |
| **Controlled 503 on Missing Model** | Missing or corrupted model checkpoints return HTTP 503 `MODEL_UNAVAILABLE` with remediation instructions. | Verified via `test_missing_model_returns_503`. |
| **Startup Failure Visibility** | Model loading failures during application lifespan log `CRITICAL` errors and surface on `GET /ready` as `status="degraded"` with `startup_error`. | Verified via `test_readiness_probe_reflects_model_status`. |
| **Frontend Connection** | `RoomApiClient` verifies connectivity and schema matching with active backend. | Verified via automated client health check and static build. |

---

## 6. Regression Check Against Frozen Frontend

1. **Narrative Controller Integrity**: Confirmed exactly **one** `<NarrativeController>` instance exists in `web/app/page.tsx:12`. No duplicate providers.
2. **Asset Paths**: Point cloud assets (`web/public/assets/roomai/exhibit-01/point-cloud/scene_reconstruction.ply` and `web/scene_reconstruction.ply`) exist and are untouched.
3. **Three.js Render Loops**: `DepthExhibitCanvas` and `SceneCanvas` retain `frameloop="demand"`. No `setState()` inside `useFrame()`.
4. **Duplicate API Protection**: `LiveInferenceExhibit` uses `activeRequestIdRef` and `AbortController` to cancel in-flight requests and drop stale responses.
5. **Visual Styling**: RoomAI design system tokens (`--obsidian: #0A0908`, `--charcoal: #151412`, `--stone: #B9B4AA`, `--oxide: #8B4937`) strictly adhered to.

---

## 7. Test Results Summary

1. **Backend Integration Suite (`tests/test_api_detect.py` & `tests/test_coordinates.py`)**: **17 passed in 11.26s** (100% pass)
2. **Frontend Typecheck (`npm --prefix web run typecheck`)**: **0 errors** (`tsc --noEmit` exited 0)
3. **Frontend Production Build (`npm --prefix web run build`)**: **Compiled successfully (Static pages 4/4, `/` size 58.8 kB)**
4. **Latency Benchmark (`scripts/benchmark_latency.py`)**: **80 warm runs completed successfully**
5. **Concurrency & VRAM Benchmark (`scripts/benchmark_concurrency.py`)**: **75 requests completed with 0 failures**

---

## 8. Staging vs Production Readiness Assessment

### A. Local Prototype Readiness: READY
- The frontend and backend run reliably on local hardware (RTX 4060 8GB).
- Model A3 loads cleanly, caches repeat queries in <2ms, and handles sequential uploads within 23–35 ms p95.
- The 2D → 3D projection is clearly documented as provisional ray projection ($Z=3.0\text{m}$).

### B. Staging Readiness: READY FOR DEPLOYMENT
- Containerization configuration is straightforward.
- Health (`/health`) and hardware readiness (`/ready`) endpoints provide telemetry for staging load balancers.
- CORS is configurable via `ROOMAI_CORS_ORIGINS`.
- Payload limits (20MB application level) and failure paths (HTTP 413, 415, 400, 503) are verified.
- Frontend builds cleanly as static Next.js export with configurable backend target.

### C. Production Readiness: NOT YET CERTIFIED (Pending Infrastructure Hardening)
The system is **not yet production-ready**. Full production deployment requires resolving the specific infrastructure blockers identified below.

---

## 9. Final Pre-Staging Deliverables

### Exact Files Changed
1. `api/main.py`: Configured CORS origin restrictions via `ROOMAI_CORS_ORIGINS` environment variable; logged startup errors critically and attached them to the inference engine.
2. `api/schemas.py`: Added `startup_error: Optional[str] = None` to `ReadinessResponse`.
3. `api/routers/health.py`: Updated `GET /ready` probe to evaluate Model A3 warmup and report startup errors.
4. `api/routers/detect.py`: Added `self.startup_error` tracking on initialization and warmup failure.
5. `tests/test_api_detect.py`: Added tests 11, 12, 13 verifying 20MB file rejection (HTTP 413), missing model handling (HTTP 503), and readiness probe diagnostics.
6. `scripts/benchmark_concurrency.py`: Rewrote benchmark to eliminate cache bleed across concurrency levels, pre-generate payloads to isolate CPU encoding time, pre-warm HTTP pipeline, and report detailed VRAM breakdowns.
7. `plans/frontend-backend-integration-report.md`: Updated with scientifically defensible concurrency wording, detailed VRAM interpretation, and three-tier readiness assessment.

### Exact Tests & Results
- `pytest tests/test_api_detect.py tests/test_coordinates.py -v`: **17 passed, 0 failed** in 11.26s.
- `npm --prefix web run typecheck`: **0 errors**.
- `npm --prefix web run build`: **Static generation 4/4 succeeded** (`/` size 58.8 kB, First Load JS 162 kB).
- `python scripts/benchmark_concurrency.py`: **75 requests, 0 failures** across C=1, 2, 4, 8.

### Concurrency Discrepancy Explanation
The previous preliminary numbers (10.91 req/s at C=1 vs 41.77 req/s at C=2) were caused by three benchmark artifacts:
1. **Accidental LRU Cache Hits**: In the initial script, identical seeds were reused across concurrency levels, causing 50% of C=2 requests to hit the in-memory cache and return in 2ms without invoking the GPU.
2. **Cold JIT / Route Warmup**: In C=1, only 5 requests were run and request #1 suffered the 216ms cold initialization of the ASGI/CUDA pipeline, which distorted the small sample average.
3. **Synchronous CPU Image Generation**: PIL JPEG encoding was previously inside the timed batch loop.
In the clean benchmark with isolated caches, pre-warmed pipelines, and pre-generated payloads:
- C=1 achieves **25.15 req/s** (p50 latency 40.35 ms).
- C=2 achieves **29.88 req/s** (p50 latency 53.04 ms).
- C=4 achieves **29.40 req/s** (p50 latency 86.52 ms).
- C=8 achieves **30.07 req/s** (p50 latency 159.24 ms).
Physical throughput plateaus at ~29–30 req/s, confirming that higher concurrency does not produce proportional throughput gains and primarily increases queueing delay.

### Exact VRAM Measurement Interpretation
"107.13 MB Peak VRAM" measures **peak active PyTorch tensor memory (`torch.cuda.max_memory_allocated()`)**, NOT total GPU process memory or total device VRAM. The full breakdown is:
- **Active PyTorch Tensors**: 82.04 MB (`torch.cuda.memory_allocated()`).
- **Peak Active PyTorch Tensors**: 107.13 MB (`torch.cuda.max_memory_allocated()`).
- **PyTorch Reserved Blocks**: 146.00 MB (`torch.cuda.memory_reserved()`).
- **Total Process / Device Memory**: 1005.00 MB (`nvidia-smi` reported device memory including CUDA driver context, cuBLAS/cuDNN workspaces, and OS display manager).

### Whether Staging Deployment Is Now Justified
**Yes, staging deployment is justified.** All operational safeguards, upload limits, error envelopes, and frontend build outputs are verified.

### Remaining Production Blockers
Before full public production deployment, the following must be resolved:
1. **Container GPU Passthrough**: Deploy backend Docker image with NVIDIA Container Toolkit (`--gpus all` or NVIDIA runtime in Kubernetes) so CUDA is available.
2. **Reverse Proxy Configuration**: Ensure reverse proxy (Nginx, Caddy, Traefik, or Cloudflare) sets `client_max_body_size 25M` so 20MB image uploads are not severed before reaching FastAPI.
3. **Strict Origin Enforcement**: Set `ROOMAI_CORS_ORIGINS` to the exact production domain(s) rather than allowing wildcards.
4. **Authentication & Rate Limiting**: Production exposure requires API key authentication or IP rate limiting (e.g. via Redis/TokenBucket) to prevent denial-of-service on the GPU queue.
5. **Process Supervision**: Run FastAPI via a production ASGI server (e.g., Uvicorn under systemd or Gunicorn with Uvicorn worker class).
