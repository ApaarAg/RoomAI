# RoomAI — Concurrency & GPU Benchmarks

**Date:** 2026-10-30
**Environment:** `tf-gpu` (Python 3.10.20), NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)
**Interpreter:** `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`
**Image:** 1600×1200 RGB (real DeepFurniture test image)
**Detector:** Model A3 (Indoor, 800px, 12-class) — production default

---

## Methodology

A dedicated benchmark script ran real-mode `DetectionService.run` through the
`GPUInferenceGate` (capacity 1) with 1, 2, and 3 concurrent threads. Each trial warmed the
YOLO cache first so the measurement reflects **steady-state**, not cold start. The wall
clock, per-request `queue_wait_ms`, per-request `inference_time_ms`, and CUDA
`memory_allocated`/`memory_reserved` were captured. Threads were joined with a 120s timeout
to catch any deadlock; none occurred.

The gate serialises heavy GPU ops with a `threading.BoundedSemaphore(1)`, so under N
concurrent requests exactly one runs at a time and the rest queue. This is the conservative,
measured-safe choice for a single 8 GB card and prevents the VRAM contention that unbounded
concurrency would cause.

---

## Results

### 1 concurrent request
| Metric | Value |
|--------|------|
| wall total | 98.5 ms |
| successful | 1/1 |
| queue_wait_ms (mean / max) | 0.0 / 0.0 |
| infer_ms (mean / max) | 88.9 / 88.9 |
| VRAM allocated | 100.0 MiB |
| VRAM reserved | 208.0 MiB |
| objects detected | 9 |
| device | cuda:0 |

### 2 concurrent requests (serialised)
| Metric | Value |
|--------|------|
| wall total | 162.9 ms |
| successful | 2/2 |
| queue_wait_ms (mean / max) | 36.2 / 72.5 |
| infer_ms (mean / max) | 76.6 / 81.8 |
| VRAM allocated | 100.0 MiB |
| VRAM reserved | 208.0 MiB |

Per request:
- req0: queue=0.0 ms, infer=71.3 ms, 9 objects, cuda:0
- req1: queue=72.5 ms, infer=81.8 ms, 9 objects, cuda:0

### 3 concurrent requests (serialised)
| Metric | Value |
|--------|------|
| wall total | 246.7 ms |
| successful | 3/3 |
| queue_wait_ms (mean / max) | 79.6 / 158.1 |
| infer_ms (mean / max) | 78.6 / 81.1 |
| VRAM allocated | 100.0 MiB |
| VRAM reserved | 208.0 MiB |

Per request:
- req0: queue=0.0 ms, infer=79.2 ms, 9 objects, cuda:0
- req1: queue=80.6 ms, infer=75.5 ms, 9 objects, cuda:0
- req2: queue=158.1 ms, infer=81.1 ms, 9 objects, cuda:0

---

## Interpretation

- **Throughput scales linearly with concurrency under the gate** because the gate serialises
  GPU work: wall ≈ N × per-request infer time. This is the intended trade-off for a single
  8 GB card — we trade peak concurrency for **guaranteed VRAM safety and 100% success**.
- **VRAM is flat** at 100 MiB allocated / 208 MiB reserved across 1, 2, and 3 concurrent
  requests. The gate prevents the VRAM growth/OOM that unbounded concurrency would cause.
- **`queue_wait_ms` is reported separately** from `inference_time_ms` so the UI can show the
  user whether a slow response was due to GPU queueing or to the model itself. The timing
  metrics dict now carries `queue_wait_ms`.
- **No deadlocks**: all threads joined within the 120s timeout; the gate releases its permit
  in a `finally` block even when the inference callable raises (verified by
  `test_gpu_gate_releases_permit_on_exception`).

---

## Failure recovery test

`test_detection_service_real_mode_missing_checkpoint_no_silent_mock` confirms that a missing
production checkpoint in real mode:
- returns `success=False`, `response=None`,
- never silently substitutes mock output,
- **releases the GPU gate permit** (the test asserts `gate.active == 0` after the failure),
so a failed inference cannot deadlock subsequent requests.

---

## Concurrency limitations

- The default capacity is **1**. Raising it (e.g. to 2) would require re-measuring VRAM
  headroom with the *largest* production checkpoint (A3 at 800px) and the segmentation/depth
  models loaded concurrently; on the 8 GB RTX 4060 this is **not recommended** without
  evidence of headroom.
- The gate is process-wide and single-machine only; it does not coordinate across separate
  Streamlit worker processes (Streamlit's default server is single-process, so this is fine
  for the intended deployment).
- Mock mode bypasses the gate because mock inference is CPU-only and deterministic; this
  keeps the UI responsive during development.

---

## Reproducing

```powershell
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pytest tests/test_deployment_hardening.py -v -p no:cacheprovider
```

(13 regression tests, including the concurrency-gate tests; all PASS.)

The ad-hoc concurrency benchmark script (`reports/_concurrency_benchmark.py`) was run during
this cycle and then removed to avoid leaving a stray script; the numbers above are its
captured output.