# ROOMAI Phase 4.3 — FastAPI HTTP Test Report

## Root cause of the stalled TestClient run

The hang was reproduced with both the RoomAI app and a minimal FastAPI app. The Python stack stopped inside `asyncio.windows_events.ProactorEventLoop._make_self_pipe()` while `socket._fallback_socketpair()` was waiting in `socket.accept()`. This happens when AnyIO starts Starlette's synchronous `TestClient` portal on Windows.

The same `/health` request completed with HTTP 200 when the probe ran outside the restricted sandbox. The sandbox blocks or stalls the local socket operation required to start the Proactor event loop. That isolates the observed stall to the execution sandbox, not to RoomAI's health route, application lifespan, database initialization, inference, or an HTTPX/Starlette mismatch.

Test environment: Python 3.10.20, FastAPI 0.134.0, Starlette 1.0.1, HTTPX 0.28.1, AnyIO 4.13.0, Pydantic 2.13.5, pytest 9.0.3. TestClient passed with this installed combination when local socket setup was permitted.

## Startup and background-work findings

- `api.main:app` has no FastAPI lifespan/startup/shutdown handler.
- `services.job_service.GLOBAL_JOB_SERVICE` starts its daemon worker during module import. It recovers interrupted jobs at construction time.
- `services.job_store.GLOBAL_JOB_STORE` opens/configures SQLite and initializes tables during module import.
- `GLOBAL_UPLOAD_SERVICE` creates the upload directory during module import.
- Health is a synchronous liveness response and does not call Torch, CUDA, SQLite, or a model. `/ready` calls device reporting and model-registry discovery.
- The global worker is not shut down through an application lifecycle hook. That is an operational/test-isolation improvement to consider separately; it was not the cause of this socketpair stall and was not changed in this validation pass.

## Tests run

Command, using the RoomAI environment with local socket access permitted:

```text
python -m pytest tests/test_api_endpoints.py tests/test_phase4_3_http_api.py tests/test_reconstruction_contract.py -q
```

Result after the final validator-report persistence change: **24 passed in 24.13 seconds**.

The route tests use service fakes for the asynchronous reconstruction boundary, so they validate HTTP request/response and persistence handoff without loading inference models. A separate real `JobService` test injects a deterministic adapter exception and verifies that the worker writes a terminal `FAILED` state and its error details. The existing real-model end-to-end tests were left intact and were not replaced with mock tests.

| Surface | Result | Evidence |
|---|---|---|
| `/health` | PASS | Existing endpoint test, HTTP 200/schema assertions |
| `/ready` | PASS | Existing readiness endpoint test, hardware/model fields |
| Model discovery | PASS | Existing registry/capability endpoint test |
| Upload and metadata | PASS | Existing upload success/metadata coverage and new workflow |
| Invalid upload extension/content | PASS | Existing endpoint tests and invalid-input test |
| Job creation | PASS | Mocked adapter/service; 202, queued state, upload path and camera intrinsics handoff asserted |
| Job status | PASS | Queued and terminal failure states asserted |
| Job results | PASS | Completed result envelope asserted |
| Artifact retrieval | PASS | GLB response status, media type and bytes asserted; real GLBs independently validated in the Khronos report |
| Invalid job inputs | PASS | Empty IDs (422), unknown model (400), unknown upload/status/results (404), unsupported export (400) |
| Submission/service failure | PASS | HTTP 500 error envelope asserted |
| Worker inference failure | PASS | Actual worker with isolated SQLite store records `FAILED` and exception details |
| Geometry and output contract | PASS | Reconstruction/export contract tests included in the same run |

The HTTP artifact route test uses a tiny route fixture; it does not claim that fixture is a valid production GLB. Binary validation is separately proven on real pipeline GLBs in `ROOMAI_GLB_VALIDATION_REPORT.md`.

## Remaining lifecycle risk

The app's import-time global worker and SQLite initialization are not managed by a lifespan context. They should eventually be moved behind an explicit start/stop lifecycle or injected application services to improve multi-process server behavior and test isolation. No lifecycle rewrite was needed to resolve this test stall or to complete the requested endpoint coverage.
