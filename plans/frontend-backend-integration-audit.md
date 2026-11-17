# Phase 0: Frontend ↔ Backend Integration Audit

**Date**: November 16, 2026  
**Auditor**: Senior Systems & Frontend-Backend Integration Engineer  
**Status**: Completed  

---

## 1. Executive Summary

This audit establishes the baseline boundary between the RoomAI frontend (`web/`, Next.js 15, Three.js 0.187, React 19) and the backend API (`api/`, FastAPI 1.4.0, PyTorch 2.6.0+cu124 on NVIDIA GeForce RTX 4060). 

The primary finding is:
**Currently, zero active API calls exist between the frontend and the backend.**
The frontend operates exclusively on static pre-rendered pipeline assets (`web/public/assets/roomai/pipeline/...`) and development mock geometries. Conversely, the FastAPI backend exposes health, file uploads, and async 3D reconstruction jobs, but **completely lacks a synchronous, low-latency object detection endpoint** exposing the audited Model A3 (`checkpoints/indoor_detector_a3_yolo11s_800px_best.pt`).

---

## 2. Current Frontend API Status

- **API Utilities / Client**: None present. There is no `web/lib/api/` directory or centralized fetch/axios wrapper.
- **Fetch / Axios Calls**: Searching `web/` for `fetch(`, `axios.`, and `XMLHttpRequest` reveals zero runtime calls to backend services. All assets are loaded via standard HTML `<img>` elements or Three.js texture loaders referencing relative URLs in `public/assets/`.
- **Upload Components**: No interactive image upload input or drag-and-drop zone currently exists in the user-facing sections (`Hero`, `TransformSection`, `EngineSection`, `ExploreSection`, `EvidenceSection`).
- **Result Viewers**: `ResultViewer.tsx` displays pre-baked static WebP sequences (`01_rgb_image.webp` through `06_ply_glb_export_concept.webp`) with GSAP ScrollTrigger scrubbing and relative depth reliefs via `DepthExhibitCanvas`.
- **Loading States**: Static or simulated only (e.g. `viewerReady`, `assetFailed` image load events).
- **Environment Variables**: `web/` lacks an `.env` or `.env.example` defining `NEXT_PUBLIC_API_URL`.
- **TypeScript API Interfaces**: No shared DTOs/types defining backend response models exist in `web/`.

---

## 3. Current Backend API Status

- **Application Framework**: FastAPI 1.4.0 mounted in `api/main.py` with standard CORS (`allow_origins=["*"]`) and response time middleware (`X-Process-Time-Ms`).
- **Existing Routers**:
  - `health` (`GET /health`, `GET /ready`): Reports service status, GPU memory, and model readiness.
  - `uploads` (`POST /api/v1/uploads`): Stores images with SHA-256 deduplication and dimensions check.
  - `jobs` (`POST /api/v1/jobs/reconstruct`, `GET /api/v1/jobs/{id}`, `DELETE /api/v1/jobs/{id}`): Queues 3D room reconstruction jobs.
  - `models` (`GET /api/v1/models`): Lists registered capabilities.
  - `experimental` (`POST /api/v1/experimental/vqa`, `POST /redesign`): Returns 501 Not Implemented.
- **Missing Detection Route**: Despite `services/detection_service.py` containing a `DetectionService` and `models/detection_model.py` containing `CustomObjectDetector`, FastAPI has no `/api/v1/detect` route.
- **Model Lifecycle**: Currently, `DetectionService` instantiates on demand without a persistent server lifespan hook preloading Model A3 into GPU VRAM at startup.

---

## 4. Existing Contracts & Discrepancies

- **Pydantic Schemas**:
  - `schemas/responses.py` contains `DetectionResponse`, `DetectionItem`, and `BoundingBox`.
  - `BoundingBox` uses normalized floats `x_min, y_min, x_max, y_max` in $[0.0, 1.0]$.
  - The prompt specifies a client-friendly contract providing normalized and/or pixel coordinates along with image dimensions and inference metadata (`model_id`, `latency_ms`).
- **Discrepancy**:
  - `schemas/responses.py` is tailored for internal services rather than clean decoupled frontend consumption (lacks client request ID, structured image metadata envelope, and explicit pixel coordinates).
  - A unified, strongly-typed contract must be codified in `api/schemas.py` and mirrored in `web/lib/api/types.ts`.

---

## 5. Mock Data Locations

1. `web/components/3d/dev/fixtureConfig.ts`:
   - Defines `USE_DEVELOPMENT_FIXTURES = process.env.NODE_ENV !== "production"`.
   - Hardcoded geometric bounds for synthetic developmental rooms (`width: 3.2, height: 2.4, depth: 3.6`).
2. `web/components/3d/SpatialStructureObject.tsx`:
   - Contains explicit flag `SPATIAL_STRUCTURE_SOURCE_VERIFIED = false` and deliberately renders `null` to avoid fabricating unverified geometry.
3. `web/components/3d/SpatialAnnotationObject.tsx`:
   - Contains hardcoded fixture position `[-1.42, 1.03, 0.12]`.
4. `models/detection_model.py`:
   - Contains `MockObjectDetector` returning hardcoded synthetic boxes for testing.

---

## 6. Duplicate / Dead Request Paths

- No dead HTTP requests are currently firing because the frontend does not yet initiate network calls.
- In `api/routers/jobs.py`, polling endpoints exist for long-running asynchronous reconstructions, but they are uncalled by any frontend component.
- The direct detection path must be synchronous and low-latency, completely separate from the asynchronous reconstruction job queue.

---

## 7. Integration Blockers & Action Plan

| Blocker | Root Cause | Resolution Strategy |
| :--- | :--- | :--- |
| **No Detection Endpoint** | FastAPI lacks `/api/v1/detect` router. | Create `api/routers/detect.py` mounting `POST /api/v1/detect` supporting direct multipart image upload. |
| **Model Cold-Start Latency** | Model A3 is not preloaded or warmed up at server boot. | Implement FastAPI `lifespan` handler preloading Model A3 and running 1 warmup pass. |
| **No Frontend Client** | Frontend has no API library or environment config. | Create `web/lib/api/client.ts` with `NEXT_PUBLIC_API_URL` and `web/.env.example`. |
| **No UI Upload Trigger** | Frontend has no interactive upload mechanism. | Add an interactive "Live Room Inference" panel to `ResultViewer`/`ExploreSection` allowing custom image analysis and sample presets. |
| **Coordinate Semantics** | YOLO normalized $[0, 1]$ coordinates vs Three.js spatial coordinates. | Implement typed coordinate unprojection transforming 2D detections into 3D SpatialAnnotationObjects. |
| **Stale Response Hazards** | Asynchronous network responses can race if user submits multiple images. | Implement `AbortController` cancellation and monotonic request tracking. |
