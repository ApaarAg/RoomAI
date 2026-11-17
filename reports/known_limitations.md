# RoomAI — Backend Known Limitations & Technical Constraints

**Date:** October 29, 2026  
**Status:** RELEASE CANDIDATE AUDIT  
**Scope:** Backend inference pipelines, computer-vision models, geometry services, and environment constraints.

---

## 1. Executive Summary

This document provides a transparent, defensible inventory of the known technical limitations, architectural trade-offs, and behavioral constraints in the RoomAI backend release candidate. These entries are documented to guide frontend integration, set user expectations, and define the roadmap for subsequent releases.

---

## 2. Detailed Technical Limitations

### 2.1 Depth Estimation Semantics (Relative vs. Metric Depth)

* **Current Implementation:** Depth Anything V2 Small (`depth_anything_v2_vits.pt`), loaded via `services/scene3d_service.py`.
* **Nature of Output:** Inverse relative depth (`relative_depth_only: true`). Pixel values correspond to relative depth ordering within the monocular image frame (0.0 = furthest surface, 1.0 = closest surface, normalized from raw inverse depth).
* **Limitation:** The model does **not** output absolute metric depth in physical units (meters or feet). It cannot directly compute physical room dimensions, square footage, or true object clearances without external metric scale anchors (such as camera intrinsics $K$, ground-plane homography, or known fiducial marker heights).
* **Impact on Frontend:**
  * Depth map visualizers must treat heatmaps as relative proximity fields.
  * Spatial clearances (e.g., "Will this 2-meter couch fit here?") cannot be authoritatively verified using raw depth output alone.
* **Mitigation & Roadmap:**
  * For relative ordering and occlusion reasoning, the existing output is reliable (sub-200 ms latency on GPU).
  * Future releases will integrate floor-plane homography or metric metric-depth models (e.g., Metric3D v2, UniDepth) when calibrated focal lengths are available in EXIF metadata.

---

### 2.2 Instance Segmentation Domain Semantics (COCO vs. Architectural Boundaries)

* **Current Implementation:** YOLO11n-seg (`yolo11n-seg.pt`), loaded via `services/segmentation_service.py`.
* **Nature of Output:** General-purpose object instance segmentation trained on the 80-class MS COCO dataset (e.g., chair, couch, bed, dining table, person, bottle).
* **Limitation:** The segmentation model does **not** recognize domain-specific architectural boundary polygons such as walls, ceilings, floor surfaces, molding, or room cavities.
* **Impact on Frontend:**
  * Calling the segmentation service will produce high-quality pixel masks for common indoor furniture items present in COCO.
  * It cannot be used to generate architectural CAD floorplans, 3D room boundary wireframes, or paint surface masks without custom architectural segmentation weights.
* **Mitigation & Roadmap:**
  * Primary object detection for architectural and exterior structures is correctly handled by specialized YOLO11s models (Group B: 10 exterior classes; Group C: 8 parking classes).
  * Future architectural surface masking will integrate ADE20K/ScanNet-trained semantic segmentation for wall/floor/ceiling parsing.

---

### 2.3 Rare Class Sample Representation (Porch Class Validation Power)

* **Current Implementation:** Exterior Detector Model B2 (`exterior_detector_b2_yolo11s_best.pt`).
* **Nature of Output:** 10 exterior architectural classes: `building`, `door`, `window`, `roof`, `balcony`, `chimney`, `garage_door`, `porch`, `stairs`, `fence`.
* **Limitation:** In the standardized validation split, the `porch` class contains only $N=24$ ground-truth annotated instances.
* **Impact on Frontend:**
  * While `porch` detection functions at runtime with a calibrated threshold ($\tau = 0.25$), its evaluation metrics (precision, recall, mAP) have wider statistical confidence intervals than dense classes such as `building` ($N=642$) or `window` ($N=2,180$).
* **Mitigation & Roadmap:**
  * Per-class threshold configuration (`config/model_config.yaml`) explicitly accounts for class imbalance with calibrated operating points.
  * Porch specialist ensemble architectures and multi-dataset expansion (Phase 19/22) are maintained as quarantined research tracks to expand porch support once validation splits reach $N \ge 150$.

---

### 2.4 Quarantined Experimental Models (B5, B6, B7, and C1)

* **Current Status:** Strict non-production quarantine (`enabled: false`, `status: "experimental_candidate"`).
* **Audited Checkpoints:**
  * `exterior_detector_b5_yolo11s_clahe_best.pt` (Phase 16 CLAHE augmentation)
  * `exterior_detector_b6_yolo11s_best.pt` (Phase 17 full backbone fine-tuning)
  * `exterior_detector_b7_yolo11s_masked_best.pt` (Phase 20 masked-loss pilot)
  * `parking_detector_c1_yolo11n_best.pt` (Nano fallback candidate)
* **Limitation:** None of these checkpoints are promoted to production default. Model B2 remains the sole production-ready default for exterior detection; Model A3 is the production-ready default for indoor detection; Model C2 is the production-ready default for parking detection.
* **Impact on Frontend:**
  * Frontend requests must not attempt to invoke B5, B6, B7, or C1 without explicit override flags in developer/research modes.
  * Production API routes systematically route to B2, A3, and C2.
* **Mitigation & Safeguard:**
  * Enforced via automated tests (`tests/test_specialist_ensemble.py::test_production_default_remains_b2`).
  * Any attempt to load an unapproved candidate triggers an explicit configuration validation warning.

---

### 2.5 Generative Staging and VLM Narrative Modules (Mocked / Stubbed)

* **Current Implementation:** `services/vlm_service.py` and `services/generation_service.py`.
* **Nature of Output:** Stubbed interfaces returning deterministic mock payloads or simulated room redesigns unless external multimodal LLM API keys (e.g., Gemini 1.5 Pro) or local Stable Diffusion weights are provisioned.
* **Limitation:** The backend does **not** currently bundle standalone local weights for high-resolution generative room inpainting or 10B+ parameter vision-language models due to the 8 GB VRAM budget constraint.
* **Impact on Frontend:**
  * Frontend features displaying automated property narrative summaries or generative restaging must clearly identify outputs as simulated/mock unless live API credentials are configured.
* **Mitigation & Roadmap:**
  * The interface schemas (`VLMAnalysisResponse`, `StagingResponse`) are fully typed and frozen. When external API keys or lightweight quantized models are attached, zero frontend schema modifications will be needed.

---

### 2.6 Concurrency and Multi-Request VRAM Management

* **Current Implementation:** PyTorch CUDA runtime on single NVIDIA GPU (`cuda:0`).
* **VRAM Allocation:** Single inference pass requires approximately 68 MB to 140 MB of dynamic VRAM. Peak reserved memory across all loaded models remains $< 600$ MB against the 8,192 MB capacity.
* **Limitation:** The backend does not implement an asynchronous Celery/Redis multi-worker queue. Simultaneous overlapping GPU inference requests from multiple concurrent HTTP threads will contend for the single CUDA stream and serialize at the PyTorch lock.
* **Impact on Frontend:**
  * If the frontend sends bursts of concurrent image analysis requests, response latency will scale linearly ($T_{\text{total}} \approx N \times 350\text{ ms}$).
* **Mitigation & Roadmap:**
  * For local desktop / single-tenant beta deployment, the current sub-350 ms latency easily sustains interactive UI updates.
  * For multi-tenant production hosting, the deployment recommendation specifies running behind a FastAPI ASGI worker pool with an asynchronous job queue (e.g., Redis + Celery/RQ) with max 2 concurrent GPU workers.

---

## 3. Summary Matrix

| Limitation ID | Component | Severity | Real-World Impact | Workaround / Fallback |
| :--- | :--- | :---: | :--- | :--- |
| **LIM-01** | Depth Anything V2 | Medium | Relative depth only; no metric measurements. | Use for visual heatmaps & depth ordering only. |
| **LIM-02** | YOLO11n-seg | Low | COCO categories; no wall/floor surface polygons. | Primary room objects detected via YOLO11s A3/B2. |
| **LIM-03** | Exterior B2 | Low | Porch class has $N=24$ validation instances. | Operating threshold calibrated at $\tau=0.25$. |
| **LIM-04** | Checkpoints B5–B7, C1 | Info | Quarantined; not available in default production routes. | Production B2, A3, C2 are verified and active. |
| **LIM-05** | VLM / GenAI Staging | Medium | Interface stubs; requires external API keys for live outputs. | Mock mode isolates UI from network dependencies. |
| **LIM-06** | GPU Concurrency | Low | Single-stream CUDA execution; latency serializes on burst. | Single-tenant UI; queueing required for multi-tenant. |

