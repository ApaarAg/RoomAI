# RoomAI — Engineering Architecture Review

**Phase:** 1–3 (Read-only architecture inspection + architecture review)
**Date:** 2026-10-30
**Reviewer:** Principal Software Architect pass
**Environment:** `tf-gpu` conda env — Python 3.10.20, torch 2.11.0+cu128, ultralytics 8.4.66, transformers 4.49.0, streamlit 1.58.0, pydantic v2, numpy 2.x. CUDA available on **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)**.

---

## 1. Existing Architecture

RoomAI is a Streamlit single-page application organised into clean, narrow-responsibility layers. There is **no hidden cross-cutting coupling**: each layer imports the layer below it and never the reverse.

```
app.py  (Streamlit entrypoint / page config / mock-mode toggle)
  │
  ├── ui/                       PRESENTATION (Streamlit widgets only)
  │     ├── upload_view.py         image upload + validation + state invalidation
  │     ├── analysis_view.py        Detection / Segmentation / 3D / Classification / VLM tabs
  │     ├── design_view.py           generative redesign studio
  │     └── components.py            banners, device sidebar, honesty notices
  │
  ├── services/                COORDINATION (validate, dispatch, render, error contract)
  │     ├── detection_service.py     resolve_detector() + DetectionService.run()
  │     ├── segmentation_service.py  SegmentationService.run()
  │     ├── scene3d_service.py       Scene3DService.run()
  │     ├── classification_service.py
  │     ├── vision_language_service.py
  │     ├── generation_service.py
  │     ├── image_service.py         draw_bounding_boxes / overlay / heatmap / side-by-side
  │     ├── ensemble_service.py        WBF + class-aware NMS fusion
  │     └── specialist_ensemble_service.py
  │
  ├── models/                  INFERENCE (own init, device, predict; no Streamlit dep)
  │     ├── interfaces.py          ABCs + ModelNotImplementedError
  │     ├── registry.py             ModelSpec + ModelRegistry (VRAM budget, LRU eviction)
  │     ├── detection_model.py       CustomObjectDetector + MockObjectDetector
  │     ├── segmentation_model.py    CustomRoomSegmenter + MockRoomSegmenter
  │     ├── scene3d_model.py          CustomScene3DModel (Depth Anything V2 Small) + Mock
  │     ├── classification_model.py    CustomRoomClassifier (stub) + Mock
  │     ├── generation_model.py         CustomDesignGenerator (stub) + Mock
  │     └── vision_language_model.py    CustomVisionLanguageModel (stub) + Mock
  │
  ├── schemas/                 CONTRACTS (Pydantic v2)
  │     ├── requests.py            AnalysisRequest / DesignRequest / VLMQuestionRequest
  │     └── responses.py           BoundingBox / DetectionItem / DetectionResponse / ...
  │
  ├── utils/                   INFRASTRUCTURE (no business logic)
  │     ├── image_validation.py    validate_image_file (size/integrity/EXIF/RGB/dimension)
  │     ├── artifact_manager.py     session-scoped unpredictable artifact paths + retention
  │     ├── gpu_concurrency.py       GPUInferenceGate (BoundedSemaphore, max_concurrent=1)
  │     ├── device.py                 get_device_info / get_optimal_device
  │     ├── timing_utils.py             StageTimer / InferenceTimer / percentiles
  │     ├── logging_config.py            structured stdout logger
  │     └── (training helpers: deepfurniture_loader, masked_detection_loss, ...)
  │
  ├── config/                  CONFIGURATION (YAML)
  │     ├── app.yaml               upload limits, design styles, gpu_concurrency, retention
  │     └── model_config.yaml        per-task checkpoints, thresholds, device, classes
  │
  ├── tests/                   VERIFICATION (pytest)
  └── scripts/                 TRAINING / AUDIT / BENCHMARK scripts (not import-time deps)
```

**Responsibility boundaries — observed and confirmed correct:**

| Layer | Owns | Does NOT own |
|-------|------|--------------|
| UI (`ui/`) | widgets, session-state keys, result display | inference, model init, coordinate math |
| Service (`services/`) | input/RGB normalisation, GPU-gate dispatch, rendering, timing, error→dict contract | neural network weights |
| Model (`models/`) | checkpoint resolution, device selection, CUDA sync, forward pass, box normalisation | Streamlit, rendering |
| Schema (`schemas/`) | field validation, bounds, optional/error states | business rules |
| Config (`config/`) | thresholds, checkpoint paths, device, classes, limits | runtime values |
| Utils (`utils/`) | validation, concurrency, artifacts, logging, timing | model logic |

## 2. Dependency Map (verified by reading every module)

- `app.py` → `ui/*`, `utils/logging_config`
- `ui/*` → `services/*`, `schemas/*`, `utils/image_validation`, `utils/device`, Streamlit
- `services/*` → `models/*`, `schemas/*`, `utils/{gpu_concurrency, artifact_manager, image_service, logging_config, timing_utils}`, `services/{ensemble_service, specialist_ensemble_service, image_service}`
- `models/*` → `schemas/*`, `models/interfaces`, `utils/{device, logging_config}`, torch/ultralytics/transformers
- `schemas/*` → pydantic only
- `utils/*` → torch (lazy/guarded), PIL, numpy, stdlib

**No circular dependencies** were found. `models/registry.py` imports only `utils/logging_config`; it does **not** import the model implementations, so the registry is metadata-only and cannot create a cycle. `services/ensemble_service.py` and `services/specialist_ensemble_service.py` import `models/detection_model` (composition), which is the expected downward edge.

## 3. Architectural Strengths

1. **Strict layering with no import cycles.** UI never calls models directly; models never import Streamlit.
2. **Honest mock/real separation.** Every model has a `Mock*` (deterministic, CPU) and a `Custom*` (real weights) variant; services pick at construction time via `mock_mode`. Responses carry `is_simulated` so the UI can badge simulated output. Real mode **never falls back to mock** on failure (verified by `tests/test_services.py::test_detection_service_mock_and_real` and `test_ui_integration.py`).
3. **Bounded GPU concurrency.** [`utils/gpu_concurrency.py`](../utils/gpu_concurrency.py:59) serialises heavy GPU work through a process-wide `GPUInferenceGate` (`BoundedSemaphore`, capacity 1) appropriate for an 8 GB single card. Mock/CPU work bypasses the gate. `GPUBusyError` is surfaced cleanly to the UI.
4. **Session-scoped, unpredictable artifact paths.** [`utils/artifact_manager.py`](../utils/artifact_manager.py:69) writes annotated images to a per-call random subdirectory under the controlled `artifacts/` root — no user-controlled output filename, no path traversal, retention + active-grace protection.
5. **Config/registry agreement.** `config/model_config.yaml` primary checkpoints (`A3`, `B2`, `C2`, `yolo11n-seg`, `depth_anything_v2_small`) match `models/registry.py` primary specs and exist on disk.
6. **Honesty disclaimers.** Depth is consistently labelled `relative_depth_only` with a "not metric distance" disclaimer; segmentation carries a COCO-pretrained scope notice; generation carries a feasibility notice.

## 4. Defects Discovered (architecture-relevant)

| ID | Severity | Layer | Defect | Status |
|----|----------|-------|--------|--------|
| DEF-1 | High | UI/state | Image-change invalidation in `ui/upload_view.py` omitted `classification_result`, `vlm_result`, `design_result` → stale results could display for a new image | **Fixed** |
| DEF-2 | High | UI/state | Clear button called `st.rerun()` *inside* the clear `for` loop → only the first key was ever cleared before rerun | **Fixed** |

No circular dependencies, dead code, global mutable inference state, or config/registry disagreements were found. The single global mutable state (`GLOBAL_MODEL_REGISTRY`, module-level `_YOLO_CACHE`) is the intended YOLO weight cache and is safe for the single-process Streamlit model.

## 5. Recommendations (next engineering improvements, not actioned)

- The `Mock*`/`Custom*` class split duplicates the per-class-threshold validation block between `CustomObjectDetector` and `MockObjectDetector`; a shared validator would remove ~30 lines of duplication without changing behaviour.
- `models/registry.py` `ModelSpec` metadata is declarative only; no runtime path enforces that a loaded checkpoint's `nc` matches `len(classes)`. A lightweight assertion in `CustomObjectDetector.load_model` would catch weight/config mismatches early.
- `services/ensemble_service.py` recomputes cluster centres via simple averaging rather than the WBF coordinate fusion used in `fuse_cluster_wbf`; this is a known simplification, not a defect.