# RoomAI — Functional Correctness Report

**Phase:** 2 (audit every implemented user-facing feature)
**Date:** 2026-10-30
**Environment:** `tf-gpu` conda env — torch 2.11.0+cu128, CUDA available, RTX 4060 (8 GB).
**Test baseline:** 143 passed → **146 passed** after fixes (full suite, 18.58s).

---

## Audit method

Each feature was traced from the Streamlit widget (`ui/`) → service (`services/`) → model (`models/`) → schema (`schemas/`), then exercised with the real production checkpoint on CUDA via `tests/test_ui_integration.py` (already in-repo) and a new real-inference smoke benchmark (`scripts/smoke_test_real_inference.py`). Mock vs real behaviour was verified separately.

## 1. Image upload & validation — ✅ Correct

[`utils/image_validation.py`](../utils/image_validation.py:32) `validate_image_file`:
- **Supported formats** — validated against `allowed_extensions` (`.jpg/.jpeg/.png/.webp`). ✅
- **File size** — 0-byte and over-limit rejected with clear messages. ✅
- **Corrupt files** — `Image.verify()` + decode guard; returns "corrupted/unreadable". ✅
- **EXIF orientation** — `ImageOps.exif_transpose(img)` applied before RGB conversion. ✅
- **RGB / grayscale / transparency** — non-RGB modes converted to RGB; RGBA alpha flattened by `convert("RGB")`. Verified by `tests/test_ui_integration.py::test_image_channel_normalization`. ✅
- **Decompression bombs** — caught (`Image.DecompressionBombError`); verified by `test_decompression_bomb_rejection`. ✅
- **Dimensions** — over `max_dimension` is **downsampled** (preserves aspect, LANCZOS) rather than rejected. ✅
- **Image replacement / stale invalidation** — `ui/upload_view.py` derives `active_id = f"{name}_{w}x{h}"` and clears all per-image result keys when it changes. **(Fixed this cycle — see DEF-1/DEF-2.)**

## 2. Object detection — ✅ Correct (A3 indoor / B2 exterior / C2 parking)

[`models/detection_model.py`](../models/detection_model.py:39) `CustomObjectDetector`:
- **Production model selection** — `config/model_config.yaml` + `models/registry.py` agree: Indoor → **A3** (`indoor_detector_a3_yolo11s_800px_best.pt`, `is_primary`), Exterior → **B2** (`exterior_detector_b2_yolo11s_best.pt`), Parking → **C2** (`parking_detector_c2_yolo11s_pilot_best.pt`). ✅
- **Preprocessing** — RGB conversion in service; `imgsz` auto-derived (800 for A3, 640 otherwise). ✅
- **Coordinate transform** — YOLO `xyxy` pixel boxes → normalised `[0,1]` by `img_w/img_h`, clamped, `x_max>=x_min` enforced. ✅
- **Class mapping** — `r.names[cls_id]`; per-class thresholds (experimental) lower-cased for lookup. ✅
- **Confidence** — `float(box.conf[0])`, filtered by effective threshold. ✅
- **Postprocessing** — empty results return `total_objects=0` cleanly (verified `test_empty_results_and_error_handling`). ✅
- **Visualization** — `draw_bounding_boxes` with label pills + confidence %. ✅
- **Response schema** — `DetectionResponse` with `model_name`, `checkpoint`, `inference_time_ms`, `device`, `is_simulated=False`. ✅

Real-inference smoke results (single held-out image, warm cache, CUDA):

| Model | Device | is_simulated | objects | warm inference (ms) | VRAM Δ (MB) |
|-------|--------|--------------|---------|----------------------|-------------|
| Indoor A3 (800px) | cuda:0 | false | 6 | 2649* | 68 |
| Exterior B2 | cuda:0 | false | 70 | 302 | 36 |
| Parking C2 | cuda:0 | false | 1 | 190 | 36 |

\* A3 "warm" figure includes a first-call CUDA warm-up/jit path on the 800px input; the model-reported `inference_time_ms` (CUDA-synced) was **523.61 ms**. B2/C2 steady-state matches the model-reported times (236/142 ms).

## 3. Segmentation — ✅ Correct (generic pretrained; clearly distinguished)

[`models/segmentation_model.py`](../models/segmentation_model.py:28) `CustomRoomSegmenter`:
- Uses **pretrained YOLO11n-seg** (`checkpoints/yolo11n-seg.pt`) — **generic COCO instance segmentation, NOT RoomAI-trained.**
- **Mask resize** — `cv2.resize(..., INTER_NEAREST)` to original `h,w` when mask dims differ. ✅
- **Alignment** — composite mask shape `(h,w)`; overlay resizes defensively. ✅
- **Overlay** — `create_segmentation_overlay` alpha-blends per-uid colour where mask>0. ✅
- **Coverage** — `pixel_count / (w*h) * 100`, clamped `[0,100]`. ✅
- **Limitation notice** — `render_segmentation_limitation_notice` clearly states COCO scope and that architectural boundaries are unsupported. ✅
- Real run: 2 masks, 412.97 ms warm, +11.62 MB VRAM. `is_simulated=false`.

## 4. Depth estimation — ✅ Correct (relative only; never metric)

[`models/scene3d_model.py`](../models/scene3d_model.py:21) `CustomScene3DModel` (Depth Anything V2 Small):
- **Loading** — `AutoImageProcessor`/`AutoModelForDepthEstimation.from_pretrained`; TF suppressed (`TRANSFORMERS_NO_TF=1`). ✅
- **Normalization** — processor handles preprocessing; output bicubic-interpolated back to `(orig_h, orig_w)`. ✅
- **Output dimensions** — match input exactly (verified `test_custom_scene3d_real_inference_shape_and_range`: `(240,320)`). ✅
- **Range** — min-max normalised to `[0,1]`; `nan_to_num` guards non-finite. ✅
- **Visualization** — `create_depth_heatmap` (plasma-like gradient). ✅
- **Honesty** — `relative_depth_only=True`, disclaimer explicitly says "Not metric distance in meters". ✅
- Real run: `depth_min=0.4727`, `depth_max=8.233`, peak VRAM 389.3 MB, 363 ms warm, load 3297 ms (transformers first-load).

## 5. Classification / VLM / Generation — correctly classified as NOT implemented (stubs)

These three features have **no trained weights**; the `Custom*` classes raise `ModelNotImplementedError` on `load_model`/`predict`, and the `Mock*` classes produce deterministic simulated output. The UI correctly surfaces `is_implemented=False` with a "Model Implementation Pending" alert when real mode is selected without weights.

| Feature | Real impl | Mock impl | UI behaviour |
|----------|-----------|------------|--------------|
| Style classification | `CustomRoomClassifier` → `ModelNotImplementedError` | `MockRoomClassifier` (deterministic) | real → "implementation pending" alert; mock → simulated badge + result |
| Vision Q&A (VLM) | `CustomVisionLanguageModel` → `ModelNotImplementedError` | `MockVisionLanguageModel` (keyword-based) | real → "implementation pending"; mock → simulated answer |
| Generative redesign | `CustomDesignGenerator` → `ModelNotImplementedError` | `MockDesignGenerator` (PIL enhancement) | real → "implementation pending"; mock → simulated concept + honesty notice |

No feature is presented as operational when its real weights are absent — the `is_implemented` contract is honoured throughout.

## 6. State & data integrity — ✅ Correct (after fix)

- **Image change invalidates all results** — `active_image_id` mismatch clears all 9 result keys **(DEF-1 fixed: previously omitted classification/vlm/design)**.
- **Clear button clears all + reruns once** — **(DEF-2 fixed: `st.rerun()` was inside the pop loop)**.
- **Task-group switch** — detection results are namespaced `detection_result_{indoor,exterior,parking}` so switching groups does not show another group's boxes.
- **Failed inference** — services return `success=False` with `response=None`; UI shows error + retry; no stale success survives.
- **File collisions** — `new_session_artifact_root` uses a 22-char random token per call; `new_artifact_path` adds a random filename; no overwrite risk.
- **Cross-session** — results live in Streamlit `session_state` (per-browser), not shared.

## 7. Defects found and fixed

| ID | Severity | Defect | Root cause | Fix | Regression test |
|----|----------|--------|-------------|-----|-----------------|
| DEF-1 | High (data integrity) | Stale classification/VLM/design results displayed for a new image | `ui/upload_view.py` invalidation lists omitted those 3 keys | Added `classification_result`, `vlm_result`, `design_result` to both clear lists | `tests/test_state_invalidation.py::test_inference_result_keys_are_complete`, `test_image_change_invalidates_all_result_keys` |
| DEF-2 | High (data integrity) | Clear button only cleared `current_image` before rerun | `st.rerun()` nested inside the `for k in [...]` pop loop | Dedented `st.rerun()` to the loop's level | `tests/test_state_invalidation.py::test_clear_button_clears_all_and_reruns_once` (AST-based) |

## 8. Model integrity (Phase 10) — ✅ Verified

- A3, B2, C2 remain the production primaries (config + registry agree; checkpoints present on disk).
- B5/B6/B7 remain `enabled: false` / `status: experimental_candidate`; **not** promoted.
- Segmentation = generic pretrained YOLO11n-seg (no RoomAI-trained seg weights claimed).
- Depth = relative only; never described as metric.
- No checkpoints/weights/datasets/labels were modified.