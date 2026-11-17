# RoomAI — Final Engineering Validation

**Date:** 2026-10-30
**Environment:** `tf-gpu` conda env — Python 3.10.20, torch 2.11.0+cu128, ultralytics 8.4.66, transformers 4.49.0, streamlit 1.58.0, pydantic v2, numpy 2.x. CUDA on **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)**, 24 GB RAM.
**Scope:** Engineering correctness/stability/maintainability. Public deployment, auth, Cloudflare, rate limiting, and production infrastructure are explicitly **out of scope** (per instructions).

---

## 1. Existing architecture (verified)

Single-process Streamlit app with clean, acyclic, downward-only layering:
`app.py → ui → services → models → schemas`; `utils` is infrastructure-only; `config` is declarative YAML. No circular dependencies, no Streamlit imports inside the model layer, no inference logic duplicated in the UI. Full detail in `reports/engineering_architecture_review.md`.

## 2. Defects discovered

| ID | Severity | Defect | Root cause |
|----|----------|--------|-------------|
| DEF-1 | High (data integrity) | Stale `classification_result`/`vlm_result`/`design_result` displayed for a new image after switching images | `ui/upload_view.py` invalidation lists omitted those 3 keys |
| DEF-2 | High (data integrity) | Clear-image button only cleared `current_image` before rerun (remaining keys survived) | `st.rerun()` nested inside the `for k in [...]` pop loop |

No other Critical/High defects were found across the audited layers (image validation, detection A3/B2/C2, segmentation, depth, error handling, state integrity, GPU/concurrency, code quality).

## 3. Root causes

Both defects stem from a single module (`ui/upload_view.py`) maintaining an incomplete enumeration of per-image session-state result keys, plus an indentation bug in the Clear-button handler that short-circuited the clear loop.

## 4. Corrections implemented

- **DEF-1:** Added `classification_result`, `vlm_result`, `design_result` to **both** the image-change invalidation list and the Clear-button clear list in [`ui/upload_view.py`](../ui/upload_view.py).
- **DEF-2:** Dedented `st.rerun()` out of the clear `for` loop so all keys are popped before the script reruns.

## 5. Files changed

| File | Change |
|------|--------|
| `ui/upload_view.py` | Added 3 result keys to image-change invalidation list; added 3 result keys to Clear-button list; dedented `st.rerun()` out of the clear loop |
| `tests/test_state_invalidation.py` | **New** — 3 regression tests for state invalidation (key completeness, image-change clear, Clear-button structural correctness via AST) |
| `scripts/smoke_test_real_inference.py` | **New** — real-inference smoke benchmark (A3/B2/C2/seg/depth on CUDA) capturing load/warm/VRAM |
| `reports/engineering_architecture_review.md` | **New** |
| `reports/functional_correctness_report.md` | **New** |
| `reports/performance_optimization_report.md` | **New** |
| `reports/final_engineering_validation.md` | **New** |
| `reports/real_inference_smoke_results.json` | **New** — benchmark output |

No model weights, checkpoints, datasets, labels, or training scripts were modified.

## 6. Tests added

`tests/test_state_invalidation.py` (3 tests):
1. `test_inference_result_keys_are_complete` — the canonical result-key set is fully referenced by the production module (guards DEF-1).
2. `test_image_change_invalidates_all_result_keys` — simulates the image-change branch and asserts no result key survives (guards DEF-1).
3. `test_clear_button_clears_all_and_reruns_once` — AST-analyses the Clear-button handler: verifies the loop clears all result+identity keys, the loop body has exactly one `st.session_state.pop()`, and `st.rerun()` is at the loop's indentation level (not nested) (guards DEF-2).

## 7. Full test results

| Run | Result | Time |
|-----|--------|------|
| Baseline (before fixes) | 143 passed | 29.84 s |
| After fixes + new tests | **146 passed, 0 failed, 0 skipped** | 18.58 s |

Command: `python -m pytest -q` with the `tf-gpu` interpreter. All 146 tests green; no tests were weakened to pass.

### Mock vs real inference — clearly distinguished

- **Mock tests** (deterministic, CPU, no weights): `MockObjectDetector`/`MockRoomSegmenter`/`MockScene3DModel`/`MockRoomClassifier`/`MockVisionLanguageModel`/`MockDesignGenerator` via `mock_mode=True`; responses carry `is_simulated=True`.
- **Real inference tests** (CUDA, production checkpoints): `tests/test_ui_integration.py` (indoor A3, exterior B2, parking C2, segmentation, depth) + `tests/test_depth_estimation.py` real-mode tests + the new `scripts/smoke_test_real_inference.py`. Real responses carry `is_simulated=False`.
- **No test silently substitutes mock for real inference.** `test_services.py::test_detection_service_mock_and_real` explicitly asserts that a missing-checkpoint real service returns `success=False` and does **not** fall back to mock.

## 8. Real inference verification (CUDA)

`scripts/smoke_test_real_inference.py` on held-out images (`reports/real_inference_smoke_results.json`):

| Path | Model | Device | is_simulated | Result | Warm (ms) | VRAM Δ (MB) |
|------|-------|--------|--------------|--------|-----------|-------------|
| Indoor detection | A3 (800px) | cuda:0 | false | 6 objects | 2649* | 68 |
| Exterior detection | B2 | cuda:0 | false | 70 objects | 302 | 36 |
| Parking detection | C2 | cuda:0 | false | 1 object | 190 | 36 |
| Segmentation | YOLO11n-seg (pretrained) | cuda:0 | false | 2 masks | 413 | 12 |
| Depth | Depth Anything V2 Small | cuda:0 | false | depth ∈[0.4727, 8.233], relative_only=True, peak 389.3 MB | 363 | 1 |

\* A3 wall-clock includes a first-call CUDA warm-up on the 800px input; model-reported CUDA-synced steady inference was **523.61 ms**.

All five production paths execute correctly on CUDA with finite, normalised outputs and honest metadata.

## 9. Performance before/after

No code-level optimisation was applied (steady-state latencies are already interactive-budget on the target hardware, and the existing infra already implements weight caching, a GPU concurrency gate, and Streamlit resource caching). Before/after full-suite timing: 29.84 s → 18.58 s (faster on rerun due to warmed CUDA context; no source change targets speed). See `reports/performance_optimization_report.md`.

## 10. GPU & RAM observations

- Peak VRAM after loading all paths sequentially: **~248 MB allocated / 389 MB peak** (depth forward). All models fit the 8 GB budget with >7 GB headroom.
- No CUDA OOM across the full real-inference test run.
- Uniform resource policy: every GPU-heavy service routes through `GPUInferenceGate` (capacity 1); mock/CPU work bypasses it; no concurrency was increased.

## 11. Remaining limitations

- **Classification / VLM / Generation** are intentionally **stubs** (`ModelNotImplementedError` in real mode) — no trained weights exist. The UI correctly surfaces "Model Implementation Pending" and never presents them as operational. This is by design, not a defect.
- **Segmentation** uses generic COCO-pretrained YOLO11n-seg, not RoomAI-trained architectural segmentation. Clearly disclosed in-UI.
- **Depth** is relative-only; never described as metric. Clearly disclosed in-UI.
- Depth Anything V2 first-load (~3.3 s) is a transformers library cost, not a RoomAI bottleneck.

## 12. Technical debt

- Per-class-threshold validation is duplicated between `CustomObjectDetector` and `MockObjectDetector` (~30 lines); a shared validator would reduce duplication.
- `ModelSpec` metadata (`nc`) is not runtime-enforced against a loaded checkpoint's class count; a lightweight assertion in `load_model` would catch weight/config mismatches early.
- `services/ensemble_service.py` cluster-centre recomputation uses simple averaging rather than the WBF coordinate fusion in `fuse_cluster_wbf`; known simplification.

## 13. Recommended next engineering improvements

1. Extract the per-class-threshold validator into `utils/` and share it across `Custom*`/`Mock*` detectors.
2. Add a runtime `nc` assertion in `CustomObjectDetector.load_model` (compare `model.names` length vs config `classes`/`nc`).
3. Add a Streamlit-`ScriptRunContext`-aware integration test for the full upload → detect → segment → depth → classify → VLM → design flow (currently covered by service-level integration tests; a headless Streamlit AppTest run would add end-to-end UI coverage).
4. Consider pinning `use_fast=True` for the Depth Anything processor to remove the transformers slow-processor warning.

---

## Final verdict

**ENGINEERING VALIDATION PASSED** — with the following qualifications:

- All implemented production features (Indoor A3, Exterior B2, Parking C2 detection; COCO-pretrained segmentation; Depth Anything V2 relative depth) execute correctly and consistently on CUDA with honest metadata.
- Two High-severity data-integrity defects (DEF-1, DEF-2) were found and fixed with regression coverage.
- Full test suite: **146 passed, 0 failed, 0 skipped** (tf-gpu, CUDA).
- Classification / VLM / Generation are intentionally unimplemented (stubs) and are correctly presented as pending — not as operational.

The application is technically correct, stable, and well-structured for its implemented scope. Classification/VLM/Generation remain **FUNCTIONALLY INCOMPLETE BY DESIGN** (no weights); the implemented detection/segmentation/depth stack is **FUNCTIONALLY COMPLETE WITH KNOWN LIMITATIONS** (generic seg, relative depth).