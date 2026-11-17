# RoomAI — Final Project Consistency & Deployment Readiness Audit

**Audit date:** 2026-10-30
**Auditor:** Roo (senior software engineer / ML engineer / QA / security reviewer)
**Python environment:** `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe` (Python 3.10.20)
**Repository root:** `f:/RoomAI`
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM), 24 GB system RAM, Windows 11

---

## 1. Executive Summary

RoomAI is a Streamlit-based computer-vision studio for room/interior understanding. The
project separates UI (`ui/`), service orchestration (`services/`), Pydantic contracts
(`schemas/`), model interfaces (`models/interfaces.py`), concrete models (`models/`),
and configuration (`config/`). The inference pipeline uses Ultralytics YOLO11 for
detection/segmentation and HuggingFace Transformers (Depth Anything V2 Small) for
relative depth.

The codebase was found to be in strong shape: production model defaults (Indoor A3,
Exterior B2, Parking C2) are correctly configured and verified on disk; experimental
candidates (B5/B6) are guarded by `enabled: false` and `is_primary: false`; the
inference path refuses to silently fall back to mock mode on missing checkpoints; and
the full pytest suite (130 tests) passes on the real `tf-gpu` environment.

Two genuine discrepancies were identified and corrected:

1. A duplicate docstring line in `ui/upload_view.py` (Low).
2. The parking detector UI tab defaulted to the C1+C2 *ensemble* as the first/selectbox
   default, which was inconsistent with the indoor (A3) and exterior (B2) convention of
   placing the production single model first. This made the ensemble the apparent
   production default for parking, conflicting with the "C2 is the parking production
   default" constraint (Medium). Corrected so the C2 single model is the primary default.

A regression test (`test_parking_registry_and_ui_default_is_c2_production`) was added
that verifies both the registry primary and the UI choice ordering via AST inspection.

**Verdict:** **READY FOR LOCAL DEMO** (see §14 for full assessment and remaining
limitations).

---

## 2. Initial Git Status & Preserved User Changes

### Baseline (recorded before any edits)
- Branch: `master`
- Last 2 commits:
  - `8f4bf66 feat: real YOLO11 object detection, COCO audit, benchmarks, and fine-tuning experiment`
  - `d73ea0d feat: complete RoomAI application infrastructure, model interfaces, and test suite`

### Pre-existing modified (tracked) files — preserved, not reverted
| File | Status |
|------|--------|
| `README.md` | modified |
| `app.py` | modified |
| `config/app.yaml` | modified |
| `config/model_config.yaml` | modified |
| `models/detection_model.py` | modified |
| `models/scene3d_model.py` | modified |
| `models/segmentation_model.py` | modified |
| `schemas/responses.py` | modified |
| `services/detection_service.py` | modified |
| `services/image_service.py` | modified |
| `services/scene3d_service.py` | modified |
| `tests/conftest.py` | modified |
| `ui/analysis_view.py` | modified |
| `ui/components.py` | modified |
| `ui/upload_view.py` | modified |
| `utils/image_validation.py` | modified |

### Untracked directories/files — preserved
All untracked user work (`.agent/`, `.agents/`, `.gemini/`, `.gsd/`, `RoomAI_Data/`,
`docs/`, `scripts/*`, `services/ensemble_service.py`, `services/specialist_ensemble_service.py`,
`utils/timing_utils.py`, `utils/masked_detection_loss.py`, `conftest.py`, `pytest.ini`,
`VERSION`, `model_capabilities.yaml`, `PROJECT_RULES.md`, `GSD-STYLE.md`, all
`config/*.backup_*`, all `artifacts/*`, all `reports/*`) was left untouched.

No `git reset --hard`, `git clean`, or broad `git restore` commands were run.

### Files modified by this audit (only)
| File | Change | Severity addressed |
|------|--------|--------------------|
| `ui/upload_view.py` | Removed a duplicated docstring line (lines 11–12 were identical) | Low |
| `ui/analysis_view.py` | Reordered the parking `detector_choices` dict so the **C2 single production model** is the first (default) option, matching the indoor (A3) and exterior (B2) convention | Medium |
| `tests/test_detection_deployment_readiness.py` | Added regression test `test_parking_registry_and_ui_default_is_c2_production` | — (coverage) |

---

## 3. Project Architecture & Execution Flow

```
User browser
  → Streamlit app.py (page config, mock-mode toggle)
    → ui/upload_view.py         (upload + utils/image_validation.py: size/format/bomb/corruption + EXIF + RGB normalize + dimension clamp)
      → ui/analysis_view.py / ui/design_view.py
        → services/* (DetectionService, SegmentationService, ClassificationService, Scene3DService, VisionLanguageService, GenerationService)
          → models/* (CustomObjectDetector [Ultralytics YOLO], CustomRoomSegmenter [YOLO11n-seg], CustomScene3DModel [Depth Anything V2 Small], mock variants)
            → checkpoints/ (production + experimental .pt weights, gitignored)
          → services/image_service.py (box drawing, overlay, depth heatmap, side-by-side)
        → schemas/responses.py (Pydantic DetectionResponse / SegmentationResponse / Scene3DResponse / ...)
      → utils/device.py (CUDA detection, VRAM accounting)
      → utils/logging_config.py (structured logger)
    → models/registry.py (ModelSpec registry + GLOBAL_MODEL_REGISTRY singleton + VRAM budget enforcement)
  → config/app.yaml + config/model_config.yaml (loaded via @st.cache_data in app.py)
```

### Key architectural properties (verified)
- **UI / inference separation:** UI imports services only; services import models; models import schemas. No reverse dependencies.
- **Model caching:** `models/detection_model.py` uses a module-level `_YOLO_CACHE: Dict[(ckpt,device), YOLO]` and `models/segmentation_model.py` uses `_SEG_CACHE`; `ui/analysis_view.py` caches `DetectionService` via `@st.cache_resource`. Repeated Streamlit reruns reuse loaded weights.
- **Mock/real isolation:** Each service selects `Mock*` or `Custom*` strictly from `mock_mode`. Real mode raises `FileNotFoundError`/`ModelNotImplementedError` on missing weights — never returns mock output (verified by `test_real_mode_never_silently_substitutes_mock` and by the verification probe).
- **Session state:** `ui/upload_view.py` stores `current_image` and per-domain detection results in `st.session_state` and invalidates stale results when the active image identity changes.

---

## 4. Findings Categorized by Severity

### Critical
None. No data loss, security breach, incorrect production model, or application-wide failure was found.

### High
None. No broken core functionality, GPU OOM, serious concurrency defect, or broken inference was found.

### Medium
| ID | Finding | Root cause | Fix |
|----|---------|-------------|-----|
| M1 | Parking detection UI tab defaulted to the C1+C2 ensemble as the selectbox default (index 0), making the ensemble appear to be the parking production default. The indoor and exterior tabs correctly default to their single production models (A3, B2). | The `detector_choices` dict literal for the parking branch in `ui/analysis_view.py` was authored with the ensemble first. | Reordered so `Model C2: YOLO11s Parking/Outdoor ... [Primary Recommended]` is first; ensemble and C1 follow. Added a regression test that asserts both the registry primary and the UI ordering. |

### Low
| ID | Finding | Root cause | Fix |
|----|---------|-------------|-----|
| L1 | `ui/upload_view.py` `render_upload_view` had a duplicated docstring line (the same `"""Renders the image upload widget..."""` appeared twice consecutively). | Likely a copy-paste artifact from an earlier edit. | Removed the duplicate line, leaving a single correct docstring. |

### Informational (not modified — no genuine defect)
- `requirements.txt` comments out torch/transformers/ultralytics with a note that they are installed in the user environment. Verified present in `tf-gpu`. Not a defect.
- `.gitignore` excludes `*.pt`, `checkpoints/`, `data/`, `runs/`. Correct for this project.
- `conftest.py` (root) and `tests/conftest.py` set `TRANSFORMERS_NO_TF`/`TRANSFORMERS_NO_FLAX` and neutralize `tensorflow` in `sys.modules` to avoid NumPy 2.x / TF 1.x ABI crashes. Correct and necessary for this environment.

---

## 5. Root-Cause Analysis of Corrected Issues

### M1 — Parking production-default UI ordering
- **Source:** `ui/analysis_view.py`, parking `detector_choices` literal.
- **Why a genuine defect:** The project's model constraint explicitly states "Parking detector: C2" as the production default and "B5/B6/B7 and other experimental checkpoints must remain experimental." The parking UI offered three options and placed the ensemble first; because Streamlit `st.selectbox` defaults to `index=0`, the ensemble — not the C2 single model — was the effective default. This created an inconsistency between the registry primary (`parking_c2_pilot`, `is_primary=True`), `config/model_config.yaml` (`parking_detection.checkpoint_path = checkpoints/parking_detector_c2_yolo11s_pilot_best.pt`), and the UI. It did not change inference correctness if the user left the slider untouched (the ensemble still uses C2), but it misrepresented the production posture and could lead a future maintainer to treat the ensemble as canonical.
- **Reproduction:** Open the app, select the "Parking" task group; the default highlighted architecture was the ensemble.
- **Correction:** Reorder the dict so the C2 single model is first and labeled `[Primary Recommended]`; the ensemble is relabeled without "(Recommended)".
- **Regression coverage:** `test_parsing_registry_and_ui_default_is_c2_production` parses `ui/analysis_view.py` with `ast`, locates the parking `detector_choices` dict, and asserts the first key references the C2 production model and maps to `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt`.

### L1 — Duplicate docstring
- **Source:** `ui/upload_view.py` lines 11–12.
- **Why a genuine defect:** A function with two consecutive string literals silently makes the second a no-op expression statement; harmless at runtime but misleading and a maintainability smell flagged by static analysis.
- **Correction:** Removed the duplicate; the single canonical docstring remains.

---

## 6. Files Modified and Why

| File | Why |
|------|-----|
| `ui/upload_view.py` | Removed duplicate docstring (L1). |
| `ui/analysis_view.py` | Reordered parking `detector_choices` so C2 production single model is the default (M1). |
| `tests/test_detection_deployment_readiness.py` | Added regression test `test_parking_registry_and_ui_default_is_c2_production` to lock the parking production-default ordering against future drift. |

No other files were modified. No working code was refactored for cosmetics.

---

## 7. Regression Tests Added

- `tests/test_detection_deployment_readiness.py::test_parking_registry_and_ui_default_is_c2_production`
  - Asserts `GLOBAL_MODEL_REGISTRY.get_primary_for_task("parking_detection").model_id == "parking_c2_pilot"` with `is_primary=True` and the C2 checkpoint path.
  - Asserts the first key/value of the parking `detector_choices` dict in `ui/analysis_view.py` references the C2 production model and checkpoint (via AST parsing, so the test does not require a running Streamlit runtime).

---

## 8. Commands Executed

| # | Command | Purpose | Exit |
|---|---------|---------|------|
| 1 | `git status` | Baseline status | 0 |
| 2 | `git log --oneline -20` | Commit history | 0 |
| 3 | `dir checkpoints`, `dir data\...` | Checkpoint & dataset existence | 0 |
| 4 | `C:\...\tf-gpu\python.exe -m pytest -q --tb=line -p no:cacheprovider` | **Baseline full suite** | 0 (129 passed, 18.69s) |
| 5 | `C:\...\tf-gpu\python.exe -m pytest tests/test_detection_deployment_readiness.py::test_parking_registry_and_ui_default_is_c2_production -v` | New regression test | 0 (passed, 2.73s) |
| 6 | `C:\...\tf-gpu\python.exe -m pytest -q --tb=line -p no:cacheprovider` | **Post-fix full suite** | 0 (130 passed, 39.71s) |
| 7 | `reports/_audit_verification_probe.py` (tf-gpu) | Config/registry/real-GPU/missing-ckpt/VRAM/startup probe | 0 (ALL PASSED) |
| 8 | `git status --short` | Confirm only intended files changed | 0 |

All commands used the explicit `tf-gpu` interpreter as required.

---

## 9. Actual Test Results & Exit Codes

### Baseline (before changes)
- `pytest -q`: **129 passed in 18.69s**, exit 0.

### Post-fix (after changes)
- `pytest -q`: **130 passed in 39.71s**, exit 0. (The +1 is the new regression test. The longer wall time is due to real-GPU tests being exercised again in this run.)

### New regression test
- `test_parking_registry_and_ui_default_is_c2_production`: **PASSED**, 2.73s, exit 0.

### Verification probe (`reports/_audit_verification_probe.py`, executed then removed)
- Section 1 Configuration loading & checkpoint existence: **PASSED** (A3, B2, C2, yolo11n-seg all exist).
- Section 2 Registry primary selection: **PASSED** (indoor=`indoor_a3_800px`, exterior=`exterior_b2_finetuned`, parking=`parking_c2_pilot`; B5/B6 `is_primary=False`).
- Section 3 Real GPU inference: **PASSED**
  - Indoor A3: 9 objects, 680.1 ms, `cuda:0`, `is_simulated=False`, checkpoint `indoor_detector_a3_yolo11s_800px_best.pt`.
  - Exterior B2: 1 object, 311.1 ms, `cuda:0`, `is_simulated=False`, checkpoint `exterior_detector_b2_yolo11s_best.pt`.
  - Parking C2: 1 object, 177.0 ms, `cuda:0`, `is_simulated=False`, checkpoint `parking_detector_c2_yolo11s_pilot_best.pt`.
- Section 4 Missing-checkpoint failure: **PASSED** — `FileNotFoundError`, no silent mock fallback, `success=False`, `response=None`.
- Section 5 Repeated inference & VRAM: **PASSED** — 5 repeated inferences, mean 30.1 ms (min 25.4 / max 42.4); post-run VRAM allocated 140.6 MiB, reserved 214.0 MiB (well under the 4500 MiB budget; no leak).
- Section 6 App import/startup: **PASSED** — `app.py` imports cleanly.

---

## 10. Model Registry & Checkpoint Verification

| Task | Primary model_id | Checkpoint (rel) | On disk | is_primary | Config match |
|------|------------------|------------------|---------|------------|---------------|
| Indoor detection | `indoor_a3_800px` | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | ✅ (18,724 KiB) | ✅ True | ✅ `detection.checkpoint_path` |
| Exterior detection | `exterior_b2_finetuned` | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | ✅ (18,713 KiB) | ✅ True | ✅ `exterior_detection.checkpoint_path` |
| Parking detection | `parking_c2_pilot` | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | ✅ (18,708 KiB) | ✅ True | ✅ `parking_detection.checkpoint_path` |
| Segmentation | `yolo11_seg_instance` | `checkpoints/yolo11n-seg.pt` | ✅ (6,037 KiB) | ✅ True | ✅ `segmentation.checkpoint_path` |
| Scene3D depth | `depth_anything_v2_small` | `checkpoints/depth_anything_v2_small` | ✅ (dir) | ✅ True | ✅ `scene3d.checkpoint_path` |

### Experimental guards
- `exterior_b5_transfer` (`checkpoints/exterior_detector_b5_transfer_best.pt`): `is_primary=False`, config `exterior_detection_b5_candidate.enabled=false`, `status="experimental_candidate"`.
- `exterior_b6_expanded` (`checkpoints/exterior_detector_b6_expanded_best.pt`): `is_primary=False`, config `exterior_detection_b6_candidate.enabled=false`, `status="experimental_candidate"`.
- `exterior_detection_b4_experimental` config block: `enabled=false`, `status="experimental_candidate"`.
- No experimental checkpoint can become the production default: `GLOBAL_MODEL_REGISTRY.get_primary_for_task(...)` only returns `is_primary=True` specs, and the UI defaults now match.

### Taxonomy consistency
- Indoor: 12 classes (`cabinet#shelf, table, chair#stool, lamp, door, bed, sofa, plant, decoration, curtain, home-appliance, window`) — consistent across `config/model_config.yaml`, `models/registry.py`, and `docs/frontend_integration_contract.md`.
- Exterior: 10 classes — consistent across config, registry, docs.
- Parking: 8 classes — consistent across config, registry, docs.

---

## 11. GPU & VRAM Observations

- CUDA available: **NVIDIA GeForce RTX 4060 Laptop GPU**.
- Inference executed on `cuda:0` for all three production detectors; `is_simulated=False` confirmed.
- `models/registry.py` `ModelRegistry.enforce_vram_budget()` evicts least-recently-used models above 4500 MiB and calls `torch.cuda.empty_cache()` — appropriate, not blanket.
- Module-level caches (`_YOLO_CACHE`, `_SEG_CACHE`) plus `@st.cache_resource` for `DetectionService` prevent redundant reloads across Streamlit reruns.
- Repeated inference (5x) showed stable VRAM (140.6 MiB allocated / 214.0 MiB reserved after the loop) — **no memory leak detected**.
- `CustomObjectDetector` synchronizes the CUDA stream before/after timing so `inference_time_ms` is a genuine GPU measurement.

---

## 12. Security Findings

### Addressed / adequate
- **Upload validation** (`utils/image_validation.py`): enforces allowed extensions, max file size (15 MB default), max dimension (4096px) with graceful downsampling, `Image.verify()` integrity, and `Image.DecompressionBombError` handling. EXIF orientation is applied. Invalid/corrupt uploads return a clear `ImageValidationResult(is_valid=False)` with a non-revealing error message.
- **Mock transparency**: every response schema carries `is_simulated`; the UI renders a prominent "SIMULATED OUTPUT (MOCK MODE)" badge when active.
- **No secrets in code**: no hardcoded API keys/tokens found in the inspected source; model weights are gitignored.
- **Error messages**: service error messages are generic ("Object detection failed: ...") and do not dump stack traces to the UI; the verification probe's traceback was from the log, not user-facing.
- **Path handling**: checkpoint resolution in `models/detection_model.py` and `models/segmentation_model.py` resolves relative paths against the repo root and raises `FileNotFoundError` when missing; no user-controlled path is concatenated into filesystem operations.
- **Generated file naming**: `services/detection_service.py` writes to `artifacts/detections/detection_<ms>.png` with a collision counter, preventing overwrite races.

### Remaining risks (noted, not fixed — see §13)
- **No authentication/authorization**: Streamlit app has no login; suitable for local demo only.
- **No rate limiting / concurrent-user isolation**: Streamlit's single-threaded server model mitigates, but under a tunnel this is not a hardened deployment.
- **`artifacts/detections/` accumulates**: each run writes a new PNG; no automatic cleanup. Safe for local demo, should be rotated for long-running or multi-user deployments.
- **`config/app.yaml` `mock_mode: false`**: the app boots in real-inference mode by default; correct for the intended production posture, but means a missing checkpoint surfaces as an error in the UI rather than a silent mock — which is the desired behavior.
- **`torch`/`transformers`/`ultralytics` not pinned in `requirements.txt`**: documented as installed in the user environment; not a security issue but a reproducibility note.

---

## 13. Remaining Limitations

1. **No auth / no network hardening** — the app is intended for local or restricted private demo; it must not be exposed publicly without a reverse proxy + auth + TLS.
2. **Unbounded `artifacts/detections/` growth** — no retention policy; acceptable for local demo.
3. **VLM and generation features are stubbed** — `VisionLanguageService(mock_mode=False)` returns `is_implemented=False` because `CustomRoomClassifier`/VLM/generator real implementations are not connected (weights absent). This is honest behavior, not a defect.
4. **Segmentation uses generic COCO YOLO11n-seg** — not room-architecture-aware; documented in `ui/components.py` limitation notice and `docs/frontend_integration_contract.md`.
5. **Depth is relative only** — Depth Anything V2 Small produces affine-invariant relative depth, not metric distance; clearly disclaimed in the UI and schema.
6. **Concurrency not load-tested beyond a single Streamlit session** — out of scope for a local demo readiness verdict.
7. **`requirements.txt` does not pin ML runtime versions** — environment is managed via the `tf-gpu` conda env; documented, not changed.

---

## 14. Final Readiness Assessment

| Criterion | Status |
|-----------|--------|
| No unresolved Critical defects | ✅ |
| No unresolved High defects in core functionality | ✅ |
| Production model defaults correct (A3 / B2 / C2) | ✅ verified in registry, config, UI, and real inference |
| Configuration & checkpoint paths consistent | ✅ |
| Complete test suite executed (tf-gpu) | ✅ 130 passed |
| Real inference verified on production checkpoints | ✅ A3, B2, C2 on `cuda:0`, `is_simulated=False` |
| Missing-checkpoint failure behavior | ✅ explicit `FileNotFoundError`, no silent mock |
| Application startup verified | ✅ `app.py` imports cleanly |
| Repeated inference & VRAM stability | ✅ no leak over 5 runs |
| Regression coverage for corrections | ✅ added |
| Existing user work preserved | ✅ all pre-existing modifications + untracked files untouched |
| Known limitations documented | ✅ §13 |

### Verdict: **READY FOR LOCAL DEMO**

The application is reliable for a local demonstration on the `tf-gpu` environment with the
RTX 4060. It is **not** ready for unrestricted public deployment: it lacks authentication,
rate limiting, and a managed artifact-retention policy. A restricted private demo behind
an authenticated tunnel is the appropriate next stage after adding auth and a cleanup
policy for `artifacts/detections/`.

---

## Appendix A — What was fixed

1. Removed a duplicate docstring in `ui/upload_view.py` (`render_upload_view`).
2. Reordered the parking detector choices in `ui/analysis_view.py` so the **C2 production
   single model** is the default selectbox option, matching the indoor (A3) and exterior
   (B2) convention and the "C2 is the parking production default" constraint.
3. Added `test_parking_registry_and_ui_default_is_c2_production` to lock the parking
   production-default ordering against future regressions.

## Appendix B — What remains broken
Nothing in core functionality. Remaining items are documented limitations (§13), not
defects: no auth, unbounded artifact growth, and stubbed VLM/generation features that
honestly report `is_implemented=False` in real mode.

## Appendix C — Next stage recommendation
Before any external exposure, add: (1) authentication on the Streamlit/tunnel entrypoint,
(2) a bounded retention/cleanup job for `artifacts/detections/`, (3) explicit
`max_workers` / upload concurrency limits, and (4) pin `torch`/`transformers`/`ultralytics`
in `requirements.txt` for reproducibility. None of these were implemented here because they
are deployment-environment concerns outside the local-demo scope and several would require
new dependencies or architectural choices needing approval.
