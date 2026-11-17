# RoomAI Production Readiness Audit — Findings Ledger

**Audit Date:** 2026-10-29  
**Target Application:** RoomAI (Backend, Services, Models, and Streamlit UI)  
**Environment:** Windows 11, Conda `tf-gpu` (Python 3.10.20), PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
**Repository Commit:** `8f4bf662eb73518866f516710b68a99c6b03f37a`

---

## 1. Executive Summary of Audit Findings

A rigorous audit was conducted across the codebase, configuration files, test suites, service contracts, and interactive frontend. Four defects were identified, analyzed, and remediated:

| Finding ID | Severity | Area | Status | Summary |
| :--- | :--- | :--- | :--- | :--- |
| **FINDING-P1-01** | **P1 (Critical)** | Frontend (`ui/upload_view.py`) | **RESOLVED** | Stale inference artifacts remained cached in `st.session_state` when a user uploaded or selected a new image without explicitly clicking clear. |
| **FINDING-P2-01** | **P2 (Major)** | Testing Harness (`pytest.ini`, `scripts/`) | **RESOLVED** | Running bare `pytest` caused discovery timeouts because standalone validation scripts in `scripts/test_*.py` lacked entrypoint guards and ran long-running subprocesses at import time. |
| **FINDING-P2-02** | **P2 (Major)** | Visualization (`services/image_service.py`) | **RESOLVED** | 1-indexed segmentation mask IDs caused an off-by-one color shift against the 0-indexed UI legend palette (`PALETTE[uid % len]` vs `HEX_PALETTE[idx % len]`). |
| **FINDING-P2-03** | **P2 (Major)** | Security & Input (`utils/image_validation.py`) | **RESOLVED** | Image validation did not explicitly catch Pillow `DecompressionBombError`, exposing the application to generic unhandled decode errors or denial-of-service on malicious images. |
| **FINDING-P3-01** | **P3 (Informational)** | MLOps / Environment | **DOCUMENTED** | PATH collision: Global `python` invoked Miniconda base environment (`transformers`/`huggingface_hub` mismatch) instead of `tf-gpu` environment. Documented in deployment runbook. |
| **FINDING-P3-02** | **P3 (Informational)** | Pipeline Performance | **DOCUMENTED** | Detection request latency is dominated by synchronous PNG compression and disk I/O on 1600x1200 images (~728ms) rather than GPU inference (~58ms). |

---

## 2. Detailed Findings and Corrective Actions

### FINDING-P1-01: Stale Inference Results Across Image Upload / Selection
- **Severity:** P1 (Critical UX / Integrity)
- **Component:** `ui/upload_view.py`
- **Description:**  
  When an end-user switched benchmark sample photographs or uploaded a new image, `st.session_state["current_image"]` was updated, but the downstream cached results (`detection_result`, `detection_result_indoor`, `detection_result_exterior`, `detection_result_parking`, `segmentation_result`, `scene3d_result`) remained in session memory. Consequently, the user continued to see detection bounding boxes, segmentation masks, and depth maps computed for the *previous* image until they manually ran a new inference or clicked "Clear".
- **Reproduction:**
  1. Select "Indoor Room" sample.
  2. Navigate to "Object Detection", execute inference. Result is displayed.
  3. Return to "Image Upload", select "Exterior Architecture" sample.
  4. Navigate to "Object Detection": the indoor boxes and metrics were still displayed on screen until a re-run.
- **Root Cause:**  
  No tracking mechanism existed to monitor changes in the active image identity.
- **Remediation:**  
  In `ui/upload_view.py`, computed an active image fingerprint:
  ```python
  active_id = f"{img_name}_{img_w}x{img_h}"
  if st.session_state.get("active_image_id") != active_id:
      for k in [
          "detection_result",
          "detection_result_indoor",
          "detection_result_exterior",
          "detection_result_parking",
          "segmentation_result",
          "scene3d_result",
      ]:
          st.session_state.pop(k, None)
      st.session_state["active_image_id"] = active_id
  ```
- **Verification:**  
  Session state invalidation verified; switching images cleanly clears downstream analysis widgets.

---

### FINDING-P2-01: Pytest Test Discovery Collision and Timeout
- **Severity:** P2 (Major)
- **Component:** `pytest.ini`, `scripts/test_streamlit_startup.py`, `scripts/test_load_depth_anything.py`, `scripts/test_midas_load.py`
- **Description:**  
  Executing standard `python -m pytest -q` without specifying the `tests/` directory failed with:
  ```
  subprocess.TimeoutExpired: Command ... timed out after 5 seconds
  ```
  during the test collection phase.
- **Root Cause:**  
  1. The repository root lacked a `pytest.ini` configuring `testpaths = tests`. Pytest defaulted to scanning all subdirectories recursively.
  2. The standalone verification scripts in `scripts/` were named `test_*.py` and contained top-level execution code without `if __name__ == "__main__":` entrypoint guards. Pytest imported them during discovery, launching a real background Streamlit server on port 8503 and blocking collection.
- **Remediation:**  
  1. Created `pytest.ini` with:
     ```ini
     [pytest]
     testpaths = tests
     python_files = test_*.py
     python_classes = Test*
     python_functions = test_*
     ```
  2. Refactored `scripts/test_streamlit_startup.py`, `scripts/test_load_depth_anything.py`, and `scripts/test_midas_load.py` into dedicated functions guarded by `if __name__ == "__main__":`.
- **Verification:**  
  Bare `pytest -q` now executes cleanly, discovering and running only the 129 formal unit/integration tests in `tests/` with 0 collection errors.

---

### FINDING-P2-02: Segmentation Mask Color vs UI Legend Mismatch
- **Severity:** P2 (Major)
- **Component:** `services/image_service.py` (line 94), `models/segmentation_model.py` (line 133-146)
- **Description:**  
  In the segmentation visualization, the color rendered onto the image overlay did not match the color badge displayed in the UI legend table.
  For example, for the first detected object (`idx = 0`):
  - UI Legend: Royal Blue (`HEX_PALETTE[0] = #3B82F6`)
  - Image Overlay: Emerald Green (`PALETTE[1] = (16, 185, 129)`)
- **Root Cause:**  
  In `models/segmentation_model.py`:
  - `mask_id = (idx % 254) + 1` (1-indexed, starting at 1).
  - `color_hex = HEX_PALETTE[idx % len(HEX_PALETTE)]` (0-indexed, starting at index 0).
  In `services/image_service.py`:
  - `color = PALETTE[int(uid) % len(PALETTE)]`. For `uid = 1`, `int(uid) % 8 = 1`, which picked `PALETTE[1]` (Green) instead of `PALETTE[0]` (Blue).
- **Remediation:**  
  In `services/image_service.py`, corrected the indexing:
  ```python
  color = PALETTE[(int(uid) - 1) % len(PALETTE)]
  ```
  Now `uid = 1` evaluates to `(1 - 1) % 8 = 0`, selecting `PALETTE[0]` (Royal Blue), in exact 1:1 alignment with `HEX_PALETTE[0]`.
  Also updated `MockRoomSegmenter` metadata to reference `HEX_PALETTE` directly.
- **Verification:**  
  Added regression test `test_segmentation_palette_alignment` in `tests/test_ui_integration.py`. Verified mask color and legend hex match across all 8 palette entries.

---

### FINDING-P2-03: Lack of Explicit Decompression Bomb Protection
- **Severity:** P2 (Major Security / Robustness)
- **Component:** `utils/image_validation.py`
- **Description:**  
  `validate_image_file` verified file size and dimensions, but did not catch `PIL.Image.DecompressionBombError` when processing extremely large or malicious image payloads designed to trigger memory exhaustion.
- **Root Cause:**  
  Pillow raises `Image.DecompressionBombError` (a subclass of `ValueError`) during `verify()` or `Image.open()`. Catching only generic `Exception` resulted in unhandled or generic "Error decoding image" responses rather than an actionable, clear error message.
- **Remediation:**  
  Added explicit exception handling for `Image.DecompressionBombError` in both `verify()` and `open()` stages, returning:
  `"Image dimensions exceed safe decompression limits (possible decompression bomb)."`
- **Verification:**  
  Added regression test `test_decompression_bomb_rejection` in `tests/test_ui_integration.py`. Verified that simulated bombs are safely rejected with the expected error message.

---

### FINDING-P3-01: Python Environment PATH Sensitivity
- **Severity:** P3 (Informational / Operational)
- **Component:** Environment configuration / CLI execution
- **Description:**  
  If a developer or CI pipeline executes `python` without activating the `tf-gpu` Conda environment, the system defaults to the Miniconda base environment (`C:\Users\apaar\miniconda3\python.exe`), where `huggingface-hub` is version `1.24.0` while `transformers` expects `<1.0`. This produces an `ImportError` on HuggingFace-dependent depth estimation.
- **Remediation & Recommendation:**  
  Documented in `reports/production_audit/deployment_checklist.md`: The application and test suites must always be executed using the activated Conda environment `tf-gpu` or the explicit binary path `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`.

---

### FINDING-P3-02: Latency Profile Dominated by PNG Disk I/O
- **Severity:** P3 (Informational / Performance)
- **Component:** `services/detection_service.py`
- **Description:**  
  Empirical multi-pass latency profiling revealed that the forward-pass inference of Model B2 is 35.54 ms (3.59 FPS full pipeline, ~28 FPS raw inference). However, overall request time on 1600x1200 images averages ~794 ms. The breakdown proves that 728.71 ms (91.7% of total time) is spent encoding and saving uncompressed PNG annotated images to disk (`artifacts/detections/`).
- **Recommendation:**  
  For high-throughput real-world production deployments, save images asynchronously in a background thread or adopt JPEG encoding (`quality=85`) rather than blocking synchronous PNG writes.

