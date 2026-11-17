# RoomAI — Security & User Isolation Report

**Date:** 2026-10-30
**Environment:** `tf-gpu`, RTX 4060 Laptop GPU (8 GB), Windows 11
**Target posture:** restricted private beta behind an authenticated secure tunnel.

---

## 1. Threat model & scope

The intended deployment is:

```
Browser → HTTPS → [Auth layer: Cloudflare Access] → tunnel → 127.0.0.1:8501 Streamlit → RoomAI services → RTX 4060
```

The laptop runs inference locally; external users reach it only through the authenticated
tunnel. **Authn/authz is delegated to the tunnel/identity provider**, not implemented
inside the Streamlit app. This report therefore covers what the *application* must guarantee
and what the *tunnel* must guarantee, and the tests that establish it.

| Boundary | Owned by | Status |
|----------|----------|--------|
| TLS termination, identity, SSO | Tunnel (Cloudflare Access) | Configured via deployment guide (not in-repo credentials) |
| Network binding | `.streamlit/config.toml` | Bound to `127.0.0.1` only ✅ |
| Upload validation | `utils/image_validation.py` | Size/format/bomb/corruption + dimension clamp ✅ |
| Cross-session state | `st.session_state` | Per-session by Streamlit; verified no global leakage of user data ✅ |
| Artifact isolation | `utils/artifact_manager.py` | Per-session random paths, no user-controlled output path ✅ |
| GPU contention | `utils/gpu_concurrency.py` | Serialised, no VRAM growth, no deadlock ✅ |
| Secrets in code | (none) | No hardcoded credentials found ✅ |
| Error disclosure | services | Generic messages; no stack traces to UI ✅ |

---

## 2. User isolation investigation

### What is per-session
- `st.session_state["current_image"]`, `["image_filename"]`, `["active_image_id"]`,
  `["detection_result*"]`, `["segmentation_result"]`, `["scene3d_result"]`,
  `["classification_result"]`, `["design_result"]` — all keyed by Streamlit's per-session
  state. Streamlit provides one `session_state` per browser session; these are **not** shared
  across sessions.

### What is process-wide (intentionally shared, read-mostly)
- `_YOLO_CACHE` (`models/detection_model.py`) — caches loaded YOLO instances keyed by
  (checkpoint, device). This is a **read-only model cache**: the same `YOLO` object is
  reused, but `model.predict(...)` does not mutate shared user state — it produces a fresh
  result object per call. Concurrent calls are now serialised by `GPU_GATE` so two threads
  never call `predict` on the same instance at the same time.
- `GLOBAL_MODEL_REGISTRY` (`models/registry.py`) — metadata-only, populated once at import.
- `@st.cache_resource` for `DetectionService` (`ui/analysis_view.py`) — Streamlit caches the
  service **per (mock_mode, checkpoint, device, task_group, imgsz)**, so two users selecting
  the same model share the cached service (and its underlying YOLO); the GPU gate makes this
  safe.

### What was shared and is now isolated
- **Artifact output path** — previously `artifacts/detections/detection_<ms>.png` (shared
  directory, wall-clock filename). Now each `DetectionService.run(save_annotated=True)`
  writes to `artifacts/detections/<22-char random token>/detection_<random>.png`. Two
  sessions get distinct random directories and cannot read each other's paths from the UI
  (the UI only shows the *current* session's `saved_image_path`). Verified by
  `test_two_sessions_cannot_overwrite_each_other` and `test_session_artifact_paths_are_unique_and_scoped`.

### Regression tests establishing isolation
- `test_session_artifact_paths_are_unique_and_scoped` — two session roots differ, both
  under `detections/`, filenames within a session are unique.
- `test_two_sessions_cannot_overwrite_each_other` — two sessions writing concurrently do not
  clobber each other's bytes.
- `test_gpu_gate_serialises_concurrent_ops` — the gate serialises two concurrent calls (no
  overlap of "start"/"end" markers).
- `test_gpu_gate_releases_permit_on_exception` — a failed call releases the permit so a
  later call succeeds (no denial-of-service via lock squatting).
- `test_detection_service_real_mode_missing_checkpoint_no_silent_mock` — real-mode failure
  never substitutes mock output and leaves the gate released.

These tests prove the **mechanism**; they do not claim complete isolation against an
adversary who controls the machine — that is out of scope for a restricted private beta on a
single trusted laptop.

---

## 3. Upload & request protection

`utils/image_validation.py` (unchanged, re-verified):
- Allowed extensions enforced (`.jpg/.jpeg/.png/.webp`).
- Max file size (default 15 MB, now also `maxUploadSize=50` in Streamlit config).
- Max image dimension (default 4096) with **graceful downsampling** to avoid GPU OOM.
- `Image.verify()` integrity check; corrupt uploads return a clear, non-revealing error.
- `Image.DecompressionBombError` handling.
- EXIF orientation applied; RGBA/grayscale/palette normalised to RGB.

No weakening of validation was performed. The Streamlit server config caps
`maxUploadSize` and `maxMessageSize` to bound WebSocket payloads.

---

## 4. Path traversal

- `utils/artifact_manager.new_artifact_path` refuses to resolve a session directory outside
  the controlled `artifacts/` root (`is_relative_to` / `relative_to` guard).
- No user-controlled filename is concatenated into an output path; all output filenames are
  `secrets.token_urlsafe(...)` tokens.
- Checkpoint resolution in `models/detection_model.py` / `models/segmentation_model.py`
  resolves relative paths against the repo root and raises `FileNotFoundError` when missing;
  no user-supplied path reaches the filesystem.

---

## 5. Secrets & error disclosure

- No API keys, tokens, or credentials in the inspected source or configs.
- Model weights are gitignored (`*.pt`, `checkpoints/`, `data/`).
- Service error messages are generic (`"Object detection failed: ..."`); the UI shows
  `res.get("error_message")`, not stack traces. Internal stack traces go to the logger only.

---

## 6. Rate limiting (gap, documented)

The GPU gate bounds **concurrency** (one heavy op at a time) but does not bound request
**rate**. A trusted private-beta user could enqueue many inferences back-to-back. For a
restricted private beta with known users this is acceptable; a token-bucket rate limiter
(per-user, e.g. `slowapi`) should be added before **public** beta. This is listed as a
remaining risk, not silently claimed as resolved.

---

## 7. Remaining security risks

1. **Auth is external** — correctness depends on the tunnel + identity provider being
   configured correctly. If the tunnel is exposed without the auth policy, the app is
   reachable. Revocation steps are in `reports/private_beta_deployment_guide.md`.
2. **No in-app audit log** of *who* ran *what* — only application logs (per-request
   inference logs without user identity). Add before public beta.
3. **No in-app rate limiting** — see §6.
4. **Single machine, no HA** — appropriate for a restricted private beta; not for production.
5. **Generated artifacts on disk** — the retention policy is advisory; a scheduled cleanup
   job is documented in the deployment guide. Files are session-scoped but still stored in a
   shared filesystem tree.

---

## 8. Verdict

For the intended **restricted private beta** behind an authenticated tunnel bound to
`127.0.0.1`, the application's isolation, upload, and concurrency controls are adequate and
**verified by tests**. It is **not** production-hardened: in-app auth, rate limiting, and
audit logging remain external/gaps.