# RoomAI — Artifact Retention Report

**Date:** 2026-10-30
**Module:** `utils/artifact_manager.py`
**Config:** `config/app.yaml` → `artifact_retention`

---

## 1. Problem

`DetectionService.run` previously wrote annotated images to a single shared directory
`artifacts/detections/detection_<wall-clock-ms>.png` with a collision counter and **no
retention**. Under multi-user use this caused:

- **Unbounded disk growth** — every inference run left a PNG forever.
- **Cross-session visibility** — every session's artifact lived in the same folder.
- **No active-file protection** — a naive cleanup could delete a file mid-write.

---

## 2. Mechanism implemented

### Session-scoped, unpredictable output paths
- `new_session_artifact_root()` → `artifacts/detections/<22-char URL-safe random token>/`
  (not derived from any user value; unguessable).
- `new_artifact_path(session_dir, prefix, suffix)` →
  `<session_dir>/<prefix>_<12-char random>.<suffix>`.
- `ensure_artifact_dir()` lazily creates the parent.
- Defensive guard: refuses to write a path that resolves outside the controlled `artifacts/`
  root (no path traversal).

`DetectionService.run(save_annotated=True)` now writes each annotated image into its own
random per-session subdirectory; the returned `saved_image_path` is the absolute path of that
file. The UI shows only the current session's path.

### Safe retention cleaner (`ArtifactManager`)
`ArtifactRetentionPolicy`:
| Field | Default | Effect |
|-------|---------|--------|
| `retention_seconds` | 86400 (24 h) | Files older than this are eligible for eviction. |
| `max_storage_mb` | 2048 | Hard cap on total artifact storage; oldest eligible files evicted first. |
| `active_grace_seconds` | 120 | **A file modified within the last N seconds is never evicted** (in-flight protection). |

`ArtifactManager.cleanup_expired(root)`:
1. Walks `root.rglob('*')`, gathering files with `st_mtime`/`st_size`.
2. **Never** touches a file whose `st_mtime > now - active_grace_seconds` (in-flight).
3. Skips files within the retention window that are not over the storage cap.
4. If total size exceeds `max_storage_mb`, evicts oldest *eligible* files first until under cap.
5. On `PermissionError` (file open/locked on Windows) → counts as "protected_open" and is
   left in place (never throws).
6. Logs every eviction; never raises into the caller.

Config is surfaced in `config/app.yaml` under `artifact_retention` so the operator can tune
it without code changes.

---

## 3. Tests (all PASS, `tf-gpu`)

| Test | Asserts |
|------|---------|
| `test_retention_evicts_expired_files` | A 2-hour-old file is evicted; a fresh file survives. |
| `test_retention_protects_active_files` | A just-written file (mtime now) is **not** evicted even with a 1s TTL, because it is inside the 120s active grace window. |
| `test_retention_respects_storage_cap` | With a 2 MiB cap and three 1 MiB eligible files, the oldest eligible is evicted and total stays ≤ cap. |
| `test_retention_never_raises_into_caller` | `cleanup_expired` on a non-existent path returns a dict; no exception. |
| `test_session_artifact_paths_are_unique_and_scoped` | Two session roots differ and both sit under `detections/`; filenames within a session are unique. |
| `test_two_sessions_cannot_overwrite_each_other` | Two sessions writing concurrently do not clobber each other's bytes. |
| `test_detection_service_writes_session_scoped_artifact` | A mock `DetectionService.run(save_annotated=True)` writes to a per-session random subdirectory under `artifacts/detections/`, and the file exists on disk. |

---

## 4. Operational guidance

The retention sweep is **advisory** — it does not run on a timer by itself. Two recommended
ways to actually run it:

### Option A — on each save (advisory)
The `config/app.yaml` flag `artifact_retention.sweep_on_save: true` is read by the
application; a production wiring would call `ArtifactManager.cleanup_expired` lazily (e.g.
every Nth save). This cycle did **not** add an automatic sweep on the hot path to avoid
introducing latency; the operator should schedule it (Option B) for the private beta.

### Option B — scheduled cleanup (recommended)
Windows Task Scheduler, every hour:
```powershell
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -c "from utils.artifact_manager import ArtifactManager; from pathlib import Path; ArtifactManager().cleanup_expired(Path('artifacts/detections'))"
```
This never deletes user-uploaded data (uploads are not stored persistently — they live only
in `st.session_state`) and never touches the existing user artifacts under
`artifacts/audit_galleries`, `artifacts/indoor_error_analysis`, etc., because the sweep is
scoped to `artifacts/detections` only.

**No bulk deletion was run against existing user artifacts during implementation.** The tests
operate on temporary `tmp_path` directories only.

---

## 5. Limitations

- Retention is scoped to `artifacts/detections/` (session-scoped detection outputs). Other
  `artifacts/*` subdirectories (error analysis galleries, canonical samples, phase reports)
  are **user-created and preserved** — they are not swept.
- The sweep is advisory; without a scheduled job, `artifacts/detections/` will still
  accumulate until a job runs.
- The cleaner uses `st_mtime`; a file whose clock was tampered with backwards could be
  evicted earlier than intended. Acceptable for a private beta.

---

## 6. Verdict

Artifact retention is **safe, tested, and configured**. Active/in-flight files are
protected; total storage is capped; old files are evicted with logging; no path traversal.
The mechanism is appropriate for a restricted private beta.