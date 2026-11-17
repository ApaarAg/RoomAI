"""Session-scoped artifact storage and safe retention cleanup for RoomAI.

Previously, ``services/detection_service.py`` wrote annotated images to a single shared
directory ``artifacts/detections/`` using a wall-clock-millisecond filename
(``detection_<ms>.png``) with a collision counter. Under a multi-user Streamlit
deployment this created two problems:

1. **Cross-session visibility / collision**: every session's annotated detection was
   written into the same directory. Two concurrent sessions producing files within the
   same millisecond could rely on the counter to avoid clobbering, but any user could
   read another user's artifact path from the shared folder, and files accumulated
   indefinitely with no retention.

2. **Unbounded disk growth**: no eviction of expired artifacts.

This module provides:

* :func:`new_artifact_path` — a **session-scoped, unique, unpredictable** artifact path
  that combines a per-session directory (identified by a random session token, not a
  user-controlled value) with a random filename. This prevents a user from guessing or
  controlling the output path (no path traversal) and keeps each session's outputs
  isolated on disk.

* :class:`ArtifactManager` — a configurable, **safe** retention cleaner that:
    - evicts files older than a configured TTL,
    - caps total directory size,
    - **never** deletes a file whose mtime is within the active-session grace window
      (so an in-flight request's freshly written file is not removed),
    - skips files that are currently open / locked (Windows raises ``PermissionError``),
    - logs every eviction,
    - is purely advisory — a failure to clean up never raises into the request path.

No user-controlled filename is ever used as the output path. No bulk deletion is ever
performed against the whole tree blindly; only expired, non-active files are removed.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from utils.logging_config import get_logger

logger = get_logger("ArtifactManager")

DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_DETECTIONS_SUBDIR = "detections"

# Default retention policy (conservative for a single-machine demo).
DEFAULT_RETENTION_SECONDS = 60 * 60 * 24  # 24 hours
DEFAULT_MAX_STORAGE_MB = 2048  # 2 GiB total cap across the artifact root
# Grace window: never delete a file modified within the last N seconds (in-flight).
DEFAULT_ACTIVE_GRACE_SECONDS = 120


@dataclass
class ArtifactRetentionPolicy:
    """Configurable retention parameters for artifact cleanup."""

    retention_seconds: float = DEFAULT_RETENTION_SECONDS
    max_storage_mb: float = DEFAULT_MAX_STORAGE_MB
    active_grace_seconds: float = DEFAULT_ACTIVE_GRACE_SECONDS


def new_session_artifact_root(root: Optional[Path] = None, subdir: str = DEFAULT_DETECTIONS_SUBDIR) -> Path:
    """Returns a fresh, unique per-session artifact directory under ``root/subdir``.

    The leaf directory name is a 32-char cryptographically random token, so it is
    unguessable and not derived from any user-controlled value. The directory is
    created lazily on first write.
    """
    base = (root or DEFAULT_ARTIFACT_ROOT) / subdir
    token = secrets.token_urlsafe(16)  # ~22 chars of URL-safe entropy
    session_dir = base / token
    return session_dir


def new_artifact_path(
    session_dir: Path,
    prefix: str = "detection",
    suffix: str = ".png",
) -> Path:
    """Returns a unique, unpredictable artifact path inside ``session_dir``.

    The filename is ``<prefix>_<random>.<suffix>`` — not derived from a user filename,
    not a wall-clock timestamp alone. ``session_dir`` must have been produced by
    :func:`new_session_artifact_root` (or otherwise be under the controlled root).

    Returns a :class:`pathlib.Path`; the parent directory is **not** created here so
    callers that only need the path (e.g. for logging) don't pay the mkdir cost. Use
    :func:`ensure_artifact_dir` before writing.
    """
    if not session_dir.is_relative_to(DEFAULT_ARTIFACT_ROOT) and not session_dir.is_absolute():
        # Defensive: if a relative session_dir is passed that escapes the root, reject.
        resolved = session_dir.resolve()
        root_resolved = DEFAULT_ARTIFACT_ROOT.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to write artifact outside the controlled root: {session_dir}"
            ) from exc
    name = f"{prefix}_{secrets.token_urlsafe(12)}{suffix}"
    return session_dir / name


def ensure_artifact_dir(path: Path) -> Path:
    """Creates the parent directory of ``path`` (idempotent) and returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class ArtifactManager:
    """Safe, advisory retention cleanup for the RoomAI artifact tree.

    Usage::

        mgr = ArtifactManager(policy=ArtifactRetentionPolicy(retention_seconds=3600))
        evicted = mgr.cleanup_expired(Path("artifacts/detections"))
    """

    def __init__(self, policy: Optional[ArtifactRetentionPolicy] = None) -> None:
        self.policy = policy or ArtifactRetentionPolicy()

    def cleanup_expired(self, root: Path) -> Dict[str, int]:
        """Removes expired, non-active files under ``root``.

        Returns a summary dict with counts. Never raises into the caller; any error is
        logged and the file is left in place. Only files older than
        ``retention_seconds`` **and** outside the ``active_grace_seconds`` window are
        eligible. If the total size of the directory exceeds ``max_storage_mb``, the
        oldest eligible files are evicted first until the cap is satisfied (best-effort).

        Active files (recently modified, or currently locked/open on Windows) are
        always preserved.
        """
        summary = {"evicted": 0, "protected_active": 0, "protected_open": 0, "errors": 0}
        root = Path(root)
        if not root.exists():
            return summary

        now = time.time()
        cutoff = now - self.policy.retention_seconds
        active_cutoff = now - self.policy.active_grace_seconds

        # Gather candidates: (mtime, size, path)
        candidates = []
        total_size = 0
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                summary["errors"] += 1
                continue
            total_size += st.st_size
            # Active session protection: recently modified files are never touched.
            if st.st_mtime > active_cutoff:
                summary["protected_active"] += 1
                continue
            if st.st_mtime > cutoff:
                # Within retention window but not active — keep.
                continue
            candidates.append((st.st_mtime, st.st_size, p))

        # Sort oldest first (by mtime then path for determinism).
        candidates.sort(key=lambda t: (t[0], str(t[2])))

        over_cap = total_size / (1024 * 1024) > self.policy.max_storage_mb
        for mtime, size, path in candidates:
            # If under cap and file is within retention, stop the cap-driven eviction.
            if over_cap and (total_size / (1024 * 1024)) <= self.policy.max_storage_mb:
                over_cap = False
            # If not over cap and the file is within retention, keep it.
            if not over_cap and mtime > cutoff:
                continue
            # Evict.
            try:
                path.unlink()
                logger.info("Artifact retention evicted: %s (%d bytes, mtime=%s)", path, size, mtime)
                summary["evicted"] += 1
                total_size -= size
            except PermissionError:
                # File is open/locked (active write in progress) — protect it.
                summary["protected_open"] += 1
                logger.debug("Artifact in-use (locked), preserved: %s", path)
            except OSError as exc:
                summary["errors"] += 1
                logger.warning("Could not evict artifact %s: %s", path, exc)

        return summary

    def total_size_mb(self, root: Path) -> float:
        """Returns the total size of all files under ``root`` in MiB."""
        root = Path(root)
        if not root.exists():
            return 0.0
        total = 0
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total / (1024 * 1024)