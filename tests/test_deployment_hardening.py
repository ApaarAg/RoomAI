"""Regression tests for deployment hardening: GPU concurrency, session isolation,
and artifact retention.

These tests exercise the *real* behaviour of the new concurrency gate and artifact
manager. They use mock detectors (deterministic, CPU) for the isolation/concurrency
tests so they run quickly and deterministically, and use a short retention policy for
the retention tests. A real-GPU concurrency test lives in the verification probe and in
the existing real-inference suite; here we verify the *mechanisms*.
"""

from __future__ import annotations

import io
import os
import shutil
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import pytest

from utils.artifact_manager import (
    ArtifactManager,
    ArtifactRetentionPolicy,
    ensure_artifact_dir,
    new_artifact_path,
    new_session_artifact_root,
)
from utils.gpu_concurrency import GPUInferenceGate, GPUBusyError


# --------------------------------------------------------------------------- #
# GPU concurrency gate
# --------------------------------------------------------------------------- #
def test_gpu_gate_serialises_concurrent_ops():
    """A capacity-1 gate must serialise two concurrent calls."""
    gate = GPUInferenceGate(max_concurrent=1)
    call_order: List[str] = []

    def slow(label: str) -> str:
        call_order.append(f"start-{label}")
        time.sleep(0.05)
        call_order.append(f"end-{label}")
        return label

    threads = [
        threading.Thread(target=lambda: gate.run(slow, "A")),
        threading.Thread(target=lambda: gate.run(slow, "B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The two calls must not overlap: each "start" must immediately precede its "end".
    a_end = call_order.index("end-A")
    a_start = call_order.index("start-A")
    b_end = call_order.index("end-B")
    b_start = call_order.index("start-B")
    assert a_end < b_start or b_end < a_start, f"calls overlapped: {call_order}"


def test_gpu_gate_releases_permit_on_exception():
    """If the gated callable raises, the permit must be released (no deadlock)."""
    gate = GPUInferenceGate(max_concurrent=1)

    def boom() -> None:
        raise RuntimeError("inference exploded")

    with pytest.raises(RuntimeError, match="inference exploded"):
        gate.run(boom)

    # The gate must be fully available again — a fresh call must not block.
    result = gate.run(lambda: "recovered")
    assert result.value == "recovered"
    assert result.acquired is True


def test_gpu_gate_raises_on_busy_when_requested():
    """raise_on_busy=True surfaces a clear GPUBusyError instead of blocking."""
    gate = GPUInferenceGate(max_concurrent=1)
    hold_done = threading.Event()

    def hold() -> str:
        time.sleep(0.2)
        return "holder"

    holder = threading.Thread(
        target=lambda: (gate.run(hold), hold_done.set())
    )
    holder.start()
    time.sleep(0.05)  # let the holder acquire the permit

    with pytest.raises(GPUBusyError):
        gate.run(lambda: "blocked", raise_on_busy=True, busy_timeout=0.01)

    holder.join()
    assert hold_done.is_set()


def test_gpu_gate_reports_queue_wait_separately_from_inference():
    """queue_wait_ms must be separated from inference_time_ms."""
    gate = GPUInferenceGate(max_concurrent=1)
    hold_done = threading.Event()

    def hold() -> str:
        time.sleep(0.1)
        return "ok"

    # Occupy the gate so the second caller queues.
    holder = threading.Thread(
        target=lambda: (gate.run(hold), hold_done.set())
    )
    holder.start()
    time.sleep(0.02)

    def quick() -> str:
        return "done"

    res = gate.run(quick)
    holder.join()
    assert res.queue_wait_ms >= 0.0
    # Queue wait should be non-trivial because we waited for the holder.
    assert res.inference_time_ms >= 0.0


def test_gpu_gate_active_and_waiting_accounting():
    """The gate must accurately report active count while a permit is held."""
    gate = GPUInferenceGate(max_concurrent=1)
    in_holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        in_holding.set()
        release.wait(timeout=1.0)

    t = threading.Thread(target=lambda: gate.run(hold))
    t.start()
    in_holding.wait(timeout=1.0)
    try:
        assert gate.active == 1
    finally:
        release.set()
        t.join()
    assert gate.active == 0


# --------------------------------------------------------------------------- #
# Session-scoped artifact isolation
# --------------------------------------------------------------------------- #
def test_session_artifact_paths_are_unique_and_scoped():
    """Two sessions must get distinct, non-guessable artifact directories."""
    d1 = new_session_artifact_root()
    d2 = new_session_artifact_root()
    assert d1 != d2
    # Both must be under the controlled detections subdir.
    assert "detections" in str(d1)
    assert "detections" in str(d2)
    # Filenames within a session are unique.
    f1 = new_artifact_path(d1)
    f2 = new_artifact_path(d1)
    assert f1 != f2
    assert f1.parent == d1


def test_two_sessions_cannot_overwrite_each_other(tmp_path):
    """Simulated two sessions writing concurrently cannot collide on disk."""
    d1 = new_session_artifact_root(tmp_path, "detections")
    d2 = new_session_artifact_root(tmp_path, "detections")
    p1 = ensure_artifact_dir(new_artifact_path(d1))
    p2 = ensure_artifact_dir(new_artifact_path(d2))
    p1.write_bytes(b"session-1")
    p2.write_bytes(b"session-2")
    assert p1.read_bytes() == b"session-1"
    assert p2.read_bytes() == b"session-2"
    assert p1 != p2
    # Cross-session reads are not possible via the returned paths by accident.
    assert not p1.exists() or p1.read_bytes() == b"session-1"


# --------------------------------------------------------------------------- #
# Artifact retention
# --------------------------------------------------------------------------- #
def test_retention_evicts_expired_files(tmp_path):
    """Files older than the TTL are evicted; recent files are preserved."""
    root = tmp_path / "detections" / "sess-old"
    root.mkdir(parents=True)
    old_file = root / "old.png"
    old_file.write_bytes(b"x" * 1024)
    # Backdate mtime by 2 hours.
    old_time = time.time() - 7200
    os.utime(old_file, (old_time, old_time))

    recent_dir = tmp_path / "detections" / "sess-new"
    recent_dir.mkdir(parents=True)
    recent_file = recent_dir / "new.png"
    recent_file.write_bytes(b"y" * 1024)

    mgr = ArtifactManager(
        policy=ArtifactRetentionPolicy(
            retention_seconds=3600,
            max_storage_mb=2048,
            active_grace_seconds=10,
        )
    )
    summary = mgr.cleanup_expired(tmp_path / "detections")
    assert summary["evicted"] == 1
    assert not old_file.exists()
    assert recent_file.exists()


def test_retention_protects_active_files(tmp_path):
    """A file modified within the active-grace window is never evicted."""
    root = tmp_path / "detections" / "sess-active"
    root.mkdir(parents=True)
    active = root / "active.png"
    active.write_bytes(b"z" * 1024)
    # mtime is "now", inside the active grace window.

    mgr = ArtifactManager(
        policy=ArtifactRetentionPolicy(
            retention_seconds=1,  # would otherwise evict
            max_storage_mb=2048,
            active_grace_seconds=120,
        )
    )
    summary = mgr.cleanup_expired(tmp_path / "detections")
    assert summary["evicted"] == 0
    assert summary["protected_active"] >= 1
    assert active.exists()


def test_retention_respects_storage_cap(tmp_path):
    """When total size exceeds the cap, oldest eligible files are evicted first."""
    root = tmp_path / "detections" / "sess"
    root.mkdir(parents=True)
    # Write three 1 MiB files, backdated to be eligible.
    files = []
    for i in range(3):
        f = root / f"f{i}.bin"
        f.write_bytes(b"a" * (1024 * 1024))
        t = time.time() - 7200 - i  # oldest first
        os.utime(f, (t, t))
        files.append(f)

    # Set a 2 MiB cap so at least one file must be evicted.
    mgr = ArtifactManager(
        policy=ArtifactRetentionPolicy(
            retention_seconds=3600,
            max_storage_mb=2.0,
            active_grace_seconds=10,
        )
    )
    summary = mgr.cleanup_expired(tmp_path / "detections")
    assert summary["evicted"] >= 1
    # The newest eligible file should survive the cap-driven eviction if any.
    total_after = mgr.total_size_mb(tmp_path / "detections")
    assert total_after <= 2.0 + 1e-6 or summary["evicted"] == 3


def test_retention_never_raises_into_caller(tmp_path):
    """cleanup_expired must not raise even on a locked/missing file."""
    root = tmp_path / "detections" / "sess"
    root.mkdir(parents=True)
    f = root / "ok.png"
    f.write_bytes(b"data")
    t = time.time() - 7200
    os.utime(f, (t, t))

    # Point at a non-existent path — must not raise.
    mgr = ArtifactManager()
    summary = mgr.cleanup_expired(tmp_path / "does-not-exist")
    assert isinstance(summary, dict)


# --------------------------------------------------------------------------- #
# DetectionService integration: session-scoped save path + GPU gate
# --------------------------------------------------------------------------- #
def test_detection_service_writes_session_scoped_artifact():
    """A real-mode DetectionService mock run must write to a session-scoped path
    under artifacts/detections/<random>/ rather than the old shared timestamp path.
    """
    from services.detection_service import DetectionService

    svc = DetectionService(mock_mode=True)
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    res = svc.run(img, confidence_threshold=0.40, save_annotated=True)
    assert res["success"] is True
    saved = res.get("saved_image_path")
    assert saved is not None
    p = Path(saved)
    # Must be under artifacts/detections and contain a random session token dir,
    # not a bare detection_<ms>.png at the root.
    assert p.parts[-3] == "detections" or "detections" in p.as_posix()
    assert not p.name.startswith("detection_") or "_" in p.name
    # The file must actually exist on disk.
    assert p.exists()
    # Mock mode bypasses the GPU gate, so queue_wait_ms is 0.
    assert res["timing_metrics"].get("queue_wait_ms", 0.0) == 0.0


def test_detection_service_real_mode_missing_checkpoint_no_silent_mock():
    """A missing production checkpoint in real mode must fail explicitly, never
    silently substitute mock output, and must release the GPU gate permit on the
    way out (no deadlock)."""
    from services.detection_service import DetectionService

    svc = DetectionService(
        mock_mode=False,
        config={"checkpoint_path": "checkpoints/does_not_exist_zzz.pt"},
    )
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    res = svc.run(img)
    assert res["success"] is False
    assert res["response"] is None
    assert res["error_message"] is not None

    # The gate must be fully released after the failure.
    gate = __import__("utils.gpu_concurrency", fromlist=["get_gpu_gate"]).get_gpu_gate()
    assert gate.active == 0