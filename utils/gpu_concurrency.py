"""Thread-safe GPU inference concurrency control for RoomAI.

Streamlit executes concurrent user script runs in separate threads within a single
process. RoomAI's detection / segmentation / depth models share a module-level YOLO
cache (``_YOLO_CACHE``) and a global :class:`~models.registry.ModelRegistry` singleton.
While the loaded Ultralytics YOLO / Transformers model objects are themselves safe to
*call* ``predict`` on from multiple threads, unbounded concurrent CUDA forward passes on
an 8 GB RTX 4060 risk VRAM contention, out-of-memory errors, and latency collapse.

This module provides a single process-wide :class:`GPUInferenceGate` that bounds the
number of simultaneous "heavy" GPU inference operations via a
:class:`threading.BoundedSemaphore`. The default capacity is **1** — only one heavy
GPU forward pass may execute at a time — which is the conservative, measured-safe
choice for a single 8 GB GPU. Lightweight CPU-only work (mock inference, image
validation, rendering) does **not** acquire the gate.

Design constraints honoured:
* Thread-safe (BoundedSemaphore is implemented on top of a condition variable).
* Releases the permit even if the inference callable raises — via ``try/finally``.
* Reports a measured ``queue_wait_ms`` separately from the inference time.
* Returns a clear, user-facing "busy/queue" message when capacity is reached and a
  caller opts into ``raise_on_busy``.
* No distributed queue / Redis — single-process only, matching the Streamlit model.
* Avoids deadlocks: each ``acquire`` is paired with exactly one ``release`` in the
  same frame; permits are never held across nested acquisitions.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from utils.logging_config import get_logger

logger = get_logger("GPUConcurrency")

# Single process-wide gate. Default capacity = 1 concurrent heavy GPU op.
_DEFAULT_MAX_CONCURRENT_GPU_OPS = 1

# Sentinel for the "all permits exhausted" user message.
_BUSY_MESSAGE = (
    "All GPU inference slots are currently busy serving another request. "
    "Please retry in a moment."
)


@dataclass
class GPURunResult:
    """Outcome of a gated GPU inference call."""

    value: Any
    queue_wait_ms: float
    inference_time_ms: float
    acquired: bool


class GPUInferenceGate:
    """Bounds concurrent heavy GPU inference with a counting semaphore.

    Parameters
    ----------
    max_concurrent:
        Maximum number of simultaneously executing heavy GPU operations. Defaults to
        1, which serialises GPU work on a single 8 GB card and prevents VRAM
        contention between concurrent Streamlit sessions.
    """

    def __init__(self, max_concurrent: int = _DEFAULT_MAX_CONCURRENT_GPU_OPS) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self._sem = threading.BoundedSemaphore(max_concurrent)
        # Tracks how many callers are currently waiting (not holding) a permit.
        self._waiting = 0
        self._waiting_lock = threading.Lock()
        # Tracks how many permits are currently held.
        self._active = 0
        self._active_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Introspection (thread-safe, approximate — for diagnostics/tests)
    # ------------------------------------------------------------------ #
    @property
    def active(self) -> int:
        with self._active_lock:
            return self._active

    @property
    def waiting(self) -> int:
        with self._waiting_lock:
            return self._waiting

    @property
    def capacity_remaining(self) -> int:
        """Approximate remaining capacity (informational only)."""
        return self.max_concurrent - self.active

    # ------------------------------------------------------------------ #
    # Gated execution
    # ------------------------------------------------------------------ #
    def run(
        self,
        fn: Callable[..., Any],
        *args: Any,
        raise_on_busy: bool = False,
        busy_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> GPURunResult:
        """Execute ``fn(*args, **kwargs)`` under the GPU concurrency gate.

        Parameters
        ----------
        fn:
            The heavy GPU inference callable.
        raise_on_busy:
            If True, raise :class:`GPUBusyError` instead of blocking when no permit is
            available within ``busy_timeout`` seconds.
        busy_timeout:
            Maximum seconds to wait for a permit when ``raise_on_busy`` is True.
            ``None`` means wait indefinitely (only valid when ``raise_on_busy`` is
            False — blocking behaviour).
        """
        t_queue_start = time.perf_counter()

        acquired = False
        try:
            if raise_on_busy:
                # Non-blocking / timed acquisition: surface a clear message if busy.
                if not self._try_acquire(timeout=busy_timeout):
                    logger.warning(
                        "GPU gate busy (%d/%d active); raising GPUBusyError.",
                        self.active,
                        self.max_concurrent,
                    )
                    raise GPUBusyError(_BUSY_MESSAGE)
            else:
                self._blocking_acquire()
            acquired = True

            queue_wait_ms = (time.perf_counter() - t_queue_start) * 1000.0

            t_infer_start = time.perf_counter()
            try:
                value = fn(*args, **kwargs)
            finally:
                # Synchronise CUDA stream before timing so inference_time_ms is real
                # GPU time, not just CPU dispatch time.
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
            inference_time_ms = (time.perf_counter() - t_infer_start) * 1000.0

            return GPURunResult(
                value=value,
                queue_wait_ms=queue_wait_ms,
                inference_time_ms=inference_time_ms,
                acquired=True,
            )
        finally:
            if acquired:
                self._release()

    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """Context manager for acquiring the GPU permit."""
        self._blocking_acquire()
        try:
            yield
        finally:
            self._release()

    # ------------------------------------------------------------------ #
    # Low-level permit handling
    # ------------------------------------------------------------------ #
    def _blocking_acquire(self) -> None:
        with self._waiting_lock:
            self._waiting += 1
        try:
            self._sem.acquire()
        finally:
            with self._waiting_lock:
                self._waiting -= 1
            with self._active_lock:
                self._active += 1

    def _try_acquire(self, timeout: Optional[float]) -> bool:
        with self._waiting_lock:
            self._waiting += 1
        try:
            got = self._sem.acquire(timeout=timeout) if timeout is not None else self._sem.acquire(blocking=False)
        finally:
            with self._waiting_lock:
                self._waiting -= 1
        if got:
            with self._active_lock:
                self._active += 1
        return bool(got)

    def _release(self) -> None:
        with self._active_lock:
            self._active = max(0, self._active - 1)
        try:
            self._sem.release()
        except ValueError:
            # BoundedSemaphore raises ValueError if released more than acquired.
            # This should never happen given our bookkeeping, but guard defensively.
            logger.error("GPU gate over-release detected; ignoring.", exc_info=True)


class GPUBusyError(RuntimeError):
    """Raised when the GPU concurrency gate is saturated and ``raise_on_busy`` is set."""


# Process-wide singleton gate. Importing modules share this instance so that all
# Streamlit script threads funnel through a single bounded queue.
GPU_GATE = GPUInferenceGate(max_concurrent=_DEFAULT_MAX_CONCURRENT_GPU_OPS)


def get_gpu_gate() -> GPUInferenceGate:
    """Returns the process-wide :class:`GPUInferenceGate` singleton."""
    return GPU_GATE