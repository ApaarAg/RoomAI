"""Timing and progress utilities for RoomAI training, preparation, and inference.

Fulfills product requirements:
- Displays stage name, estimated duration, items/epochs, disk & GPU expectations, and assumptions at start.
- Displays elapsed time, progress percentage, completed vs total, ETA, metrics, and GPU VRAM during execution.
- Displays actual duration, estimated vs actual, items processed, latency percentiles, and output paths at completion.
- Provides granular inference component breakdown (loading, preprocessing, inference, fusion, postprocessing).
"""

import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

# Reconfigure stdout/stderr for Windows UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def format_duration(seconds: float) -> str:
    """Formats seconds into HH:MM:SS or MM:SS format."""
    if seconds < 0:
        return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_current_gpu_vram_gb() -> Optional[float]:
    """Returns currently allocated GPU memory in GB, or None if unavailable."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            return round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        except Exception:
            return None
    return None


class StageTimer:
    """Tracks and reports stage progress, ETA, and resource usage with structured console output."""

    def __init__(self):
        self.stage_name: str = ""
        self.start_time: float = 0.0
        self.estimated_sec: float = 0.0
        self.total_items: int = 0
        self.completed_items: int = 0
        self.expected_gpu_gb: float = 0.0
        self.assumptions: str = ""
        self.is_active: bool = False
        self._last_report_time: float = 0.0

    def start_stage(
        self,
        stage_name: str,
        estimated_sec: float,
        total_items: int = 0,
        expected_gpu_vram_gb: float = 0.0,
        assumptions: str = "Empirical estimation based on baseline hardware benchmark",
    ) -> None:
        """Starts a workflow stage and displays initial metadata and assumptions."""
        self.stage_name = stage_name
        self.estimated_sec = estimated_sec
        self.total_items = total_items
        self.completed_items = 0
        self.expected_gpu_gb = expected_gpu_vram_gb
        self.assumptions = assumptions
        self.start_time = time.time()
        self._last_report_time = self.start_time
        self.is_active = True

        border = "=" * 78
        print(f"\n{border}")
        print(f">> [STAGE START]: {self.stage_name}")
        print(f"   Estimated Duration : {format_duration(self.estimated_sec)} ({self.estimated_sec:.1f}s)")
        if self.total_items > 0:
            print(f"   Total Items/Epochs : {self.total_items}")
        if self.expected_gpu_gb > 0:
            print(f"   Expected GPU VRAM  : ~{self.expected_gpu_gb:.1f} GB")
        print(f"   Estimation Rationale: {self.assumptions}")
        print(f"{border}\n")

    def update(
        self,
        completed_items: int,
        current_metrics: Optional[Dict[str, Any]] = None,
        step_name: Optional[str] = None,
        min_interval_sec: float = 0.5,
    ) -> None:
        """Updates progress and computes dynamic ETA based on observed throughput."""
        if not self.is_active:
            return

        now = time.time()
        self.completed_items = completed_items
        elapsed = now - self.start_time

        # Rate-limit updates to avoid flooding terminal unless completed
        if completed_items < self.total_items and (now - self._last_report_time) < min_interval_sec:
            return
        self._last_report_time = now

        pct = (completed_items / self.total_items * 100) if self.total_items > 0 else 0.0
        items_per_sec = (completed_items / elapsed) if elapsed > 0 else 0.0

        if items_per_sec > 0 and self.total_items > completed_items:
            remaining_items = self.total_items - completed_items
            eta_sec = remaining_items / items_per_sec
            eta_str = format_duration(eta_sec)
        else:
            eta_str = "00:00"

        gpu_vram = get_current_gpu_vram_gb()
        gpu_str = f" | GPU VRAM: {gpu_vram:.2f} GB" if gpu_vram is not None else ""

        metrics_str = ""
        if current_metrics:
            metrics_str = " | " + ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in current_metrics.items())

        step_prefix = f"[{step_name}] " if step_name else ""
        print(
            f"   >> {step_prefix}{pct:5.1f}% ({completed_items}/{self.total_items}) | "
            f"Elapsed: {format_duration(elapsed)} | ETA: {eta_str}{gpu_str}{metrics_str}"
        )

    def finish_stage(
        self,
        artifacts: Optional[List[str]] = None,
        failures: Optional[List[str]] = None,
        skipped: int = 0,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Finalizes the stage, reports variance between estimated and actual time, and prints summary."""
        if not self.is_active:
            return {}

        now = time.time()
        actual_duration = now - self.start_time
        variance_sec = actual_duration - self.estimated_sec
        variance_str = f"{'+' if variance_sec >= 0 else '-'}{format_duration(abs(variance_sec))}"

        self.is_active = False
        border = "=" * 78
        print(f"\n{border}")
        print(f">> [STAGE COMPLETE]: {self.stage_name}")
        print(f"   Actual Duration    : {format_duration(actual_duration)} ({actual_duration:.1f}s)")
        print(f"   Estimated Duration : {format_duration(self.estimated_sec)} (Diff: {variance_str})")
        if self.total_items > 0:
            print(f"   Items Processed    : {self.completed_items}/{self.total_items} (Skipped: {skipped})")
        if failures:
            print(f"   [!] Failures ({len(failures)}): {failures[:3]}")
        if extra_metrics:
            print("   Metrics:")
            for k, v in extra_metrics.items():
                print(f"     * {k}: {v:.4f}" if isinstance(v, float) else f"     * {k}: {v}")
        if artifacts:
            print("   Output Artifacts:")
            for a in artifacts:
                print(f"     [FILE] {a}")
        print(f"{border}\n")

        return {
            "stage_name": self.stage_name,
            "actual_duration_sec": round(actual_duration, 2),
            "estimated_duration_sec": round(self.estimated_sec, 2),
            "variance_sec": round(variance_sec, 2),
            "completed_items": self.completed_items,
            "total_items": self.total_items,
            "artifacts": artifacts or [],
        }


class InferenceTimer:
    """Measures component-level latencies (loading, preprocess, inference, fusion, postprocess)."""

    def __init__(self):
        self.load_times: List[float] = []
        self.preprocess_times: List[float] = []
        self.inference_times: List[float] = []
        self.fusion_times: List[float] = []
        self.postprocess_times: List[float] = []
        self.total_times: List[float] = []

    def record(
        self,
        load_ms: float = 0.0,
        preprocess_ms: float = 0.0,
        inference_ms: float = 0.0,
        fusion_ms: float = 0.0,
        postprocess_ms: float = 0.0,
    ) -> Dict[str, float]:
        """Records a single inference iteration and returns breakdown."""
        total_ms = load_ms + preprocess_ms + inference_ms + fusion_ms + postprocess_ms
        if load_ms > 0:
            self.load_times.append(load_ms)
        self.preprocess_times.append(preprocess_ms)
        self.inference_times.append(inference_ms)
        self.fusion_times.append(fusion_ms)
        self.postprocess_times.append(postprocess_ms)
        self.total_times.append(total_ms)

        return {
            "load_ms": round(load_ms, 2),
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "fusion_ms": round(fusion_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "total_ms": round(total_ms, 2),
            "throughput_fps": round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0,
        }

    def summarize(self) -> Dict[str, Any]:
        """Calculates statistical summary across all recorded runs."""
        if not self.total_times:
            return {"count": 0}

        totals = np.array(self.total_times)
        return {
            "count": len(totals),
            "mean_total_ms": round(float(np.mean(totals)), 2),
            "median_total_ms": round(float(np.median(totals)), 2),
            "p90_total_ms": round(float(np.percentile(totals, 90)), 2),
            "p95_total_ms": round(float(np.percentile(totals, 95)), 2),
            "min_total_ms": round(float(np.min(totals)), 2),
            "max_total_ms": round(float(np.max(totals)), 2),
            "mean_preprocess_ms": round(float(np.mean(self.preprocess_times)), 2) if self.preprocess_times else 0.0,
            "mean_inference_ms": round(float(np.mean(self.inference_times)), 2) if self.inference_times else 0.0,
            "mean_fusion_ms": round(float(np.mean(self.fusion_times)), 2) if self.fusion_times else 0.0,
            "mean_postprocess_ms": round(float(np.mean(self.postprocess_times)), 2) if self.postprocess_times else 0.0,
            "mean_load_ms": round(float(np.mean(self.load_times)), 2) if self.load_times else 0.0,
            "mean_fps": round(float(1000.0 / np.mean(totals)), 1) if np.mean(totals) > 0 else 0.0,
        }
