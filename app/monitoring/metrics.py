"""RoomAI Metrics Collection.

Prometheus-compatible metrics and structured telemetry.
"""

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import psutil
import torch


@dataclass
class JobMetrics:
    """Metrics for a single job."""
    job_id: str
    model_id: str
    status: str = "QUEUED"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gpu_memory_peak_mb: float = 0.0
    cpu_memory_peak_mb: float = 0.0
    error: Optional[str] = None


class MetricsCollector:
    """Collects and exposes system and job metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._jobs: Dict[str, JobMetrics] = {}
        self._lock = threading.Lock()
        self._system_metrics: deque = deque(maxlen=1000)
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._start_time = time.time()

    def record_job_start(self, job_id: str, model_id: str) -> None:
        """Record job start."""
        with self._lock:
            self._jobs[job_id] = JobMetrics(job_id=job_id, model_id=model_id, status="PROCESSING")
            self._jobs[job_id].started_at = time.time()
            self._increment("jobs_started_total")

    def record_job_stage(
        self,
        job_id: str,
        stage: str,
        progress: float,
        duration_ms: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Record job stage progress."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].stages[stage] = {
                    "progress": progress,
                    "duration_ms": duration_ms,
                    "timestamp": time.time(),
                    **extra,
                }
                self._jobs[job_id].status = stage

    def record_job_complete(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Record job completion."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.completed_at = time.time()
                job.status = status
                job.error = error

                # Calculate total duration
                if job.started_at:
                    total_ms = (job.completed_at - job.started_at) * 1000
                    self._histogram("job_duration_ms", total_ms)

                self._increment(f"jobs_{status.lower()}_total")

    def record_gpu_memory(self, job_id: str, allocated_mb: float, reserved_mb: float) -> None:
        """Record GPU memory usage for a job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.gpu_memory_peak_mb = max(job.gpu_memory_peak_mb, allocated_mb)

            self._histogram("gpu_allocated_mb", allocated_mb)
            self._histogram("gpu_reserved_mb", reserved_mb)

    def record_cpu_memory(self, job_id: str, used_mb: float) -> None:
        """Record CPU memory usage for a job."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.cpu_memory_peak_mb = max(job.cpu_memory_peak_mb, used_mb)

    def record_inference_latency(self, model_id: str, duration_ms: float) -> None:
        """Record model inference latency."""
        self._increment(f"inference_requests_total", labels={"model": model_id})
        self._histogram(f"inference_latency_ms", duration_ms, labels={"model": model_id})

    def record_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Record API request metrics."""
        self._increment("api_requests_total", labels={"method": method, "path": path, "status": str(status_code)})
        self._histogram("api_latency_ms", duration_ms, labels={"method": method, "path": path})

    def record_model_load(self, model_id: str, duration_ms: float, success: bool) -> None:
        """Record model loading metrics."""
        self._increment("model_loads_total", labels={"model": model_id, "success": str(success)})
        if success:
            self._histogram("model_load_duration_ms", duration_ms, labels={"model": model_id})

    def record_export(self, job_id: str, format: str, size_bytes: int, valid: bool) -> None:
        """Record export metrics."""
        self._increment("exports_total", labels={"format": format, "valid": str(valid)})
        self._histogram("export_size_bytes", size_bytes, labels={"format": format})

    def _increment(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._make_key(name, labels)
        self._counters[key] += 1

    def _histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        # Keep only last 1000 values per histogram
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / (1024*1024),
            "memory_available_mb": psutil.virtual_memory().available / (1024*1024),
            "disk_percent": psutil.disk_usage("/").percent,
            "uptime_seconds": time.time() - self._start_time,
        }

        if torch.cuda.is_available():
            metrics.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024*1024),
                "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024*1024),
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024*1024),
            })
            try:
                free, total = torch.cuda.mem_get_info()
                metrics["gpu_free_mb"] = free / (1024*1024)
                metrics["gpu_total_mb"] = total / (1024*1024)
            except Exception:
                pass

        self._system_metrics.append(metrics)
        return metrics

    def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific job."""
        with self._lock:
            if job_id not in self._jobs:
                return None
            job = self._jobs[job_id]
            return {
                "job_id": job.job_id,
                "model_id": job.model_id,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "duration_ms": (job.completed_at - job.started_at) * 1000 if job.started_at and job.completed_at else None,
                "stages": job.stages,
                "gpu_memory_peak_mb": job.gpu_memory_peak_mb,
                "cpu_memory_peak_mb": job.cpu_memory_peak_mb,
                "error": job.error,
            }

    def get_all_job_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics for all jobs."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
            return [self.get_job_metrics(j.job_id) for j in jobs]

    def get_counters(self) -> Dict[str, int]:
        """Get all counters."""
        with self._lock:
            return dict(self._counters)

    def get_histograms(self) -> Dict[str, Dict[str, float]]:
        """Get histogram summaries (min, max, mean, p50, p95, p99)."""
        import numpy as np
        with self._lock:
            result = {}
            for key, values in self._histograms.items():
                if values:
                    arr = np.array(values)
                    result[key] = {
                        "count": len(arr),
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        "mean": float(arr.mean()),
                        "p50": float(np.percentile(arr, 50)),
                        "p95": float(np.percentile(arr, 95)),
                        "p99": float(np.percentile(arr, 99)),
                    }
            return result

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Counters
        for key, value in self.get_counters().items():
            lines.append(f"# TYPE {key} counter")
            lines.append(f"{key} {value}")

        # Histograms (summary)
        for key, stats in self.get_histograms().items():
            base = key.split("{")[0]
            labels = "{" + key.split("{")[1] if "{" in key else ""
            lines.append(f"# TYPE {base} summary")
            for stat, value in stats.items():
                if stat != "count":
                    lines.append(f'{base}_{stat}{labels} {value}')
            lines.append(f'{base}_count{labels} {stats["count"]}')

        # System metrics (latest)
        if self._system_metrics:
            latest = self._system_metrics[-1]
            lines.append("# TYPE system_cpu_percent gauge")
            lines.append(f"system_cpu_percent {latest['cpu_percent']}")
            lines.append("# TYPE system_memory_percent gauge")
            lines.append(f"system_memory_percent {latest['memory_percent']}")
            if "gpu_allocated_mb" in latest:
                lines.append("# TYPE gpu_allocated_mb gauge")
                lines.append(f"gpu_allocated_mb {latest['gpu_allocated_mb']}")

        return "\n".join(lines) + "\n"


# Singleton
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector