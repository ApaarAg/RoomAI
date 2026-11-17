"""RoomAI GPU Worker.

Single-threaded GPU job executor with cooperative cancellation and resource management.
"""

import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

import torch

from app.core.config import settings
from app.core.exceptions import GPUError, JobError, CancellationError
from app.core.logging import get_logger, log_job_event, TraceContext
from app.services.reconstruction.pipeline import ReconstructionService, get_reconstruction_service
from app.services.job_store import JobStore, get_job_store
from app.monitoring.metrics import MetricsCollector, get_metrics_collector


logger = get_logger("GPUWorker")


@dataclass
class JobTask:
    """A reconstruction job task."""
    job_id: str
    image_paths: List[str]
    model_id: str
    options: Dict[str, Any]
    created_at: float


class GPUWorker:
    """Background worker that processes reconstruction jobs sequentially on GPU."""

    def __init__(
        self,
        reconstruction_service: Optional[ReconstructionService] = None,
        job_store: Optional[JobStore] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.reconstruction = reconstruction_service or get_reconstruction_service()
        self.job_store = job_store or get_job_store()
        self.metrics = metrics or get_metrics_collector()

        self._queue: queue.Queue = queue.Queue()
        self._cancelled_jobs: set = set()
        self._active_job_id: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

    def start(self) -> None:
        """Start the worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Worker already running")
            return

        self._shutdown = False
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="GPUWorker",
        )
        self._worker_thread.start()
        logger.info("GPU Worker started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker thread gracefully."""
        self._shutdown = True
        self._queue.put(None)  # Wake up worker
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        logger.info("GPU Worker stopped")

    def pause(self) -> None:
        """Pause job processing (finish current job first)."""
        self._pause_event.clear()
        logger.info("GPU Worker paused")

    def resume(self) -> None:
        """Resume job processing."""
        self._pause_event.set()
        logger.info("GPU Worker resumed")

    def enqueue_job(
        self,
        image_paths: List[str],
        model_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a job to the queue."""
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        task = JobTask(
            job_id=job_id,
            image_paths=image_paths,
            model_id=model_id,
            options=options or {},
            created_at=time.time(),
        )
        self._queue.put(task)

        # Create job in store
        self.job_store.create_job(
            job_id=job_id,
            model_id=model_id,
            image_paths=image_paths,
            options=options or {},
        )

        self.metrics.record_job_start(job_id, model_id)
        logger.info(f"Enqueued job {job_id} (model: {model_id}, views: {len(image_paths)})")
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of a queued or running job."""
        self._cancelled_jobs.add(job_id)

        if job_id == self._active_job_id:
            logger.info(f"Cancellation requested for active job {job_id}")
            return True

        # If queued, mark as cancelled in store
        job = self.job_store.get_job(job_id)
        if job and job.get("status") == "QUEUED":
            self.job_store.update_job(
                job_id=job_id,
                status="CANCELLED",
                stage="Cancelled while queued",
            )
            logger.info(f"Cancelled queued job {job_id}")
            return True

        return False

    def get_queue_size(self) -> int:
        """Get number of queued jobs."""
        return self._queue.qsize()

    def get_active_job(self) -> Optional[str]:
        """Get currently active job ID."""
        return self._active_job_id

    def _worker_loop(self) -> None:
        """Main worker loop."""
        logger.info("GPU Worker loop started")

        while not self._shutdown:
            try:
                # Wait for pause to be lifted
                self._pause_event.wait()

                # Get next task (blocking with timeout)
                try:
                    task = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if task is None:  # Shutdown signal
                    break

                if task.job_id in self._cancelled_jobs:
                    logger.info(f"Skipping cancelled job {task.job_id}")
                    self._cancelled_jobs.discard(task.job_id)
                    self._queue.task_done()
                    continue

                self._active_job_id = task.job_id

                try:
                    self._execute_job(task)
                except Exception as exc:
                    logger.error(f"Job {task.job_id} failed with exception: {exc}", exc_info=True)
                    self._handle_job_failure(task.job_id, exc)
                finally:
                    self._active_job_id = None
                    self._queue.task_done()

                    # Clean up cancelled set
                    self._cancelled_jobs.discard(task.job_id)

                    # GPU memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as exc:
                logger.critical(f"Unexpected error in worker loop: {exc}", exc_info=True)
                time.sleep(1.0)

        logger.info("GPU Worker loop ended")

    def _execute_job(self, task: JobTask) -> None:
        """Execute a single reconstruction job."""
        job_id = task.job_id

        # Load images
        images: List[Image.Image] = []
        for path_str in task.image_paths:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Input image not found: {path_str}")
            img = Image.open(path).convert("RGB")
            images.append(img)

        # Update job status
        self.job_store.update_job(
            job_id=job_id,
            status="VALIDATING",
            stage="Validating inputs",
            progress=0.1,
        )

        # Validate model
        from app.services.model_registry.registry import get_model_registry_service
        registry = get_model_registry_service()
        available, reason = registry.is_model_available(task.model_id)
        if not available:
            raise JobError(f"Model '{task.model_id}' unavailable: {reason}", job_id=job_id)

        # Run reconstruction with progress tracking
        def progress_callback(stage: str, progress: float):
            if job_id in self._cancelled_jobs:
                raise CancellationError(job_id)
            self.job_store.update_job(
                job_id=job_id,
                status=stage.upper(),
                stage=stage,
                progress=progress,
            )
            self.metrics.record_job_stage(job_id, stage, progress)

        self.metrics.record_job_start(job_id, task.model_id)

        result = self.reconstruction.reconstruct(
            job_id=job_id,
            images=images,
            model_id=task.model_id,
            options=task.options,
            progress_callback=progress_callback,
        )

        # Job completed successfully
        self.job_store.update_job(
            job_id=job_id,
            status="COMPLETED",
            stage="Reconstruction completed successfully",
            progress=1.0,
            timing_updates={
                "completed_at": time.time(),
                "elapsed_ms": result["timing"]["total_seconds"] * 1000,
            },
            results_summary={
                "num_points": result["scene_data"].get("num_points", 0),
                "exports": list(result["exports"].keys()),
                "depth_artifacts": result.get("depth_artifacts", {}),
            },
            artifacts={
                fmt: {"path": info["path"], "size_bytes": info["size_bytes"]}
                for fmt, info in result["exports"].items()
            } | result.get("depth_artifacts", {}),
        )

        self.metrics.record_job_complete(job_id, "COMPLETED")

        # Record export metrics
        for fmt, info in result["exports"].items():
            self.metrics.record_export(job_id, fmt, info["size_bytes"], info.get("valid", False))

    def _handle_job_failure(self, job_id: str, exc: Exception) -> None:
        """Handle job failure and update store."""
        error_info = {
            "code": type(exc).__name__,
            "message": str(exc),
            "remediation": "Check server logs for details",
        }

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            error_info["code"] = "CUDA_OUT_OF_MEMORY"
            error_info["message"] = "GPU memory exhausted"
            error_info["remediation"] = "Reduce input resolution, decrease view count, or select a smaller model"

        self.job_store.update_job(
            job_id=job_id,
            status="FAILED",
            stage=f"Failed: {exc}",
            error=error_info,
        )

        self.metrics.record_job_complete(job_id, "FAILED", error=str(exc))

        logger.error(f"Job {job_id} failed: {exc}")


# Singleton
_gpu_worker: Optional[GPUWorker] = None


def get_gpu_worker() -> GPUWorker:
    global _gpu_worker
    if _gpu_worker is None:
        _gpu_worker = GPUWorker()
        _gpu_worker.start()
    return _gpu_worker


def shutdown_gpu_worker() -> None:
    global _gpu_worker
    if _gpu_worker:
        _gpu_worker.stop()
        _gpu_worker = None
