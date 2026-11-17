"""Job Management Endpoints."""

from fastapi import APIRouter, HTTPException, status, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional
from PIL import Image

from app.api.dependencies import (
    get_gpu_worker, get_job_store, get_model_registry, get_artifact_store,
    require_model_available
)
from app.workers.gpu_worker import GPUWorker
from app.services.job_store import JobStore
from app.services.model_registry.registry import ModelRegistryService
from app.storage.artifacts import ArtifactStore
from app.api.schemas import (
    ReconstructionJobRequest, JobSubmissionResponse, JobStatusResponse,
    JobResultResponse, JobCancellationResponse,
)


router = APIRouter(prefix="/api/v1/jobs", tags=["Reconstruction Jobs"])


@router.post("/reconstruct", response_model=JobSubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_reconstruction_job(
    req: ReconstructionJobRequest,
    background_tasks: BackgroundTasks,
    worker: GPUWorker = Depends(get_gpu_worker),
    registry: ModelRegistryService = Depends(get_model_registry),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
    model_id: str = Depends(require_model_available),
):
    """Submit a 3D reconstruction job."""
    # Resolve model ID (handles aliases)
    resolved_model_id = registry.resolve_model_id(model_id)

    # Verify model availability
    caps = registry.get_model(resolved_model_id).get("capabilities", {})
    max_views = caps.get("max_views", 1)
    min_views = caps.get("min_views", 1)

    if len(req.upload_ids) > max_views:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{resolved_model_id}' supports max {max_views} views, received {len(req.upload_ids)}."
        )
    if len(req.upload_ids) < min_views:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{resolved_model_id}' requires at least {min_views} views, received {len(req.upload_ids)}."
        )

    # Load images from uploads
    images: List[Image.Image] = []
    image_paths: List[str] = []

    for uid in req.upload_ids:
        path = artifact_store.get_upload_path(uid)
        if not path or not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Upload '{uid}' not found.",
            )
        img = Image.open(path).convert("RGB")
        images.append(img)
        image_paths.append(str(path))

    # Enqueue job
    job_id = worker.enqueue_job(
        image_paths=image_paths,
        model_id=resolved_model_id,
        options={
            "refine_depth": req.refine_depth,
            "downsample_points": req.downsample_points,
            "upload_ids": req.upload_ids,
        },
    )

    return JobSubmissionResponse(
        status="accepted",
        job_id=job_id,
        job_status="QUEUED",
        model_id=resolved_model_id,
        view_count=len(images),
        poll_url=f"/api/v1/jobs/{job_id}",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
):
    """Get job status and progress."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(**job)


@router.get("/{job_id}/results", response_model=JobResultResponse)
def get_job_results(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
):
    """Get full job results."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    return JobResultResponse(
        **job,
        is_completed=job.get("status") == "COMPLETED",
    )


@router.get("/{job_id}/exports/{fmt}")
def download_export(
    job_id: str,
    fmt: str,
    job_store: JobStore = Depends(get_job_store),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
):
    """Download exported 3D asset."""
    fmt_clean = fmt.lower().strip()
    if fmt_clean not in ("glb", "ply"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{fmt}'. Supported: glb, ply.",
        )

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    if job["status"] != "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job '{job_id}' is {job['status']}. Exports only available when COMPLETED.",
        )

    artifacts = job.get("artifacts", {})
    if fmt_clean not in artifacts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export '{fmt_clean}' not found for job '{job_id}'.",
        )

    file_path = Path(artifacts[fmt_clean]["path"])
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export file missing.")

    media_type = "model/gltf-binary" if fmt_clean == "glb" else "application/octet-stream"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=f"roomai_{job_id}_{fmt_clean}.{fmt_clean}",
    )


@router.post("/{job_id}/cancel", response_model=JobCancellationResponse)
def cancel_job(
    job_id: str,
    worker: GPUWorker = Depends(get_gpu_worker),
    job_store: JobStore = Depends(get_job_store),
):
    """Cancel a job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    success = worker.cancel_job(job_id)

    if not success:
        return JobCancellationResponse(
            job_id=job_id,
            status=job["status"],
            cancelled=False,
            message=f"Job '{job_id}' could not be cancelled (status: {job['status']}).",
        )

    return JobCancellationResponse(
        job_id=job_id,
        status="CANCELLED",
        cancelled=True,
        message=f"Cancellation requested for job '{job_id}'.",
    )


@router.get("", response_model=List[dict])
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    job_store: JobStore = Depends(get_job_store),
):
    """List reconstruction jobs."""
    return job_store.list_jobs(limit=limit, status=status_filter, offset=offset)
