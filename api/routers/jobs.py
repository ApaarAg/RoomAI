"""Reconstruction Job Management and Asset Export Endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from PIL import Image

from api.schemas import (
    JobCancellationResponse,
    JobResultResponse,
    JobStatusResponse,
    JobSubmissionResponse,
    ReconstructionJobRequest,
)
from models.adapters import RECONSTRUCTION_REGISTRY
from services.job_service import GLOBAL_JOB_SERVICE
from services.job_store import GLOBAL_JOB_STORE
from services.upload_service import GLOBAL_UPLOAD_SERVICE

router = APIRouter(prefix="/api/v1/jobs", tags=["Reconstruction Jobs"])


@router.post("/reconstruct", response_model=JobSubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_reconstruction_job(req: ReconstructionJobRequest) -> JobSubmissionResponse:
    """Submits a single-view or multi-view 3D room reconstruction job.

    Validates:
    - All specified upload IDs exist.
    - View count is compatible with selected model adapter capabilities.
    - Target model adapter is recognized and ready.

    Returns HTTP 202 Accepted with job tracking ID immediately.
    """
    if not req.upload_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No upload IDs provided.")

    # 1. Verify model adapter
    model_id = req.model_id or RECONSTRUCTION_REGISTRY.get_default_model_id()
    try:
        adapter = RECONSTRUCTION_REGISTRY.get_adapter(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    caps = adapter.get_capabilities()
    if len(req.upload_ids) > caps.max_views:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_id}' supports a maximum of {caps.max_views} views; received {len(req.upload_ids)}.",
        )
    if len(req.upload_ids) < caps.min_views:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_id}' requires at least {caps.min_views} views; received {len(req.upload_ids)}.",
        )

    # 2. Check model availability
    avail, reason = adapter.is_available()
    if not avail:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model adapter '{model_id}' is currently unavailable: {reason}",
        )

    # 3. Resolve images from disk
    images: List[Image.Image] = []
    for uid in req.upload_ids:
        img = GLOBAL_UPLOAD_SERVICE.load_upload_image(uid)
        if not img:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Upload '{uid}' could not be found or loaded.",
            )
        images.append(img)

    # 4. Enqueue job
    try:
        source_upload_paths = [
            str(path) for uid in req.upload_ids
            if (path := GLOBAL_UPLOAD_SERVICE.get_upload_path(uid)) is not None
        ]
        job_id = GLOBAL_JOB_SERVICE.submit_reconstruction_job(
            images=images,
            model_id=model_id,
            options={
                "refine_depth": req.refine_depth,
                "downsample_points": req.downsample_points,
                "upload_ids": req.upload_ids,
                "source_upload_paths": source_upload_paths,
                "camera_intrinsics": req.camera_intrinsics,
            },
        )
        return JobSubmissionResponse(
            status="accepted",
            job_id=job_id,
            job_status="QUEUED",
            model_id=model_id,
            view_count=len(images),
            poll_url=f"/api/v1/jobs/{job_id}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit reconstruction job: {exc}",
        )


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Polls progress, state, and execution timing for a reconstruction job."""
    status_dict = GLOBAL_JOB_SERVICE.get_job_status(job_id)
    if not status_dict:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(**status_dict)


@router.get("/{job_id}/results", response_model=JobResultResponse)
def get_job_results(job_id: str) -> JobResultResponse:
    """Retrieves full 3D scene data, objects, surfaces, and bounding boxes for a completed job."""
    res = GLOBAL_JOB_SERVICE.get_job_results(job_id)
    if not res:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    return JobResultResponse(**res)


@router.get("/{job_id}/exports/{fmt}")
def download_export(job_id: str, fmt: str):
    """Streams validated binary glTF 2.0 (GLB) or ASCII PLY 3D reconstruction files."""
    fmt_clean = fmt.lower().strip()
    if fmt_clean not in ("glb", "ply"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported export format '{fmt}'. Supported formats: 'glb', 'ply'.",
        )

    file_path, media_type = GLOBAL_JOB_SERVICE.get_export_file(job_id, fmt_clean)
    if not file_path or not file_path.exists():
        job = GLOBAL_JOB_STORE.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
        if job["status"] != "COMPLETED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job '{job_id}' is in status '{job['status']}'. Exports are only available once COMPLETED.",
            )
        acceptance = ((job.get("results_summary") or {}).get("asset_acceptance") or {}).get("status")
        if acceptance == "REJECTED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job '{job_id}' completed computation, but its reconstruction was rejected by validation.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export file '{fmt_clean}' for job '{job_id}' could not be located.",
        )

    download_filename = f"roomai_{job_id}_{fmt_clean}.{fmt_clean}"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=download_filename,
    )


@router.post("/{job_id}/cancel", response_model=JobCancellationResponse)
def cancel_job(job_id: str) -> JobCancellationResponse:
    """Requests cancellation of a queued or actively running reconstruction job."""
    success = GLOBAL_JOB_SERVICE.cancel_job(job_id)
    if not success:
        job = GLOBAL_JOB_STORE.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
        return JobCancellationResponse(
            job_id=job_id,
            status=job["status"],
            cancelled=False,
            message=f"Job '{job_id}' could not be cancelled because it is in status '{job['status']}'.",
        )

    return JobCancellationResponse(
        job_id=job_id,
        status="CANCELLED",
        cancelled=True,
        message=f"Cancellation requested for job '{job_id}'.",
    )


@router.get("", response_model=List[dict])
def list_recent_jobs(limit: int = Query(default=20, ge=1, le=100)) -> List[dict]:
    """Lists recent reconstruction jobs."""
    return GLOBAL_JOB_STORE.list_jobs(limit=limit)

