"""Experimental/Hunyuan Endpoints."""

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from typing import List, Optional
from PIL import Image

from app.api.dependencies import get_gpu_worker, get_job_store, get_model_registry, get_artifact_store
from app.workers.gpu_worker import GPUWorker
from app.services.job_store import JobStore
from app.services.model_registry.registry import ModelRegistryService
from app.storage.artifacts import ArtifactStore
from app.api.schemas import ReconstructionJobRequest, JobSubmissionResponse


router = APIRouter(prefix="/api/v1/experimental", tags=["Experimental"])


@router.post("/hunyuan/reconstruct", response_model=JobSubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_hunyuan_reconstruction(
    req: ReconstructionJobRequest,
    worker: GPUWorker = Depends(get_gpu_worker),
    registry: ModelRegistryService = Depends(get_model_registry),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
):
    """Submit a HunyuanWorld-Mirror reconstruction job (strict, no fallback)."""
    # Force Hunyuan model
    hunyuan_models = ["hunyuan_world_mirror_1b", "worldmirror1_baseline", "worldmirror1_lora_r8", "worldmirror1_stage2"]
    model_id = req.model_id or "worldmirror1_lora_r8"

    if model_id not in hunyuan_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model for Hunyuan endpoint: {model_id}. Must be one of {hunyuan_models}."
        )

    # Check availability (strict - no fallback)
    available, reason = registry.is_model_available(model_id)
    if not available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hunyuan model '{model_id}' unavailable: {reason}. No fallback available.",
        )

    # Load images
    images: List[Image.Image] = []
    image_paths: List[str] = []

    for uid in req.upload_ids:
        path = artifact_store.get_upload_path(uid)
        if not path or not path.exists():
            raise HTTPException(status_code=404, detail=f"Upload '{uid}' not found.")
        img = Image.open(path).convert("RGB")
        images.append(img)
        image_paths.append(str(path))

    # Enqueue with strict options
    job_id = worker.enqueue_job(
        image_paths=image_paths,
        model_id=model_id,
        options={
            "refine_depth": req.refine_depth,
            "downsample_points": req.downsample_points,
            "upload_ids": req.upload_ids,
            "strict_no_fallback": True,
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


@router.post("/hunyuan/verify/{model_id}")
def verify_hunyuan_model(
    model_id: str,
    registry: ModelRegistryService = Depends(get_model_registry),
):
    """Verify Hunyuan model checkpoint and architecture compatibility."""
    try:
        spec = registry.get_model(model_id)
        ok, reason = registry.verify_checkpoint(model_id)
        available, avail_reason = registry.is_model_available(model_id)

        return {
            "model_id": model_id,
            "name": spec.get("name"),
            "version": spec.get("version"),
            "status": spec.get("status"),
            "checkpoint_verified": ok,
            "verification_error": reason,
            "hardware_available": available,
            "availability_reason": avail_reason,
            "checkpoint_path": str(registry.get_checkpoint_path(model_id)),
            "parameter_count": spec.get("parameter_count"),
            "trainable_partition": spec.get("trainable_partition"),
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))