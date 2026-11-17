"""Health and Readiness Endpoints."""

from fastapi import APIRouter, Depends
import torch

from app.core.config import settings
from app.api.dependencies import get_model_registry
from app.services.model_registry.registry import ModelRegistryService
from app.monitoring.metrics import get_metrics_collector
from app.api.schemas import HealthResponse, ReadinessResponse


router = APIRouter(tags=["Diagnostics"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Service liveness probe."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        service="RoomAI Backend API",
        timestamp=time.time(),
    )


@router.get("/ready", response_model=ReadinessResponse)
def get_readiness(registry: ModelRegistryService = Depends(get_model_registry)):
    """Hardware readiness and model availability probe."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    total_vram_gb = 0.0
    free_vram_gb = 0.0

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = round(props.total_memory / (1024**3), 2)
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        free_vram_gb = round(free_mb / 1024, 2)

    models = registry.list_models()
    ready_models = [m["model_id"] for m in models if m.get("status") in ("PRODUCTION_READY", "REFERENCE", "VALIDATION_CANDIDATE")]

    # Check at least one model has verified checkpoint
    checkpoints_ready = False
    for model_id in ready_models:
        ok, _ = registry.verify_checkpoint(model_id)
        if ok:
            checkpoints_ready = True
            break

    is_ready = checkpoints_ready and (cuda_available or len(ready_models) == 0)
    status_str = "ready" if is_ready else "degraded"

    return ReadinessResponse(
        status=status_str,
        cuda_available=cuda_available,
        device_name=device_name,
        free_vram_gb=free_vram_gb,
        total_vram_gb=total_vram_gb,
        checkpoints_ready=checkpoints_ready,
        models_ready=ready_models,
    )


@router.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    metrics = get_metrics_collector()
    return PlainTextResponse(content=metrics.get_prometheus_metrics(), media_type="text/plain")


import time