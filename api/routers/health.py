"""Health and Readiness Diagnostic Endpoints."""

import time
from fastapi import APIRouter
import torch

from api.routers import detect
from api.schemas import HealthResponse, ReadinessResponse
from models.adapters import RECONSTRUCTION_REGISTRY
from utils.device import get_device_info

router = APIRouter(tags=["Diagnostics"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Service liveness probe."""
    return HealthResponse(
        status="healthy",
        version="1.4.0",
        service="RoomAI Backend API",
        timestamp=time.time(),
    )


@router.get("/ready", response_model=ReadinessResponse)
def get_readiness() -> ReadinessResponse:
    """Hardware readiness and model availability probe."""
    dev_info = get_device_info()
    models = RECONSTRUCTION_REGISTRY.list_models()
    ready_models = [m.model_id for m in models if m.status == "PRODUCTION_READY"]

    # Check Model A3 engine status and surface any startup failure
    a3_engine = detect.GLOBAL_MODEL_A3_ENGINE
    if a3_engine.is_warmed and a3_engine.startup_error is None:
        if a3_engine.model_name not in ready_models:
            ready_models.append(a3_engine.model_name)

    startup_err = a3_engine.startup_error
    is_ready = len(ready_models) > 0 and a3_engine.is_warmed and (startup_err is None)
    status_str = "ready" if is_ready else "degraded"

    return ReadinessResponse(
        status=status_str,
        cuda_available=dev_info.get("cuda_available", False),
        device_name=dev_info.get("device_name", "CPU"),
        free_vram_gb=dev_info.get("free_vram_gb", 0.0),
        total_vram_gb=dev_info.get("total_vram_gb", 0.0),
        checkpoints_ready=is_ready,
        models_ready=ready_models,
        startup_error=startup_err,
    )

