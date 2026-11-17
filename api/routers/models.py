"""Model Discovery and Capability Query Endpoints."""

from fastapi import APIRouter
from api.schemas import ModelListResponse
from models.adapters import RECONSTRUCTION_REGISTRY

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("", response_model=ModelListResponse)
def list_available_models() -> ModelListResponse:
    """Discovers installed 3D reconstruction models, hardware requirements, and capabilities."""
    models = RECONSTRUCTION_REGISTRY.list_models()
    default_id = RECONSTRUCTION_REGISTRY.get_default_model_id()

    return ModelListResponse(
        status="success",
        default_model_id=default_id,
        models=models,
    )

