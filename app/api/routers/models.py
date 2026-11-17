"""Model Discovery Endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.api.dependencies import get_model_registry
from app.services.model_registry.registry import ModelRegistryService
from app.api.schemas import ModelListResponse, ModelCapabilities


router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("", response_model=ModelListResponse)
def list_available_models(
    registry: ModelRegistryService = Depends(get_model_registry),
):
    """List all registered models with capabilities."""
    models = registry.list_models()
    try:
        status_info = registry.get_model_status("roomai_composite_v1")
        default_id = "roomai_composite_v1" if status_info.get("available") else list(registry.get_all_aliases().values())[0]
    except Exception:
        default_id = list(registry.get_all_aliases().values())[0] if registry.get_all_aliases() else "worldmirror1_baseline"

    # Convert to ModelCapabilities
    capabilities = []
    for m in models:
        caps = m.get("capabilities", {})
        capabilities.append(ModelCapabilities(
            model_id=m["model_id"],
            name=m.get("name", ""),
            version=m.get("version", ""),
            status=m.get("status", ""),
            supports_single_view=caps.get("supports_single_view", True),
            supports_multi_view=caps.get("supports_multi_view", False),
            min_views=caps.get("min_views", 1),
            max_views=caps.get("max_views", 1),
            supports_metric_depth=caps.get("supports_metric_depth", True),
            supports_surface_normals=caps.get("supports_surface_normals", False),
            supports_3dgs=caps.get("supports_3dgs", False),
            supports_mesh_glb=caps.get("supports_mesh_glb", True),
            supports_point_cloud_ply=caps.get("supports_point_cloud_ply", True),
            device=caps.get("device", "cuda"),
            memory_required_mb=caps.get("memory_required_mb", 1500),
            description=m.get("role", ""),
        ))

    return ModelListResponse(
        status="success",
        default_model_id=default_id,
        models=capabilities,
    )


@router.get("/{model_id}")
def get_model_details(
    model_id: str,
    registry: ModelRegistryService = Depends(get_model_registry),
):
    """Get detailed model information and status."""
    try:
        status_info = registry.get_model_status(model_id)
        return {
            "status": "success",
            **status_info,
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{model_id}/verify")
def verify_model_checkpoint(
    model_id: str,
    registry: ModelRegistryService = Depends(get_model_registry),
):
    """Verify model checkpoint integrity."""
    try:
        ok, reason = registry.verify_checkpoint(model_id)
        return {
            "model_id": model_id,
            "verified": ok,
            "reason": reason,
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))