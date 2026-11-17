"""Tests for Reconstruction Model Adapters and Registry."""

from PIL import Image
import pytest

from models.adapters import (
    HunyuanMirrorAdapter,
    ModelCapabilities,
    ReconstructionModelAdapter,
    ReconstructionModelRegistry,
    RoomAICompositeAdapter,
)


def test_registry_lists_capabilities():
    """Registry must report all available models with typed capabilities."""
    registry = ReconstructionModelRegistry(mock_mode=True)
    models = registry.list_models()
    assert len(models) >= 2

    model_ids = [m.model_id for m in models]
    assert "roomai_composite_v1" in model_ids
    assert "hunyuan_world_mirror_1b" in model_ids


def test_composite_adapter_capabilities():
    """Composite adapter must declare metric depth and single/multi-view support."""
    adapter = RoomAICompositeAdapter(mock_mode=True)
    caps = adapter.get_capabilities()
    assert caps.supports_single_view is True
    assert caps.supports_multi_view is True
    assert caps.supports_metric_depth is True
    assert caps.min_views == 1
    assert caps.max_views == 10
    assert caps.status == "PRODUCTION_READY"


def test_composite_adapter_input_validation():
    """Composite adapter must reject empty and excessive input sequences."""
    adapter = RoomAICompositeAdapter(mock_mode=True)

    # Empty
    valid, err = adapter.validate_inputs([])
    assert valid is False
    assert "At least one" in err

    # Too many views (> 10)
    excess_imgs = [Image.new("RGB", (64, 64)) for _ in range(11)]
    valid, err = adapter.validate_inputs(excess_imgs)
    assert valid is False
    assert "Maximum supported" in err

    # Valid
    valid, err = adapter.validate_inputs([Image.new("RGB", (64, 64))])
    assert valid is True
    assert err is None


def test_hunyuan_adapter_status_and_capabilities():
    """HunyuanWorld-Mirror adapter must report status and enforce 6 views max."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    adapter = HunyuanMirrorAdapter(mock_mode=False)
    caps = adapter.get_capabilities()
    assert caps.status in ("STAGED", "UNAVAILABLE")
    assert caps.max_views == 6
    assert caps.supports_surface_normals is True
    assert caps.supports_3dgs is False
    assert caps.supports_metric_depth is False
    assert caps.supports_mesh_glb is False

    if not adapter.is_available()[0]:
        # Reconstruct must raise clear message if weights or VRAM are unavailable
        with pytest.raises(NotImplementedError) as exc:
            adapter.reconstruct([Image.new("RGB", (64, 64))])
        assert "unavailable" in str(exc.value).lower()

