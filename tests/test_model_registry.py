"""Unit tests for RoomAI Model Registry and Dynamic VRAM Routing."""

from pathlib import Path
import pytest

from models.registry import GLOBAL_MODEL_REGISTRY, ModelRegistry, ModelSpec

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_model_registry_default_models_registered():
    """Verify that all task domains have registered model specifications."""
    reg = GLOBAL_MODEL_REGISTRY
    indoor_models = reg.list_by_task("indoor_detection")
    exterior_models = reg.list_by_task("exterior_detection")
    parking_models = reg.list_by_task("parking_detection")
    seg_models = reg.list_by_task("segmentation")

    assert len(indoor_models) >= 3
    assert len(exterior_models) >= 2
    assert len(parking_models) >= 2
    assert len(seg_models) >= 1


def test_model_registry_primary_models():
    """Verify primary model resolution for each task."""
    reg = GLOBAL_MODEL_REGISTRY

    indoor_primary = reg.get_primary_for_task("indoor_detection")
    assert indoor_primary is not None
    assert indoor_primary.model_id == "indoor_a3_800px"
    assert indoor_primary.exists()
    assert indoor_primary.input_resolution == (800, 800)

    # Verify A2 rollback model remains registered and available
    indoor_a2 = reg.get_model_spec("indoor_a2_finetuned")
    assert indoor_a2 is not None
    assert indoor_a2.exists()
    assert indoor_a2.input_resolution == (640, 640)

    exterior_primary = reg.get_primary_for_task("exterior_detection")
    assert exterior_primary is not None
    assert exterior_primary.model_id == "exterior_b2_finetuned"
    assert exterior_primary.exists()
    assert exterior_primary.is_primary is True

    # Verify B5 and B6 candidates are registered and ready
    exterior_b5 = reg.get_model_spec("exterior_b5_transfer")
    assert exterior_b5 is not None
    assert exterior_b5.exists()
    assert exterior_b5.is_primary is False

    exterior_b6 = reg.get_model_spec("exterior_b6_expanded")
    assert exterior_b6 is not None
    assert exterior_b6.exists()
    assert exterior_b6.is_primary is False

    parking_primary = reg.get_primary_for_task("parking_detection")
    assert parking_primary is not None
    assert parking_primary.model_id == "parking_c2_pilot"
    assert parking_primary.exists()

    seg_primary = reg.get_primary_for_task("segmentation")
    assert seg_primary is not None
    assert seg_primary.exists()


def test_model_registry_vram_headroom_check():
    """Verify VRAM headroom reporting returns valid numerical values."""
    reg = GLOBAL_MODEL_REGISTRY
    info = reg.check_vram_headroom()
    assert "allocated_mb" in info
    assert "reserved_mb" in info
    assert "max_budget_mb" in info
    assert info["max_budget_mb"] == 4500.0


def test_custom_segmenter_with_pretrained_checkpoint(sample_pil_image):
    """Verify CustomRoomSegmenter loads and predicts when checkpoint is explicitly provided."""
    from models.segmentation_model import CustomRoomSegmenter
    from schemas.responses import SegmentationResponse

    seg = CustomRoomSegmenter(config={"checkpoint_path": "checkpoints/yolo11n-seg.pt", "confidence_threshold": 0.25})
    assert seg.checkpoint_path == "checkpoints/yolo11n-seg.pt"
    resp, mask = seg.predict(sample_pil_image)
    assert isinstance(resp, SegmentationResponse)
    assert resp.is_simulated is False
    assert mask is not None
    assert mask.shape == (sample_pil_image.height, sample_pil_image.width)
