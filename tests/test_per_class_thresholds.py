"""Unit tests for Phase 11 Controlled Per-Class Threshold Calibration.

Tests:
1. Production Default: B2 remains active production exterior model; B4 is strictly experimental.
2. Candidate Isolation: exterior_detection_b4_experimental is disabled by default.
3. Default Uniform Threshold: Standard behavior when per-class thresholds are disabled.
4. Per-Class Filtering: Real/mock filtering using class-specific thresholds.
5. Unknown Class Fallback: Unrecognized class gracefully falls back to default threshold.
6. Invalid Threshold Validation: Catches non-dict, non-numeric, and out-of-bounds (0, 1] thresholds.
7. Mock Mode Separation: Mock detector maintains is_simulated=True and separate codepaths.
8. Strict Failure / No Silent Fallback: Missing checkpoint raises error without falling back.
9. Rollback Verification: Production B2 config loads and executes with uniform 0.25 threshold.
"""

from pathlib import Path
import pytest
from PIL import Image
import yaml

from models.detection_model import CustomObjectDetector, MockObjectDetector
from services.detection_service import DetectionService, resolve_detector


@pytest.fixture
def sample_pil_image():
    """Generates a simple 640x640 RGB image for testing."""
    return Image.new("RGB", (640, 640), color=(128, 128, 128))


def test_production_default_remains_b2():
    """Verify B2 remains active production model and B4 is not enabled by default."""
    config_path = Path("config/model_config.yaml")
    assert config_path.exists(), "config/model_config.yaml must exist."

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Active production config
    ext_cfg = cfg.get("exterior_detection", {})
    assert ext_cfg.get("enabled") is True, "Production exterior detection must be enabled."
    assert "exterior_detector_b2_yolo11s_best.pt" in ext_cfg.get("checkpoint_path", ""), (
        "Production exterior detector checkpoint must remain Model B2."
    )
    assert ext_cfg.get("status") == "production_ready", "Production exterior detector status must be 'production_ready'."

    # Experimental B4 config
    exp_cfg = cfg.get("exterior_detection_b4_experimental", {})
    assert exp_cfg.get("enabled") is False, "B4 candidate must be disabled (opt-in) by default."
    assert exp_cfg.get("status") == "experimental_candidate", "B4 candidate status must be 'experimental_candidate'."
    assert exp_cfg.get("use_per_class_thresholds") is True, "B4 experimental config must specify use_per_class_thresholds: true."
    assert "porch" in exp_cfg.get("per_class_thresholds", {}), "B4 experimental per_class_thresholds must include porch."
    assert exp_cfg["per_class_thresholds"]["porch"] == 0.17, "Porch candidate threshold must be 0.17."


def test_mock_detector_per_class_filtering(sample_pil_image):
    """Test MockObjectDetector filters items based on per-class thresholds."""
    # Mock items: couch (0.92), dining table (0.86), chair (0.81), potted plant (0.77), tv (0.89)
    config = {
        "use_per_class_thresholds": True,
        "per_class_thresholds": {
            "couch": 0.95,       # Should filter out couch (0.92 < 0.95)
            "chair": 0.80,       # Should keep chair (0.81 >= 0.80)
            "tv": 0.90,          # Should filter out tv (0.89 < 0.90)
            "dining table": 0.85 # Should keep dining table (0.86 >= 0.85)
            # potted plant is not specified: should fall back to confidence_threshold
        },
    }
    detector = MockObjectDetector(config)
    # Default confidence_threshold=0.80 -> potted plant (0.77 < 0.80) should be dropped
    res = detector.predict(sample_pil_image, confidence_threshold=0.80)

    assert res.is_simulated is True, "Mock detector must always mark output as simulated."
    labels = [d.label for d in res.detections]
    assert "couch" not in labels, "Couch (0.92) should be filtered out by threshold 0.95."
    assert "tv" not in labels, "TV (0.89) should be filtered out by threshold 0.90."
    assert "potted plant" not in labels, "Potted plant (0.77) should be filtered out by fallback threshold 0.80."
    assert "chair" in labels, "Chair (0.81) should be retained by threshold 0.80."
    assert "dining table" in labels, "Dining table (0.86) should be retained by threshold 0.85."


def test_mock_detector_unknown_class_fallback(sample_pil_image):
    """Test that items with classes not in per_class_thresholds fall back to confidence_threshold."""
    config = {
        "use_per_class_thresholds": True,
        "per_class_thresholds": {
            "unknown_future_class": 0.10,
        },
    }
    detector = MockObjectDetector(config)
    # With confidence_threshold=0.85, couch (0.92), dining table (0.86), tv (0.89) pass; chair (0.81) & potted plant (0.77) fail
    res = detector.predict(sample_pil_image, confidence_threshold=0.85)
    labels = [d.label for d in res.detections]
    assert "couch" in labels
    assert "dining table" in labels
    assert "tv" in labels
    assert "chair" not in labels
    assert "potted plant" not in labels


def test_invalid_per_class_thresholds_validation():
    """Test validation catches missing, empty, non-numeric, or out-of-range thresholds."""
    # Missing per_class_thresholds dict when enabled
    with pytest.raises(ValueError, match="missing or empty"):
        CustomObjectDetector({"use_per_class_thresholds": True})

    with pytest.raises(ValueError, match="missing or empty"):
        CustomObjectDetector({"use_per_class_thresholds": True, "per_class_thresholds": {}})

    # Non-numeric threshold
    with pytest.raises(ValueError, match="Invalid threshold value"):
        CustomObjectDetector({
            "use_per_class_thresholds": True,
            "per_class_thresholds": {"window": "not_a_number"},
        })

    # Out of bounds: 0.0 or negative
    with pytest.raises(ValueError, match="must be in"):
        CustomObjectDetector({
            "use_per_class_thresholds": True,
            "per_class_thresholds": {"window": 0.0},
        })

    with pytest.raises(ValueError, match="must be in"):
        CustomObjectDetector({
            "use_per_class_thresholds": True,
            "per_class_thresholds": {"window": -0.15},
        })

    # Out of bounds: > 1.0
    with pytest.raises(ValueError, match="must be in"):
        CustomObjectDetector({
            "use_per_class_thresholds": True,
            "per_class_thresholds": {"window": 1.25},
        })


def test_strict_failure_no_silent_fallback():
    """Verify that a non-existent checkpoint raises FileNotFoundError and does not fallback to mock."""
    config = {
        "checkpoint_path": "checkpoints/non_existent_detector_xyz.pt",
        "device": "cpu",
    }
    detector = CustomObjectDetector(config)
    with pytest.raises(FileNotFoundError):
        detector.load_model()


def test_detection_service_with_per_class_thresholds(sample_pil_image):
    """Verify DetectionService correctly passes and tracks per_class_thresholds in mock mode."""
    config = {
        "use_per_class_thresholds": True,
        "per_class_thresholds": {
            "couch": 0.90,
            "dining table": 0.90,
            "tv": 0.90,
            "chair": 0.90,
            "potted plant": 0.90,
        },
    }
    service = DetectionService(mock_mode=True, config=config)
    result = service.run(sample_pil_image, confidence_threshold=0.25, save_annotated=False)

    assert result["success"] is True
    assert result["timing_metrics"]["use_per_class_thresholds"] is True
    # In mock items, only couch has confidence 0.92 >= 0.90
    assert result["response"].total_objects == 1
    assert result["response"].detections[0].label == "couch"


def test_production_b2_rollback_and_uniform_behavior(sample_pil_image):
    """Verify that B2 config without per_class_thresholds runs in standard uniform mode."""
    with open("config/model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prod_config = cfg["exterior_detection"]
    assert prod_config.get("use_per_class_thresholds", False) is False, "Production B2 must not have per_class_thresholds enabled."

    service = DetectionService(mock_mode=True, config=prod_config)
    result = service.run(sample_pil_image, confidence_threshold=0.80, save_annotated=False)

    assert result["success"] is True
    assert result["timing_metrics"]["use_per_class_thresholds"] is False
    # All simulated items >= 0.80 (couch 0.92, dining table 0.86, chair 0.81, tv 0.89) -> 4 items
    assert result["response"].total_objects == 4
