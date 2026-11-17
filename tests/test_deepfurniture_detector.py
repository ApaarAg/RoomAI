"""Comprehensive integration tests for fine-tuned DeepFurniture YOLO11n detector."""

from pathlib import Path
from PIL import Image
import pytest
import yaml

from models.detection_model import (
    CustomObjectDetector,
    MockObjectDetector,
    get_cached_yolo,
    clear_detector_cache,
)
from schemas.responses import DetectionResponse
from services.detection_service import DetectionService

REPO_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = REPO_ROOT / "data" / "deepfurniture" / "deepfurniture.yaml"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "deepfurniture_yolo11n_finetune_best.pt"

EXPECTED_CLASSES = [
    "cabinet#shelf",
    "table",
    "chair#stool",
    "lamp",
    "door",
    "bed",
    "sofa",
    "plant",
    "decoration",
    "curtain",
    "home-appliance",
]


def test_checkpoint_exists_and_class_mapping():
    """Verify fine-tuned checkpoint exists and output head matches deepfurniture.yaml 11 classes."""
    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}"
    assert YAML_PATH.exists(), f"Missing dataset YAML: {YAML_PATH}"

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yaml_names = list(cfg["names"].values())
    assert len(yaml_names) == 11
    assert yaml_names == EXPECTED_CLASSES

    detector = CustomObjectDetector()
    detector.load_model()
    assert detector.model is not None
    assert len(detector.model.names) == 11
    for idx, name in enumerate(EXPECTED_CLASSES):
        assert detector.model.names[idx] == name


def test_real_inference_response_schema(sample_pil_image: Image.Image):
    """Verify real inference produces non-simulated DetectionResponse matching strict schema."""
    detector = CustomObjectDetector()
    response = detector.predict(sample_pil_image, confidence_threshold=0.20)

    assert isinstance(response, DetectionResponse)
    assert response.is_simulated is False
    assert response.checkpoint == "deepfurniture_yolo11n_finetune_best.pt"
    assert response.inference_time_ms > 0
    assert response.device in ["cuda:0", "cpu"]
    assert response.total_objects == len(response.detections)

    for item in response.detections:
        assert 0 <= item.class_id <= 10
        assert item.label in EXPECTED_CLASSES
        assert item.confidence >= 0.20
        b = item.box
        assert 0.0 <= b.x_min <= 1.0
        assert 0.0 <= b.y_min <= 1.0
        assert 0.0 <= b.x_max <= 1.0
        assert 0.0 <= b.y_max <= 1.0
        assert b.x_max >= b.x_min
        assert b.y_max >= b.y_min


def test_confidence_filtering(sample_pil_image: Image.Image):
    """Verify confidence filtering behaves monotonically."""
    detector = CustomObjectDetector()
    resp_low = detector.predict(sample_pil_image, confidence_threshold=0.10)
    resp_high = detector.predict(sample_pil_image, confidence_threshold=0.95)

    assert resp_low.total_objects >= resp_high.total_objects
    for item in resp_high.detections:
        assert item.confidence >= 0.95


def test_mock_mode_preserved(sample_pil_image: Image.Image):
    """Verify MockObjectDetector retains deterministic mock behavior and is_simulated=True."""
    mock_detector = MockObjectDetector()
    mock_resp = mock_detector.predict(sample_pil_image)

    assert mock_resp.is_simulated is True
    assert mock_resp.checkpoint == "mock_simulated.pt"
    assert mock_resp.total_objects > 0
    labels = [d.label for d in mock_resp.detections]
    assert "couch" in labels
    assert "dining table" in labels


def test_cached_model_loading():
    """Verify get_cached_yolo returns identical model instance from cache."""
    clear_detector_cache()
    det1 = CustomObjectDetector()
    det1.load_model()
    m1 = det1.model

    det2 = CustomObjectDetector()
    det2.load_model()
    m2 = det2.model

    assert m1 is m2, "Model instance was reloaded instead of using cache!"


def test_missing_checkpoint_fails_cleanly(sample_pil_image: Image.Image):
    """Verify missing checkpoint raises clear error without silent fallback."""
    det = CustomObjectDetector(config={"checkpoint_path": "checkpoints/nonexistent_model.pt"})
    with pytest.raises((FileNotFoundError, RuntimeError)) as exc_info:
        det.predict(sample_pil_image)
    assert "nonexistent_model.pt" in str(exc_info.value)


def test_detection_service_integration_with_finetuned_checkpoint(sample_pil_image: Image.Image):
    """Verify DetectionService end-to-end execution in real mode with fine-tuned checkpoint."""
    service = DetectionService(mock_mode=False)
    result = service.run(sample_pil_image, confidence_threshold=0.25, save_annotated=False)

    assert result["success"] is True
    assert result["is_implemented"] is True
    assert result["error_message"] is None
    assert result["annotated_image"] is not None
    resp = result["response"]
    assert resp.is_simulated is False
    assert resp.checkpoint == "deepfurniture_yolo11n_finetune_best.pt"
