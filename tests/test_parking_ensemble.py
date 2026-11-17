"""Unit and integration tests for RoomAI Group C (Parking & Outdoor Property) detectors and ensemble."""

from pathlib import Path
from PIL import Image
import pytest

from models.detection_model import CustomObjectDetector
from schemas.responses import BoundingBox, DetectionItem, DetectionResponse
from services.detection_service import DetectionService, resolve_detector
from services.ensemble_service import EnsembleDetector

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_parking_checkpoints_exist():
    """Verify that both Group C pilot checkpoints exist in checkpoints/."""
    c1_path = REPO_ROOT / "checkpoints" / "parking_detector_c1_yolo11n_pilot_best.pt"
    c2_path = REPO_ROOT / "checkpoints" / "parking_detector_c2_yolo11s_pilot_best.pt"
    assert c1_path.exists(), f"Missing {c1_path}"
    assert c2_path.exists(), f"Missing {c2_path}"


def test_parking_dataset_yaml_and_splits():
    """Verify that parking_canonical.yaml exists and points to valid splits."""
    yaml_path = REPO_ROOT / "data" / "parking_canonical" / "parking_canonical.yaml"
    manifest_path = REPO_ROOT / "data" / "parking_canonical" / "split_manifest.json"
    assert yaml_path.exists()
    assert manifest_path.exists()

    train_dir = REPO_ROOT / "data" / "parking_canonical" / "images" / "train"
    val_dir = REPO_ROOT / "data" / "parking_canonical" / "images" / "val"
    test_dir = REPO_ROOT / "data" / "parking_canonical" / "images" / "test"

    assert len(list(train_dir.glob("*.jpg"))) == 510
    assert len(list(val_dir.glob("*.jpg"))) == 120
    assert len(list(test_dir.glob("*.jpg"))) == 120


def test_resolve_parking_detector_single():
    """Verify resolve_detector instantiates single parking detector."""
    config = {
        "checkpoint_path": "checkpoints/parking_detector_c2_yolo11s_pilot_best.pt",
        "task_group": "parking",
        "detector_mode": "single",
    }
    model, load_ms = resolve_detector(config, mock_mode=False)
    assert isinstance(model, CustomObjectDetector)
    assert load_ms >= 0.0


def test_resolve_parking_detector_ensemble():
    """Verify resolve_detector instantiates parking EnsembleDetector with WBF."""
    config = {
        "checkpoint_path": "ensemble_parking",
        "task_group": "parking",
        "detector_mode": "ensemble",
    }
    model, load_ms = resolve_detector(config, mock_mode=False)
    assert isinstance(model, EnsembleDetector)
    assert model.fusion_method == "wbf"
    assert len(model.models) == 2


def test_parking_ensemble_predict_mock():
    """Verify parking detection service works in mock mode without disk error."""
    service = DetectionService(
        mock_mode=True,
        config={"checkpoint_path": "ensemble_parking", "task_group": "parking"}
    )
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    res = service.run(img)
    assert res["success"] is True
    assert res["response"] is not None
    assert len(res["response"].detections) > 0
