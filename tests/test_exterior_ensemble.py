"""Unit and integration tests for RoomAI Group B (Exterior & Building Components) detectors and ensemble."""

from pathlib import Path
from PIL import Image
import pytest

from models.detection_model import CustomObjectDetector
from schemas.responses import BoundingBox, DetectionItem, DetectionResponse
from services.detection_service import DetectionService, resolve_detector
from services.ensemble_service import EnsembleDetector

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_exterior_checkpoints_exist():
    """Verify that Group B checkpoints (pilot and fine-tuned) exist in checkpoints/."""
    b1_path = REPO_ROOT / "checkpoints" / "exterior_detector_b1_yolo11n_pilot_best.pt"
    b2_pilot_path = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_pilot_best.pt"
    b2_best_path = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
    assert b1_path.exists(), f"Missing {b1_path}"
    assert b2_pilot_path.exists(), f"Missing {b2_pilot_path}"
    assert b2_best_path.exists(), f"Missing {b2_best_path}"


def test_exterior_dataset_yaml_and_splits():
    """Verify that exterior_canonical.yaml exists and points to valid splits."""
    yaml_path = REPO_ROOT / "data" / "exterior_canonical" / "exterior_canonical.yaml"
    manifest_path = REPO_ROOT / "data" / "exterior_canonical" / "split_manifest.json"
    assert yaml_path.exists()
    assert manifest_path.exists()

    train_dir = REPO_ROOT / "data" / "exterior_canonical" / "images" / "train"
    val_dir = REPO_ROOT / "data" / "exterior_canonical" / "images" / "val"
    test_dir = REPO_ROOT / "data" / "exterior_canonical" / "images" / "test"

    assert len(list(train_dir.glob("*.jpg"))) == 458
    assert len(list(val_dir.glob("*.jpg"))) == 134
    assert len(list(test_dir.glob("*.jpg"))) == 134


def test_resolve_exterior_detector_single():
    """Verify resolve_detector instantiates single exterior detector."""
    config = {
        "checkpoint_path": "checkpoints/exterior_detector_b2_yolo11s_pilot_best.pt",
        "task_group": "exterior",
        "detector_mode": "single",
    }
    model, load_ms = resolve_detector(config, mock_mode=False)
    assert isinstance(model, CustomObjectDetector)
    assert load_ms >= 0.0


def test_resolve_exterior_detector_ensemble():
    """Verify resolve_detector instantiates exterior EnsembleDetector with WBF."""
    config = {
        "checkpoint_path": "ensemble_exterior",
        "task_group": "exterior",
        "detector_mode": "ensemble",
    }
    model, load_ms = resolve_detector(config, mock_mode=False)
    assert isinstance(model, EnsembleDetector)
    assert model.fusion_method == "wbf"
    assert len(model.models) == 2


def test_exterior_ensemble_predict_mock():
    """Verify exterior detection service works in mock mode without disk error."""
    service = DetectionService(
        mock_mode=True,
        config={"checkpoint_path": "ensemble_exterior", "task_group": "exterior"}
    )
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    res = service.run(img)
    assert res["success"] is True
    assert res["response"] is not None
    assert len(res["response"].detections) > 0


def test_exterior_b6_checkpoint_and_dataset_integrity():
    """Verify that Model B6 checkpoint exists and b6 expanded dataset has correct splits."""
    b6_path = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
    assert b6_path.exists(), f"Missing B6 checkpoint {b6_path}"

    b6_data_dir = REPO_ROOT / "data" / "exterior_b6_expanded"
    assert (b6_data_dir / "dataset.yaml").exists()
    assert (b6_data_dir / "b6_manifest.json").exists()

    val_imgs = list((b6_data_dir / "images" / "val").glob("*.jpg"))
    test_imgs = list((b6_data_dir / "images" / "test").glob("*.jpg"))
    train_imgs = list((b6_data_dir / "images" / "train").glob("*.jpg"))

    assert len(val_imgs) == 154
    assert len(test_imgs) == 154
    assert len(train_imgs) == 1302


def test_resolve_exterior_b6_candidate():
    """Verify resolve_detector instantiates Model B6 candidate."""
    config = {
        "checkpoint_path": "checkpoints/exterior_detector_b6_expanded_best.pt",
        "task_group": "exterior",
        "detector_mode": "single",
        "confidence_threshold": 0.25,
    }
    model, load_ms = resolve_detector(config, mock_mode=False)
    assert isinstance(model, CustomObjectDetector)
    assert load_ms >= 0.0


def test_exterior_b6_service_real_and_mock_inference():
    """Verify DetectionService executes successfully with Model B6 in both mock and real mode."""
    # 1. Mock Mode
    mock_service = DetectionService(
        mock_mode=True,
        config={
            "checkpoint_path": "checkpoints/exterior_detector_b6_expanded_best.pt",
            "task_group": "exterior",
        }
    )
    img = Image.new("RGB", (640, 640), color=(120, 140, 160))
    mock_res = mock_service.run(img)
    assert mock_res["success"] is True
    assert mock_res["response"].is_simulated is True

    # 2. Real Mode
    real_service = DetectionService(
        mock_mode=False,
        config={
            "checkpoint_path": "checkpoints/exterior_detector_b6_expanded_best.pt",
            "task_group": "exterior",
            "confidence_threshold": 0.25,
            "device": "cuda:0",
        }
    )
    real_res = real_service.run(img)
    assert real_res["success"] is True
    assert real_res["response"] is not None
    assert real_res["response"].is_simulated is False

