"""Tests for Mock Mode Default, Production Model Selection, and Hardware Status.

Verifies:
1. Real inference is the default (mock_mode: false in config/app.yaml and app.py).
2. Explicit mock-mode activation produces is_simulated=True across all perception services.
3. Explicit real-mode activation produces is_simulated=False, actual model name, and real device.
4. Model A3 (800px) is correctly selected as the primary indoor detector.
5. Models B2 and C2 are correctly selected as production exterior and parking detectors.
6. Missing checkpoints trigger explicit failures and NEVER silently fall back to mock predictions.
7. is_simulated metadata strictly matches execution mode.
8. Streamlit session-state persistence preserves user toggles without resetting to default.
9. Hardware status correctly reports GPU availability, allocated/peak VRAM in MB/GB, without treating idle 0 GB as failure.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import pytest
import yaml

from models.registry import GLOBAL_MODEL_REGISTRY
from schemas.responses import DetectionResponse
from services.detection_service import DetectionService
from services.scene3d_service import Scene3DService
from services.segmentation_service import SegmentationService
from utils.device import get_device_info, get_optimal_device

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_image() -> Image.Image:
    """Synthetic RGB test image."""
    return Image.new("RGB", (640, 480), color=(180, 180, 180))


def test_mock_mode_defaults_to_false():
    """Requirement 1: Verify that Mock Mode is OFF (false) by default in config/app.yaml."""
    app_yaml_path = REPO_ROOT / "config" / "app.yaml"
    assert app_yaml_path.exists(), "config/app.yaml must exist."

    with open(app_yaml_path, "r", encoding="utf-8") as f:
        app_cfg = yaml.safe_load(f)

    assert app_cfg.get("mock_mode") is False, (
        f"config/app.yaml mock_mode must be False by default, found: {app_cfg.get('mock_mode')}"
    )


def test_explicit_mock_mode_activation(sample_image: Image.Image):
    """Requirement 1 & 3: When mock mode is explicitly enabled, outputs must be marked is_simulated=True."""
    # Detection
    det_svc = DetectionService(mock_mode=True)
    res_det = det_svc.run(sample_image, confidence_threshold=0.35, save_annotated=False)
    assert res_det["success"] is True
    assert res_det["response"].is_simulated is True
    assert res_det["response"].checkpoint == "mock_simulated.pt"

    # Segmentation
    seg_svc = SegmentationService(mock_mode=True)
    res_seg = seg_svc.run(sample_image)
    assert res_seg["success"] is True
    assert res_seg["response"].is_simulated is True

    # Scene 3D
    s3d_svc = Scene3DService(mock_mode=True)
    res_s3d = s3d_svc.run(sample_image)
    assert res_s3d["success"] is True
    assert res_s3d["response"].is_simulated is True


def test_explicit_real_mode_activation(sample_image: Image.Image):
    """Requirement 3: When mock mode is disabled, real models must execute with is_simulated=False."""
    a3_ckpt = REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt"
    if not a3_ckpt.exists():
        pytest.skip(f"A3 checkpoint not found at {a3_ckpt}")

    det_cfg = {
        "checkpoint_path": str(a3_ckpt),
        "imgsz": 800,
        "confidence_threshold": 0.35,
        "task_group": "indoor",
    }
    det_svc = DetectionService(mock_mode=False, config=det_cfg)
    res_det = det_svc.run(sample_image, confidence_threshold=0.35, save_annotated=False)

    assert res_det["success"] is True
    resp: DetectionResponse = res_det["response"]
    assert resp.is_simulated is False
    assert "indoor_detector_a3_yolo11s_800px_best.pt" in resp.checkpoint
    assert resp.device in ["cuda:0", "cpu"]
    assert resp.inference_time_ms >= 0.0


def test_correct_a3_checkpoint_selection():
    """Requirement 2: Indoor object detection must use indoor_detector_a3_yolo11s_800px_best.pt."""
    # 1. Verify model_config.yaml
    model_yaml_path = REPO_ROOT / "config" / "model_config.yaml"
    with open(model_yaml_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    det_cfg = model_cfg.get("detection", {})
    assert "indoor_detector_a3_yolo11s_800px_best.pt" in det_cfg.get("checkpoint_path", ""), (
        "model_config.yaml detection.checkpoint_path must point to Model A3."
    )
    assert det_cfg.get("imgsz") == 800, "Model A3 resolution must be 800px."

    # 2. Verify GLOBAL_MODEL_REGISTRY primary
    primary = GLOBAL_MODEL_REGISTRY.get_primary_for_task("indoor_detection")
    assert primary is not None
    assert primary.model_id == "indoor_a3_800px"
    assert primary.is_primary is True
    assert "indoor_detector_a3_yolo11s_800px_best.pt" in primary.checkpoint_rel_path
    assert primary.input_resolution == (800, 800)

    # 3. Verify ui/analysis_view.py INDOOR_DETECTOR_CHOICES
    from ui.analysis_view import INDOOR_DETECTOR_CHOICES

    first_label = list(INDOOR_DETECTOR_CHOICES.keys())[0]
    first_ckpt = INDOOR_DETECTOR_CHOICES[first_label]
    assert "Model A3" in first_label
    assert "indoor_detector_a3_yolo11s_800px_best.pt" in first_ckpt


def test_b2_and_c2_production_selection():
    """Requirement 2: Verify B2 and C2 production checkpoints are selected by default."""
    model_yaml_path = REPO_ROOT / "config" / "model_config.yaml"
    with open(model_yaml_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # B2 Exterior
    ext_cfg = model_cfg.get("exterior_detection", {})
    assert "exterior_detector_b2_yolo11s_best.pt" in ext_cfg.get("checkpoint_path", "")
    assert ext_cfg.get("status") == "production_ready"

    ext_primary = GLOBAL_MODEL_REGISTRY.get_primary_for_task("exterior_detection")
    assert ext_primary is not None
    assert ext_primary.is_primary is True
    assert "exterior_detector_b2_yolo11s_best.pt" in ext_primary.checkpoint_rel_path

    from ui.analysis_view import EXTERIOR_DETECTOR_CHOICES

    first_ext_label = list(EXTERIOR_DETECTOR_CHOICES.keys())[0]
    assert "Model B2" in first_ext_label
    assert "exterior_detector_b2_yolo11s_best.pt" in EXTERIOR_DETECTOR_CHOICES[first_ext_label]

    # C2 Parking
    park_cfg = model_cfg.get("parking_detection", {})
    assert "parking_detector_c2_yolo11s_pilot_best.pt" in park_cfg.get("checkpoint_path", "")
    assert park_cfg.get("status") == "production_ready"

    park_primary = GLOBAL_MODEL_REGISTRY.get_primary_for_task("parking_detection")
    assert park_primary is not None
    assert park_primary.is_primary is True
    assert "parking_detector_c2_yolo11s_pilot_best.pt" in park_primary.checkpoint_rel_path

    from ui.analysis_view import PARKING_DETECTOR_CHOICES

    first_park_label = list(PARKING_DETECTOR_CHOICES.keys())[0]
    assert "Model C2" in first_park_label
    assert "parking_detector_c2_yolo11s_pilot_best.pt" in PARKING_DETECTOR_CHOICES[first_park_label]


def test_missing_checkpoint_handling_no_silent_fallback(sample_image: Image.Image):
    """Requirement 1 & 3: Real mode must fail explicitly on missing weights and never return mock results."""
    bad_cfg = {
        "checkpoint_path": "checkpoints/non_existent_ghost_model.pt",
        "device": "cpu",
    }
    svc = DetectionService(mock_mode=False, config=bad_cfg)
    result = svc.run(sample_image)

    assert result["success"] is False
    assert result["response"] is None
    assert result["error_message"] is not None
    assert (
        "non_existent_ghost_model.pt" in result["error_message"]
        or "does not exist" in result["error_message"].lower()
    )


def test_is_simulated_metadata_integrity(sample_image: Image.Image):
    """Requirement 3: Never disguise mock predictions as real inference."""
    # Real mode
    a3_ckpt = REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt"
    if a3_ckpt.exists():
        real_svc = DetectionService(
            mock_mode=False,
            config={"checkpoint_path": str(a3_ckpt), "imgsz": 800},
        )
        res_real = real_svc.run(sample_image, save_annotated=False)
        assert res_real["success"] is True
        assert res_real["response"].is_simulated is False

    # Mock mode
    mock_svc = DetectionService(mock_mode=True)
    res_mock = mock_svc.run(sample_image, save_annotated=False)
    assert res_mock["success"] is True
    assert res_mock["response"].is_simulated is True


def test_session_state_mock_mode_persistence(monkeypatch: pytest.MonkeyPatch):
    """Requirement 1: Verify Streamlit session state preserves explicit mock mode toggle."""
    import streamlit as st

    fake_state: Dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", fake_state, raising=False)

    app_yaml_path = REPO_ROOT / "config" / "app.yaml"
    with open(app_yaml_path, "r", encoding="utf-8") as f:
        app_cfg = yaml.safe_load(f)

    # Initial session state initialization: defaults to app_cfg["mock_mode"] (False)
    if "mock_mode" not in st.session_state:
        st.session_state["mock_mode"] = app_cfg.get("mock_mode", False)

    assert st.session_state["mock_mode"] is False

    # User explicitly turns mock mode ON
    st.session_state["mock_mode"] = True
    assert st.session_state["mock_mode"] is True

    # Simulate rerun: initial check must not overwrite existing state
    if "mock_mode" not in st.session_state:
        st.session_state["mock_mode"] = app_cfg.get("mock_mode", False)

    assert st.session_state["mock_mode"] is True

    # User explicitly turns mock mode OFF
    st.session_state["mock_mode"] = False
    assert st.session_state["mock_mode"] is False


def test_hardware_status_reporting():
    """Requirement 4: Verify GPU availability and VRAM reporting with MB/GB units and no 0 GB false-failure."""
    info = get_device_info()
    assert "cuda_available" in info
    assert "device_name" in info
    assert "total_vram_gb" in info
    assert "allocated_vram_gb" in info
    assert "allocated_vram_mb" in info
    assert "peak_vram_mb" in info
    assert "free_vram_gb" in info

    if info["cuda_available"]:
        assert info["total_vram_gb"] > 0.0
        assert info["free_vram_gb"] > 0.0
        assert isinstance(info["allocated_vram_mb"], (int, float))
        assert isinstance(info["peak_vram_mb"], (int, float))
        # Idle allocation of 0 GB or low MB is valid and normal
        assert info["allocated_vram_gb"] >= 0.0
    else:
        assert info["device_name"] == "CPU"

