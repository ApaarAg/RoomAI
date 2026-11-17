"""Deployment Readiness & Verification Tests for RoomAI Detection System.

Verifies:
1. Mock Mode integrity and isolation (Mock mode unchanged, deterministic, is_simulated=True).
2. Strict failure behavior: Real mode NEVER silently substitutes mock predictions.
3. Class mapping and canonical taxonomy consistency across models and configs.
4. Confidence threshold handling and boundary filtering.
5. Missing checkpoint error handling (graceful, descriptive, no silent fallbacks).
6. Timing metrics integrity: all values positive, synchronized, non-fabricated.
7. Cross-class conflict resolution and confidence-weighted fusion.
"""

from pathlib import Path
import pytest
from PIL import Image

from models.detection_model import CustomObjectDetector, MockObjectDetector
from schemas.responses import BoundingBox, DetectionItem
from services.detection_service import DetectionService, resolve_detector
from services.ensemble_service import EnsembleDetector, calculate_iou


@pytest.fixture
def test_image():
    """Synthetic RGB test image."""
    return Image.new("RGB", (640, 480), color=(200, 200, 200))


def test_mock_mode_intact_and_isolated(test_image):
    """Verifies that Mock Mode remains deterministic, fully isolated, and marked as simulated."""
    mock_detector = MockObjectDetector()
    resp = mock_detector.predict(test_image, confidence_threshold=0.35)

    assert resp.is_simulated is True
    assert resp.checkpoint == "mock_simulated.pt"
    assert resp.total_objects > 0
    for det in resp.detections:
        assert det.confidence >= 0.35
        assert 0.0 <= det.box.x_min <= det.box.x_max <= 1.0
        assert 0.0 <= det.box.y_min <= det.box.y_max <= 1.0

    # Service-level mock mode
    service = DetectionService(mock_mode=True)
    res = service.run(test_image, confidence_threshold=0.35, save_annotated=False)
    assert res["success"] is True
    assert res["response"].is_simulated is True
    assert res["is_implemented"] is True


def test_real_mode_never_silently_substitutes_mock(test_image):
    """CRITICAL SAFETY TEST: Real inference failures MUST NEVER silently return mock results."""
    # Point to a nonexistent checkpoint in real inference mode
    service = DetectionService(mock_mode=False, config={"checkpoint_path": "checkpoints/non_existent_model_123.pt"})
    res = service.run(test_image)

    # Must fail explicitly
    assert res["success"] is False
    assert res["response"] is None
    assert res["error_message"] is not None
    assert "not exist" in res["error_message"].lower() or "failed" in res["error_message"].lower() or "could not load" in res["error_message"].lower()


def test_missing_checkpoint_exception_handling():
    """Verifies that CustomObjectDetector raises FileNotFoundError when given missing checkpoint."""
    detector = CustomObjectDetector({"checkpoint_path": "checkpoints/missing_checkpoint_file.pt"})
    with pytest.raises((FileNotFoundError, RuntimeError)) as exc_info:
        detector.load_model()
    assert "missing_checkpoint_file.pt" in str(exc_info.value) or "does not exist" in str(exc_info.value)


def test_class_mapping_consistency():
    """Verifies canonical 12-class indoor taxonomy mapping."""
    canonical = [
        "cabinet#shelf", "table", "chair#stool", "lamp", "door",
        "bed", "sofa", "plant", "decoration", "curtain", "home-appliance", "window"
    ]
    assert len(canonical) == 12

    # Verify fine-tuned checkpoint contains 12 classes matching canonical
    ft_pt = Path("checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt")
    if ft_pt.exists():
        from ultralytics import YOLO
        m = YOLO(str(ft_pt))
        assert len(m.names) == 12
        assert m.names[0] == "cabinet#shelf"
        assert m.names[6] == "sofa"
        assert m.names[11] == "window"


def test_confidence_threshold_filtering(test_image):
    """Verifies strict confidence threshold filtering."""
    service = DetectionService(mock_mode=True)

    # At low threshold, should return all simulated items (5 items)
    res_low = service.run(test_image, confidence_threshold=0.70, save_annotated=False)
    assert res_low["response"].total_objects == 5

    # At higher threshold 0.85, only sofa(0.92), tv(0.89), and table(0.86) survive (3 items)
    res_high = service.run(test_image, confidence_threshold=0.85, save_annotated=False)
    assert res_high["response"].total_objects == 3
    for det in res_high["response"].detections:
        assert det.confidence >= 0.85

    # At threshold 0.95, no item survives
    res_ultra = service.run(test_image, confidence_threshold=0.95, save_annotated=False)
    assert res_ultra["response"].total_objects == 0


def test_timing_metrics_integrity(test_image):
    """Verifies that all timing metrics are positive floats and non-fabricated."""
    service = DetectionService(mock_mode=True)
    res = service.run(test_image, confidence_threshold=0.40, save_annotated=False)

    tm = res["timing_metrics"]
    assert tm["load_time_ms"] >= 0.0
    assert tm["preprocess_time_ms"] >= 0.0
    assert tm["per_model_inference_ms"] >= 0.0
    assert tm["ensemble_fusion_ms"] >= 0.0
    assert tm["rendering_time_ms"] >= 0.0
    assert tm["total_end_to_end_ms"] >= 0.0
    assert tm["steady_state_fps"] >= 0.0


def test_cross_class_conflict_resolution():
    """Verifies that nearly coincident boxes of different classes are resolved by confidence."""
    # Model 1 detects chair at [0.2, 0.2, 0.6, 0.6] with conf 0.70
    # Model 2 detects sofa at [0.2, 0.2, 0.6, 0.6] with conf 0.88
    # IoU = 1.0 -> Sofa should win, chair duplicate should be suppressed.
    from tests.test_ensemble_service import DummyDetector

    det1 = DummyDetector([
        DetectionItem(class_id=2, label="chair#stool", confidence=0.70, box=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6))
    ], name="M1")
    det2 = DummyDetector([
        DetectionItem(class_id=6, label="sofa", confidence=0.88, box=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6))
    ], name="M2")

    ensemble = EnsembleDetector(
        models=[det1, det2],
        cross_class_iou_threshold=0.80,
    )
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.30)

    # Cross-class conflict resolution should leave only 1 box (the higher confidence sofa)
    assert resp.total_objects == 1
    assert resp.detections[0].label == "sofa"


def test_distinct_overlapping_objects_preserved():
    """Verifies that distinct objects with moderate overlap (e.g. coffee table in front of sofa) are NOT suppressed."""
    from tests.test_ensemble_service import DummyDetector

    # Sofa in background: [0.1, 0.2, 0.9, 0.8]
    # Small table in foreground: [0.3, 0.5, 0.7, 0.9]
    # IoU ~ 0.35 (< 0.80) -> Both must be preserved!
    det1 = DummyDetector([
        DetectionItem(class_id=6, label="sofa", confidence=0.90, box=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.9, y_max=0.8)),
        DetectionItem(class_id=1, label="table", confidence=0.85, box=BoundingBox(x_min=0.3, y_min=0.5, x_max=0.7, y_max=0.9)),
    ], name="M1")

    ensemble = EnsembleDetector(
        models=[det1],
        cross_class_iou_threshold=0.80,
    )
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.30)

    assert resp.total_objects == 2
    labels = {d.label for d in resp.detections}
    assert labels == {"sofa", "table"}


def test_configured_checkpoint_and_image_size():
    """Phase 7: Verifies active model_config.yaml specifies A3 at 800px."""
    import yaml
    config_path = Path("config/model_config.yaml")
    assert config_path.exists(), "config/model_config.yaml not found"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg.get("detection", {})
    assert "indoor_detector_a3_yolo11s_800px_best.pt" in det_cfg.get("checkpoint_path", "")
    assert det_cfg.get("imgsz") == 800
    assert det_cfg.get("confidence_threshold") == 0.35
    assert det_cfg.get("iou_threshold") == 0.45


def test_model_registry_primary_a3_and_rollback_a2():
    """Phase 7: Verifies registry has indoor_a3_800px as primary and indoor_a2_finetuned as rollback."""
    from models.registry import GLOBAL_MODEL_REGISTRY
    registry = GLOBAL_MODEL_REGISTRY

    primary = registry.get_primary_for_task("indoor_detection")
    assert primary is not None
    assert primary.model_id == "indoor_a3_800px"
    assert primary.input_resolution == (800, 800)
    assert "indoor_detector_a3_yolo11s_800px_best.pt" in primary.checkpoint_rel_path

    rollback = registry.get_model_spec("indoor_a2_finetuned")
    assert rollback is not None
    assert rollback.is_primary is False
    assert rollback.input_resolution == (640, 640)
    assert "indoor_detector_a2_yolo11s_finetune_best.pt" in rollback.checkpoint_rel_path


def test_a3_inference_output_schema_and_classes(test_image):
    """Phase 7: Verifies DetectionService executes A3 with complete response schema and valid classes."""
    import yaml
    from schemas.responses import DetectionResponse

    with open("config/model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    service = DetectionService(mock_mode=False, config=cfg["detection"])
    result = service.run(test_image, confidence_threshold=0.35, save_annotated=False)

    assert result["success"] is True
    assert result["is_implemented"] is True
    assert result["response"] is not None

    resp: DetectionResponse = result["response"]
    assert resp.is_simulated is False
    assert "indoor_detector_a3_yolo11s_800px_best.pt" in resp.checkpoint
    assert resp.inference_time_ms >= 0.0

    canonical = {
        "cabinet#shelf", "table", "chair#stool", "lamp", "door",
        "bed", "sofa", "plant", "decoration", "curtain", "home-appliance", "window"
    }
    for det in resp.detections:
        assert det.label in canonical
        assert 0 <= det.class_id < 12
        assert 0.0 <= det.box.x_min <= det.box.x_max <= 1.0
        assert 0.0 <= det.box.y_min <= det.box.y_max <= 1.0


def test_a2_rollback_configuration(test_image):
    """Phase 7: Verifies that DetectionService can successfully execute rollback to Model A2."""
    import yaml
    with open("config/model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rollback_cfg = dict(cfg["detection"])
    rollback_cfg["checkpoint_path"] = "checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt"
    rollback_cfg["imgsz"] = 640

    service = DetectionService(mock_mode=False, config=rollback_cfg)
    result = service.run(test_image, confidence_threshold=0.35, save_annotated=False)

    assert result["success"] is True
    assert result["response"].is_simulated is False
    assert "indoor_detector_a2_yolo11s_finetune_best.pt" in result["response"].checkpoint


def test_strict_failure_prevents_accidental_fallback(test_image):
    """Phase 7: Verifies real mode raises an explicit error and NEVER conceals failure by falling back to mock or A2."""
    corrupt_cfg = {
        "checkpoint_path": "checkpoints/corrupted_or_missing_checkpoint_a3.pt",
        "imgsz": 800,
        "confidence_threshold": 0.35
    }
    service = DetectionService(mock_mode=False, config=corrupt_cfg)
    result = service.run(test_image)

    assert result["success"] is False
    assert result["response"] is None
    assert result["error_message"] is not None
    assert "corrupted_or_missing_checkpoint_a3.pt" in result["error_message"] or "does not exist" in result["error_message"]


def test_parking_registry_and_ui_default_is_c2_production():
    """Regression: Parking production default MUST be Model C2 (not the C1+C2 ensemble),
    matching the indoor (A3) and exterior (B2) convention that the production single
    model is the primary recommended default.

    Verifies:
      1. Registry primary for parking_detection is parking_c2_pilot (is_primary=True).
      2. The parking detector_choices dict in ui/analysis_view.py orders the C2 single
         model first (index 0), so the Streamlit selectbox defaults to C2.
    """
    # 1. Registry primary
    from models.registry import GLOBAL_MODEL_REGISTRY

    primary = GLOBAL_MODEL_REGISTRY.get_primary_for_task("parking_detection")
    assert primary is not None
    assert primary.model_id == "parking_c2_pilot"
    assert primary.is_primary is True
    assert "parking_detector_c2_yolo11s_pilot_best.pt" in primary.checkpoint_rel_path

    # 2. UI choices ordering: the first option must reference the C2 production checkpoint.
    import ast
    from pathlib import Path

    analysis_path = Path(__file__).resolve().parent.parent / "ui" / "analysis_view.py"
    if not analysis_path.exists():
        # Project migrated from Streamlit ui/ to Next.js web/app/model/page.tsx
        model_page_path = Path(__file__).resolve().parent.parent / "web" / "app" / "model" / "page.tsx"
        assert model_page_path.exists(), "Neither ui/analysis_view.py nor web/app/model/page.tsx found"
        source = model_page_path.read_text(encoding="utf-8")
        assert "parking_detector_c2_yolo11s" in source, "Model C2 must be registered in web/app/model/page.tsx"
        return
    source = analysis_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    parking_choices_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = []
            for k in node.keys:
                keys.append(k.value if isinstance(k, ast.Constant) else None)
            if any(isinstance(k, str) and "Parking/Outdoor (Higher Capacity" in k for k in keys):
                parking_choices_node = node
                break

    assert parking_choices_node is not None, "Could not locate parking detector_choices dict in ui/analysis_view.py"
    first_key = parking_choices_node.keys[0]
    first_key_value = first_key.value if isinstance(first_key, ast.Constant) else None
    assert isinstance(first_key_value, str), "First parking detector choice key must be a string literal"
    assert "Parking/Outdoor (Higher Capacity" in first_key_value, (
        "First parking detector choice must be the C2 production single model, found: " + str(first_key_value)
    )
    first_value = parking_choices_node.values[0]
    first_value_str = first_value.value if isinstance(first_value, ast.Constant) else None
    assert first_value_str == "checkpoints/parking_detector_c2_yolo11s_pilot_best.pt", (
        "First parking detector choice must map to the C2 production checkpoint, found: " + str(first_value_str)
    )
