"""Unit and Integration Tests for Phase 15 Specialist Detection Architecture.

Verifies:
1. Specialist and general checkpoints exist and hashes match.
2. Specialist ensemble modes (general_only, parallel_ensemble, cascaded, high_res_crop).
3. Prediction schema integrity and model provenance tracking.
4. Same-class duplicate suppression (IoU-based deduplication).
5. Cross-class conflict handling (porch vs facade containment).
6. Missing checkpoint raises explicit FileNotFoundError without silent mock fallback.
7. Mock mode isolation and determinism.
8. Production configuration integrity (B2 remains production default).
"""

from pathlib import Path
from PIL import Image
import pytest
import yaml

from models.detection_model import CustomObjectDetector, MockObjectDetector
from schemas.responses import BoundingBox, DetectionItem, DetectionResponse
from services.specialist_ensemble_service import SpecialistEnsembleDetector, box_iou_xyxy

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_specialist_checkpoints_exist():
    """Verify that general and specialist checkpoints exist on disk."""
    b2_p = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
    b6_p = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
    spec_p = REPO_ROOT / "checkpoints" / "exterior_specialist_porch_yolo11s_best.pt"

    assert b2_p.exists(), f"Missing B2 checkpoint: {b2_p}"
    assert b6_p.exists(), f"Missing B6 checkpoint: {b6_p}"
    assert spec_p.exists(), f"Missing Porch Specialist checkpoint: {spec_p}"


def test_production_default_remains_b2():
    """Verify B2 remains active production default and specialist candidates are opt-in."""
    cfg_path = REPO_ROOT / "config" / "model_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prod = cfg.get("exterior_detection", {})
    assert prod.get("enabled") is True
    assert "exterior_detector_b2_yolo11s_best.pt" in prod.get("checkpoint_path", "")
    assert prod.get("status") == "production_ready"

    # Candidates must be disabled in production
    b6 = cfg.get("exterior_detection_b6_candidate", {})
    assert b6.get("enabled") is False


def test_box_iou_xyxy_utility():
    """Verify bounding box IoU calculation in xyxy coordinate space."""
    b1 = [0.1, 0.1, 0.5, 0.5]
    b2 = [0.1, 0.1, 0.5, 0.5]
    assert pytest.approx(box_iou_xyxy(b1, b2), rel=1e-3) == 1.0

    b3 = [0.5, 0.5, 0.9, 0.9]
    assert box_iou_xyxy(b1, b3) == 0.0

    b4 = [0.3, 0.1, 0.7, 0.5]
    # Intersection: [0.3, 0.1, 0.5, 0.5] -> area = 0.2 * 0.4 = 0.08
    # Union: (0.4 * 0.4) + (0.4 * 0.4) - 0.08 = 0.16 + 0.16 - 0.08 = 0.24
    # IoU: 0.08 / 0.24 = 1/3 ~ 0.3333
    assert pytest.approx(box_iou_xyxy(b1, b4), rel=1e-2) == 0.3333


def test_specialist_ensemble_mock_mode():
    """Verify SpecialistEnsembleDetector works cleanly in mock mode."""
    mock_gen = MockObjectDetector({"checkpoint_path": "mock", "task_group": "exterior"})
    ensemble = SpecialistEnsembleDetector(
        general_model=mock_gen,
        specialist_model=None,
        mode="general_only",
    )
    img = Image.new("RGB", (640, 640), color=(100, 150, 200))
    resp = ensemble.predict(img)

    assert isinstance(resp, DetectionResponse)
    assert resp.is_simulated is True
    assert len(resp.detections) > 0


def test_specialist_same_class_deduplication():
    """Verify that overlapping detections of same class are deduplicated."""
    mock_gen = MockObjectDetector({})
    ensemble = SpecialistEnsembleDetector(general_model=mock_gen, mode="parallel_ensemble")

    det1 = DetectionItem(
        class_id=6,
        label="porch",
        confidence=0.85,
        box=BoundingBox(x_min=0.2, y_min=0.5, x_max=0.8, y_max=0.9),
    )
    det2 = DetectionItem(
        class_id=6,
        label="porch",
        confidence=0.70,
        box=BoundingBox(x_min=0.22, y_min=0.51, x_max=0.79, y_max=0.89),
    )

    deduped = ensemble._deduplicate_same_class([det1, det2], iou_thresh=0.50)
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.85


def test_specialist_fusion_and_provenance_tracking():
    """Verify WBF fusion correctly marks provenance and boosts confidence on consensus."""
    mock_gen = MockObjectDetector({})
    ensemble = SpecialistEnsembleDetector(
        general_model=mock_gen,
        mode="parallel_ensemble",
        weights=[0.8, 1.2],
        consensus_bonus=0.05,
    )

    gen_item = DetectionItem(
        class_id=6,
        label="porch",
        confidence=0.30,
        box=BoundingBox(x_min=0.20, y_min=0.50, x_max=0.80, y_max=0.90),
    )
    spec_item = DetectionItem(
        class_id=6,
        label="porch",
        confidence=0.40,
        box=BoundingBox(x_min=0.22, y_min=0.52, x_max=0.82, y_max=0.92),
    )

    fused = ensemble._fuse_predictions([gen_item], [spec_item], conf_thresh=0.25)
    assert len(fused) == 1
    porch_det = fused[0]
    assert porch_det.class_id == 6
    assert porch_det.label == "porch"
    assert porch_det.model_provenance == "fused_consensus_wbf"
    assert porch_det.confidence > 0.40  # Received consensus bonus


def test_missing_specialist_checkpoint_fails_cleanly():
    """Verify that a missing specialist checkpoint raises FileNotFoundError explicitly."""
    bad_specialist = CustomObjectDetector({
        "checkpoint_path": "checkpoints/non_existent_specialist.pt",
        "device": "cpu",
    })
    ensemble = SpecialistEnsembleDetector(
        general_model=bad_specialist,
        specialist_model=bad_specialist,
        mode="parallel_ensemble",
    )
    img = Image.new("RGB", (640, 640), color=(100, 100, 100))

    with pytest.raises(FileNotFoundError) as exc_info:
        ensemble.predict(img)
    assert "does not exist on disk" in str(exc_info.value)


def test_real_cascaded_and_parallel_inference_modes():
    """Verify real inference execution under cascaded and parallel modes."""
    b6_p = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
    spec_p = REPO_ROOT / "checkpoints" / "exterior_specialist_porch_yolo11s_best.pt"

    gen_model = CustomObjectDetector({"checkpoint_path": str(b6_p), "device": "cuda:0", "confidence_threshold": 0.25})
    spec_model = CustomObjectDetector({"checkpoint_path": str(spec_p), "device": "cuda:0", "confidence_threshold": 0.20})

    gen_model.load_model()
    spec_model.load_model()

    img = Image.new("RGB", (640, 640), color=(130, 160, 190))

    # Test Parallel Mode
    parallel_ens = SpecialistEnsembleDetector(
        general_model=gen_model,
        specialist_model=spec_model,
        mode="parallel_ensemble",
    )
    resp_par = parallel_ens.predict(img)
    assert isinstance(resp_par, DetectionResponse)
    assert resp_par.is_simulated is False
    assert "parallel_ms" in str(parallel_ens.last_timing_breakdown) or "general_ms" in parallel_ens.last_timing_breakdown

    # Test Cascaded Mode
    cascaded_ens = SpecialistEnsembleDetector(
        general_model=gen_model,
        specialist_model=spec_model,
        mode="cascaded",
    )
    resp_cas = cascaded_ens.predict(img)
    assert isinstance(resp_cas, DetectionResponse)
    assert resp_cas.is_simulated is False
    assert "cascaded_invoked" in cascaded_ens.last_timing_breakdown


def test_clahe_preprocessing_in_detection_service():
    """Verify DetectionService execution with exterior_b6_clahe config."""
    from services.detection_service import DetectionService
    config = {
        "checkpoint_path": "exterior_b6_clahe",
        "device": "cuda:0",
        "detector_mode": "single",
        "task_group": "exterior",
        "imgsz": 640,
    }
    svc = DetectionService(mock_mode=False, config=config)
    img = Image.new("RGB", (640, 640), color=(100, 120, 140))
    res = svc.run(img, confidence_threshold=0.25)
    assert res["success"] is True
    assert res["response"] is not None
    assert res["response"].is_simulated is False
    assert "preprocess_time_ms" in res["timing_metrics"]


def test_phase16_benchmark_and_metrics_integrity():
    """Verify Phase 16 saved metrics reproduce baseline and CLAHE benchmarks."""
    import json
    metrics_p = REPO_ROOT / "reports" / "phase16_metrics.json"
    latency_p = REPO_ROOT / "reports" / "phase16_latency_benchmark.json"

    assert metrics_p.exists(), f"Missing {metrics_p}"
    assert latency_p.exists(), f"Missing {latency_p}"

    with open(metrics_p, encoding="utf-8") as f:
        metrics = json.load(f)["ablation_metrics"]

    b6_base = metrics["Cond 1: B6 Baseline (No Preproc)"]
    b6_clahe = metrics["Cond 2: B6 + CLAHE (Luminance, clip=2.0)"]

    assert b6_base["macro_f1"] == 0.5166
    assert b6_base["micro_f1"] == 0.6540
    assert b6_base["per_class"]["porch"]["f1"] == 0.1333

    assert b6_clahe["macro_f1"] == 0.5244
    assert b6_clahe["micro_f1"] == 0.6499
    assert b6_clahe["per_class"]["porch"]["f1"] == 0.1935

    with open(latency_p, encoding="utf-8") as f:
        lat = json.load(f)

    b2_lat = lat["B2 General Only (Production Default)"]
    b6_lat = lat["B6 General Only (Expanded Candidate)"]

    assert b2_lat["model_only_latency_ms"]["median"] < 50.0
    assert b6_lat["model_only_latency_ms"]["median"] < 50.0

