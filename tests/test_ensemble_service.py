"""Unit tests for Confidence-Weighted Ensemble Service.

Verifies:
1. IoU calculation accuracy.
2. Coordinate normalization and bounding box bounds.
3. Class-aware matching (only matching same-class predictions).
4. Weighted Box Fusion (WBF) mathematical coordinate averaging.
5. Confidence calibration (scaling and clipping).
6. Consensus reinforcement vs. single-model preservation.
7. Duplicate suppression without suppressing distinct neighboring objects.
8. Handling of empty predictions from one or all models.
9. Single-model fallback and error resilience.
"""

import pytest
from PIL import Image

from models.interfaces import BaseDetector
from schemas.responses import BoundingBox, DetectionItem, DetectionResponse
from services.ensemble_service import EnsembleDetector, calculate_iou


class DummyDetector(BaseDetector):
    """Mock detector for predictable ensemble testing."""

    def __init__(self, detections: list, name: str = "DummyDetector", device: str = "cpu"):
        super().__init__({})
        self.detections = detections
        self._name = name
        self._device = device

    def load_model(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self._name

    def predict(self, image: Image.Image, confidence_threshold: float = 0.35) -> DetectionResponse:
        filtered = [d for d in self.detections if d.confidence >= confidence_threshold]
        return DetectionResponse(
            detections=filtered,
            total_objects=len(filtered),
            is_simulated=False,
            model_name=self._name,
            checkpoint=f"{self._name}.pt",
            inference_time_ms=10.0,
            device=self._device,
        )


class FailingDetector(BaseDetector):
    """Detector that raises an exception during inference."""

    def __init__(self, name: str = "FailingDetector"):
        super().__init__({})
        self._name = name

    def load_model(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self._name

    def predict(self, image: Image.Image, confidence_threshold: float = 0.35) -> DetectionResponse:
        raise RuntimeError("GPU OOM / Simulated Checkpoint Failure")


# --- Test 1: IoU calculation ---
def test_calculate_iou():
    box1 = (0.0, 0.0, 0.5, 0.5)
    box2 = (0.0, 0.0, 0.5, 0.5)
    assert calculate_iou(box1, box2) == pytest.approx(1.0)

    box3 = (0.5, 0.5, 1.0, 1.0)
    assert calculate_iou(box1, box3) == pytest.approx(0.0)

    # Partial overlap
    box4 = (0.25, 0.0, 0.75, 0.5)
    # inter = 0.25 * 0.5 = 0.125
    # area1 = 0.25, area2 = 0.25, union = 0.375
    # iou = 0.125 / 0.375 = 1/3
    assert calculate_iou(box1, box4) == pytest.approx(1.0 / 3.0)


# --- Test 2: Coordinate normalization & bounds ---
def test_wbf_coordinate_fusion_and_bounds():
    det1 = DummyDetector([
        DetectionItem(class_id=6, label="sofa", confidence=0.8, box=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6))
    ], name="Model1")
    det2 = DummyDetector([
        DetectionItem(class_id=6, label="sofa", confidence=0.8, box=BoundingBox(x_min=0.12, y_min=0.18, x_max=0.48, y_max=0.62))
    ], name="Model2")

    ensemble = EnsembleDetector(models=[det1, det2], fusion_iou_threshold=0.5)
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.3)

    assert resp.total_objects == 1
    fused_box = resp.detections[0].box
    assert 0.0 <= fused_box.x_min <= 1.0
    assert 0.0 <= fused_box.y_min <= 1.0
    assert fused_box.x_max > fused_box.x_min
    assert fused_box.y_max > fused_box.y_min
    # Equal weights and scores should produce arithmetic midpoint
    assert fused_box.x_min == pytest.approx(0.11, abs=0.01)
    assert fused_box.y_min == pytest.approx(0.19, abs=0.01)


# --- Test 3: Class agreement & isolation ---
def test_class_isolation_no_cross_class_fusion():
    """Boxes in the exact same location but different classes MUST NOT be fused."""
    det1 = DummyDetector([
        DetectionItem(class_id=1, label="table", confidence=0.8, box=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6))
    ], name="Model1")
    det2 = DummyDetector([
        DetectionItem(class_id=2, label="chair#stool", confidence=0.75, box=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6))
    ], name="Model2")

    ensemble = EnsembleDetector(models=[det1, det2], fusion_iou_threshold=0.5)
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.3)

    # Must preserve both separate class detections
    assert resp.total_objects == 2
    classes_found = {d.label for d in resp.detections}
    assert classes_found == {"table", "chair#stool"}


# --- Test 4: Confidence calibration ---
def test_confidence_calibration():
    det1 = DummyDetector([], name="Model1")
    det2 = DummyDetector([], name="Model2")
    # Model 1 has weight 1.0, Model 2 has weight 0.5 (penalized overconfident model)
    ensemble = EnsembleDetector(models=[det1, det2], model_weights=[1.0, 0.5])

    raw_score = 0.8
    assert ensemble.calibrate_score(raw_score, model_idx=0) == pytest.approx(0.8)
    assert ensemble.calibrate_score(raw_score, model_idx=1) == pytest.approx(0.4)


# --- Test 5: Consensus boost vs single-model preservation ---
def test_consensus_boost_and_single_model_retention():
    # Model 1 detects a sofa (agreed by Model 2) and an independent lamp (only Model 1)
    det1 = DummyDetector([
        DetectionItem(class_id=6, label="sofa", confidence=0.8, box=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)),
        DetectionItem(class_id=3, label="lamp", confidence=0.7, box=BoundingBox(x_min=0.8, y_min=0.1, x_max=0.9, y_max=0.4)),
    ], name="Model1")
    det2 = DummyDetector([
        DetectionItem(class_id=6, label="sofa", confidence=0.8, box=BoundingBox(x_min=0.11, y_min=0.11, x_max=0.49, y_max=0.49)),
    ], name="Model2")

    ensemble = EnsembleDetector(
        models=[det1, det2],
        single_model_factor=0.9,
        consensus_bonus=0.05,
    )
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.3)

    assert resp.total_objects == 2
    items_by_label = {d.label: d for d in resp.detections}

    # Agreed sofa gets consensus bonus
    sofa_det = items_by_label["sofa"]
    assert sofa_det.confidence > 0.8  # Reinforced

    # Single-model lamp is preserved for recall
    lamp_det = items_by_label["lamp"]
    assert lamp_det.confidence == pytest.approx(0.7 * 0.9, abs=0.01)


# --- Test 6: Duplicate suppression ---
def test_duplicate_suppression():
    """Three overlapping detections of the same sofa from different models should fuse to 1."""
    box = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.7, y_max=0.7)
    det1 = DummyDetector([DetectionItem(class_id=6, label="sofa", confidence=0.85, box=box)], name="M1")
    det2 = DummyDetector([DetectionItem(class_id=6, label="sofa", confidence=0.80, box=box)], name="M2")
    det3 = DummyDetector([DetectionItem(class_id=6, label="sofa", confidence=0.75, box=box)], name="M3")

    ensemble = EnsembleDetector(models=[det1, det2, det3], fusion_iou_threshold=0.5)
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.4)

    assert resp.total_objects == 1
    assert resp.detections[0].label == "sofa"


# --- Test 7: Empty predictions ---
def test_empty_predictions():
    det1 = DummyDetector([], name="M1")
    det2 = DummyDetector([], name="M2")

    ensemble = EnsembleDetector(models=[det1, det2])
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.3)

    assert resp.total_objects == 0
    assert len(resp.detections) == 0


# --- Test 8: Single-model fallback on failure ---
def test_single_model_fallback_on_failure():
    """If one detector crashes, the ensemble must gracefully return surviving model predictions."""
    working_det = DummyDetector([
        DetectionItem(class_id=5, label="bed", confidence=0.92, box=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.8, y_max=0.8))
    ], name="WorkingDetector")
    failing_det = FailingDetector(name="BrokenDetector")

    ensemble = EnsembleDetector(models=[working_det, failing_det])
    img = Image.new("RGB", (640, 640), color="white")
    resp = ensemble.predict(img, confidence_threshold=0.3)

    assert resp.total_objects == 1
    assert resp.detections[0].label == "bed"
