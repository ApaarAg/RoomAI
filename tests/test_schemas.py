"""Tests for Pydantic request and response schemas."""

import pytest
from pydantic import ValidationError

from schemas.requests import AnalysisRequest, DesignRequest, VLMQuestionRequest
from schemas.responses import (
    BoundingBox,
    ClassificationItem,
    ClassificationResponse,
    DetectionItem,
    DetectionResponse,
    DesignGenerationResponse,
    SegmentationMask,
    SegmentationResponse,
    VLMAnswerResponse,
)


def test_bounding_box_valid():
    """Valid normalized bounding box coordinates should pass."""
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
    assert box.x_min == 0.1
    left, top, right, bottom = box.to_pixel_coords(1000, 1000)
    assert (left, top, right, bottom) == (100, 200, 800, 900)


def test_bounding_box_invalid_bounds():
    """x_max less than x_min or out of [0, 1] range should raise ValidationError."""
    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.8, y_min=0.1, x_max=0.2, y_max=0.5)

    with pytest.raises(ValidationError):
        BoundingBox(x_min=-0.1, y_min=0.0, x_max=0.5, y_max=0.5)

    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.0, y_min=0.0, x_max=1.5, y_max=1.0)


def test_detection_response_schema():
    """DetectionResponse should validate detection items."""
    item = DetectionItem(
        class_id=1,
        label="sofa",
        confidence=0.95,
        box=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.7),
    )
    resp = DetectionResponse(detections=[item], total_objects=1, is_simulated=True)
    assert resp.total_objects == 1
    assert resp.detections[0].label == "sofa"
    assert resp.is_simulated is True


def test_classification_response_schema():
    """ClassificationResponse validates predicted class and scores."""
    resp = ClassificationResponse(
        predicted_class="Living Room",
        confidence=0.89,
        top_classes=[
            ClassificationItem(class_name="Living Room", score=0.89),
            ClassificationItem(class_name="Bedroom", score=0.11),
        ],
        is_simulated=False,
    )
    assert resp.predicted_class == "Living Room"
    assert resp.confidence == 0.89
    assert len(resp.top_classes) == 2


def test_segmentation_response_schema():
    """SegmentationResponse validates region metadata."""
    mask = SegmentationMask(
        category="floor",
        coverage_percentage=35.0,
        mask_shape=(480, 640),
        color_hex="#10B981",
    )
    resp = SegmentationResponse(
        masks=[mask], categories_found=["floor"], is_simulated=True
    )
    assert len(resp.masks) == 1
    assert resp.categories_found == ["floor"]


def test_vlm_schemas():
    """VLMQuestionRequest and VLMAnswerResponse validation."""
    req = VLMQuestionRequest(question="Where is the natural light coming from?")
    assert req.question == "Where is the natural light coming from?"

    with pytest.raises(ValidationError):
        VLMQuestionRequest(question="")

    resp = VLMAnswerResponse(
        question=req.question,
        answer="From the large left-hand windows.",
        is_simulated=True,
    )
    assert "left-hand windows" in resp.answer


def test_design_request_schema():
    """DesignRequest enforces guidance scale and style limits."""
    req = DesignRequest(target_style="Minimalist", custom_prompt="Use beige and wood.")
    assert req.target_style == "Minimalist"

    with pytest.raises(ValidationError):
        DesignRequest(guidance_scale=25.0)  # Exceeds max 20.0
