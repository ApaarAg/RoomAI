"""Tests for service classes and image rendering logic."""

import numpy as np
from PIL import Image

from schemas.requests import DesignRequest
from schemas.responses import BoundingBox, DetectionItem
from services.classification_service import ClassificationService
from services.detection_service import DetectionService
from services.generation_service import GenerationService
from services.image_service import (
    create_depth_heatmap,
    create_segmentation_overlay,
    create_side_by_side_comparison,
    draw_bounding_boxes,
)
from services.scene3d_service import Scene3DService
from services.segmentation_service import SegmentationService
from services.vision_language_service import VisionLanguageService


def test_detection_service_mock_and_real(sample_pil_image: Image.Image):
    """Test DetectionService in both mock mode and real YOLO inference mode."""
    # 1. Mock mode
    mock_svc = DetectionService(mock_mode=True)
    res_mock = mock_svc.run(sample_pil_image, confidence_threshold=0.40)
    assert res_mock["success"] is True
    assert res_mock["is_implemented"] is True
    assert res_mock["annotated_image"] is not None
    assert res_mock["response"].is_simulated is True
    assert res_mock["response"].total_objects > 0

    # 2. Real YOLO inference mode
    real_svc = DetectionService(mock_mode=False, config={"checkpoint_path": "yolo11n.pt", "device": "cuda:0"})
    res_real = real_svc.run(sample_pil_image, confidence_threshold=0.25)
    assert res_real["success"] is True
    assert res_real["is_implemented"] is True
    assert res_real["annotated_image"] is not None
    assert res_real["saved_image_path"] is not None
    assert res_real["response"].is_simulated is False
    assert res_real["response"].checkpoint == "yolo11n.pt"

    # 3. Real mode failure handling (must not fall back to mock)
    err_svc = DetectionService(mock_mode=False, config={"checkpoint_path": "missing_weights.pt"})
    res_err = err_svc.run(sample_pil_image)
    assert res_err["success"] is False
    assert res_err["response"] is None
    assert "failed" in res_err["error_message"].lower() or "could not load" in res_err["error_message"].lower()


def test_segmentation_service_mock_and_real(sample_pil_image: Image.Image):
    """Test SegmentationService in both modes."""
    mock_svc = SegmentationService(mock_mode=True)
    res_mock = mock_svc.run(sample_pil_image)
    assert res_mock["success"] is True
    assert res_mock["overlay_image"] is not None
    assert res_mock["mask_array"] is not None

    real_svc = SegmentationService(mock_mode=False)
    res_real = real_svc.run(sample_pil_image)
    assert res_real["success"] is False
    assert res_real["is_implemented"] is False


def test_classification_service_mock_and_real(sample_pil_image: Image.Image):
    """Test ClassificationService in both modes."""
    mock_svc = ClassificationService(mock_mode=True)
    res_mock = mock_svc.run(sample_pil_image)
    assert res_mock["success"] is True
    assert res_mock["response"].predicted_class == "Living Room"

    real_svc = ClassificationService(mock_mode=False)
    res_real = real_svc.run(sample_pil_image)
    assert res_real["success"] is False
    assert res_real["is_implemented"] is False


def test_vlm_service_mock_and_real(sample_pil_image: Image.Image):
    """Test VisionLanguageService question handling."""
    mock_svc = VisionLanguageService(mock_mode=True)
    # Valid question
    res_mock = mock_svc.run(sample_pil_image, "What style is this room?")
    assert res_mock["success"] is True
    assert len(res_mock["response"].answer) > 0

    # Empty question
    res_empty = mock_svc.run(sample_pil_image, "   ")
    assert res_empty["success"] is False
    assert "cannot be empty" in res_empty["error_message"].lower()

    # Real mode
    real_svc = VisionLanguageService(mock_mode=False)
    res_real = real_svc.run(sample_pil_image, "What style is this room?")
    assert res_real["success"] is False
    assert res_real["is_implemented"] is False


def test_generation_service_mock_and_real(sample_pil_image: Image.Image):
    """Test GenerationService redesign creation."""
    req = DesignRequest(target_style="Modern", custom_prompt="Minimalist art and lighting")
    mock_svc = GenerationService(mock_mode=True)
    res_mock = mock_svc.run(sample_pil_image, req)
    assert res_mock["success"] is True
    assert res_mock["generated_image"] is not None
    assert res_mock["comparison_image"] is not None

    real_svc = GenerationService(mock_mode=False)
    res_real = real_svc.run(sample_pil_image, req)
    assert res_real["success"] is False
    assert res_real["is_implemented"] is False


def test_scene3d_service_mock_and_real(sample_pil_image: Image.Image):
    """Test Scene3DService experimental depth estimation."""
    mock_svc = Scene3DService(mock_mode=True)
    res_mock = mock_svc.run(sample_pil_image)
    assert res_mock["success"] is True
    assert res_mock["heatmap_image"] is not None

    real_svc = Scene3DService(mock_mode=False)
    res_real = real_svc.run(sample_pil_image)
    assert res_real["success"] is False
    assert res_real["is_implemented"] is False


def test_image_service_helpers(sample_pil_image: Image.Image):
    """Test image overlay and composition functions."""
    # Bounding boxes
    items = [
        DetectionItem(
            class_id=1,
            label="sofa",
            confidence=0.91,
            box=BoundingBox(x_min=0.2, y_min=0.3, x_max=0.7, y_max=0.8),
        )
    ]
    bbox_img = draw_bounding_boxes(sample_pil_image, items)
    assert bbox_img.size == sample_pil_image.size

    # Segmentation overlay
    mask_arr = np.zeros((sample_pil_image.height, sample_pil_image.width), dtype=np.uint8)
    mask_arr[100:200, 100:200] = 1
    overlay_img = create_segmentation_overlay(sample_pil_image, mask_arr)
    assert overlay_img.size == sample_pil_image.size

    # Side-by-side comparison
    comp_img = create_side_by_side_comparison(sample_pil_image, bbox_img)
    assert comp_img.width >= sample_pil_image.width * 2

    # Depth heatmap
    depth_arr = np.ones((100, 100), dtype=np.float32) * 50.0
    heatmap = create_depth_heatmap(depth_arr)
    assert heatmap.size == (100, 100)
