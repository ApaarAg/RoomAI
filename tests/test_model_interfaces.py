"""Tests for model abstract interfaces and mock implementations."""

from PIL import Image
import pytest

from models.classification_model import CustomRoomClassifier, MockRoomClassifier
from models.detection_model import CustomObjectDetector, MockObjectDetector
from models.generation_model import CustomDesignGenerator, MockDesignGenerator
from models.interfaces import (
    BaseClassifier,
    BaseDetector,
    BaseGenerator,
    BaseMLModel,
    BaseScene3D,
    BaseSegmenter,
    BaseVisionLanguage,
    ModelNotImplementedError,
)
from models.scene3d_model import CustomScene3DModel, MockScene3DModel
from models.segmentation_model import CustomRoomSegmenter, MockRoomSegmenter
from models.vision_language_model import (
    CustomVisionLanguageModel,
    MockVisionLanguageModel,
)
from schemas.requests import DesignRequest


def test_abstract_interfaces_cannot_be_instantiated():
    """Abstract interfaces must enforce implementation of required methods."""
    for base_cls in [
        BaseMLModel,
        BaseDetector,
        BaseSegmenter,
        BaseClassifier,
        BaseVisionLanguage,
        BaseGenerator,
        BaseScene3D,
    ]:
        with pytest.raises(TypeError):
            base_cls()


def test_custom_models_raise_not_implemented_for_unimplemented(sample_pil_image: Image.Image):
    """Custom model stubs (classifier, segmenter, VLM, generator, 3D) must cleanly raise ModelNotImplementedError."""
    # Segmenter
    seg = CustomRoomSegmenter()
    with pytest.raises(ModelNotImplementedError):
        seg.predict(sample_pil_image)

    # Classifier
    cls = CustomRoomClassifier()
    with pytest.raises(ModelNotImplementedError):
        cls.predict(sample_pil_image)

    # VLM
    vlm = CustomVisionLanguageModel()
    with pytest.raises(ModelNotImplementedError):
        vlm.ask(sample_pil_image, "Is there a sofa?")

    # Generator
    gen = CustomDesignGenerator()
    with pytest.raises(ModelNotImplementedError):
        gen.generate(sample_pil_image, DesignRequest(target_style="Modern"))

    # 3D
    s3d = CustomScene3DModel()
    with pytest.raises(ModelNotImplementedError):
        s3d.reconstruct(sample_pil_image)


def test_custom_detector_real_inference(sample_pil_image: Image.Image):
    """CustomObjectDetector must perform real inference and return is_simulated=False."""
    det = CustomObjectDetector(config={"checkpoint_path": "yolo11n.pt", "device": "cuda:0"})
    res = det.predict(sample_pil_image, confidence_threshold=0.25)
    assert res.is_simulated is False
    assert res.checkpoint == "yolo11n.pt"
    assert res.inference_time_ms > 0
    assert res.device in ["cuda:0", "cpu"]


def test_custom_detector_invalid_checkpoint(sample_pil_image: Image.Image):
    """CustomObjectDetector must fail explicitly with invalid weights and never return simulated data."""
    det = CustomObjectDetector(config={"checkpoint_path": "nonexistent_weights_xyz.pt"})
    with pytest.raises(Exception):
        det.predict(sample_pil_image)


def test_mock_detector(sample_pil_image: Image.Image):
    """Mock detector should return simulated detection results."""
    mock_det = MockObjectDetector()
    res = mock_det.predict(sample_pil_image, confidence_threshold=0.40)
    assert res.is_simulated is True
    assert res.total_objects > 0
    assert len(res.detections) == res.total_objects


def test_mock_classifier(sample_pil_image: Image.Image):
    """Mock classifier should return top predictions."""
    mock_cls = MockRoomClassifier()
    res = mock_cls.predict(sample_pil_image)
    assert res.is_simulated is True
    assert res.predicted_class == "Living Room"
    assert res.confidence > 0.80


def test_mock_segmenter(sample_pil_image: Image.Image):
    """Mock segmenter should return masks and numpy mask array."""
    mock_seg = MockRoomSegmenter()
    res, mask_arr = mock_seg.predict(sample_pil_image)
    assert res.is_simulated is True
    assert mask_arr is not None
    assert mask_arr.shape == (sample_pil_image.height, sample_pil_image.width)
    assert len(res.masks) > 0


def test_mock_vlm(sample_pil_image: Image.Image):
    """Mock VLM should return answer response."""
    mock_vlm = MockVisionLanguageModel()
    res = mock_vlm.ask(sample_pil_image, "How is the lighting?")
    assert res.is_simulated is True
    assert "lighting" in res.answer.lower() or "daylight" in res.answer.lower()


def test_mock_generator(sample_pil_image: Image.Image):
    """Mock generator should return enhanced image and response."""
    mock_gen = MockDesignGenerator()
    req = DesignRequest(target_style="Scandinavian", custom_prompt="Warm wood tones")
    gen_img, res = mock_gen.generate(sample_pil_image, req)
    assert res.is_simulated is True
    assert gen_img.size == sample_pil_image.size
    assert res.style == "Scandinavian"
