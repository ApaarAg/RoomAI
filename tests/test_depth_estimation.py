"""Tests for Depth Estimation and 3D Scene Module."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from models.interfaces import ModelNotImplementedError
from models.scene3d_model import CustomScene3DModel, MockScene3DModel
from services.image_service import create_depth_heatmap
from services.scene3d_service import Scene3DService

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CHECKPOINT = "checkpoints/depth_anything_v2_small"


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Fixture returning a standard RGB test image with varied spatial pattern."""
    arr = np.zeros((240, 320, 3), dtype=np.uint8)
    arr[:120, :, 0] = 200  # Red upper half (simulating distant wall)
    arr[120:, :, 1] = 200  # Green lower half (simulating foreground floor)
    return Image.fromarray(arr)


def test_mock_scene3d_model(test_image_rgb: Image.Image):
    """MockScene3DModel must deterministically return simulated output."""
    mock_model = MockScene3DModel()
    assert mock_model.is_ready() is True
    assert "Mock" in mock_model.get_model_name()

    depth_map, response = mock_model.reconstruct(test_image_rgb)
    assert depth_map is not None
    assert depth_map.shape == (240, 320)
    assert response.is_simulated is True
    assert response.depth_available is True
    assert response.device == "cpu"


def test_custom_scene3d_unconfigured_raises():
    """Unconfigured CustomScene3DModel must raise ModelNotImplementedError."""
    unconfigured = CustomScene3DModel()
    assert unconfigured.is_ready() is False
    with pytest.raises(ModelNotImplementedError):
        unconfigured.load_model()

    with pytest.raises(ModelNotImplementedError):
        unconfigured.reconstruct(Image.new("RGB", (64, 64)))


def test_custom_scene3d_missing_checkpoint_raises():
    """CustomScene3DModel with nonexistent checkpoint path must raise ModelNotImplementedError."""
    model = CustomScene3DModel(config={"checkpoint_path": "nonexistent_dir/weights.pth"})
    with pytest.raises(ModelNotImplementedError):
        model.load_model()


def test_custom_scene3d_invalid_input(test_image_rgb: Image.Image):
    """CustomScene3DModel must reject None input with ValueError."""
    if not Path(LOCAL_CHECKPOINT).exists():
        pytest.skip("Local Depth Anything checkpoint not found")

    model = CustomScene3DModel(config={"checkpoint_path": LOCAL_CHECKPOINT, "device": "cpu"})
    with pytest.raises(ValueError):
        model.reconstruct(None)  # type: ignore


def test_custom_scene3d_real_inference_shape_and_range(test_image_rgb: Image.Image):
    """Depth Anything V2 Small must return correctly sized, normalized, finite relative depth."""
    if not Path(LOCAL_CHECKPOINT).exists():
        pytest.skip("Local Depth Anything checkpoint not found")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CustomScene3DModel(config={"checkpoint_path": LOCAL_CHECKPOINT, "device": device})
    assert model.is_ready() is True

    depth_map, response = model.reconstruct(test_image_rgb)

    # 1. Output dimensions match input image dimensions (H, W)
    assert depth_map is not None
    assert depth_map.shape == (240, 320)

    # 2. Strict numerical finiteness
    assert not np.isnan(depth_map).any(), "Depth map contains NaNs"
    assert not np.isinf(depth_map).any(), "Depth map contains Infs"

    # 3. Relative normalization in [0.0, 1.0]
    assert depth_map.min() >= 0.0
    assert depth_map.max() <= 1.0

    # 4. Honesty and metadata verification
    assert response.is_simulated is False
    assert response.depth_available is True
    assert response.relative_depth_only is True
    assert response.inference_time_ms > 0.0
    assert "Depth Anything V2" in response.model_name
    # Disclaimer now mentions relative inverse depth (not metric meters)
    assert "relative" in response.disclaimer.lower() or "inverse" in response.disclaimer.lower()


def test_custom_scene3d_cpu_fallback(test_image_rgb: Image.Image):
    """CustomScene3DModel must support explicit CPU execution."""
    if not Path(LOCAL_CHECKPOINT).exists():
        pytest.skip("Local Depth Anything checkpoint not found")

    model_cpu = CustomScene3DModel(config={"checkpoint_path": LOCAL_CHECKPOINT, "device": "cpu"})
    depth_map, response = model_cpu.reconstruct(test_image_rgb)

    assert depth_map.shape == (240, 320)
    assert response.device == "cpu"
    assert response.is_simulated is False


def test_scene3d_service_real_and_mock(test_image_rgb: Image.Image):
    """Scene3DService must support mock mode and real mode correctly."""
    # Mock mode test
    mock_service = Scene3DService(mock_mode=True)
    mock_res = mock_service.run(test_image_rgb)
    assert mock_res["success"] is True
    assert mock_res["is_implemented"] is True
    assert mock_res["response"].is_simulated is True
    assert mock_res["heatmap_image"] is not None

    # Real mode without config: should report not implemented
    unconf_service = Scene3DService(mock_mode=False)
    unconf_res = unconf_service.run(test_image_rgb)
    assert unconf_res["success"] is False
    assert unconf_res["is_implemented"] is False

    # Real mode with active checkpoint
    if Path(LOCAL_CHECKPOINT).exists():
        real_service = Scene3DService(
            mock_mode=False,
            config={
                "enabled": True,
                "checkpoint_path": LOCAL_CHECKPOINT,
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            },
        )
        real_res = real_service.run(test_image_rgb)
        assert real_res["success"] is True
        assert real_res["is_implemented"] is True
        assert real_res["response"].is_simulated is False
        assert real_res["heatmap_image"].size == (320, 240)
        assert real_res["timing"]["inference_time_ms"] > 0


def test_create_depth_heatmap_rendering(test_image_rgb: Image.Image):
    """create_depth_heatmap must convert 2D array to 3-channel RGB PIL Image."""
    dummy_depth = np.linspace(0.0, 1.0, 240 * 320, dtype=np.float32).reshape(240, 320)
    heatmap = create_depth_heatmap(dummy_depth)

    assert isinstance(heatmap, Image.Image)
    assert heatmap.size == (320, 240)  # PIL size is (width, height)
    assert heatmap.mode == "RGB"
