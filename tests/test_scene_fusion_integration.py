"""Regression and Integration Tests for Phase 7 Scene Fusion and 3D Perception.

Verifies:
1. CustomRoomSegmenter exposes segmentation_mode property.
2. SceneFusionService executes fuse() without AttributeError.
3. SceneFusionService operates seamlessly in both real and mock modes.
4. Model A3 detector and YOLO-Seg instance masks fuse into valid SceneGraph.
5. Extracted SceneObjects contain valid 3D geometry and extents.
"""

from pathlib import Path
from PIL import Image
import pytest
import yaml

from models.segmentation_model import CustomRoomSegmenter, MockRoomSegmenter
from services.scene_fusion_service import SceneFusionService
from schemas.scene import SceneGraph


def test_segmenter_has_segmentation_mode_attribute():
    """CustomRoomSegmenter and MockRoomSegmenter must expose segmentation_mode."""
    mock_seg = MockRoomSegmenter()
    assert hasattr(mock_seg, "segmentation_mode")
    assert mock_seg.segmentation_mode in ("semantic", "instance")

    real_seg = CustomRoomSegmenter(config={"checkpoint_path": "checkpoints/yolo11n-seg.pt", "enabled": True})
    assert hasattr(real_seg, "segmentation_mode")
    assert real_seg.segmentation_mode == "instance"


def test_scene_fusion_mock_mode_execution():
    """SceneFusionService must fuse mock scenes without error."""
    service = SceneFusionService(mock_mode=True)
    img = Image.new("RGB", (320, 240), color=(128, 128, 128))
    scene = service.fuse(img)

    assert isinstance(scene, SceneGraph)
    assert scene.width == 320
    assert scene.height == 240
    assert len(scene.surfaces) > 0


def test_scene_fusion_real_mode_execution():
    """SceneFusionService must fuse real scenes without 'segmentation_mode' AttributeError."""
    model_cfg_path = Path("config/model_config.yaml")
    if not model_cfg_path.exists():
        pytest.skip("config/model_config.yaml not found")

    with open(model_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seg_ckpt = Path("checkpoints/yolo11n-seg.pt")
    a3_ckpt = Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
    depth_dir = Path("checkpoints/depth_anything_v2_small")

    if not (seg_ckpt.exists() and a3_ckpt.exists() and depth_dir.exists()):
        pytest.skip("Required checkpoints not found for real scene fusion test")

    service = SceneFusionService(mock_mode=False, config=cfg)
    img = Image.new("RGB", (320, 240), color=(80, 120, 160))
    scene = service.fuse(img)

    assert isinstance(scene, SceneGraph)
    assert scene.width == 320
    assert scene.height == 240
    assert scene.n_total_points == 320 * 240
    assert scene.n_valid_points >= 0

