"""Unit and Integration Tests for HunyuanWorld-Mirror RoomAI Integration."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from schemas.scene import SceneGraph
from services.roomai_backend import RoomAIBackend, validate_input_image
from services.hunyuan_mirror_service import C_MODEL_TO_ROOMAI
from utils.export_3d import validate_glb_bytes, validate_ply_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hunyuan_coordinate_transformation_invariants():
    """Validates that C = diag(1, -1, -1) correctly flips OpenGL +Y-up and +Z-back to OpenCV +Y-down and +Z-forward."""
    # Test point in model space: [1.0, 2.0, -3.0] (X=right, Y=up, Z=forward into scene in OpenGL)
    p_model = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    p_roomai = p_model @ C_MODEL_TO_ROOMAI.T

    assert p_roomai[0] == 1.0, "X-axis must remain unchanged (+X right)"
    assert p_roomai[1] == -2.0, "Y-axis must invert (+Y up -> +Y down)"
    assert p_roomai[2] == 3.0, "Z-axis must invert (+Z back -> +Z forward)"


def test_reconstruct_room_hunyuan_mock_mode():
    """Validates reconstruct_room_hunyuan mock execution and downstream 3D export."""
    backend = RoomAIBackend()
    img = Image.new("RGB", (518, 518), color=(180, 160, 140))

    res = backend.reconstruct_room_hunyuan(img, mock_mode=True)
    assert res["status"] == "success"
    assert res["success"] is True
    assert res["capability_status"] == "PRODUCTION_CANDIDATE"
    assert res["scene_graph"] is not None

    sg: SceneGraph = res["scene_graph"]
    assert sg.depth_mode == "metric"
    assert len(sg.point_cloud["xyz"]) > 0

    # Validate asset exports
    exports = backend.export_3d_assets(sg)
    assert exports["status"] == "success"
    assert exports["glb_valid"] is True
    assert exports["ply_valid"] is True
    assert len(exports["glb"]) > 0
    assert len(exports["ply"]) > 0

    # Validate GLB structure
    glb_val = validate_glb_bytes(exports["glb"])
    assert glb_val["is_valid"] is True

    # Validate PLY structure
    ply_val = validate_ply_bytes(exports["ply"])
    assert ply_val["is_valid"] is True


def test_reconstruct_room_hunyuan_invalid_inputs():
    """Validates error handling for empty or invalid inputs."""
    backend = RoomAIBackend()
    res = backend.reconstruct_room_hunyuan([], mock_mode=True)
    assert res["status"] == "error"
    assert res["success"] is False
    assert res["error"]["code"] == "NO_IMAGES_PROVIDED"


def test_reconstruct_room_hunyuan_real_inference_no_fallback():
    """Validates real Hunyuan inference without mock mode and without fallback."""
    from services.hunyuan_mirror_service import get_hunyuan_adapter
    adapter = get_hunyuan_adapter()
    if not adapter.is_ready:
        pytest.skip("Hunyuan model not initialized in this test runner context.")
    backend = RoomAIBackend()
    img = Image.new("RGB", (518, 518), color=(200, 180, 160))
    res = backend.reconstruct_room_hunyuan(img, mock_mode=False, allow_composite_fallback=False)
    assert res["status"] == "success"
    assert res["success"] is True
    assert res["capability_status"] == "PRODUCTION_CANDIDATE"
    assert res["scene_graph"].model_name == "Tencent HunyuanWorld-Mirror 1.0"
    assert len(res["scene_graph"].point_cloud["xyz"]) > 0


def test_reconstruct_room_hunyuan_disables_silent_fallback():
    """Regression test: Proves that an unready/failing Hunyuan model explicitly fails with an error envelope and NEVER silently calls the composite pipeline."""
    backend = RoomAIBackend()
    from unittest.mock import patch

    # Mock adapter as unready
    with patch("services.hunyuan_mirror_service.HunyuanWorldMirrorAdapter.is_ready", new=False):
        img = Image.new("RGB", (518, 518), color=(120, 120, 120))
        # Default behavior: allow_composite_fallback=False
        res = backend.reconstruct_room_hunyuan(img, mock_mode=False, allow_composite_fallback=False)
        assert res["status"] == "error", "Must return error when adapter is not ready"
        assert res["success"] is False
        assert res["error"]["code"] == "HUNYUAN_ADAPTER_NOT_READY"
        assert res["scene_graph"] is None, "Must not return a fallback composite scene graph"

        # Explicit fallback behavior only when explicitly requested
        res_fallback = backend.reconstruct_room_hunyuan(img, mock_mode=True, allow_composite_fallback=True)
        # Note: in mock_mode=True, reconstruct_room_hunyuan handles mock directly


