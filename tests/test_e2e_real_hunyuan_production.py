"""
RoomAI Phase 12 — End-to-End Real Hunyuan Production Validation
Strictly validates real Hunyuan model inference, 3D point cloud generation,
GLB and PLY binary export validation, job lifecycle, and failure handling.
Zero mock mode. Zero fallback.
"""

import io
import time
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.roomai_backend import RoomAIBackend
from schemas.scene import SceneGraph
from utils.export_3d import validate_glb_bytes, validate_ply_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]

def test_real_hunyuan_strict_inference_and_3d_export():
    """Validates real HunyuanWorld-Mirror 1.0 inference with strict zero-fallback enforcement."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for real Hunyuan model execution")

    backend = RoomAIBackend()

    # Load authentic indoor test image
    test_img_path = REPO_ROOT / "research/3d_reconstruction/data/indoor_verified_dataset/Bright_Room/frame_0000_rgb.png"
    if test_img_path.exists():
        img = Image.open(test_img_path).convert("RGB")
    else:
        img = Image.new("RGB", (518, 518), color=(210, 190, 170))

    # 1. Run real reconstruction with strict zero-fallback enforcement (allow_composite_fallback=False by default)
    t0 = time.time()
    res = backend.reconstruct_room_hunyuan(
        img,
        confidence_threshold=0.3,
        max_points=20000,
        mock_mode=False,
        allow_composite_fallback=False,
    )
    elapsed = time.time() - t0

    assert res["status"] == "success", f"Reconstruction failed: {res.get('error')}"
    assert res["success"] is True
    assert res["capability_status"] == "PRODUCTION_CANDIDATE"

    # 2. Validate SceneGraph and Point Cloud
    sg: SceneGraph = res["scene_graph"]
    assert sg.model_name == "Tencent HunyuanWorld-Mirror 1.0", f"Unexpected model name: {sg.model_name}"
    assert sg.depth_mode == "metric"
    assert "xyz" in sg.point_cloud
    assert "rgb" in sg.point_cloud

    pts_xyz = np.array(sg.point_cloud["xyz"])
    pts_rgb = np.array(sg.point_cloud["rgb"])

    assert len(pts_xyz) > 0, "Point cloud XYZ must not be empty"
    assert len(pts_xyz) == len(pts_rgb), "Point cloud XYZ and RGB lengths must match"
    assert np.isfinite(pts_xyz).all(), "Point cloud coordinates must be finite"
    assert (pts_rgb >= 0.0).all() and (pts_rgb <= 1.0).all(), "RGB colors must be normalized in [0, 1]"

    print(f"\nReal Hunyuan Inference Succeeded in {elapsed:.2f}s!")
    print(f"Generated {len(pts_xyz):,} 3D points in metric canonical frame.")

    # 3. Validate GLB and PLY Binary Exports
    exports = backend.export_3d_assets(sg, max_points=10000)
    assert exports["status"] == "success"
    assert exports["glb_valid"] is True
    assert exports["ply_valid"] is True

    glb_bytes = exports["glb"]
    ply_bytes = exports["ply"]

    assert len(glb_bytes) > 1000, f"GLB binary output too small: {len(glb_bytes)} bytes"
    assert len(ply_bytes) > 1000, f"PLY binary output too small: {len(ply_bytes)} bytes"

    glb_audit = validate_glb_bytes(glb_bytes)
    assert glb_audit["is_valid"] is True, f"GLB validation failed: {glb_audit.get('error')}"
    assert glb_audit["mesh_count"] >= 1, "GLB must contain at least 1 mesh"
    assert glb_bytes[:4] == b"glTF", "GLB binary must begin with glTF header magic"

    ply_audit = validate_ply_bytes(ply_bytes)
    assert ply_audit["is_valid"] is True, f"PLY validation failed: {ply_audit.get('error')}"
    assert ply_audit["vertex_count"] > 0, "PLY must contain valid vertices"
    assert ply_bytes[:3] == b"ply", "PLY binary must begin with ply header magic"

    print(f"GLB binary valid: {len(glb_bytes):,} bytes | PLY binary valid: {len(ply_bytes):,} bytes")

def test_real_hunyuan_strict_failure_handling():
    """Validates that empty inputs or invalid configurations fail cleanly with informative errors."""
    backend = RoomAIBackend()

    # Empty images list must fail cleanly without throwing unhandled exceptions
    res = backend.reconstruct_room_hunyuan([], mock_mode=False, strict_no_fallback=True)
    assert res["status"] == "error"
    assert res["success"] is False
    assert res["error"]["code"] == "NO_IMAGES_PROVIDED"

if __name__ == "__main__":
    test_real_hunyuan_strict_inference_and_3d_export()
    test_real_hunyuan_strict_failure_handling()
    print("ALL REAL PRODUCTION TESTS PASSED!")
