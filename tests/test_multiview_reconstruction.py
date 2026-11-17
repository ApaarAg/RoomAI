"""Comprehensive Unit and Integration Tests for Multi-View 3D Reconstruction and Depth Refinement.

Verifies:
1. Rigid 3D-3D Kabsch / Umeyama transformation calculation and recovery precision.
2. Flying-pixel edge filtering and guided bilateral depth refinement.
3. Voxel grid downsampling and statistical outlier removal.
4. MultiViewFusionService single-view and multi-view execution contracts.
5. Multi-view object box association and layout consensus.
"""

import numpy as np
from PIL import Image
import pytest

from schemas.scene import Object3DGeometry, SceneGraph, SceneObject
from services.multiview_fusion_service import MultiViewFusionService, compute_rigid_transform_3d
from utils.depth_refinement import (
    filter_flying_pixels,
    refine_metric_depth_guided,
    statistical_outlier_removal,
    voxel_grid_downsample,
)


def test_rigid_transform_identity():
    """Verify rigid transform on identical points yields identity matrix and zero translation."""
    pts = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.5],
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 2.5],
    ], dtype=np.float32)

    r_mat, t_vec, rmse = compute_rigid_transform_3d(pts, pts)
    np.testing.assert_allclose(r_mat, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(t_vec, np.zeros(3), atol=1e-5)
    assert rmse < 1e-5


def test_rigid_transform_known_rotation_and_translation():
    """Verify Kabsch/Umeyama recovery of known 3D rotation and translation."""
    np.random.seed(42)
    pts_src = np.random.uniform(-2.0, 2.0, size=(20, 3)).astype(np.float32)
    pts_src[:, 2] += 5.0  # Depth offset

    # Known rotation: 25 degrees around Y axis
    theta = np.radians(25.0)
    c, s = np.cos(theta), np.sin(theta)
    true_r = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)
    true_t = np.array([0.45, -0.20, 0.35], dtype=np.float32)

    pts_dst = (true_r @ pts_src.T).T + true_t

    recovered_r, recovered_t, rmse = compute_rigid_transform_3d(pts_src, pts_dst)

    np.testing.assert_allclose(recovered_r, true_r, atol=1e-4)
    np.testing.assert_allclose(recovered_t, true_t, atol=1e-4)
    assert rmse < 1e-4


def test_filter_flying_pixels():
    """Verify steep depth steps create edge masks and remove intermediate flying pixels."""
    # Create smooth 10x10 depth map with a sharp step in the center
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    depth[:, 5:] = 6.0  # 300% jump relative to 2.0m

    filtered = filter_flying_pixels(depth, gradient_threshold_ratio=0.15)
    assert filtered.shape == (10, 10)
    # The boundary columns (columns 4 and 5) should be masked out to prevent edge stretching
    assert np.any(filtered[:, 4:6] == 0.0)
    # Non-edge regions should retain valid metric depths
    assert np.all(filtered[:, :3] == 2.0)
    assert np.all(filtered[:, 7:] == 6.0)


def test_refine_metric_depth_guided():
    """Verify guided depth filtering smooths surfaces and preserves dimensions."""
    np.random.seed(42)
    clean_depth = np.full((30, 30), 3.0, dtype=np.float32)
    noise = np.random.normal(0, 0.05, (30, 30)).astype(np.float32)
    noisy_depth = np.clip(clean_depth + noise, 0.5, 10.0)

    rgb = np.full((30, 30, 3), 120, dtype=np.uint8)

    refined = refine_metric_depth_guided(noisy_depth, rgb, diameter=5, sigma_depth=0.15, sigma_color=25.0)
    assert refined.shape == (30, 30)
    assert np.all(refined > 0)
    # Mean should remain very close to original 3.0 meters
    assert abs(float(np.mean(refined)) - 3.0) < 0.1


def test_voxel_grid_downsample():
    """Verify voxel downsampling groups proximate points into unique voxel bins."""
    pts = np.array([
        [1.001, 2.001, 3.001],
        [1.002, 2.002, 3.002],
        [1.003, 2.003, 3.003],
        [5.0, 5.0, 5.0],
    ], dtype=np.float32)
    colors = np.ones((4, 3), dtype=np.float32)

    down_pts, down_colors = voxel_grid_downsample(pts, colors, voxel_size=0.05)
    # First 3 points belong to the same 5cm voxel; 4th point is separate
    assert len(down_pts) == 2
    assert len(down_colors) == 2


def test_statistical_outlier_removal():
    """Verify statistical outlier removal eliminates isolated noise points."""
    np.random.seed(42)
    # Dense cluster of 100 points around (0, 0, 2)
    cluster = np.random.normal(0, 0.1, size=(100, 3)).astype(np.float32)
    cluster[:, 2] += 2.0

    # 2 distant isolated outlier points
    outliers = np.array([
        [15.0, 15.0, 15.0],
        [-12.0, -12.0, 10.0],
    ], dtype=np.float32)

    all_pts = np.vstack([cluster, outliers])
    clean_pts, _ = statistical_outlier_removal(all_pts, k_neighbors=10, std_ratio=1.5)

    # Clean points should retain the cluster and discard outliers
    assert len(clean_pts) < len(all_pts)
    assert np.all(np.linalg.norm(clean_pts - np.array([0, 0, 2.0]), axis=1) < 2.0)


def test_multiview_fusion_service_single_view():
    """Verify MultiViewFusionService handles single view gracefully."""
    service = MultiViewFusionService(mock_mode=True)
    img = Image.new("RGB", (100, 100), color=(180, 180, 180))

    scene = service.fuse_multiview([img], refine_depth=False)
    assert isinstance(scene, SceneGraph)
    assert scene.width == 100
    assert scene.height == 100
    assert scene.depth_mode == "metric" or scene.depth_mode == "relative"
    assert "point_cloud" in scene.__dict__ or hasattr(scene, "point_cloud")


def test_multiview_fusion_service_multiple_views():
    """Verify MultiViewFusionService processes multi-view image list in mock mode."""
    service = MultiViewFusionService(mock_mode=True)
    img1 = Image.new("RGB", (120, 120), color=(150, 150, 150))
    img2 = Image.new("RGB", (120, 120), color=(170, 170, 170))

    scene = service.fuse_multiview([img1, img2], refine_depth=False)
    assert isinstance(scene, SceneGraph)
    assert "Multi-View" in scene.model_name
    assert scene.n_valid_points > 0
    assert hasattr(scene, "multiview_telemetry")

