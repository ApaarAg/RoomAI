"""
Comprehensive Unit-Test Suite for Official WorldMirror Scene Normalization.

Tests all 12 criteria mandated by Phase 8:
1. Normalization factor computed from valid world points.
2. Invalid pixels excluded from scale estimation.
3. Multiple views share a consistent scene normalization factor.
4. Depth normalized using the same scene factor.
5. Point maps normalized using the same scene factor.
6. Camera translations normalized using the same factor.
7. Camera rotations remain unchanged.
8. Intrinsics remain unchanged.
9. Surface normals retain unit length.
10. Denormalization restores original metric coordinates.
11. No NaN or Inf is introduced.
12. Camera-to-world and world-to-camera conventions remain valid.
Plus synthetic geometric test scene with known physical dimensions.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add script directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))
from scene_normalization import (
    compute_world_coordinates,
    compute_scene_scale_factor,
    normalize_scene_coordinates,
    denormalize_scene_coordinates,
)


class TestSceneNormalization(unittest.TestCase):
    def setUp(self):
        """Create a synthetic geometric test scene: a 4m x 3m x 2.5m indoor room."""
        np.random.seed(42)
        self.H, self.W = 64, 64
        self.K = np.array([
            [50.0,  0.0, 32.0],
            [ 0.0, 50.0, 32.0],
            [ 0.0,  0.0,  1.0],
        ], dtype=np.float32)

        # View 1: Camera at (0, 0, 1) looking along +Z
        self.c2w_1 = np.eye(4, dtype=np.float32)
        self.c2w_1[:3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Flat wall at Z_cam = 4.0 meters
        self.depth_1 = np.full((self.H, self.W), 4.0, dtype=np.float32)
        self.mask_1 = np.ones((self.H, self.W), dtype=bool)

        # Compute 3D world points for View 1
        self.pts_1, _ = compute_world_coordinates(self.depth_1, self.K, self.c2w_1, c2w=True)

        # Unit normals pointing along -Z (facing camera)
        self.normals_1 = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.normals_1[..., 2] = -1.0

        # View 2: Camera translated to (1.5, 0.5, 2.0) with 30 deg rotation around Y
        theta = np.deg2rad(30.0)
        R_y = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ], dtype=np.float32)
        self.c2w_2 = np.eye(4, dtype=np.float32)
        self.c2w_2[:3, :3] = R_y
        self.c2w_2[:3, 3] = np.array([1.5, 0.5, 2.0], dtype=np.float32)

        # View 2 depths varying between 2.5m and 5.5m
        self.depth_2 = np.linspace(2.5, 5.5, self.H * self.W, dtype=np.float32).reshape(self.H, self.W)
        self.mask_2 = np.ones((self.H, self.W), dtype=bool)
        self.pts_2, _ = compute_world_coordinates(self.depth_2, self.K, self.c2w_2, c2w=True)
        self.normals_2 = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.normals_2[..., 2] = -1.0

        self.views = [
            {
                "pts3d": self.pts_1.copy(),
                "depthmap": self.depth_1.copy(),
                "valid_mask": self.mask_1.copy(),
                "camera_poses": self.c2w_1.copy(),
                "camera_intrs": self.K.copy(),
                "normals": self.normals_1.copy(),
            },
            {
                "pts3d": self.pts_2.copy(),
                "depthmap": self.depth_2.copy(),
                "valid_mask": self.mask_2.copy(),
                "camera_poses": self.c2w_2.copy(),
                "camera_intrs": self.K.copy(),
                "normals": self.normals_2.copy(),
            },
        ]

    def test_01_normalization_factor_from_valid_world_points(self):
        """1. Normalization factor computed from valid world points."""
        factor, bad_case = compute_scene_scale_factor(self.views, norm_mode="median_dis")
        self.assertFalse(bad_case)
        self.assertGreater(factor, 0.0)

        # Manually compute expected median
        all_pts = np.concatenate([self.pts_1.reshape(-1, 3), self.pts_2.reshape(-1, 3)], axis=0)
        expected_dis = np.linalg.norm(all_pts, ord=2, axis=-1)
        expected_factor = float(np.median(expected_dis))
        self.assertAlmostEqual(factor, expected_factor, places=5)

    def test_02_invalid_pixels_excluded_from_scale(self):
        """2. Invalid pixels excluded from scale estimation."""
        views_with_invalid = [
            {
                "pts3d": self.pts_1.copy(),
                "depthmap": self.depth_1.copy(),
                "valid_mask": self.mask_1.copy(),
                "camera_poses": self.c2w_1.copy(),
            }
        ]
        # Invalidate half of the pixels and fill them with extreme outlier values (e.g. 1000m)
        views_with_invalid[0]["valid_mask"][:32, :] = False
        views_with_invalid[0]["pts3d"][:32, :] = 9999.0

        clean_pts = self.pts_1[32:, :].reshape(-1, 3)
        expected_clean_factor = float(np.median(np.linalg.norm(clean_pts, ord=2, axis=-1)))

        factor, _ = compute_scene_scale_factor(views_with_invalid, norm_mode="median_dis")
        self.assertAlmostEqual(factor, expected_clean_factor, places=5)

    def test_03_consistent_scene_factor_across_views(self):
        """3. Multiple views share a consistent scene normalization factor."""
        norm_views, factor = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        self.assertEqual(len(norm_views), 2)
        self.assertEqual(norm_views[0]["norm_factor"], factor)
        self.assertEqual(norm_views[1]["norm_factor"], factor)

    def test_04_depth_normalized_by_scene_factor(self):
        """4. Depth normalized using the same scene factor."""
        norm_views, factor = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        np.testing.assert_allclose(norm_views[0]["depthmap"], self.depth_1 / factor, rtol=1e-5)
        np.testing.assert_allclose(norm_views[1]["depthmap"], self.depth_2 / factor, rtol=1e-5)

    def test_05_point_maps_normalized_by_scene_factor(self):
        """5. Point maps normalized using the same scene factor."""
        norm_views, factor = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        np.testing.assert_allclose(norm_views[0]["pts3d"], self.pts_1 / factor, rtol=1e-5)
        np.testing.assert_allclose(norm_views[1]["pts3d"], self.pts_2 / factor, rtol=1e-5)

    def test_06_camera_translations_normalized(self):
        """6. Camera translations normalized using the same factor."""
        norm_views, factor = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        np.testing.assert_allclose(norm_views[0]["camera_poses"][:3, 3], self.c2w_1[:3, 3] / factor, rtol=1e-5)
        np.testing.assert_allclose(norm_views[1]["camera_poses"][:3, 3], self.c2w_2[:3, 3] / factor, rtol=1e-5)

    def test_07_camera_rotations_unchanged(self):
        """7. Camera rotations remain unchanged."""
        norm_views, _ = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        np.testing.assert_allclose(norm_views[0]["camera_poses"][:3, :3], self.c2w_1[:3, :3], atol=1e-7)
        np.testing.assert_allclose(norm_views[1]["camera_poses"][:3, :3], self.c2w_2[:3, :3], atol=1e-7)

    def test_08_intrinsics_unchanged(self):
        """8. Intrinsics remain unchanged."""
        norm_views, _ = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        np.testing.assert_allclose(norm_views[0]["camera_intrs"], self.K, atol=1e-7)
        np.testing.assert_allclose(norm_views[1]["camera_intrs"], self.K, atol=1e-7)

    def test_09_surface_normals_retain_unit_length(self):
        """9. Surface normals retain unit length."""
        norm_views, _ = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        n1 = norm_views[0]["normals"]
        n2 = norm_views[1]["normals"]
        norm_lens_1 = np.linalg.norm(n1, ord=2, axis=-1)
        norm_lens_2 = np.linalg.norm(n2, ord=2, axis=-1)
        np.testing.assert_allclose(norm_lens_1, 1.0, atol=1e-6)
        np.testing.assert_allclose(norm_lens_2, 1.0, atol=1e-6)

    def test_10_denormalization_exact_recovery(self):
        """10. Denormalization restores original metric coordinates."""
        norm_views, factor = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        restored_views = denormalize_scene_coordinates(norm_views, norm_factor=factor)

        np.testing.assert_allclose(restored_views[0]["depthmap"], self.depth_1, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(restored_views[1]["depthmap"], self.depth_2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(restored_views[0]["pts3d"], self.pts_1, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(restored_views[1]["pts3d"], self.pts_2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(restored_views[0]["camera_poses"][:3, 3], self.c2w_1[:3, 3], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(restored_views[1]["camera_poses"][:3, 3], self.c2w_2[:3, 3], rtol=1e-5, atol=1e-5)

    def test_11_no_nan_or_inf_introduced(self):
        """11. No NaN or Inf is introduced."""
        norm_views, factor = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        for v in norm_views:
            self.assertTrue(np.isfinite(v["pts3d"]).all())
            self.assertTrue(np.isfinite(v["depthmap"]).all())
            self.assertTrue(np.isfinite(v["camera_poses"]).all())
        self.assertTrue(np.isfinite(factor))

    def test_12_camera_to_world_and_world_to_camera_conventions(self):
        """12. Camera-to-world and world-to-camera conventions remain valid under normalization."""
        norm_views, factor = normalize_scene_coordinates(self.views, norm_mode="median_dis")
        c2w_norm = norm_views[0]["camera_poses"]

        # Re-derive world points from normalized depth and normalized camera pose
        pts_norm_rederived, _ = compute_world_coordinates(
            norm_views[0]["depthmap"],
            norm_views[0]["camera_intrs"],
            c2w_norm,
            c2w=True,
        )
        # Should match normalized point map exactly
        np.testing.assert_allclose(pts_norm_rederived, norm_views[0]["pts3d"], rtol=1e-5, atol=1e-5)

        # Invert c2w to get w2c
        w2c_norm = np.linalg.inv(c2w_norm)
        # World origin in camera coordinates
        origin_w = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        origin_cam = w2c_norm @ origin_w
        self.assertTrue(np.isfinite(origin_cam).all())


if __name__ == "__main__":
    unittest.main()

