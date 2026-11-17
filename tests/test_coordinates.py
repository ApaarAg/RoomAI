"""Unit tests verifying 2D-to-3D Coordinate Unprojection invariants for RoomAI."""

import math
import pytest


def bbox_to_ndc(norm_x1: float, norm_y1: float, norm_x2: float, norm_y2: float):
    norm_u = (norm_x1 + norm_x2) / 2.0
    norm_v = (norm_y1 + norm_y2) / 2.0
    x_ndc = (norm_u - 0.5) * 2.0
    y_ndc = -(norm_v - 0.5) * 2.0  # Y inverted for Three.js OpenGL conventions
    return x_ndc, y_ndc


def unproject_to_3d(norm_x1, norm_y1, norm_x2, norm_y2, depth_m=3.0, fov_deg=34.0, aspect=16/9):
    x_ndc, y_ndc = bbox_to_ndc(norm_x1, norm_y1, norm_x2, norm_y2)
    tan_half_fov = math.tan(math.radians(fov_deg * 0.5))
    x_cam = x_ndc * depth_m * tan_half_fov * aspect
    y_cam = y_ndc * depth_m * tan_half_fov
    z_cam = -depth_m
    return x_cam, y_cam, z_cam


class TestCoordinateTransformation:
    def test_image_center_maps_to_optical_axis(self):
        """Center of image (0.5, 0.5) must project exactly along the optical center (X=0, Y=0)."""
        x_ndc, y_ndc = bbox_to_ndc(0.4, 0.4, 0.6, 0.6)
        assert abs(x_ndc) < 1e-6
        assert abs(y_ndc) < 1e-6

        x_cam, y_cam, z_cam = unproject_to_3d(0.4, 0.4, 0.6, 0.6, depth_m=3.0)
        assert abs(x_cam) < 1e-6
        assert abs(y_cam) < 1e-6
        assert z_cam == -3.0

    def test_top_left_corner_maps_to_negative_x_positive_y(self):
        """Top-left corner (0.0, 0.0) has negative X and positive Y in Three.js camera space."""
        x_ndc, y_ndc = bbox_to_ndc(0.0, 0.0, 0.1, 0.1)
        # Center is at (0.05, 0.05) -> NDC should be negative for X, positive for Y
        assert x_ndc < 0.0
        assert y_ndc > 0.0

        x_cam, y_cam, z_cam = unproject_to_3d(0.0, 0.0, 0.1, 0.1, depth_m=2.5)
        assert x_cam < 0.0
        assert y_cam > 0.0
        assert z_cam == -2.5

    def test_bottom_right_corner_maps_to_positive_x_negative_y(self):
        """Bottom-right corner has positive X and negative Y in Three.js camera space."""
        x_ndc, y_ndc = bbox_to_ndc(0.9, 0.9, 1.0, 1.0)
        assert x_ndc > 0.0
        assert y_ndc < 0.0

        x_cam, y_cam, z_cam = unproject_to_3d(0.9, 0.9, 1.0, 1.0, depth_m=4.0)
        assert x_cam > 0.0
        assert y_cam < 0.0
        assert z_cam == -4.0

    def test_depth_scaling_is_linear(self):
        """Doubling distance Z must double metric lateral offset (pinhole projection)."""
        x1, y1, _ = unproject_to_3d(0.6, 0.6, 0.8, 0.8, depth_m=2.0)
        x2, y2, _ = unproject_to_3d(0.6, 0.6, 0.8, 0.8, depth_m=4.0)
        assert pytest.approx(x2, rel=1e-5) == x1 * 2.0
        assert pytest.approx(y2, rel=1e-5) == y1 * 2.0
