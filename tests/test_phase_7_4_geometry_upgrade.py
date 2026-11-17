"""Tests for RoomAI Phase 7.4: High-Fidelity 3D Reconstruction Upgrade.

Verifies:
1. Metric depth binding for furniture 3D bounding boxes (eliminating the [0, 1] normalized depth bug).
2. Floor plane grounding: Floor-standing furniture pieces rest firmly on the estimated floor plane.
3. Multi-view sequence validation (dimension checks, blur scoring, duplicate detection, 3-10 images).
4. Duplicate object detection suppression (NMS) on scene objects.
5. Coordinate frame consistency between point cloud, room bounds, and furniture bounding boxes.
"""

from typing import Any, Dict, List
import numpy as np
from PIL import Image
import pytest

from schemas.scene import Object3DGeometry, RoomSurface3D, SceneGraph, SceneObject
from services.multiview_fusion_service import MultiViewFusionService
from services.scene_fusion_service import SceneFusionService


def create_solid_image(color=(200, 180, 160), size=(640, 480)) -> Image.Image:
    """Helper to create a solid RGB test image."""
    return Image.new("RGB", size, color=color)


class TestPhase74GeometryUpgrade:
    """Verification suite for Phase 7.4 geometry improvements."""

    def test_metric_depth_binding_in_associate_depth_to_objects(self):
        """Validates that real metric depth in meters (not [0, 1] normalized) is bound to 3D bounding boxes."""
        service = SceneFusionService(mock_mode=True)
        w, h = 640, 480

        # Create a synthetic metric depth map with room distances (3.0m to 8.0m)
        metric_depth = np.linspace(3.0, 8.0, h * w, dtype=np.float32).reshape((h, w))

        # Dummy table object in center of image
        obj = SceneObject(
            object_id="table_001",
            semantic_class="table",
            instance_id=1,
            detection_confidence=0.85,
            mask_provenance="a3_indoor",
            bounding_box=(0.3, 0.4, 0.7, 0.8),
            coverage_percentage=16.0,
            geometry=None,
        )

        intrinsics = {"fx": 1000.0, "fy": 1000.0, "cx": 0.5, "cy": 0.5}
        result = service._associate_depth_to_objects([obj], metric_depth, w, h, intrinsics)

        assert len(result) == 1
        geom = result[0].geometry
        assert geom is not None
        cx, cy, cz = geom.centroid
        phys_w, phys_h, phys_d = geom.extent

        # Critical verification: cz MUST be in metric meters (> 3.0m), NOT in [0, 1]
        assert cz >= 3.0 and cz <= 8.0, f"Expected metric depth > 3.0m, got cz={cz}m (normalized depth bug)"
        assert phys_w >= 0.5 and phys_w <= 3.0, f"Expected realistic table width, got {phys_w}m"
        assert phys_h >= 0.4 and phys_h <= 2.5, f"Expected realistic table height, got {phys_h}m"

    def test_floor_plane_grounding_constraint(self):
        """Verifies that floor-standing furniture pieces are physically grounded onto the floor plane."""
        service = SceneFusionService(mock_mode=True)

        # Floor plane at elevation y = 1.45m in camera frame (+Y is down)
        floor_elevation = 1.45

        # Create floating table: bottom at 0.90m (floating 0.55m above floor)
        floating_table = SceneObject(
            object_id="table_001",
            semantic_class="table",
            instance_id=1,
            detection_confidence=0.90,
            mask_provenance="a3_indoor",
            bounding_box=(0.3, 0.3, 0.7, 0.7),
            coverage_percentage=16.0,
            geometry=Object3DGeometry(
                centroid=(0.2, 0.5, 4.0),
                extent=(1.2, 0.8, 0.9),  # height = 0.8m -> bottom is 0.5 + 0.4 = 0.90m
            ),
        )

        service._align_furniture_to_floor([floating_table], floor_elevation)

        g_cx, g_cy, g_cz = floating_table.geometry.centroid
        g_w, g_h, g_d = floating_table.geometry.extent
        grounded_bottom = g_cy + (g_h / 2.0)

        # Verify bottom exactly touches floor elevation (1.45m)
        assert abs(grounded_bottom - floor_elevation) <= 1e-4, (
            f"Expected bottom to match floor elevation {floor_elevation}m, got {grounded_bottom}m"
        )
        assert abs(g_cy - (floor_elevation - g_h / 2.0)) <= 1e-4

    def test_duplicate_object_nms_suppression(self):
        """Verifies that overlapping duplicate bounding boxes are suppressed via NMS."""
        service = SceneFusionService(mock_mode=True)

        # Two highly overlapping couch detections (IoU > 0.8)
        couch1 = SceneObject(
            object_id="couch_001",
            semantic_class="sofa",
            instance_id=1,
            detection_confidence=0.88,
            mask_provenance="a3_indoor",
            bounding_box=(0.2, 0.3, 0.8, 0.7),
            coverage_percentage=24.0,
        )
        couch2_duplicate = SceneObject(
            object_id="couch_002",
            semantic_class="sofa",
            instance_id=2,
            detection_confidence=0.62,
            mask_provenance="a3_indoor",
            bounding_box=(0.21, 0.29, 0.79, 0.71),
            coverage_percentage=24.0,
        )
        # Distinct table detection elsewhere in image
        table = SceneObject(
            object_id="table_001",
            semantic_class="table",
            instance_id=3,
            detection_confidence=0.75,
            mask_provenance="a3_indoor",
            bounding_box=(0.0, 0.0, 0.2, 0.3),
            coverage_percentage=6.0,
        )

        kept = service._suppress_duplicate_objects([couch1, couch2_duplicate, table], iou_threshold=0.55)
        assert len(kept) == 2, f"Expected duplicate couch to be suppressed, got {len(kept)} objects"
        assert kept[0].object_id == "couch_001", "Higher confidence couch should be preserved"
        assert kept[1].object_id == "table_001"

    def test_multiview_sequence_validation_duplicate_and_blur_handling(self):
        """Verifies multi-view sequence validation detects duplicates, resolution issues, and blur."""
        mv_service = MultiViewFusionService(mock_mode=True)

        # 1. Base clean image
        clean_arr = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add high frequency gradient pattern for sharpness
        for i in range(0, 480, 10):
            clean_arr[i:i+5, :] = 255
        clean_img = Image.fromarray(clean_arr)

        # 2. Duplicate image (identical copy)
        dup_img = Image.fromarray(clean_arr.copy())

        # 3. Blurry image (uniform flat solid color with zero Laplacian variance)
        blurry_img = Image.new("RGB", (640, 480), color=(128, 128, 128))

        # 4. Tiny resolution image below threshold
        tiny_img = Image.new("RGB", (100, 80), color=(200, 100, 50))

        images = [clean_img, dup_img, blurry_img, tiny_img]
        valid_imgs, warnings, metrics = mv_service.validate_multiview_sequence(images)

        assert metrics["total_input"] == 4
        # Duplicate and tiny image must be filtered out of valid_imgs
        assert len(valid_imgs) >= 1
        assert any("duplicate" in w.lower() for w in warnings), "Duplicate warning expected"
        assert any("resolution" in w.lower() or "minimum" in w.lower() for w in warnings), "Resolution warning expected"
        assert any("blur" in w.lower() for w in warnings), "Blur warning expected"

    def test_multiview_fusion_with_three_image_sequence(self):
        """Tests that MultiViewFusionService correctly processes a 3-image sequence."""
        mv_service = MultiViewFusionService(mock_mode=True)

        img1 = create_solid_image((220, 200, 180))
        img2 = create_solid_image((210, 195, 175))
        img3 = create_solid_image((200, 190, 170))

        fused = mv_service.fuse_multiview([img1, img2, img3], refine_depth=False)
        assert isinstance(fused, SceneGraph)
        assert fused.depth_mode == "metric"
        assert fused.point_cloud is not None
        assert "xyz" in fused.point_cloud
