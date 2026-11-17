"""Comprehensive Verification and Integration Tests for Phase 7.6 Backend Completion & Frontend Handoff.

Validates:
1. ModelRegistry completeness: all production models registered, including metric depth.
2. Production service contracts: real inference is default, mock_mode defaults to False across all services.
3. SceneGraph schema completeness: depth_map, heatmap_image, multiview_telemetry, and provenance.
4. RoomAIBackend facade: unified single-entry API for frontend integration.
5. Export assets integrity: valid GLB and PLY bytes generation.
6. Graceful degradation: offline VLM and generative models return structured flags without crashing.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from models.registry import GLOBAL_MODEL_REGISTRY, ModelSpec
from schemas.scene import SceneGraph, SceneObject, RoomSurface3D, Object3DGeometry
from services.classification_service import ClassificationService
from services.detection_service import DetectionService
from services.generation_service import GenerationService
from services.multiview_fusion_service import MultiViewFusionService
from services.roomai_backend import RoomAIBackend, BACKEND
from services.scene3d_service import Scene3DService
from services.scene_fusion_service import SceneFusionService
from services.segmentation_service import SegmentationService
from services.vision_language_service import VisionLanguageService


def test_model_registry_includes_metric_depth():
    """Verify depth_anything_v2_metric_indoor_small is registered as primary metric depth model."""
    spec = GLOBAL_MODEL_REGISTRY.get_spec("depth_anything_v2_metric_indoor_small")
    assert spec is not None, "depth_anything_v2_metric_indoor_small must be registered in ModelRegistry"
    assert spec.task_id == "scene3d_depth_metric"
    assert spec.is_primary is True
    assert "metric" in spec.checkpoint_rel_path.lower()
    assert spec.exists(), "Metric depth checkpoint must exist on disk"

    # Also check relative depth spec
    rel_spec = GLOBAL_MODEL_REGISTRY.get_spec("depth_anything_v2_small")
    assert rel_spec is not None
    assert rel_spec.task_id == "scene3d_depth"
    assert rel_spec.is_primary is True


def test_services_default_to_real_inference_not_mock():
    """All production services must default mock_mode to False."""
    cls_svc = ClassificationService()
    assert cls_svc.mock_mode is False, "ClassificationService must default mock_mode to False"

    gen_svc = GenerationService()
    assert gen_svc.mock_mode is False, "GenerationService must default mock_mode to False"

    det_svc = DetectionService()
    assert det_svc.mock_mode is False, "DetectionService must default mock_mode to False"

    seg_svc = SegmentationService()
    assert seg_svc.mock_mode is False, "SegmentationService must default mock_mode to False"

    sf_svc = SceneFusionService()
    assert sf_svc.mock_mode is False, "SceneFusionService must default mock_mode to False"

    mv_svc = MultiViewFusionService()
    assert mv_svc.mock_mode is False, "MultiViewFusionService must default mock_mode to False"

    vlm_svc = VisionLanguageService()
    assert vlm_svc.mock_mode is False, "VisionLanguageService must default mock_mode to False"


def test_scene_graph_first_class_fields():
    """SceneGraph schema must natively support depth maps, heatmaps, and multiview telemetry."""
    depth_arr = np.zeros((10, 10), dtype=np.float32)
    telemetry = [{"R": np.eye(3).tolist(), "rmse_m": 0.05}]
    provenance = [{"view_index": 0, "registered": True}]

    scene = SceneGraph(
        width=10,
        height=10,
        model_name="TestModel",
        depth_mode="metric",
        depth_map=depth_arr,
        multiview_telemetry=telemetry,
        multiview_per_view_provenance=provenance,
    )

    assert scene.depth_map is not None
    assert scene.multiview_telemetry == telemetry
    assert scene.multiview_per_view_provenance == provenance

    # Verify Pydantic serialization
    data = scene.model_dump()
    assert data["width"] == 10
    assert data["depth_mode"] == "metric"
    assert "multiview_telemetry" in data


def test_roomai_backend_health_check():
    """Backend facade must report hardware and checkpoint status without throwing."""
    backend = RoomAIBackend()
    health = backend.get_system_health()

    assert "cuda_available" in health
    assert "total_vram_mb" in health
    assert "vram_headroom_mb" in health
    assert "checkpoints_status" in health

    status = health["checkpoints_status"]
    assert status["indoor_a3"] is True
    assert status["exterior_b2"] is True
    assert status["parking_c2"] is True
    assert status["yolo11_seg"] is True
    assert status["depth_anything_v2_metric"] is True
    assert status["depth_anything_v2_relative"] is True


def test_roomai_backend_detection_mock_mode():
    """Backend facade detect_objects executes in mock mode."""
    backend = RoomAIBackend()
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))

    res = backend.detect_objects(img, mock_mode=True)
    assert res["success"] is True
    assert res["response"] is not None
    assert res["response"].is_simulated is True
    assert res["annotated_image"] is not None


def test_roomai_backend_segmentation_mock_mode():
    """Backend facade segment_scene executes in mock mode."""
    backend = RoomAIBackend()
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))

    res = backend.segment_scene(img, mock_mode=True)
    assert res["success"] is True
    assert res["response"] is not None
    assert res["response"].is_simulated is True
    assert res["overlay_image"] is not None


def test_roomai_backend_depth_estimation_mock_mode():
    """Backend facade estimate_depth executes in mock mode."""
    backend = RoomAIBackend()
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))

    res_metric = backend.estimate_depth(img, mode="metric", mock_mode=True)
    assert res_metric["success"] is True
    assert res_metric["depth_map"] is not None
    assert res_metric["heatmap_image"] is not None


def test_roomai_backend_3d_reconstruction_single_view_mock():
    """Backend facade reconstruct_room_3d executes with single image in mock mode."""
    backend = RoomAIBackend()
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))

    res = backend.reconstruct_room_3d(img, mock_mode=True)
    assert res["success"] is True
    assert res["scene_graph"] is not None
    assert len(res["surfaces"]) > 0
    assert res["room_layout"] is not None


def test_roomai_backend_3d_reconstruction_multi_view_mock():
    """Backend facade reconstruct_room_3d executes with multiple images in mock mode."""
    backend = RoomAIBackend()
    img1 = Image.new("RGB", (320, 240), color=(100, 100, 100))
    img2 = Image.new("RGB", (320, 240), color=(120, 120, 120))

    res = backend.reconstruct_room_3d([img1, img2], mock_mode=True)
    assert res["success"] is True
    assert res["scene_graph"] is not None
    assert res["point_cloud"] is not None


def test_roomai_backend_export_3d_assets():
    """Backend facade export_3d_assets generates valid non-empty GLB and PLY bytes."""
    backend = RoomAIBackend()
    mock_cloud = {
        "xyz": [[0.0, 0.0, 1.0], [0.5, 0.2, 1.2], [-0.3, -0.1, 0.9]],
        "rgb": [[0.8, 0.2, 0.1], [0.2, 0.8, 0.3], [0.1, 0.2, 0.9]],
    }
    mock_objects = [
        SceneObject(
            object_id="chair_001",
            semantic_class="chair",
            instance_id=1,
            detection_confidence=0.92,
            mask_provenance="mock",
            bounding_box=(0.1, 0.1, 0.5, 0.5),
            coverage_percentage=12.0,
            geometry=Object3DGeometry(
                centroid=(0.1, 0.2, 1.1),
                extent=(0.6, 0.8, 0.6),
            ),
        )
    ]
    scene = SceneGraph(
        width=320,
        height=240,
        model_name="MockReconstructor",
        point_cloud=mock_cloud,
        objects=mock_objects,
    )

    exports = backend.export_3d_assets(scene)
    assert "glb" in exports
    assert "ply" in exports

    # Check GLB header magic (b"glTF")
    assert exports["glb"][:4] == b"glTF", "GLB must begin with glTF binary magic"
    assert len(exports["glb"]) > 100

    # Check PLY ASCII header (b"ply\n")
    assert exports["ply"].startswith(b"ply\n"), "PLY must begin with ASCII ply header"
    assert len(exports["ply"]) > 50


def test_roomai_backend_vlm_graceful_degradation():
    """Backend facade ask_vlm handles offline weights gracefully without crashing."""
    backend = RoomAIBackend()
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))

    # In real mode, when weights are offline, is_implemented must be False and no exception raised
    res = backend.ask_vlm(img, "What color is the sofa?", mock_mode=False)
    assert res["is_implemented"] is False
    assert res["success"] is False
    assert "not yet implemented" in res["error_message"].lower() or "missing" in res["error_message"].lower()

    # In mock mode, is_implemented must be True and return an answer
    res_mock = backend.ask_vlm(img, "What color is the sofa?", mock_mode=True)
    assert res_mock["success"] is True
    assert res_mock["response"] is not None
    assert len(res_mock["response"].answer) > 10


def test_roomai_backend_capabilities():
    """Verify backend get_capabilities() returns comprehensive capability matrix."""
    backend = RoomAIBackend()
    caps = backend.get_capabilities()
    assert isinstance(caps, dict)
    assert "indoor_detection" in caps
    assert caps["indoor_detection"]["status"] == "PRODUCTION_READY"
    assert "single_view_3d_reconstruction" in caps
    assert caps["single_view_3d_reconstruction"]["status"] == "PRODUCTION_READY"
    assert "multi_view_3d_reconstruction" in caps
    assert caps["multi_view_3d_reconstruction"]["status"] == "EXPERIMENTAL"
    assert "vlm_qa" in caps
    assert caps["vlm_qa"]["status"] == "UNAVAILABLE"
    assert "generative_redesign" in caps
    assert caps["generative_redesign"]["status"] == "UNAVAILABLE"


def test_roomai_backend_reconstruct_multiview_convenience():
    """Verify reconstruct_room_multiview returns envelope with EXPERIMENTAL capability_status."""
    backend = RoomAIBackend()
    img1 = Image.new("RGB", (320, 240), color=(100, 100, 100))
    img2 = Image.new("RGB", (320, 240), color=(120, 120, 120))

    res = backend.reconstruct_room_multiview([img1, img2], mock_mode=True)
    assert res["success"] is True
    assert res["capability_status"] == "EXPERIMENTAL"
    assert any("EXPERIMENTAL" in w for w in res["warnings"])
    assert res["status"] == "success"
    assert "data" in res


def test_export_validators_and_empty_inputs():
    """Verify GLB/PLY validators and safe empty array handling."""
    from utils.export_3d import export_point_cloud_ply, export_scene_glb, validate_ply_bytes, validate_glb_bytes

    # Empty point cloud should export valid 0-vertex PLY without throwing AxisError
    empty_ply = export_point_cloud_ply([], None)
    assert isinstance(empty_ply, bytes)
    val_empty_ply = validate_ply_bytes(empty_ply)
    assert val_empty_ply["is_valid"] is True
    assert val_empty_ply["vertex_count"] == 0

    # Corrupted PLY
    assert validate_ply_bytes(b"invalid data")["is_valid"] is False

    # Empty geometry GLB
    empty_glb = export_scene_glb()
    assert isinstance(empty_glb, bytes)
    val_empty_glb = validate_glb_bytes(empty_glb)
    assert val_empty_glb["is_valid"] is True

    # Corrupted GLB
    assert validate_glb_bytes(b"corrupt")["is_valid"] is False


def test_service_input_sanitization_none_image():
    """Verify all perception services gracefully handle None image without crashing."""
    det_svc = DetectionService(mock_mode=True)
    det_res = det_svc.run(None)
    assert det_res["success"] is False
    assert "None" in det_res["error_message"]

    seg_svc = SegmentationService(mock_mode=True)
    seg_res = seg_svc.run(None)
    assert seg_res["success"] is False
    assert "None" in seg_res["error_message"]

    s3d_svc = Scene3DService(mock_mode=True)
    s3d_res = s3d_svc.run(None)
    assert s3d_res["success"] is False
    assert "None" in s3d_res["error_message"]

    sf_svc = SceneFusionService(mock_mode=True)
    with pytest.raises(ValueError, match="Input image cannot be None"):
        sf_svc.fuse(None)


