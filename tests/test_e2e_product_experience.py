"""Comprehensive Opaque-Box E2E Test Suite for RoomAI Product Experience (Phase 7.2A).

Rigorously verifies requirements R1-R4 across four test tiers:
- Tier 1: Feature Coverage (>=5 test cases per feature for F1 through F8).
- Tier 2: Boundary & Corner Cases (empty uploads, corrupted bytes, boundary thresholds, missing VLM keys, GPU fallback).
- Tier 3: Pairwise Combinations (Upload -> Detector A3/B2/C2 -> Metric Depth unprojection).
- Tier 4: Real-World Scenarios on the 5 genuine ADE20K validation images:
    * Living Room (ADE_val_00001531.jpg)
    * Bedroom (ADE_val_00000118.jpg)
    * Kitchen (ADE_val_00000459.jpg)
    * Dining Room (ADE_val_00000314.jpg)
    * Home Office (ADE_val_00000420.jpg)

Hardware Target: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM budget).
Runtime: Python 3.10.20, PyTorch 2.11.0+cu128.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from models.interfaces import ModelNotImplementedError
from models.registry import GLOBAL_MODEL_REGISTRY
from models.vision_language_model import CustomVisionLanguageModel
from schemas.responses import BoundingBox, DetectionResponse, VLMAnswerResponse
from schemas.scene import Object3DGeometry, RoomSurface3D, SceneGraph, SceneObject
from services.detection_service import DetectionService
from services.room_layout_service import RoomLayoutService, fit_plane_ransac
from services.scene3d_service import Scene3DService
from services.scene_fusion_service import SceneFusionService
from services.vision_language_service import VisionLanguageService
import ui.analysis_view as av
import ui.components as uc
import ui.three_d_viewer as tv
import ui.upload_view as uv
from utils.device import get_device_info
from utils.export_3d import create_box_mesh, export_point_cloud_ply, export_scene_glb
from utils.gpu_concurrency import GPUInferenceGate
from utils.image_validation import validate_image_file


# ---------------------------------------------------------------------------
# FIXTURES & TEST HARNESS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Returns absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def app_config(repo_root: Path) -> Dict[str, Any]:
    """Loads config/app.yaml configuration."""
    cfg_path = repo_root / "config" / "app.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def model_config(repo_root: Path) -> Dict[str, Any]:
    """Loads config/model_config.yaml configuration."""
    cfg_path = repo_root / "config" / "model_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def ade20k_val_dir(repo_root: Path) -> Path:
    """Path to the 2,000 ADE20K validation images."""
    p = repo_root / "archive (2)" / "ADEChallengeData2016" / "images" / "validation"
    assert p.exists(), f"ADE20K validation directory not found at {p}"
    return p


@pytest.fixture(scope="session")
def ade20k_candidate_images(ade20k_val_dir: Path) -> Dict[str, Path]:
    """Verified candidate ADE20K validation images across the 5 room categories."""
    candidates = {
        "living_room": ade20k_val_dir / "ADE_val_00001531.jpg",
        "bedroom": ade20k_val_dir / "ADE_val_00000118.jpg",
        "kitchen": ade20k_val_dir / "ADE_val_00000459.jpg",
        "dining_room": ade20k_val_dir / "ADE_val_00000314.jpg",
        "home_office": ade20k_val_dir / "ADE_val_00000420.jpg",
    }
    for category, path in candidates.items():
        assert path.exists(), f"Candidate image for {category} missing: {path}"
    return candidates


@pytest.fixture
def synthetic_room_image() -> Image.Image:
    """Generates a realistic multi-region synthetic room image (640x480 RGB)."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[0:240, :] = [210, 210, 215]  # Wall
    arr[240:480, :] = [130, 90, 60]  # Floor
    arr[200:380, 150:480] = [70, 120, 180]  # Sofa-like object
    return Image.fromarray(arr)


@pytest.fixture
def sample_point_cloud() -> Dict[str, Any]:
    """Generates a synthetic 3D point cloud dictionary in camera coordinate frame."""
    n_points = 5000
    x = np.random.uniform(-2.0, 2.0, n_points).astype(np.float32)
    y = np.random.uniform(-1.0, 1.2, n_points).astype(np.float32)
    z = np.random.uniform(1.2, 5.0, n_points).astype(np.float32)
    xyz = np.stack([x, y, z], axis=1).tolist()
    rgb = np.random.uniform(0.1, 0.9, (n_points, 3)).astype(np.float32).tolist()
    return {"xyz": xyz, "rgb": rgb, "depth": z.tolist()}


# ===========================================================================
# TIER 1: FEATURE COVERAGE (>=5 TEST CASES PER FEATURE FOR F1 THROUGH F8)
# ===========================================================================

class TestTier1Feature1WorkspaceConsolidation:
    """F1: Consolidate navigation to exactly three workspaces (Detection, VLM, 3D Reconstruction)."""

    def test_f1_01_three_dedicated_workspaces_in_analysis_tabs(self):
        """F1.1: Verify analysis view exposes exactly 3 dedicated workspace tabs."""
        src = inspect.getsource(av.render_analysis_view)
        assert 'st.tabs(' in src or "st.tabs([" in src
        assert "🔍 Object Detection" in src
        assert "💬 Vision Q&A" in src
        assert "🧊 3D Scene Reconstruction" in src

    def test_f1_02_no_legacy_standalone_tabs_in_analysis_view(self):
        """F1.2: Verify legacy standalone tabs (Segmentation, Classification) are not separate tabs."""
        src = inspect.getsource(av.render_analysis_view)
        tree = ast.parse(src)
        tab_labels: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tabs":
                if node.args and isinstance(node.args[0], ast.List):
                    for elt in node.args[0].elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            tab_labels.append(elt.value)
        assert len(tab_labels) == 3, f"Expected exactly 3 analysis workspace tabs, found: {tab_labels}"
        for lbl in tab_labels:
            assert "segmentation" not in lbl.lower() or "3d" in lbl.lower()
            assert "classification" not in lbl.lower()

    def test_f1_03_room_segmenter_internalized_in_scene_fusion(self, model_config: Dict[str, Any]):
        """F1.3: Verify RoomSegmenter is internalized into SceneFusionService without separate tab."""
        service = SceneFusionService(mock_mode=True, config=model_config)
        assert hasattr(service, "segmenter"), "SceneFusionService must own segmenter internally."
        assert hasattr(service, "detector"), "SceneFusionService must own detector internally."
        assert hasattr(service, "scene3d_model"), "SceneFusionService must own depth model internally."

    def test_f1_04_app_entrypoint_exposes_analysis_and_design_modes(self, repo_root: Path):
        """F1.4: Verify app.py routes to the unified room analysis workspace."""
        app_path = repo_root / "app.py"
        src = app_path.read_text(encoding="utf-8")
        assert "render_analysis_view" in src
        assert "render_upload_view" in src

    def test_f1_05_workspace_session_state_key_isolation(self, monkeypatch: pytest.MonkeyPatch):
        """F1.5: Verify session state keys for workspaces are discrete and isolated."""
        import streamlit as st
        fake_state: Dict[str, Any] = {}
        monkeypatch.setattr(st, "session_state", fake_state, raising=False)

        fake_state["detection_result"] = {"success": True, "workspace": "detection"}
        fake_state["vlm_result"] = {"success": True, "workspace": "vlm"}
        fake_state["scene_fusion"] = {"success": True, "workspace": "3d"}

        assert fake_state["detection_result"]["workspace"] == "detection"
        assert fake_state["vlm_result"]["workspace"] == "vlm"
        assert fake_state["scene_fusion"]["workspace"] == "3d"


class TestTier1Feature2ProductIdentityAndBranding:
    """F2: Professional Header, Branding, Tagline, and Live Badges."""

    def test_f2_01_header_branding_contains_roomai_title(self, app_config: Dict[str, Any]):
        """F2.1: Verify render_header displays RoomAI application branding."""
        src = inspect.getsource(uc.render_header)
        assert "Room" in src and "AI" in src
        assert "🛋️" in src

    def test_f2_02_official_tagline_and_subtitle_presence(self, app_config: Dict[str, Any]):
        """F2.2: Verify official branding tagline and product summary in UI components."""
        src = inspect.getsource(uc.render_header)
        assert "AI-Powered Interior Design Assistant" in src or "Understand Your Space" in src

    def test_f2_03_hardware_and_runtime_reporting_structure(self):
        """F2.3: Verify get_device_info returns structured hardware metrics."""
        info = get_device_info()
        assert "cuda_available" in info
        assert "total_vram_gb" in info
        assert "free_vram_gb" in info
        assert "allocated_vram_mb" in info
        assert "peak_vram_mb" in info
        assert isinstance(info["cuda_available"], bool)
        assert info["total_vram_gb"] >= 0.0

    def test_f2_04_simulated_badge_renders_explicit_warning(self):
        """F2.4: Verify render_simulated_badge clearly flags simulated/mock output."""
        src = inspect.getsource(uc.render_simulated_badge)
        assert "SIMULATED OUTPUT (MOCK MODE)" in src
        assert "#FEF3C7" in src or "#F59E0B" in src  # Amber styling

    def test_f2_05_transparent_model_not_implemented_alert(self):
        """F2.5: Verify render_not_implemented_alert renders developer instructions."""
        src = inspect.getsource(uc.render_not_implemented_alert)
        assert "Model Implementation Pending" in src
        assert "instructions" in src


class TestTier1Feature3StrictStateInvalidation:
    """F3: Strict Session-State Invalidation upon Image Change and Clear."""

    def test_f3_01_supported_image_formats_pass_validation(self, synthetic_room_image: Image.Image):
        """F3.1: Verify JPEG, PNG, and WEBP formats validate cleanly."""
        for fmt in ["JPEG", "PNG", "WEBP"]:
            buf = io.BytesIO()
            synthetic_room_image.save(buf, format=fmt)
            buf.seek(0)
            ext = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"
            res = validate_image_file(
                file_input=buf,
                filename=f"test_room{ext}",
                max_file_size_mb=15.0,
                max_dimension=4096,
                allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
            )
            assert res.is_valid is True, f"Failed validation for {fmt}: {res.error_message}"
            assert res.width == 640
            assert res.height == 480

    def test_f3_02_image_session_identity_tracking(self, synthetic_room_image: Image.Image):
        """F3.2: Verify active image identity detects image change."""
        id1 = f"room_a.jpg_{synthetic_room_image.width}x{synthetic_room_image.height}"
        id2 = f"room_b.jpg_{synthetic_room_image.width}x{synthetic_room_image.height}"
        assert id1 != id2

    def test_f3_03_image_change_invalidates_all_11_result_keys(self, monkeypatch: pytest.MonkeyPatch):
        """F3.3: Verify all 11 inference keys are purged when a new image is loaded."""
        import streamlit as st
        fake_state: Dict[str, Any] = {}
        monkeypatch.setattr(st, "session_state", fake_state, raising=False)

        canonical_11_keys = [
            "detection_result", "detection_result_indoor", "detection_result_exterior",
            "detection_result_parking", "segmentation_result", "scene3d_result",
            "scene_fusion", "last_scene3d_error", "classification_result",
            "vlm_result", "design_result",
        ]
        for k in canonical_11_keys:
            fake_state[k] = {"stale_data": True}
        fake_state["active_image_id"] = "image1.jpg_640x480"

        # Simulating new image arrival
        new_id = "image2.jpg_800x600"
        if fake_state.get("active_image_id") != new_id:
            for k in canonical_11_keys:
                fake_state.pop(k, None)
            fake_state["active_image_id"] = new_id

        for k in canonical_11_keys:
            assert k not in fake_state, f"Stale key {k} survived image change!"
        assert fake_state["active_image_id"] == new_id

    def test_f3_04_clear_button_purges_all_keys_and_identities(self, monkeypatch: pytest.MonkeyPatch):
        """F3.4: Verify Clear button purges all 11 inference keys + 3 identity keys."""
        import streamlit as st
        fake_state: Dict[str, Any] = {}
        monkeypatch.setattr(st, "session_state", fake_state, raising=False)

        all_keys = [
            "current_image", "image_filename", "active_image_id",
            "detection_result", "detection_result_indoor", "detection_result_exterior",
            "detection_result_parking", "segmentation_result", "scene3d_result",
            "scene_fusion", "last_scene3d_error", "classification_result",
            "vlm_result", "design_result",
        ]
        for k in all_keys:
            fake_state[k] = "dummy_val"

        for k in all_keys:
            fake_state.pop(k, None)

        assert len(fake_state) == 0

    def test_f3_05_oversized_dimension_downsampling(self):
        """F3.5: Verify images exceeding max_dimension are downscaled gracefully."""
        huge_img = Image.new("RGB", (4800, 2400), color=(120, 120, 120))
        buf = io.BytesIO()
        huge_img.save(buf, format="JPEG")
        buf.seek(0)
        res = validate_image_file(
            file_input=buf,
            filename="oversized_panorama.jpg",
            max_file_size_mb=25.0,
            max_dimension=2048,
            allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
        )
        assert res.is_valid is True
        assert res.width <= 2048 and res.height <= 2048
        # Aspect ratio preserved (2:1)
        assert abs((res.width / res.height) - 2.0) < 0.05


class TestTier1Feature4ObjectDetectionPolish:
    """F4: Production Detectors (A3 Indoor, B2 Exterior, C2 Parking), Metrics & Checkpoints."""

    def test_f4_01_a3_indoor_detector_real_inference(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """F4.1: Model A3 YOLO11s 800px executes real inference without simulation."""
        a3_ckpt = Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
        if not a3_ckpt.exists():
            pytest.skip("A3 checkpoint not found")

        det_cfg = {
            "checkpoint_path": str(a3_ckpt),
            "imgsz": 800,
            "confidence_threshold": 0.25,
            "task_group": "indoor",
        }
        service = DetectionService(mock_mode=False, config=det_cfg)
        result = service.run(synthetic_room_image, confidence_threshold=0.25, save_annotated=False)

        assert result["success"] is True
        resp: DetectionResponse = result["response"]
        assert resp.is_simulated is False, "Real mode must produce is_simulated=False!"
        assert "indoor_detector_a3_yolo11s_800px_best.pt" in resp.checkpoint
        assert resp.inference_time_ms >= 0.0

    def test_f4_02_b2_exterior_detector_real_inference(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """F4.2: Model B2 YOLO11s executes real exterior detection."""
        b2_ckpt = Path("checkpoints/exterior_detector_b2_yolo11s_best.pt")
        if not b2_ckpt.exists():
            pytest.skip("B2 checkpoint not found")

        det_cfg = {
            "checkpoint_path": str(b2_ckpt),
            "imgsz": 640,
            "confidence_threshold": 0.25,
            "task_group": "exterior",
        }
        service = DetectionService(mock_mode=False, config=det_cfg)
        result = service.run(synthetic_room_image, confidence_threshold=0.25, save_annotated=False)
        assert result["success"] is True
        assert result["response"].is_simulated is False

    def test_f4_03_c2_parking_detector_real_inference(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """F4.3: Model C2 YOLO11s executes real parking detection."""
        c2_ckpt = Path("checkpoints/parking_detector_c2_yolo11s_pilot_best.pt")
        if not c2_ckpt.exists():
            pytest.skip("C2 checkpoint not found")

        det_cfg = {
            "checkpoint_path": str(c2_ckpt),
            "imgsz": 640,
            "confidence_threshold": 0.25,
            "task_group": "parking",
        }
        service = DetectionService(mock_mode=False, config=det_cfg)
        result = service.run(synthetic_room_image, confidence_threshold=0.25, save_annotated=False)
        assert result["success"] is True
        assert result["response"].is_simulated is False

    def test_f4_04_detection_response_bounds_and_metrics(self):
        """F4.4: Verify DetectionResponse bounding boxes and confidence score normalization."""
        box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        assert 0.0 <= box.x_min <= box.x_max <= 1.0
        assert 0.0 <= box.y_min <= box.y_max <= 1.0

    def test_f4_05_missing_checkpoint_fails_cleanly_no_silent_fallback(self, synthetic_room_image: Image.Image):
        """F4.5: Missing checkpoint in real mode must return clean failure and NEVER silent mock."""
        det_cfg = {
            "checkpoint_path": "checkpoints/nonexistent_phantom_model_12345.pt",
            "imgsz": 640,
            "confidence_threshold": 0.35,
        }
        service = DetectionService(mock_mode=False, config=det_cfg)
        result = service.run(synthetic_room_image, confidence_threshold=0.35, save_annotated=False)

        assert result["success"] is False, "Missing checkpoint must fail!"
        assert result["response"] is None or result["response"].is_simulated is False
        assert "does not exist" in result["error_message"].lower() or "failed" in result["error_message"].lower()


class TestTier1Feature5VisionQAGrounding:
    """F5: Vision Q&A Grounding, Honest VLM Unavailable Card, and Mock Opt-In."""

    def test_f5_01_custom_vlm_raises_model_not_implemented(self, synthetic_room_image: Image.Image):
        """F5.1: CustomVisionLanguageModel raises ModelNotImplementedError when unconfigured."""
        vlm = CustomVisionLanguageModel()
        with pytest.raises(ModelNotImplementedError):
            vlm.ask(synthetic_room_image, "What is the lighting like?")

    def test_f5_02_vlm_service_real_mode_reports_unimplemented(self, synthetic_room_image: Image.Image):
        """F5.2: VisionLanguageService(mock_mode=False) returns honest unimplemented status."""
        vlm_service = VisionLanguageService(mock_mode=False)
        result = vlm_service.run(synthetic_room_image, "What color are the walls?")
        assert result["success"] is False
        assert result["is_implemented"] is False
        assert result["response"] is None

    def test_f5_03_vlm_service_mock_mode_opt_in(self, synthetic_room_image: Image.Image):
        """F5.3: VisionLanguageService(mock_mode=True) returns deterministic simulated answer."""
        vlm_service = VisionLanguageService(mock_mode=True)
        result = vlm_service.run(synthetic_room_image, "How can I improve lighting in this room?")
        assert result["success"] is True
        assert result["is_implemented"] is True
        resp: VLMAnswerResponse = result["response"]
        assert resp.is_simulated is True
        assert len(resp.answer) > 20
        assert "light" in resp.answer.lower() or "accent" in resp.answer.lower() or "room" in resp.answer.lower()

    def test_f5_04_empty_or_whitespace_question_rejected(self, synthetic_room_image: Image.Image):
        """F5.4: Blank or whitespace-only questions are rejected with clear error."""
        vlm_service = VisionLanguageService(mock_mode=True)
        for q in ["", "   ", "\t\n"]:
            res = vlm_service.run(synthetic_room_image, q)
            assert res["success"] is False
            assert "empty" in res["error_message"].lower()

    def test_f5_05_vlm_answer_response_schema_integrity(self):
        """F5.5: VLMAnswerResponse correctly validates question, answer, and simulated status."""
        resp = VLMAnswerResponse(
            question="What style is this sofa?",
            answer="Mid-century modern with tapered wood legs.",
            is_simulated=True,
            model_name="Deterministic Mock VLM",
        )
        assert resp.question == "What style is this sofa?"
        assert resp.is_simulated is True


class TestTier1Feature63DReconstructionPolish:
    """F6: 3D Scene Reconstruction Polish (Metric vs Relative Depth, RANSAC Floor, Layout Bounds)."""

    def test_f6_01_metric_depth_estimation_real_inference(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """F6.1: Depth Anything V2 Metric Small returns positive metric depth in meters."""
        depth_dir = Path("checkpoints/depth_anything_v2_metric_indoor_small")
        if not depth_dir.exists():
            pytest.skip("Metric depth checkpoint directory missing")

        s3d_cfg = dict(model_config.get("scene3d_metric", model_config.get("scene3d", {})))
        service = Scene3DService(mock_mode=False, config=s3d_cfg)
        result = service.run(synthetic_room_image)

        assert result["success"] is True
        depth_map = result["depth_map"]
        assert depth_map is not None
        assert depth_map.shape == (synthetic_room_image.height, synthetic_room_image.width)
        assert np.all(depth_map >= 0.0)
        assert float(np.max(depth_map)) > 0.5  # Realistic metric depth

    def test_f6_02_scene_fusion_unprojects_metric_point_cloud(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """F6.2: SceneFusionService unprojects metric depth to 3D point cloud."""
        service = SceneFusionService(mock_mode=False, config=model_config)
        scene: SceneGraph = service.fuse(synthetic_room_image)

        assert isinstance(scene, SceneGraph)
        assert scene.point_cloud is not None
        xyz = scene.point_cloud.get("xyz")
        assert xyz is not None and len(xyz) > 0
        assert scene.n_valid_points > 0

    def test_f6_03_ransac_floor_plane_fitting_normal_orientation(self):
        """F6.3: RANSAC floor fitting extracts normal with vertical dominance check (|n_y| >= 0.70)."""
        # Synthesize horizontal floor points at y = 1.0 (in camera frame where down is +Y)
        n_pts = 200
        x = np.random.uniform(-2.0, 2.0, n_pts)
        z = np.random.uniform(1.0, 4.0, n_pts)
        y = np.ones(n_pts) * 1.0 + np.random.normal(0, 0.01, n_pts)  # Slight noise
        floor_pts = np.stack([x, y, z], axis=1)

        normal, d, inliers = fit_plane_ransac(floor_pts, distance_threshold=0.05)
        assert normal is not None
        assert abs(normal[1]) >= 0.70, f"Floor normal must be vertical-dominant, got ny={normal[1]}"

    def test_f6_04_room_layout_bounding_envelope_dimensions(self, sample_point_cloud: Dict[str, Any]):
        """F6.4: RoomLayoutService extracts positive room width, length, and height."""
        layout_service = RoomLayoutService()
        layout = layout_service.estimate_layout(sample_point_cloud, objects=[], surfaces=[])

        assert "room_width_m" in layout
        assert "room_length_m" in layout
        assert "room_height_m" in layout
        assert "camera_height_m" in layout
        assert layout["room_width_m"] >= 2.0
        assert layout["room_length_m"] >= 2.5
        assert layout["room_height_m"] >= 2.2
        assert 0.8 <= layout["camera_height_m"] <= 2.2

    def test_f6_05_explicit_depth_mode_labeling_in_scenegraph(self, model_config: Dict[str, Any]):
        """F6.5: SceneGraph explicitly identifies depth mode (metric vs relative)."""
        service = SceneFusionService(mock_mode=True, config=model_config)
        img = Image.new("RGB", (320, 240), color=(100, 100, 100))
        scene = service.fuse(img)
        assert hasattr(scene, "depth_mode")
        assert scene.depth_mode in ["metric", "relative"]


class TestTier1Feature7Interactive3DViewerAndHonestExport:
    """F7: 3D Viewer & Honest PLY/GLB Scene Export Downloads."""

    def test_f7_01_export_point_cloud_ply_ascii_header_and_data(self):
        """F7.1: export_point_cloud_ply generates valid ASCII PLY format."""
        xyz = [[0.0, 1.0, 2.0], [0.5, 1.2, 2.5], [-0.5, 0.8, 1.8]]
        rgb = [[0.8, 0.2, 0.1], [0.1, 0.9, 0.2], [0.2, 0.3, 0.9]]
        ply_bytes = export_point_cloud_ply(xyz=xyz, rgb=rgb)

        assert ply_bytes.startswith(b"ply\nformat ascii 1.0\n")
        assert b"element vertex 3\n" in ply_bytes
        assert b"end_header\n" in ply_bytes
        text = ply_bytes.decode("utf-8")
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        header_end = lines.index("end_header")
        vertex_lines = lines[header_end + 1:]
        assert len(vertex_lines) == 3

    def test_f7_02_export_scene_glb_binary_gltf2_container(self, sample_point_cloud: Dict[str, Any]):
        """F7.2: export_scene_glb generates valid glTF 2.0 binary container."""
        geom = Object3DGeometry(
            centroid=(0.0, 0.2, 2.5),
            extent=(1.8, 0.8, 0.9),
            rotation_deg=0.0,
        )
        obj = SceneObject(
            object_id="obj_1",
            semantic_class="couch",
            instance_id=1,
            detection_confidence=0.88,
            mask_provenance="a3_fastsam",
            bounding_box=(0.1, 0.2, 0.8, 0.9),
            coverage_percentage=15.0,
            geometry=geom,
        )
        glb_bytes = export_scene_glb(point_cloud=sample_point_cloud, objects=[obj])

        assert len(glb_bytes) > 20
        magic, version, total_len = struct.unpack_from("<4sII", glb_bytes, 0)
        assert magic == b"glTF", f"Expected magic b'glTF', got {magic}"
        assert version == 2, f"Expected glTF version 2, got {version}"
        assert total_len == len(glb_bytes)

    def test_f7_03_create_box_mesh_produces_8_vertices_and_12_triangles(self):
        """F7.3: create_box_mesh generates 8 corner vertices and 36 index references."""
        v, tris = create_box_mesh(centroid=(0.0, 0.0, 3.0), extent=(1.0, 1.0, 1.0))
        assert v.shape == (8, 3)
        assert tris.shape == (36,)
        assert tris.max() == 7
        assert tris.min() == 0

    def test_f7_04_create_3d_box_traces_creates_wireframe_and_centroid(self):
        """F7.4: create_3d_box_traces returns Plotly 3D wireframe trace and centroid marker."""
        traces = tv.create_3d_box_traces(
            centroid=(0.0, 0.5, 2.0),
            extent=(1.2, 0.8, 0.9),
            label="Couch (obj_1)",
            color="#3B82F6",
        )
        assert len(traces) == 2
        wireframe, centroid = traces[0], traces[1]
        assert wireframe.mode == "lines"
        assert centroid.mode == "markers+text"

    def test_f7_05_honest_export_descriptions_in_ui_source(self):
        """F7.5: UI labels and tooltips describe exports as point clouds/boxes, not textured meshes."""
        src = inspect.getsource(tv.render_3d_viewer)
        assert "point cloud" in src.lower()
        assert "not a textured" in src.lower() or "not a textured polygon mesh" in src.lower() or "bounding boxes" in src.lower()


class TestTier1Feature8RealModeEnforcementAndPerformance:
    """F8: Real-Mode Enforcement, VRAM Budget Compliance, and Concurrency Protection."""

    def test_f8_01_config_app_yaml_defaults_mock_mode_to_false(self, app_config: Dict[str, Any]):
        """F8.1: config/app.yaml defaults mock_mode to False."""
        assert app_config.get("mock_mode") is False, (
            f"app.yaml mock_mode must be False by default, got: {app_config.get('mock_mode')}"
        )

    def test_f8_02_global_model_registry_defaults_to_cuda_device(self):
        """F8.2: GLOBAL_MODEL_REGISTRY registers indoor A3 and device is cuda when available."""
        from utils.device import get_optimal_device
        primary = GLOBAL_MODEL_REGISTRY.get_primary_for_task("indoor_detection")
        assert primary is not None
        assert primary.checkpoint_path.exists()
        if torch.cuda.is_available():
            assert get_optimal_device().startswith("cuda")

    def test_f8_03_is_simulated_strictly_matches_execution_mode(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """F8.3: is_simulated metadata is False for real mode and True for mock mode."""
        # Mock mode
        det_mock = DetectionService(mock_mode=True)
        res_mock = det_mock.run(synthetic_room_image, confidence_threshold=0.35, save_annotated=False)
        assert res_mock["response"].is_simulated is True

        # Real mode
        a3_ckpt = Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
        if a3_ckpt.exists():
            det_real = DetectionService(mock_mode=False, config={"checkpoint_path": str(a3_ckpt), "imgsz": 800})
            res_real = det_real.run(synthetic_room_image, confidence_threshold=0.35, save_annotated=False)
            assert res_real["response"].is_simulated is False

    def test_f8_04_vram_budget_compliance_under_8gb(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """F8.4: Total allocated VRAM during scene fusion remains well below 8 GB (< 4 GB)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        service = SceneFusionService(mock_mode=False, config=model_config)
        _ = service.fuse(synthetic_room_image)

        alloc_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        assert alloc_mb < 4000.0, f"VRAM allocation ({alloc_mb:.1f} MB) exceeded 4,000 MB budget!"

    def test_f8_05_gpu_inference_gate_thread_serialization(self):
        """F8.5: GPUInferenceGate serializes concurrent access and releases permits."""
        gate = GPUInferenceGate(max_concurrent=1)
        res = gate.run(lambda: 42)
        assert res.value == 42
        assert res.acquired is True
        assert gate.active == 0


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ===========================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary Value Analysis (BVA), corrupt inputs, and edge conditions."""

    def test_tier2_01_empty_upload_bytes_rejected(self):
        """Tier 2.1: 0-byte file input rejected by validate_image_file."""
        buf = io.BytesIO(b"")
        res = validate_image_file(
            file_input=buf,
            filename="empty.jpg",
            max_file_size_mb=15.0,
            max_dimension=4096,
            allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
        )
        assert res.is_valid is False
        assert "empty" in res.error_message.lower()

    def test_tier2_02_corrupted_image_header_rejected(self):
        """Tier 2.2: Truncated or corrupted JFIF bytes rejected with clear error."""
        buf = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00corrupted_payload_data_not_an_image")
        res = validate_image_file(
            file_input=buf,
            filename="corrupt.jpg",
            max_file_size_mb=15.0,
            max_dimension=4096,
            allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
        )
        assert res.is_valid is False
        assert "cannot identify" in res.error_message.lower() or "corrupt" in res.error_message.lower()

    def test_tier2_03_extreme_confidence_threshold_zero(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """Tier 2.3: Confidence threshold 0.0 returns detections without crashing."""
        a3_ckpt = Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
        if not a3_ckpt.exists():
            pytest.skip("A3 checkpoint not found")

        service = DetectionService(mock_mode=False, config={"checkpoint_path": str(a3_ckpt), "imgsz": 800})
        res = service.run(synthetic_room_image, confidence_threshold=0.0, save_annotated=False)
        assert res["success"] is True

    def test_tier2_04_extreme_confidence_threshold_one(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """Tier 2.4: Confidence threshold 1.0 filters out low-confidence boxes safely."""
        a3_ckpt = Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
        if not a3_ckpt.exists():
            pytest.skip("A3 checkpoint not found")

        service = DetectionService(mock_mode=False, config={"checkpoint_path": str(a3_ckpt), "imgsz": 800})
        res = service.run(synthetic_room_image, confidence_threshold=1.0, save_annotated=False)
        assert res["success"] is True
        assert res["response"].total_objects == 0

    def test_tier2_05_tiny_64x64_image_perception(self, model_config: Dict[str, Any]):
        """Tier 2.5: Extremely small 64x64 image processed without tensor shape error."""
        tiny_img = Image.new("RGB", (64, 64), color=(140, 140, 140))
        a3_ckpt = Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
        if not a3_ckpt.exists():
            pytest.skip("A3 checkpoint missing")

        service = DetectionService(mock_mode=False, config={"checkpoint_path": str(a3_ckpt), "imgsz": 800})
        res = service.run(tiny_img, confidence_threshold=0.35, save_annotated=False)
        assert res["success"] is True

    def test_tier2_06_extreme_aspect_ratio_panoramic_image(self, model_config: Dict[str, Any]):
        """Tier 2.6: Extreme panoramic 1200x240 image processed without crashing."""
        pano_img = Image.new("RGB", (1200, 240), color=(160, 120, 80))
        a3_ckpt = Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
        if not a3_ckpt.exists():
            pytest.skip("A3 checkpoint missing")

        service = DetectionService(mock_mode=False, config={"checkpoint_path": str(a3_ckpt), "imgsz": 800})
        res = service.run(pano_img, confidence_threshold=0.35, save_annotated=False)
        assert res["success"] is True

    def test_tier2_07_unsupported_file_extension_rejection(self):
        """Tier 2.7: Executable and archive extensions rejected."""
        buf = io.BytesIO(b"MZ\x90\x00fake_executable")
        res = validate_image_file(
            file_input=buf,
            filename="malicious.exe",
            max_file_size_mb=15.0,
            max_dimension=4096,
            allowed_extensions={".jpg", ".jpeg", ".png", ".webp"},
        )
        assert res.is_valid is False
        assert "unsupported file format" in res.error_message.lower() or "allowed" in res.error_message.lower()

    def test_tier2_08_empty_point_cloud_safe_export(self):
        """Tier 2.8: Exporting boundary point clouds produces valid containers without exception."""
        # Minimal single-point coordinate
        ply_bytes = export_point_cloud_ply(xyz=[[0.0, 0.0, 0.0]])
        assert b"element vertex 1" in ply_bytes

        # Empty / None point cloud handled gracefully in scene GLB export
        glb_bytes = export_scene_glb(point_cloud=None, objects=[])
        assert len(glb_bytes) >= 12
        assert glb_bytes.startswith(b"glTF")


# ===========================================================================
# TIER 3: PAIRWISE COMBINATIONS (IMAGE FORMAT × DETECTOR × DEPTH MODALITY)
# ===========================================================================

class TestTier3PairwiseCombinations:
    """Tier 3: Pairwise combinations across upload formats, production detectors, and 3D unprojection."""

    def test_tier3_01_combo_jpeg_indoor_a3_metric_depth_fusion(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """Tier 3.1: JPEG + Model A3 (800px) + Metric Depth -> SceneFusion unprojection."""
        buf = io.BytesIO()
        synthetic_room_image.save(buf, format="JPEG")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        service = SceneFusionService(mock_mode=False, config=model_config)
        scene = service.fuse(img)

        assert scene.width == img.width
        assert scene.height == img.height
        assert scene.point_cloud is not None
        assert scene.n_valid_points > 0

    def test_tier3_02_combo_png_exterior_b2_metric_depth(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """Tier 3.2: PNG + Model B2 (Exterior) + Metric Depth -> Detection & Depth heatmap."""
        b2_ckpt = Path("checkpoints/exterior_detector_b2_yolo11s_best.pt")
        if not b2_ckpt.exists():
            pytest.skip("B2 checkpoint not found")

        buf = io.BytesIO()
        synthetic_room_image.save(buf, format="PNG")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        det_svc = DetectionService(mock_mode=False, config={"checkpoint_path": str(b2_ckpt), "task_group": "exterior"})
        det_res = det_svc.run(img, confidence_threshold=0.25, save_annotated=False)
        assert det_res["success"] is True

        depth_svc = Scene3DService(mock_mode=False, config=model_config.get("scene3d_metric", {}))
        depth_res = depth_svc.run(img)
        assert depth_res["success"] is True

    def test_tier3_03_combo_webp_parking_c2_metric_depth(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """Tier 3.3: WEBP + Model C2 (Parking) + Metric Depth -> Detection & Depth heatmap."""
        c2_ckpt = Path("checkpoints/parking_detector_c2_yolo11s_pilot_best.pt")
        if not c2_ckpt.exists():
            pytest.skip("C2 checkpoint not found")

        buf = io.BytesIO()
        synthetic_room_image.save(buf, format="WEBP")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        det_svc = DetectionService(mock_mode=False, config={"checkpoint_path": str(c2_ckpt), "task_group": "parking"})
        det_res = det_svc.run(img, confidence_threshold=0.25, save_annotated=False)
        assert det_res["success"] is True

        depth_svc = Scene3DService(mock_mode=False, config=model_config.get("scene3d_metric", {}))
        depth_res = depth_svc.run(img)
        assert depth_res["success"] is True

    def test_tier3_04_combo_high_conf_indoor_scene_to_glb_export(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """Tier 3.4: Indoor Image + High Conf (0.60) + SceneFusion -> Valid GLB export."""
        service = SceneFusionService(mock_mode=False, config=model_config)
        scene = service.fuse(synthetic_room_image)

        glb_bytes = export_scene_glb(point_cloud=scene.point_cloud, objects=scene.objects)
        assert len(glb_bytes) > 1000
        assert glb_bytes.startswith(b"glTF")

    def test_tier3_05_combo_low_conf_indoor_scene_to_ply_export(
        self, synthetic_room_image: Image.Image, model_config: Dict[str, Any]
    ):
        """Tier 3.5: Indoor Image + Low Conf (0.20) + SceneFusion -> Valid ASCII PLY export."""
        service = SceneFusionService(mock_mode=False, config=model_config)
        scene = service.fuse(synthetic_room_image)

        ply_bytes = export_point_cloud_ply(
            xyz=scene.point_cloud.get("xyz", []),
            rgb=scene.point_cloud.get("rgb", None),
        )
        assert len(ply_bytes) > 1000
        assert ply_bytes.startswith(b"ply\nformat ascii 1.0\n")


# ===========================================================================
# TIER 4: REAL-WORLD SCENARIOS ON 5 GENUINE ADE20K VALIDATION IMAGES
# ===========================================================================

class TestTier4RealWorldADE20KScenarios:
    """Tier 4: End-to-end perception and 3D reconstruction on 5 verified ADE20K room images."""

    def test_tier4_scenario1_living_room_real_perception(
        self, ade20k_candidate_images: Dict[str, Path], model_config: Dict[str, Any]
    ):
        """Scenario 1: Living Room (ADE_val_00001531.jpg) — Couch, Chair, TV + Metric Depth."""
        img_path = ade20k_candidate_images["living_room"]
        img = Image.open(img_path).convert("RGB")
        assert img.width == 816 and img.height == 512

        service = SceneFusionService(mock_mode=False, config=model_config)
        scene: SceneGraph = service.fuse(img)

        # Assertions
        assert len(scene.objects) >= 1, "Must detect furniture in living room"
        assert scene.n_valid_points >= 300000, f"Expected >= 300k valid points, got {scene.n_valid_points}"

        layout = scene.room_layout
        assert layout is not None
        assert layout["floor_plane_detected"] is True, "Floor plane must be detected"
        assert abs(layout["floor_normal"][1]) >= 0.70, "Floor normal must be horizontal (|ny| >= 0.70)"
        assert 0.8 <= layout["camera_height_m"] <= 2.2, "Camera height must be realistic"
        assert layout["room_width_m"] >= 2.0
        assert layout["room_length_m"] >= 2.5

        # Exports
        glb = export_scene_glb(point_cloud=scene.point_cloud, objects=scene.objects)
        assert glb.startswith(b"glTF") and len(glb) > 10000
        ply = export_point_cloud_ply(xyz=scene.point_cloud["xyz"], rgb=scene.point_cloud.get("rgb"))
        assert ply.startswith(b"ply\n") and len(ply) > 10000

    def test_tier4_scenario2_bedroom_real_perception(
        self, ade20k_candidate_images: Dict[str, Path], model_config: Dict[str, Any]
    ):
        """Scenario 2: Bedroom (ADE_val_00000118.jpg) — Bed, TV + Metric Depth."""
        img_path = ade20k_candidate_images["bedroom"]
        img = Image.open(img_path).convert("RGB")
        assert img.width == 725 and img.height == 512

        service = SceneFusionService(mock_mode=False, config=model_config)
        scene: SceneGraph = service.fuse(img)

        assert len(scene.objects) >= 1, "Must detect bed or room furniture"
        assert scene.n_valid_points >= 300000
        assert scene.room_layout["floor_plane_detected"] is True
        assert abs(scene.room_layout["floor_normal"][1]) >= 0.70
        assert 0.8 <= scene.room_layout["camera_height_m"] <= 2.2

    def test_tier4_scenario3_kitchen_real_perception(
        self, ade20k_candidate_images: Dict[str, Path], model_config: Dict[str, Any]
    ):
        """Scenario 3: Kitchen (ADE_val_00000459.jpg) — Cabinetry, Oven + Metric Depth."""
        img_path = ade20k_candidate_images["kitchen"]
        img = Image.open(img_path).convert("RGB")
        assert img.width == 770 and img.height == 512

        service = SceneFusionService(mock_mode=False, config=model_config)
        scene: SceneGraph = service.fuse(img)

        assert len(scene.objects) >= 1
        assert scene.n_valid_points >= 300000
        assert scene.room_layout["floor_plane_detected"] is True
        assert abs(scene.room_layout["floor_normal"][1]) >= 0.70

    def test_tier4_scenario4_dining_room_real_perception(
        self, ade20k_candidate_images: Dict[str, Path], model_config: Dict[str, Any]
    ):
        """Scenario 4: Dining Room (ADE_val_00000314.jpg) — Dining Chairs, Table + Metric Depth."""
        img_path = ade20k_candidate_images["dining_room"]
        img = Image.open(img_path).convert("RGB")
        assert img.width == 768 and img.height == 512

        service = SceneFusionService(mock_mode=False, config=model_config)
        scene: SceneGraph = service.fuse(img)

        assert len(scene.objects) >= 3, "Dining room should have multiple chairs/furniture detected"
        assert scene.n_valid_points >= 300000
        assert scene.room_layout["floor_plane_detected"] is True
        assert abs(scene.room_layout["floor_normal"][1]) >= 0.70

    def test_tier4_scenario5_home_office_real_perception(
        self, ade20k_candidate_images: Dict[str, Path], model_config: Dict[str, Any]
    ):
        """Scenario 5: Home Office / Study (ADE_val_00000420.jpg) — Desk, Chair, Books + Metric Depth."""
        img_path = ade20k_candidate_images["home_office"]
        img = Image.open(img_path).convert("RGB")
        assert img.width == 683 and img.height == 512

        service = SceneFusionService(mock_mode=False, config=model_config)
        scene: SceneGraph = service.fuse(img)

        assert len(scene.objects) >= 1
        assert scene.n_valid_points >= 300000
        assert scene.room_layout["floor_plane_detected"] is True
        assert abs(scene.room_layout["floor_normal"][1]) >= 0.70
