"""Tests for camera-aware, deterministic single-view reconstruction artifacts."""

import json
import queue
from pathlib import Path

import numpy as np
from PIL import Image

from schemas.scene import SceneGraph
from services.job_service import JobService
from services.job_store import JobStore
from utils.export_3d import (
    export_colored_point_cloud_glb,
    export_point_cloud_ply,
    extract_glb_point_cloud,
    validate_glb_bytes,
    validate_ply_bytes,
)
from utils.reconstruction_contract import (
    depth_to_xyz,
    normalize_intrinsics,
    transform_intrinsics,
    validate_reconstruction,
)


def test_known_intrinsics_unproject_and_pixel_reprojection():
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    k = {"fx": 2.0, "fy": 4.0, "cx": 0.5, "cy": 0.25}
    xyz, colors, source = depth_to_xyz(depth, rgb, k)
    np.testing.assert_allclose(xyz[0], [-0.5, -0.125, 2.0])
    np.testing.assert_allclose(colors[2], [0, 0, 1])
    assert source.tolist() == [0, 1, 2, 3]
    result = validate_reconstruction(rgb, depth, np.ones((2, 2), bool), xyz, colors, source, k, "metric_model_estimate")
    assert result["is_valid"]
    assert result["xyz_reprojection_max_error_px"] < 1e-6


def test_api_accepts_pixel_intrinsics_and_exposes_asset_acceptance():
    from api.schemas import JobStatusResponse, ReconstructionJobRequest

    request = ReconstructionJobRequest(
        upload_ids=["upl_example"],
        camera_intrinsics={"fx": 500.0, "fy": 510.0, "cx": 320.0, "cy": 240.0},
    )
    assert request.camera_intrinsics["fx"] == 500.0
    status = JobStatusResponse(
        job_id="job_example", status="COMPLETED", stage="done", progress=1.0,
        model_id="roomai_composite_v1", view_count=1, timing={},
        warnings=[], created_at=0.0, updated_at=0.0,
        asset_acceptance={"status": "REVIEW_REQUIRED", "reasons": ["scale review"]},
    )
    assert status.asset_acceptance["status"] == "REVIEW_REQUIRED"


def test_job_submission_preserves_original_bytes_and_lossless_decoded_rgb(tmp_path):
    original_path = tmp_path / "uploaded_source.jpg"
    Image.new("RGB", (17, 13), (17, 93, 201)).save(original_path, format="JPEG", quality=91)
    original_bytes = original_path.read_bytes()
    decoded = Image.open(original_path).convert("RGB")
    service = object.__new__(JobService)
    service.store = JobStore(tmp_path / "source_preservation.sqlite")
    service.jobs_dir = tmp_path / "jobs"
    service._queue = queue.Queue()
    job_id = service.submit_reconstruction_job(
        [decoded], options={"source_upload_paths": [str(original_path)]}
    )
    job_dir = service.jobs_dir / job_id
    source_path = job_dir / "source" / "view_00.jpg"
    decoded_path = job_dir / "rgb" / "view_00.png"
    inference_path = Path(service.store.get_job(job_id)["image_paths"][0])
    assert source_path.read_bytes() == original_bytes
    assert Image.open(decoded_path).convert("RGB").tobytes() == decoded.tobytes()
    assert inference_path.suffix == ".png"
    assert Image.open(inference_path).convert("RGB").tobytes() == decoded.tobytes()
    assert service.store.get_job(job_id)["options"]["source_artifacts"][0]["source_original_sha256"]


def test_invalid_depth_pixels_are_excluded_and_mask_shape_is_checked():
    depth = np.array([[1.0, np.nan], [0.0, -1.0]], dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    xyz, colors, source = depth_to_xyz(depth, rgb, {"fx": 2, "fy": 2, "cx": 0.5, "cy": 0.5})
    assert len(xyz) == len(colors) == 1
    assert source.tolist() == [0]
    try:
        depth_to_xyz(depth, rgb, {"fx": 2, "fy": 2, "cx": 0.5, "cy": 0.5}, np.ones((1, 4), bool))
    except ValueError as exc:
        assert "mask dimensions" in str(exc)
    else:
        raise AssertionError("misaligned mask must fail")


def test_intrinsics_estimation_and_resize_crop_pad_transform():
    estimated, source = normalize_intrinsics(640, 480)
    assert source.startswith("estimated_")
    assert estimated == {"fx": 640.0, "fy": 640.0, "cx": 319.5, "cy": 239.5}
    transformed = transform_intrinsics({"fx": 400, "fy": 300, "cx": 200, "cy": 100}, 0.5, 0.5, 10, 20, 4, 8)
    assert transformed == {"fx": 200.0, "fy": 150.0, "cx": 99.0, "cy": 48.0}


def test_sampling_is_deterministic_and_collapse_is_rejected():
    depth = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    k = {"fx": 10, "fy": 10, "cx": 4.5, "cy": 4.5}
    first = depth_to_xyz(depth, rgb, k, max_points=20, seed=13)
    second = depth_to_xyz(depth, rgb, k, max_points=20, seed=13)
    np.testing.assert_array_equal(first[2], second[2])
    flat_xyz = np.column_stack((np.zeros(10), np.zeros(10), np.arange(1, 11))).astype(np.float32)
    bad = validate_reconstruction(rgb, depth, np.ones_like(depth, bool), flat_xyz, np.zeros_like(flat_xyz), np.arange(10), k, "metric_model_estimate")
    assert bad["collapsed"]
    assert "collapsed_or_rank_deficient_geometry" in bad["errors"]


def test_ply_glb_color_parity_and_deterministic_subset():
    xyz = np.array([[x, y, 2.0 + x * 0.01] for x in range(20) for y in range(10)], dtype=np.float32)
    rgb = np.clip(xyz / 20.0, 0, 1)
    ply = export_point_cloud_ply(xyz.tolist(), rgb.tolist())
    glb_a = export_colored_point_cloud_glb(xyz, rgb, max_points=40, sampling_seed=7)
    glb_b = export_colored_point_cloud_glb(xyz, rgb, max_points=40, sampling_seed=7)
    assert glb_a == glb_b
    assert validate_ply_bytes(ply)["vertex_count"] == 200
    glb_result = validate_glb_bytes(glb_a)
    assert glb_result["is_valid"]
    exported_xyz, exported_rgb = extract_glb_point_cloud(glb_a)
    assert exported_xyz.shape == exported_rgb.shape == (40, 3)
    assert glb_result["json_data"]["meshes"][0]["primitives"][0]["attributes"] == {"POSITION": 0, "COLOR_0": 1}
    assert glb_result["json_data"]["extras"]["roomai"]["surface_mesh"] is False


def test_bundle_acceptance_is_separate_from_job_completion(tmp_path):
    service = object.__new__(JobService)
    image = Image.new("RGB", (16, 12), (80, 120, 160))
    depth = np.linspace(1, 3, 16 * 12, dtype=np.float32).reshape(12, 16)
    scene = SceneGraph(width=16, height=12, model_name="test metric model", depth_mode="metric", depth_map=depth,
                      point_cloud={}, n_total_points=192, n_valid_points=192)
    bundle = service._persist_and_validate_bundle(scene, [image], tmp_path, "roomai_composite_v1", {})
    assert bundle["acceptance"]["status"] == "REVIEW_REQUIRED"
    manifest = json.loads((tmp_path / "exhibit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["camera"]["intrinsics"]["calibrated"] is False
    assert manifest["depth"]["semantics"] == "metric_model_estimate"
    assert manifest["views"][0]["decoded_rgb_sha256"] == manifest["decoded_rgb"]["sha256"]
    assert (tmp_path / "depth_raw.npy").is_file()
    assert (tmp_path / "validity_mask.png").is_file()
    assert (tmp_path / "source" / "view_00.png").exists() is False


def test_relative_depth_is_rejected_not_relabelled_metric(tmp_path):
    service = object.__new__(JobService)
    image = Image.new("RGB", (8, 8), (120, 80, 40))
    scene = SceneGraph(width=8, height=8, model_name="relative depth model", depth_mode="relative",
                       depth_map=np.linspace(0.1, 0.9, 64, dtype=np.float32).reshape(8, 8),
                       point_cloud={"xyz": [[0, 0, 0.1], [0, 0, 0.2]], "rgb": [[0.5, 0.5, 0.5]] * 2})
    bundle = service._persist_and_validate_bundle(scene, [image], tmp_path, "roomai_composite_v1", {})
    assert bundle["acceptance"]["status"] == "REJECTED"
    manifest = json.loads((tmp_path / "exhibit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["depth"]["units"] is None


def test_job_can_complete_while_asset_remains_review_required(tmp_path, monkeypatch):
    from services import job_service as job_service_module

    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 12), (90, 120, 150)).save(image_path)
    store = JobStore(tmp_path / "jobs.sqlite")
    store.create_job("job_test_review", "roomai_composite_v1", [str(image_path)], {})

    class FakeAdapter:
        def is_available(self):
            return True, None

        def validate_inputs(self, images):
            return True, None

        def reconstruct(self, images, options=None, progress_callback=None):
            return SceneGraph(
                width=16, height=12, model_name="test metric model", depth_mode="metric",
                depth_map=np.linspace(1, 3, 16 * 12, dtype=np.float32).reshape(12, 16),
                point_cloud={}, n_total_points=192, n_valid_points=192,
            )

    monkeypatch.setattr(job_service_module.RECONSTRUCTION_REGISTRY, "get_adapter", lambda model_id: FakeAdapter())
    service = object.__new__(JobService)
    service.store = store
    service.jobs_dir = tmp_path / "job_artifacts"
    service._cancelled_jobs = set()
    job_dir = service.jobs_dir / "job_test_review"
    job_dir.mkdir(parents=True)
    store.update_job("job_test_review", status="QUEUED")
    service._execute_job("job_test_review", [str(image_path)], "roomai_composite_v1", {})
    job = store.get_job("job_test_review")
    assert job["status"] == "COMPLETED"
    assert job["results_summary"]["asset_acceptance"]["status"] == "REVIEW_REQUIRED"
    assert service.get_export_file("job_test_review", "glb")[0].is_file()


def test_hunyuan_native_outputs_are_preserved_but_rejected(tmp_path, monkeypatch):
    from services import job_service as job_service_module

    class AdapterWithoutCheckpoint:
        checkpoint_path = tmp_path / "not_present.ckpt"

    monkeypatch.setattr(job_service_module.RECONSTRUCTION_REGISTRY, "get_adapter", lambda model_id: AdapterWithoutCheckpoint())
    service = object.__new__(JobService)
    image = Image.new("RGB", (8, 8), (120, 90, 60))
    scene = SceneGraph(
        width=8, height=8, model_name="Hunyuan research", depth_mode="metric",
        depth_map=np.ones((8, 8), dtype=np.float32),
        point_cloud={"xyz": [[0, 0, 0.1], [0.001, 0.001, 0.101], [0.002, 0.002, 0.102]],
                     "rgb": [[0.5, 0.5, 0.5]] * 3},
        model_native_outputs={"points": np.ones((1, 8, 8, 3), dtype=np.float32),
                              "confidence": np.full((1, 8, 8), 0.8, dtype=np.float32)},
    )
    bundle = service._persist_and_validate_bundle(scene, [image], tmp_path, "hunyuan_world_mirror_1b", {})
    assert bundle["acceptance"]["status"] == "REJECTED"
    assert (tmp_path / "model_native_outputs.npz").is_file()
    manifest = json.loads((tmp_path / "exhibit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["depth"]["semantics"] == "scene_normalized"
    assert manifest["depth"]["units"] is None
    assert manifest["model_provenance"]["license_status"].startswith("review_required")
    class RejectedStore:
        @staticmethod
        def get_job(job_id):
            return {"status": "COMPLETED", "results_summary": {"asset_acceptance": {"status": "REJECTED"}}}

    service.store = RejectedStore()
    service.jobs_dir = tmp_path
    assert service.get_export_file("rejected", "glb") == (None, None)


def test_worker_inference_failure_remains_failed_without_accepted_assets(tmp_path, monkeypatch):
    from services import job_service as job_service_module

    image_path = tmp_path / "failure_input.png"
    Image.new("RGB", (8, 8)).save(image_path)
    store = JobStore(tmp_path / "failure.sqlite")
    store.create_job("job_test_failure", "roomai_composite_v1", [str(image_path)], {})

    class FailingAdapter:
        def is_available(self):
            return True, None

        def validate_inputs(self, images):
            return True, None

        def reconstruct(self, images, options=None, progress_callback=None):
            raise RuntimeError("controlled failure")

    monkeypatch.setattr(job_service_module.RECONSTRUCTION_REGISTRY, "get_adapter", lambda model_id: FailingAdapter())
    service = object.__new__(JobService)
    service.store = store
    service.jobs_dir = tmp_path / "failure_artifacts"
    service._cancelled_jobs = set()
    service._execute_job("job_test_failure", [str(image_path)], "roomai_composite_v1", {})
    job = store.get_job("job_test_failure")
    assert job["status"] == "FAILED"
    assert job["error"]["message"] == "controlled failure"
    assert job["results_summary"] is None
