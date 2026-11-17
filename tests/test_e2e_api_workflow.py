"""End-to-End Integration Test for RoomAI API Reconstruction Workflow."""

from pathlib import Path
import time
from fastapi.testclient import TestClient
import pytest

from api.main import app
from utils.export_3d import validate_glb_bytes, validate_ply_bytes

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_e2e_reconstruction_workflow_real():
    """Validates complete upload -> async job -> polling -> results -> download lifecycle on real models."""
    sample_img_path = REPO_ROOT / "artifacts" / "sample_room.jpg"
    if not sample_img_path.exists():
        pytest.skip(f"Reference test image {sample_img_path} not found.")

    # 1. Upload
    with open(sample_img_path, "rb") as f:
        files = {"file": ("sample_room.jpg", f, "image/jpeg")}
        resp_upload = client.post("/api/v1/uploads", files=files)

    assert resp_upload.status_code == 201, f"Upload failed: {resp_upload.text}"
    upload_data = resp_upload.json()
    upload_id = upload_data["upload_id"]
    assert upload_id.startswith("upl_")
    assert upload_data["width"] > 0
    assert upload_data["height"] > 0

    # 2. Submit Reconstruction Job
    job_req = {
        "upload_ids": [upload_id],
        "model_id": "roomai_composite_v1",
        "refine_depth": True,
        "downsample_points": 40000,
    }
    resp_job = client.post("/api/v1/jobs/reconstruct", json=job_req)
    assert resp_job.status_code == 202, f"Job submission failed: {resp_job.text}"
    job_data = resp_job.json()
    job_id = job_data["job_id"]
    assert job_id.startswith("job_")
    assert job_data["job_status"] == "QUEUED"

    # 3. Poll for completion (up to 120 seconds to allow for initial model weight loading)
    max_wait_sec = 120.0
    poll_interval = 1.0
    start_time = time.time()
    final_status = None

    while (time.time() - start_time) < max_wait_sec:
        resp_status = client.get(f"/api/v1/jobs/{job_id}")
        assert resp_status.status_code == 200
        st = resp_status.json()
        final_status = st["status"]
        if final_status in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll_interval)

    assert final_status == "COMPLETED", (
        f"Job {job_id} did not complete successfully (ended in {final_status}). "
        f"Error: {st.get('error')}"
    )
    assert st["progress"] == 1.0

    # 4. Fetch Results
    resp_results = client.get(f"/api/v1/jobs/{job_id}/results")
    assert resp_results.status_code == 200
    res_data = resp_results.json()
    assert res_data["is_completed"] is True
    summary = res_data["results_summary"]
    assert summary is not None
    assert summary["valid_points"] > 0
    assert "depth_range" in summary or "depth_range_m" in summary

    # 5. Download and Validate Binary GLB
    resp_glb = client.get(f"/api/v1/jobs/{job_id}/exports/glb")
    assert resp_glb.status_code == 200
    assert resp_glb.headers["content-type"] == "model/gltf-binary"
    glb_val = validate_glb_bytes(resp_glb.content)
    assert glb_val["is_valid"] is True, f"GLB validation failed: {glb_val['error']}"
    assert glb_val["mesh_count"] > 0

    # 6. Download and Validate ASCII PLY
    resp_ply = client.get(f"/api/v1/jobs/{job_id}/exports/ply")
    assert resp_ply.status_code == 200
    ply_val = validate_ply_bytes(resp_ply.content)
    assert ply_val["is_valid"] is True, f"PLY validation failed: {ply_val['error']}"
    assert ply_val["vertex_count"] > 0


def test_e2e_reconstruction_workflow_real_hunyuan():
    """Validates complete upload -> async job -> polling -> results -> download lifecycle using genuine HunyuanWorld-Mirror 1.0 foundation model without mock mode."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for real Hunyuan model execution")

    # Find authentic test image
    test_img_path = REPO_ROOT / "research/3d_reconstruction/data/indoor_verified_dataset/Bright_Room/frame_0000_rgb.png"
    if not test_img_path.exists():
        test_img_path = REPO_ROOT / "artifacts/sample_room.jpg"
    if not test_img_path.exists():
        pytest.skip("No authentic reference image found for real Hunyuan API E2E test.")

    # 1. Upload Real Image
    with open(test_img_path, "rb") as f:
        files = {"file": ("hunyuan_test_room.png", f, "image/png")}
        resp_upload = client.post("/api/v1/uploads", files=files)

    assert resp_upload.status_code == 201, f"Upload failed: {resp_upload.text}"
    upload_data = resp_upload.json()
    upload_id = upload_data["upload_id"]
    assert upload_id.startswith("upl_")

    # 2. Submit Reconstruction Job with model_id='hunyuan_world_mirror_1b'
    job_req = {
        "upload_ids": [upload_id],
        "model_id": "hunyuan_world_mirror_1b",
        "refine_depth": False,
        "downsample_points": 20000,
    }
    resp_job = client.post("/api/v1/jobs/reconstruct", json=job_req)
    assert resp_job.status_code == 202, f"Hunyuan Job submission failed: {resp_job.text}"
    job_data = resp_job.json()
    job_id = job_data["job_id"]
    assert job_id.startswith("job_")
    assert job_data["model_id"] == "hunyuan_world_mirror_1b"

    # 3. Poll for completion (up to 180 seconds for 1.22B parameter neural model on single GPU)
    max_wait_sec = 180.0
    poll_interval = 2.0
    start_time = time.time()
    final_status = None
    st = {}

    while (time.time() - start_time) < max_wait_sec:
        resp_status = client.get(f"/api/v1/jobs/{job_id}")
        assert resp_status.status_code == 200
        st = resp_status.json()
        final_status = st["status"]
        if final_status in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll_interval)

    assert final_status == "COMPLETED", (
        f"Hunyuan Job {job_id} did not complete successfully (ended in {final_status}). "
        f"Error: {st.get('error')}"
    )
    assert st["progress"] == 1.0

    # 4. Fetch Results
    resp_results = client.get(f"/api/v1/jobs/{job_id}/results")
    assert resp_results.status_code == 200
    res_data = resp_results.json()
    assert res_data["is_completed"] is True
    assert res_data["model_id"] == "hunyuan_world_mirror_1b"

    # Verify scene graph has 3D point cloud
    scene_graph = res_data.get("scene_graph")
    assert scene_graph is not None
    assert scene_graph["model_name"] == "Tencent HunyuanWorld-Mirror 1.0"
    assert res_data["results_summary"]["valid_points"] > 0
    assert len(res_data["artifacts"]["glb"]["path"]) > 0
    assert len(res_data["artifacts"]["ply"]["path"]) > 0

    # 5. Download and Validate Binary GLB
    resp_glb = client.get(f"/api/v1/jobs/{job_id}/exports/glb")
    assert resp_glb.status_code == 200
    assert resp_glb.headers["content-type"] == "model/gltf-binary"
    assert resp_glb.content[:4] == b"glTF", "GLB binary must begin with glTF magic"
    glb_val = validate_glb_bytes(resp_glb.content)
    assert glb_val["is_valid"] is True, f"GLB validation failed: {glb_val['error']}"

    # 6. Download and Validate ASCII PLY
    resp_ply = client.get(f"/api/v1/jobs/{job_id}/exports/ply")
    assert resp_ply.status_code == 200
    assert resp_ply.content[:3] == b"ply", "PLY binary must begin with ply magic"
    ply_val = validate_ply_bytes(resp_ply.content)
    assert ply_val["is_valid"] is True, f"PLY validation failed: {ply_val['error']}"
    assert ply_val["vertex_count"] > 0


