"""HTTP route coverage for the reconstruction workflow without model inference."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.main import app
from services.upload_service import UploadService
from services.job_service import JobService
from services.job_store import JobStore


def encoded_png() -> bytes:
    image = Image.new("RGB", (24, 16), (30, 80, 120))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def install_http_fakes(monkeypatch, tmp_path: Path):
    import api.routers.jobs as jobs
    import api.routers.uploads as uploads

    upload_service = UploadService(tmp_path / "uploads")
    monkeypatch.setattr(uploads, "GLOBAL_UPLOAD_SERVICE", upload_service)
    monkeypatch.setattr(jobs, "GLOBAL_UPLOAD_SERVICE", upload_service)

    class Adapter:
        def get_capabilities(self):
            return SimpleNamespace(min_views=1, max_views=2)

        def is_available(self):
            return True, None

    class Registry:
        def get_default_model_id(self):
            return "roomai_composite_v1"

        def get_adapter(self, model_id):
            if model_id != "roomai_composite_v1":
                raise KeyError(model_id)
            return Adapter()

    monkeypatch.setattr(jobs, "RECONSTRUCTION_REGISTRY", Registry())

    class JobServiceFake:
        def __init__(self):
            self.submitted = None
            self.state = "QUEUED"
            self.submit_error = None
            self.export_path = tmp_path / "valid-test.glb"
            self.export_path.write_bytes(b"glTF-test-artifact")

        def submit_reconstruction_job(self, images, model_id=None, options=None):
            if self.submit_error:
                raise RuntimeError(self.submit_error)
            self.submitted = (images, model_id, options)
            return "job_phase43_test"

        def get_job_status(self, job_id):
            if job_id != "job_phase43_test":
                return None
            failed = self.state == "FAILED"
            return {
                "job_id": job_id,
                "status": self.state,
                "stage": "Mocked API state; inference not run",
                "progress": 1.0 if failed else 0.0,
                "model_id": "roomai_composite_v1",
                "view_count": 1,
                "timing": {},
                "error": {"code": "RuntimeError", "message": "mocked inference failure"} if failed else None,
                "warnings": [],
                "asset_acceptance": None,
                "created_at": 1.0,
                "updated_at": 2.0,
            }

        def get_job_results(self, job_id):
            if job_id != "job_phase43_test":
                return None
            return {
                "job_id": job_id,
                "status": "COMPLETED",
                "is_completed": True,
                "model_id": "roomai_composite_v1",
                "timing": {},
                "results_summary": {"valid_points": 12},
                "artifacts": {},
                "scene_graph": {},
            }

        def get_export_file(self, job_id, fmt):
            if job_id == "job_phase43_test" and fmt == "glb":
                return self.export_path, "model/gltf-binary"
            return None, None

    service = JobServiceFake()
    monkeypatch.setattr(jobs, "GLOBAL_JOB_SERVICE", service)
    return service


def test_upload_to_job_status_results_and_artifact_http_flow(monkeypatch, tmp_path):
    service = install_http_fakes(monkeypatch, tmp_path)
    with TestClient(app) as client:
        upload = client.post(
            "/api/v1/uploads",
            files={"file": ("room.png", encoded_png(), "image/png")},
        )
        assert upload.status_code == 201
        upload_id = upload.json()["upload_id"]
        assert upload.json()["width"] == 24

        submitted = client.post(
            "/api/v1/jobs/reconstruct",
            json={
                "upload_ids": [upload_id],
                "model_id": "roomai_composite_v1",
                "refine_depth": False,
                "camera_intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 12.0, "cy": 8.0},
            },
        )
        assert submitted.status_code == 202
        assert submitted.json()["job_status"] == "QUEUED"
        assert service.submitted[1] == "roomai_composite_v1"
        assert service.submitted[2]["camera_intrinsics"]["fx"] == 20.0
        assert Path(service.submitted[2]["source_upload_paths"][0]).is_file()

        status = client.get("/api/v1/jobs/job_phase43_test")
        assert status.status_code == 200
        assert status.json()["status"] == "QUEUED"

        results = client.get("/api/v1/jobs/job_phase43_test/results")
        assert results.status_code == 200
        assert results.json()["results_summary"]["valid_points"] == 12

        artifact = client.get("/api/v1/jobs/job_phase43_test/exports/glb")
        assert artifact.status_code == 200
        assert artifact.headers["content-type"] == "model/gltf-binary"
        assert artifact.content.startswith(b"glTF")


def test_invalid_http_inputs_and_unknown_resources(monkeypatch, tmp_path):
    install_http_fakes(monkeypatch, tmp_path)
    with TestClient(app) as client:
        empty_ids = client.post("/api/v1/jobs/reconstruct", json={"upload_ids": []})
        assert empty_ids.status_code == 422

        unknown_model = client.post(
            "/api/v1/jobs/reconstruct",
            json={"upload_ids": ["upl_missing"], "model_id": "unknown_model"},
        )
        assert unknown_model.status_code == 400

        unknown_upload = client.post(
            "/api/v1/jobs/reconstruct",
            json={"upload_ids": ["upl_missing"], "model_id": "roomai_composite_v1"},
        )
        assert unknown_upload.status_code == 404

        bad_upload = client.post(
            "/api/v1/uploads",
            files={"file": ("not-an-image.txt", b"text", "text/plain")},
        )
        assert bad_upload.status_code == 400

        missing_status = client.get("/api/v1/jobs/job_does_not_exist")
        assert missing_status.status_code == 404
        missing_results = client.get("/api/v1/jobs/job_does_not_exist/results")
        assert missing_results.status_code == 404
        unsupported_export = client.get("/api/v1/jobs/job_phase43_test/exports/obj")
        assert unsupported_export.status_code == 400


def test_job_failure_status_and_submission_failure_are_reported(monkeypatch, tmp_path):
    service = install_http_fakes(monkeypatch, tmp_path)
    service.state = "FAILED"
    service.submit_error = "simulated service failure"
    with TestClient(app) as client:
        failed_status = client.get("/api/v1/jobs/job_phase43_test")
        assert failed_status.status_code == 200
        assert failed_status.json()["status"] == "FAILED"
        assert failed_status.json()["error"]["message"] == "mocked inference failure"

        upload = client.post(
            "/api/v1/uploads",
            files={"file": ("room.png", encoded_png(), "image/png")},
        )
        assert upload.status_code == 201
        submission = client.post(
            "/api/v1/jobs/reconstruct",
            json={"upload_ids": [upload.json()["upload_id"]], "model_id": "roomai_composite_v1"},
        )
        assert submission.status_code == 500
        assert "simulated service failure" in submission.json()["error"]["message"]


def test_real_job_worker_persists_inference_failure_without_model_weights(monkeypatch, tmp_path):
    import services.job_service as job_service_module

    class FailingAdapter:
        def is_available(self):
            return True, None

        def validate_inputs(self, images):
            return True, None

        def reconstruct(self, images, options=None, progress_callback=None):
            raise RuntimeError("deterministic mocked inference failure")

    class Registry:
        def get_default_model_id(self):
            return "roomai_composite_v1"

        def get_adapter(self, model_id):
            return FailingAdapter()

    monkeypatch.setattr(job_service_module, "RECONSTRUCTION_REGISTRY", Registry())
    service = JobService(store=JobStore(tmp_path / "worker_jobs.db"))
    service.jobs_dir = tmp_path / "worker_artifacts"
    job_id = service.submit_reconstruction_job([Image.new("RGB", (16, 16), (1, 2, 3))])

    deadline = time.monotonic() + 10
    status = service.get_job_status(job_id)
    while status["status"] not in ("FAILED", "COMPLETED", "CANCELLED") and time.monotonic() < deadline:
        time.sleep(0.05)
        status = service.get_job_status(job_id)
    assert status["status"] == "FAILED"
    assert status["error"]["code"] == "RuntimeError"
    assert status["error"]["message"] == "deterministic mocked inference failure"
