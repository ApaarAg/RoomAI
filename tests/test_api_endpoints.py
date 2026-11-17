"""Tests for RoomAI FastAPI REST Endpoints."""

import io
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from PIL import Image

from api.main import app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_health_endpoint():
    """Liveness probe must return 200 OK with service name and version."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.4.0"
    assert "timestamp" in data


def test_readiness_endpoint():
    """Readiness probe must report hardware and model state."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ready", "degraded")
    assert "cuda_available" in data
    assert "device_name" in data
    assert "free_vram_gb" in data
    assert "models_ready" in data


def test_models_discovery_endpoint():
    """Models endpoint must report registered adapters and their capabilities."""
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["default_model_id"] == "roomai_composite_v1"
    models = data["models"]
    assert len(models) >= 2

    # Check composite model
    comp = next((m for m in models if m["model_id"] == "roomai_composite_v1"), None)
    assert comp is not None
    assert comp["status"] == "PRODUCTION_READY"
    assert comp["supports_single_view"] is True
    assert comp["max_views"] == 10

    # Check foundation model
    hunyuan = next((m for m in models if m["model_id"] == "hunyuan_world_mirror_1b"), None)
    assert hunyuan is not None
    assert hunyuan["status"] in ("PRODUCTION_READY", "STAGED")
    assert hunyuan["supports_surface_normals"] is True


def test_upload_image_success():
    """Upload endpoint must accept valid image and return upload ID and dimensions."""
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_room.jpg", buf, "image/jpeg")}
    resp = client.post("/api/v1/uploads", files=files)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "success"
    assert data["upload_id"].startswith("upl_")
    assert data["width"] == 320
    assert data["height"] == 240
    assert data["format"] == "JPEG"

    # Query metadata
    meta_resp = client.get(f"/api/v1/uploads/{data['upload_id']}")
    assert meta_resp.status_code == 200
    assert meta_resp.json()["upload_id"] == data["upload_id"]


def test_upload_invalid_extension():
    """Upload endpoint must reject unsupported file extensions."""
    buf = io.BytesIO(b"fake text content")
    files = {"file": ("malicious.txt", buf, "text/plain")}
    resp = client.post("/api/v1/uploads", files=files)
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["error"]["message"]


def test_upload_corrupted_image():
    """Upload endpoint must reject corrupted images with non-image bytes."""
    buf = io.BytesIO(b"PK\x03\x04corruptedbytes")
    files = {"file": ("broken.jpg", buf, "image/jpeg")}
    resp = client.post("/api/v1/uploads", files=files)
    assert resp.status_code == 400
    assert "not a valid or readable image" in resp.json()["error"]["message"]


def test_experimental_vqa_offline():
    """VQA endpoint must honestly return 501 Not Implemented."""
    resp = client.post(
        "/api/v1/experimental/vqa",
        json={"upload_id": "dummy", "question": "What is here?"},
    )
    assert resp.status_code == 501
    err = resp.json()["error"]
    assert "UNAVAILABLE" in err["message"] or "UNAVAILABLE" in str(err)


def test_experimental_redesign_offline():
    """Generative Redesign endpoint must honestly return 501 Not Implemented."""
    resp = client.post(
        "/api/v1/experimental/redesign",
        json={"upload_id": "dummy", "style_prompt": "modern minimalist"},
    )
    assert resp.status_code == 501
    err = resp.json()["error"]
    assert "UNAVAILABLE" in err["message"] or "UNAVAILABLE" in str(err)

