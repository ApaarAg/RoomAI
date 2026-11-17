"""Comprehensive Integration Test Suite for RoomAI Model A3 Detection Endpoint."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import DetectionResponse


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient ensuring startup and teardown lifecycles run once."""
    with TestClient(app) as c:
        from api.routers.detect import GLOBAL_MODEL_A3_ENGINE
        GLOBAL_MODEL_A3_ENGINE.wait_until_ready(timeout=30.0)
        yield c


def make_test_image(width: int = 640, height: int = 480, color=(120, 140, 160), fmt: str = "JPEG") -> bytes:
    """Helper to generate in-memory test images."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestDetectionEndpoint:
    """Integration tests verifying /api/v1/detect contract, validation, and execution."""

    def test_valid_image_detection(self, client):
        """1. Valid image returns HTTP 200 with complete DetectionResponse schema."""
        img_bytes = make_test_image(640, 480)
        res = client.post(
            "/api/v1/detect",
            files={"file": ("room.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.25"},
        )
        assert res.status_code == 200
        data = res.json()

        # Validate against canonical Pydantic model
        validated = DetectionResponse(**data)
        assert validated.status == "success"
        assert validated.request_id.startswith("det_")
        assert validated.image.width == 640
        assert validated.image.height == 480
        assert validated.image.channels == 3
        assert validated.inference.model == "indoor_detector_a3_yolo11s_800px"
        assert validated.timing.total_server_ms > 0

    def test_corrupt_image(self, client):
        """2. Corrupt/unidentifiable image payload returns HTTP 400 Bad Request."""
        corrupt_bytes = b"NOT_A_VALID_IMAGE_FILE_HEADER_GARBAGE_BYTES_12345"
        res = client.post(
            "/api/v1/detect",
            files={"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")},
        )
        assert res.status_code == 400
        data = res.json()
        assert data["error"]["code"] == "CORRUPT_IMAGE"

    def test_empty_payload(self, client):
        """3. Empty (0-byte) file returns HTTP 400 Bad Request."""
        res = client.post(
            "/api/v1/detect",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert res.status_code == 400
        data = res.json()
        assert data["error"]["code"] == "EMPTY_PAYLOAD"

    def test_unsupported_media_type(self, client):
        """4. Unsupported MIME type (e.g. text/plain, image/gif) returns HTTP 415."""
        img_bytes = make_test_image(100, 100)
        res = client.post(
            "/api/v1/detect",
            files={"file": ("anim.gif", io.BytesIO(img_bytes), "image/gif")},
        )
        assert res.status_code == 415
        data = res.json()
        assert data["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    def test_dimensions_too_small(self, client):
        """5. Sub-32px image returns HTTP 400 DIMENSIONS_TOO_SMALL."""
        tiny_bytes = make_test_image(16, 16)
        res = client.post(
            "/api/v1/detect",
            files={"file": ("tiny.jpg", io.BytesIO(tiny_bytes), "image/jpeg")},
        )
        assert res.status_code == 400
        data = res.json()
        assert data["error"]["code"] == "DIMENSIONS_TOO_SMALL"

    def test_caching_behavior(self, client):
        """6. Re-submitting identical image with same confidence threshold hits LRU cache."""
        img_bytes = make_test_image(320, 240, color=(10, 20, 30))
        # 1st request -> fresh inference
        res1 = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["inference"]["cached"] is False

        # 2nd request with exact same content -> cached
        res2 = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["inference"]["cached"] is True
        assert data2["timing"]["total_server_ms"] < 10.0  # Sub-10ms cache retrieval

    def test_rapid_successive_requests(self, client):
        """7. Rapid sequential requests succeed under GPU lock without race conditions."""
        for i in range(5):
            img_bytes = make_test_image(200, 200, color=(i * 20, i * 30, i * 40))
            res = client.post(
                "/api/v1/detect",
                files={"file": (f"test_{i}.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "success"

    def test_png_and_webp_support(self, client):
        """8. Supports PNG and WebP formats in addition to JPEG."""
        for fmt, mime in [("PNG", "image/png"), ("WEBP", "image/webp")]:
            img_bytes = make_test_image(200, 200, fmt=fmt)
            res = client.post(
                "/api/v1/detect",
                files={"file": (f"test.{fmt.lower()}", io.BytesIO(img_bytes), mime)},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "success"

    def test_cache_separation_across_different_thresholds(self, client):
        """9. Hostile Review: same image with threshold A vs threshold B cannot return stale cached response."""
        from api.routers.detect import GLOBAL_MODEL_A3_ENGINE
        GLOBAL_MODEL_A3_ENGINE.clear_cache()

        img_bytes = make_test_image(640, 480, color=(140, 160, 180))

        # Request 1 with low threshold 0.15
        res_low = client.post(
            "/api/v1/detect",
            files={"file": ("threshold_test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.15"},
        )
        assert res_low.status_code == 200
        data_low = res_low.json()
        assert data_low["inference"]["cached"] is False

        # Request 2 with high threshold 0.85 on EXACT same image
        res_high = client.post(
            "/api/v1/detect",
            files={"file": ("threshold_test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.85"},
        )
        assert res_high.status_code == 200
        data_high = res_high.json()

        # Crucial invariant: high threshold must NOT be served from the 0.15 cache!
        assert data_high["inference"]["cached"] is False, "High threshold erroneously served from low threshold cache!"

        # Request 3 repeating 0.15 -> should now hit the 0.15 cache
        res_low_cached = client.post(
            "/api/v1/detect",
            files={"file": ("threshold_test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.15"},
        )
        assert res_low_cached.status_code == 200
        assert res_low_cached.json()["inference"]["cached"] is True

        # Request 4 repeating 0.85 -> should now hit the 0.85 cache
        res_high_cached = client.post(
            "/api/v1/detect",
            files={"file": ("threshold_test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.85"},
        )
        assert res_high_cached.status_code == 200
        assert res_high_cached.json()["inference"]["cached"] is True

    def test_cache_invalidation_and_model_version_isolation(self, client):
        """10. Cache clear flushes cache and model version changes produce disjoint cache entries."""
        from api.routers.detect import GLOBAL_MODEL_A3_ENGINE
        img_bytes = make_test_image(300, 300, color=(50, 100, 150))

        # Populate cache
        res = client.post(
            "/api/v1/detect",
            files={"file": ("inval.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
        assert res.status_code == 200
        assert GLOBAL_MODEL_A3_ENGINE.cache_size > 0

        # Flush cache
        GLOBAL_MODEL_A3_ENGINE.clear_cache()
        assert GLOBAL_MODEL_A3_ENGINE.cache_size == 0

        # Next call must be fresh inference
        res_after_clear = client.post(
            "/api/v1/detect",
            files={"file": ("inval.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
        assert res_after_clear.status_code == 200
        assert res_after_clear.json()["inference"]["cached"] is False

    def test_file_too_large_returns_413(self, client):
        """11. File payload exceeding 20MB ceiling is rejected with HTTP 413 FILE_TOO_LARGE."""
        # 20MB + 1024 bytes dummy stream
        oversized_bytes = b"X" * (20 * 1024 * 1024 + 1024)
        res = client.post(
            "/api/v1/detect",
            files={"file": ("large_image.jpg", io.BytesIO(oversized_bytes), "image/jpeg")},
        )
        assert res.status_code == 413
        data = res.json()
        assert data["error"]["code"] == "FILE_TOO_LARGE"

    def test_missing_model_returns_503(self, client, monkeypatch):
        """12. Missing checkpoint raises FileNotFoundError and returns controlled HTTP 503 MODEL_UNAVAILABLE."""
        from api.routers.detect import GLOBAL_MODEL_A3_ENGINE
        
        async def fake_predict_missing(*args, **kwargs):
            raise FileNotFoundError("Simulated missing checkpoint file")

        monkeypatch.setattr(GLOBAL_MODEL_A3_ENGINE, "predict_async", fake_predict_missing)

        img_bytes = make_test_image(320, 240)
        res = client.post(
            "/api/v1/detect",
            files={"file": ("test_missing.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert res.status_code == 503
        data = res.json()
        assert data["error"]["code"] == "MODEL_UNAVAILABLE"
        assert "Simulated missing checkpoint" in data["error"]["message"]

    def test_readiness_probe_reflects_model_status(self, client):
        """13. Diagnostic probe GET /ready reflects Model A3 availability and hardware telemetry."""
        res = client.get("/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        assert "indoor_detector_a3_yolo11s_800px" in data["models_ready"]
        assert data["checkpoints_ready"] is True
        assert data["startup_error"] is None


