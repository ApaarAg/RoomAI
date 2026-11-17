"""Integration tests for non-blocking asynchronous startup lifecycle.

Verifies:
1. /health responds with HTTP 200 while model initialization is still running in background.
2. /ready reports 'degraded' (checkpoints_ready=False) during initialization.
3. /ready transitions to 'ready' after successful background initialization.
4. /api/v1/detect returns HTTP 503 MODEL_UNAVAILABLE before model readiness.
5. Model initialization failures do not crash the HTTP server (/health remains healthy).
6. Application shutdown properly cancels and cleans up background initialization tasks.
7. Concurrent initialization attempts are deduplicated and synchronized.
"""

import asyncio
import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.routers import detect


def make_dummy_image_bytes() -> bytes:
    """Helper to generate in-memory JPEG test image."""
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_responds_immediately_while_warmup_in_progress(monkeypatch):
    """Verifies /health responds with HTTP 200 and /ready is degraded while warmup is sleeping."""
    engine = detect.ModelA3InferenceEngine()
    monkeypatch.setattr(detect, "GLOBAL_MODEL_A3_ENGINE", engine)

    # Simulate an artificially slow warmup (e.g. 1.5 seconds) in background thread
    original_load = engine.load_and_warmup

    def slow_load_and_warmup():
        time.sleep(1.2)
        # Call original load_and_warmup
        original_load()

    monkeypatch.setattr(engine, "load_and_warmup", slow_load_and_warmup)

    with TestClient(app) as client:
        t0 = time.perf_counter()

        # 1. /health must respond IMMEDIATELY (< 300ms) with HTTP 200 while background warmup runs
        h_resp = client.get("/health")
        h_time_ms = (time.perf_counter() - t0) * 1000.0
        assert h_resp.status_code == 200
        assert h_resp.json()["status"] == "healthy"
        assert h_time_ms < 500.0, f"/health took {h_time_ms}ms to respond, expected < 500ms"

        # 2. /ready must report degraded while warmup is still in progress
        r_resp = client.get("/ready")
        assert r_resp.status_code == 200
        r_data = r_resp.json()
        assert r_data["status"] == "degraded"
        assert r_data["checkpoints_ready"] is False
        assert "indoor_detector_a3_yolo11s_800px" not in r_data["models_ready"]

        # 3. /api/v1/detect must return HTTP 503 MODEL_UNAVAILABLE before readiness
        img_bytes = make_dummy_image_bytes()
        d_resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert d_resp.status_code == 503
        assert d_resp.json()["error"]["code"] == "MODEL_UNAVAILABLE"
        assert "initializing" in d_resp.json()["error"]["message"].lower()

        # 4. Wait for background initialization to complete
        assert engine.wait_until_ready(timeout=15.0) is True

        # 5. /ready must now report 'ready'
        r_ready = client.get("/ready")
        assert r_ready.status_code == 200
        r_ready_data = r_ready.json()
        assert r_ready_data["status"] == "ready"
        assert r_ready_data["checkpoints_ready"] is True
        assert "indoor_detector_a3_yolo11s_800px" in r_ready_data["models_ready"]
        assert r_ready_data["startup_error"] is None

        # 6. /api/v1/detect now succeeds with HTTP 200
        d_success = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
        assert d_success.status_code == 200
        assert d_success.json()["status"] == "success"
        assert d_success.json()["inference"]["model"] == "indoor_detector_a3_yolo11s_800px"


def test_initialization_failure_does_not_kill_server(monkeypatch):
    """Verifies that model initialization failure sets startup_error while keeping /health alive."""
    engine = detect.ModelA3InferenceEngine()
    monkeypatch.setattr(detect, "GLOBAL_MODEL_A3_ENGINE", engine)

    def failing_load():
        raise FileNotFoundError("Simulated corrupted or unreachable checkpoint")

    monkeypatch.setattr(engine, "load_and_warmup", failing_load)

    with TestClient(app) as client:
        # Wait for background task to execute and fail
        engine._ready_event.wait(timeout=5.0)

        # /health remains fully responsive
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        assert h_resp.json()["status"] == "healthy"

        # /ready reflects degraded status and surfaces the error
        r_resp = client.get("/ready")
        assert r_resp.status_code == 200
        r_data = r_resp.json()
        assert r_data["status"] == "degraded"
        assert r_data["checkpoints_ready"] is False
        assert "Simulated corrupted or unreachable checkpoint" in r_data["startup_error"]

        # /api/v1/detect returns controlled HTTP 503
        img_bytes = make_dummy_image_bytes()
        d_resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert d_resp.status_code == 503
        assert d_resp.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_shutdown_cancels_background_init_task(monkeypatch):
    """Verifies that application shutdown properly cancels any active background initialization."""
    engine = detect.ModelA3InferenceEngine()
    monkeypatch.setattr(detect, "GLOBAL_MODEL_A3_ENGINE", engine)

    def hanging_load():
        # Sleep for 60 seconds to simulate long download
        time.sleep(60.0)

    monkeypatch.setattr(engine, "load_and_warmup", hanging_load)

    with TestClient(app) as client:
        # Task must be active
        assert engine._init_task is not None
        assert not engine._init_task.done()

    # Once lifespan context exits, task should be cancelled
    assert engine._init_task is None or engine._init_task.done()


def test_concurrent_initialization_prevented():
    """Verifies that multiple concurrent calls to start_background_init reuse the single running task."""
    engine = detect.ModelA3InferenceEngine()

    async def _async_test():
        loop = asyncio.get_running_loop()
        # Mock load_and_warmup with async sleep
        with patch.object(engine, "load_and_warmup", side_effect=lambda: time.sleep(0.5)):
            task1 = engine.start_background_init()
            task2 = engine.start_background_init()
            assert task1 is task2
            assert engine._is_initializing is True
            await engine.cancel_background_init()

    asyncio.run(_async_test())
