"""Comprehensive tests for RoomAI Model Downloader and A3 Checkpoint Lifecycle.

Verifies:
1. Existing valid checkpoint: verified and loaded directly without triggering network download.
2. Missing checkpoint: automatic streaming download from Hugging Face URL is triggered.
3. Invalid checksum: corrupted checkpoint is deleted/rejected, failing safely with actionable error.
4. Successful checksum verification: accurate SHA-256 calculation and case-insensitive matching.
5. Environment variable overrides: ROOMAI_MODEL_URL, ROOMAI_MODEL_SHA256, and ROOMAI_MODEL_PATH.
6. Network download failures: clear actionable error raised and temporary files cleaned up.
7. Startup lifecycle & readiness: absent checkpoint automatically downloaded at startup, reaching /ready status.
"""

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import app
from utils.model_downloader import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_URL,
    ChecksumMismatchError,
    ModelDownloadError,
    calculate_sha256,
    download_file,
    ensure_model_checkpoint,
    get_model_config,
    verify_checksum,
)


@pytest.fixture
def temp_ckpt_dir(tmp_path):
    """Provides isolated checkpoint directory."""
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_successful_checksum_verification(tmp_path):
    """Verifies that calculate_sha256 and verify_checksum compute exact SHA-256 digests case-insensitively."""
    test_file = tmp_path / "test_weights.bin"
    payload = b"RoomAI_A3_Test_Payload_Weights_12345"
    test_file.write_bytes(payload)

    expected_hash = hashlib.sha256(payload).hexdigest()

    assert calculate_sha256(test_file) == expected_hash
    # Case-insensitive verification
    assert verify_checksum(test_file, expected_hash.lower()) is True
    assert verify_checksum(test_file, expected_hash.upper()) is True
    assert verify_checksum(test_file, "0000000000000000000000000000000000000000000000000000000000000000") is False


def test_environment_variable_override(monkeypatch, tmp_path):
    """Verifies that ROOMAI_MODEL_URL, ROOMAI_MODEL_SHA256, and ROOMAI_MODEL_PATH override defaults."""
    custom_url = "https://example.com/custom_a3.pt"
    custom_sha256 = "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
    custom_rel_path = "checkpoints/custom_model.pt"

    monkeypatch.setenv("ROOMAI_MODEL_URL", custom_url)
    monkeypatch.setenv("ROOMAI_MODEL_SHA256", custom_sha256)
    monkeypatch.setenv("ROOMAI_MODEL_PATH", custom_rel_path)

    url, sha, path = get_model_config()

    assert url == custom_url
    assert sha == custom_sha256
    assert path.name == "custom_model.pt"


def test_existing_valid_checkpoint_no_download(temp_ckpt_dir):
    """Verifies that when a valid checkpoint exists locally, it is used immediately without downloading."""
    ckpt_file = temp_ckpt_dir / "valid_model.pt"
    data = b"VALID_MODEL_WEIGHTS_CONTENT_BYTES"
    ckpt_file.write_bytes(data)
    valid_sha256 = hashlib.sha256(data).hexdigest()

    with patch("requests.get") as mock_get:
        resolved = ensure_model_checkpoint(
            target_path=ckpt_file,
            url="https://invalid-host-should-never-be-contacted.test/model.pt",
            expected_sha256=valid_sha256,
        )

        assert resolved == ckpt_file
        assert resolved.exists()
        # requests.get must NOT have been called
        mock_get.assert_not_called()


def test_missing_checkpoint_triggers_download(temp_ckpt_dir):
    """Verifies that when the checkpoint is absent, automatic streaming download is executed."""
    ckpt_file = temp_ckpt_dir / "missing_model.pt"
    data = b"DOWNLOADED_MODEL_STREAMING_BYTES"
    expected_sha256 = hashlib.sha256(data).hexdigest()

    # Mock streaming response
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [data[:10], data[10:]]
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp) as mock_get:
        resolved = ensure_model_checkpoint(
            target_path=ckpt_file,
            url="https://huggingface.co/test/model.pt",
            expected_sha256=expected_sha256,
        )

        assert resolved == ckpt_file
        assert resolved.exists()
        assert resolved.read_bytes() == data
        mock_get.assert_called_once()


def test_corrupted_existing_checkpoint_deleted_and_raises(temp_ckpt_dir):
    """Verifies that if an existing checkpoint has an invalid checksum, it is deleted and fails clearly."""
    ckpt_file = temp_ckpt_dir / "corrupted_model.pt"
    ckpt_file.write_bytes(b"CORRUPTED_BYTES_CONTENT")
    expected_sha256 = "6CBC9E58F87156E07DD0E4E712B8AA96097AEB514A473621139A54CB83CEC576"

    with pytest.raises(ChecksumMismatchError) as exc_info:
        ensure_model_checkpoint(
            target_path=ckpt_file,
            url="https://test.url/model.pt",
            expected_sha256=expected_sha256,
        )

    # Must fail with clear actionable message
    assert "Corrupted model checkpoint" in str(exc_info.value)
    # Corrupted file must be deleted from disk
    assert not ckpt_file.exists()


def test_download_checksum_mismatch_discards_temp_and_fails(temp_ckpt_dir):
    """Verifies that if a downloaded stream does not match expected checksum, temp file is deleted."""
    ckpt_file = temp_ckpt_dir / "bad_download.pt"
    bad_data = b"UNEXPECTED_DOWNLOADED_DATA"
    expected_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [bad_data]
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ChecksumMismatchError) as exc_info:
            ensure_model_checkpoint(
                target_path=ckpt_file,
                url="https://test.url/model.pt",
                expected_sha256=expected_sha256,
            )

        assert "Downloaded checkpoint failed SHA-256 verification" in str(exc_info.value)
        assert not ckpt_file.exists()
        # Verify no temporary files remain
        temp_files = list(temp_ckpt_dir.glob(".tmp_*"))
        assert len(temp_files) == 0


def test_download_network_failure_cleans_up_and_fails_clearly(temp_ckpt_dir):
    """Verifies that network errors raise ModelDownloadError with clear message and cleanup."""
    ckpt_file = temp_ckpt_dir / "failed_network.pt"

    with patch("requests.get", side_effect=requests.ConnectionError("Connection timed out")):
        with pytest.raises(ModelDownloadError) as exc_info:
            ensure_model_checkpoint(
                target_path=ckpt_file,
                url="https://down.host/model.pt",
                expected_sha256="1234",
            )

        assert "Failed to download model checkpoint" in str(exc_info.value)
        assert "Connection timed out" in str(exc_info.value)
        assert not ckpt_file.exists()


def test_invalid_checksum_startup_fails_safely(monkeypatch, tmp_path):
    """Verifies that if startup encounters invalid checksum, startup fails safely into degraded state."""
    from api.routers import detect

    corrupt_file = tmp_path / "bad_startup.pt"
    corrupt_file.write_bytes(b"INVALID_HEADER_GARBAGE")

    monkeypatch.setenv("ROOMAI_MODEL_PATH", str(corrupt_file))
    monkeypatch.setenv(
        "ROOMAI_MODEL_SHA256",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )

    # Re-initialize engine instance with bad path
    engine = detect.ModelA3InferenceEngine()
    monkeypatch.setattr(detect, "GLOBAL_MODEL_A3_ENGINE", engine)

    # Startup lifespan should capture startup error safely
    with TestClient(app) as client:
        # 1. Health liveness remains 200 healthy
        h_res = client.get("/health")
        assert h_res.status_code == 200
        assert h_res.json()["status"] == "healthy"

        # 2. Readiness reflects degraded state and surfaces startup error
        r_res = client.get("/ready")
        assert r_res.status_code == 200
        r_data = r_res.json()
        assert r_data["status"] == "degraded"
        assert r_data["checkpoints_ready"] is False
        assert r_data["startup_error"] is not None
        assert "Corrupted model checkpoint" in r_data["startup_error"]

        # 3. Detection endpoint returns HTTP 503 MODEL_UNAVAILABLE on valid image
        from PIL import Image
        img = Image.new("RGB", (320, 240), color=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        valid_img_bytes = buf.getvalue()

        d_res = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", io.BytesIO(valid_img_bytes), "image/jpeg")},
        )
        assert d_res.status_code == 503
        assert d_res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_clean_startup_absent_checkpoint_downloads_and_reaches_readiness(monkeypatch, tmp_path):
    """Clean-startup test: when checkpoint is absent, app downloads it, verifies hash, and reaches ready state."""
    from api.routers import detect

    clean_dir = tmp_path / "clean_env" / "checkpoints"
    clean_ckpt = clean_dir / "indoor_detector_a3_yolo11s_800px_best.pt"

    # Ensure clean starting state: checkpoint does not exist
    assert not clean_ckpt.exists()

    monkeypatch.setenv("ROOMAI_MODEL_PATH", str(clean_ckpt))
    # Keep default HF URL and expected SHA-256

    engine = detect.ModelA3InferenceEngine()
    monkeypatch.setattr(detect, "GLOBAL_MODEL_A3_ENGINE", engine)

    # Boot FastAPI application
    with TestClient(app) as client:
        # 1. Probe readiness: model should be fully initialized, warmed up, and ready
        r_res = client.get("/ready")
        assert r_res.status_code == 200
        r_data = r_res.json()
        assert r_data["status"] == "ready"
        assert r_data["checkpoints_ready"] is True
        assert "indoor_detector_a3_yolo11s_800px" in r_data["models_ready"]
        assert r_data["startup_error"] is None

        # 2. Checkpoint file must now exist on disk and have verified SHA-256
        assert clean_ckpt.exists()
        assert clean_ckpt.stat().st_size == 19173786
        assert verify_checksum(clean_ckpt, DEFAULT_MODEL_SHA256) is True

        # 3. Predict on a valid image
        from PIL import Image
        img = Image.new("RGB", (640, 480), color=(150, 150, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        d_res = client.post(
            "/api/v1/detect",
            files={"file": ("test_clean.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
        assert d_res.status_code == 200
        d_data = d_res.json()
        assert d_data["status"] == "success"
        assert d_data["inference"]["model"] == "indoor_detector_a3_yolo11s_800px"

    # 4. Subsequent ensure_model_checkpoint calls must NOT re-download
    with patch("requests.get") as mock_get:
        re_resolved = ensure_model_checkpoint(clean_ckpt)
        assert re_resolved == clean_ckpt
        mock_get.assert_not_called()

