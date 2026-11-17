"""Tests for image validation and format checks."""

import io
import numpy as np
from PIL import Image

from utils.image_validation import validate_image_file


def test_valid_image_bytes(sample_image_bytes: bytes):
    """Valid JPEG bytes should pass validation and return RGB PIL image."""
    res = validate_image_file(sample_image_bytes, filename="living_room.jpg")
    assert res.is_valid is True
    assert res.image is not None
    assert res.width == 640
    assert res.height == 480
    assert res.channels == 3
    assert res.error_message is None


def test_unsupported_file_extension(sample_image_bytes: bytes):
    """Files with disallowed extensions should fail validation."""
    res = validate_image_file(sample_image_bytes, filename="room.pdf")
    assert res.is_valid is False
    assert "Unsupported file format" in res.error_message


def test_corrupted_image(corrupted_image_bytes: bytes):
    """Corrupted bytes should fail validation gracefully."""
    res = validate_image_file(corrupted_image_bytes, filename="corrupted.jpg")
    assert res.is_valid is False
    assert "corrupted" in res.error_message.lower() or "integrity" in res.error_message.lower()


def test_empty_file_upload():
    """Empty 0-byte input should fail validation."""
    res = validate_image_file(b"", filename="empty.jpg")
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()


def test_oversized_file():
    """File exceeding max size in MB should be rejected."""
    # Create a 2 MB dummy byte buffer and set limit to 1 MB
    oversized_data = b"0" * (2 * 1024 * 1024)
    res = validate_image_file(oversized_data, filename="large.jpg", max_file_size_mb=1.0)
    assert res.is_valid is False
    assert "exceeds maximum limit" in res.error_message


def test_downscaling_large_dimensions():
    """Images exceeding max_dimension should be safely resized preserving aspect ratio."""
    large_img = Image.fromarray(np.zeros((3000, 4000, 3), dtype=np.uint8))
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")
    raw = buf.getvalue()

    res = validate_image_file(raw, filename="giant.png", max_dimension=1000)
    assert res.is_valid is True
    assert res.width <= 1000
    assert res.height <= 1000
