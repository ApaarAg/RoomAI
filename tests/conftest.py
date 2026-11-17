"""Pytest fixtures for RoomAI tests."""

import io
from typing import Dict, Any
import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Returns a simple synthetic 640x480 RGB PIL image for testing."""
    img_array = np.zeros((480, 640, 3), dtype=np.uint8)
    # Give it some variation
    img_array[:240, :, 0] = 200  # Reddish top half
    img_array[240:, :, 1] = 180  # Greenish bottom half
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def sample_image_bytes(sample_pil_image: Image.Image) -> bytes:
    """Returns valid JPEG encoded bytes."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """Returns corrupted image bytes."""
    return b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00CORRUPT_BYTES_TRUNCATED"


@pytest.fixture
def mock_app_config() -> Dict[str, Any]:
    """Returns a test application configuration."""
    return {
        "mock_mode": True,
        "upload": {
            "max_file_size_mb": 5.0,
            "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp"],
            "max_image_dimension": 2048,
        },
        "design": {
            "styles": ["Scandinavian", "Modern", "Minimalist"],
            "default_style": "Scandinavian",
            "default_prompt": "Redesign room with oak wood and natural lighting.",
        },
    }
