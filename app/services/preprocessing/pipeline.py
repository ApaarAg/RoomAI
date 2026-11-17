"""RoomAI Preprocessing Pipeline.

Reproducible image preprocessing for model inputs.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageOps
import numpy as np


class PreprocessingPipeline:
    """Configurable preprocessing pipeline for reconstruction models."""

    # Supported formats
    SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "JPG"}
    MAX_DIMENSION = 8192
    MAX_PIXELS = 100_000_000  # 100MP decompression bomb limit
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.version = "1.0.0"

    def validate_file(self, file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """Validate uploaded file."""
        # Check size
        if len(file_bytes) > self.MAX_FILE_SIZE:
            return False, f"File size {len(file_bytes)} exceeds {self.MAX_FILE_SIZE} bytes"

        # Check extension
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in {"jpg", "jpeg", "png", "webp"}:
            return False, f"Unsupported extension: .{ext}"

        return True, None

    def decode_image(self, file_bytes: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
        """Decode image bytes to PIL Image with metadata."""
        try:
            img = Image.open(file_bytes if isinstance(file_bytes, (str, Path)) else
                           __import__("io").BytesIO(file_bytes))
            img.load()  # Force load to catch corruption
        except Exception as exc:
            raise ValueError(f"Failed to decode image: {exc}")

        # Get metadata before any transformation
        metadata = {
            "original_format": img.format,
            "original_mode": img.mode,
            "original_size": img.size,
            "exif": dict(img.getexif()) if hasattr(img, "getexif") else {},
        }

        return img, metadata

    def correct_orientation(self, img: Image.Image) -> Image.Image:
        """Apply EXIF orientation correction."""
        try:
            return ImageOps.exif_transpose(img)
        except Exception:
            return img  # If no EXIF or error, return as-is

    def convert_to_rgb(self, img: Image.Image) -> Image.Image:
        """Ensure image is RGB."""
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def validate_dimensions(self, img: Image.Image) -> Tuple[bool, Optional[str]]:
        """Check image dimensions are within limits."""
        w, h = img.size
        if w <= 0 or h <= 0:
            return False, f"Invalid dimensions: {w}x{h}"
        if w > self.MAX_DIMENSION or h > self.MAX_DIMENSION:
            return False, f"Dimensions {w}x{h} exceed maximum {self.MAX_DIMENSION}"
        if w * h > self.MAX_PIXELS:
            return False, f"Total pixels {w*h} exceeds limit {self.MAX_PIXELS}"
        return True, None

    def compute_sha256(self, file_bytes: bytes) -> str:
        """Compute SHA-256 of file content."""
        return hashlib.sha256(file_bytes).hexdigest()

    def resize_for_model(
        self,
        img: Image.Image,
        target_size: Tuple[int, int] = (518, 518),
        preserve_aspect: bool = True,
    ) -> Image.Image:
        """Resize image for model input with patch alignment."""
        if preserve_aspect:
            # Square crop to preserve aspect ratio
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

        # Resize to target (must be multiple of patch_size)
        patch_size = self.config.get("patch_size", 14)
        target_w = (target_size[0] // patch_size) * patch_size
        target_h = (target_size[1] // patch_size) * patch_size

        return img.resize((target_w, target_h), Image.Resampling.BILINEAR)

    def normalize_image(
        self,
        img: Image.Image,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """Normalize image to model input range."""
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        return arr  # HWC format

    def to_tensor(self, arr: np.ndarray) -> "torch.Tensor":
        """Convert HWC numpy array to CHW tensor."""
        import torch
        return torch.from_numpy(arr).permute(2, 0, 1)

    def process_for_model(
        self,
        file_bytes: bytes,
        filename: str,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full preprocessing pipeline for a model."""
        model_config = model_config or {}
        target_size = model_config.get("input_resolution", (518, 518))
        patch_size = model_config.get("patch_size", 14)

        # 1. Validate
        valid, error = self.validate_file(file_bytes, filename)
        if not valid:
            raise ValueError(error)

        # 2. Decode
        img, metadata = self.decode_image(file_bytes)

        # 3. Orientation correction
        img = self.correct_orientation(img)

        # 4. Convert to RGB
        img = self.convert_to_rgb(img)

        # 5. Validate dimensions
        valid, error = self.validate_dimensions(img)
        if not valid:
            raise ValueError(error)

        # 6. Resize for model
        img_resized = self.resize_for_model(img, target_size)

        # 7. Normalize
        normalized = self.normalize_image(img_resized)

        # 8. To tensor
        tensor = self.to_tensor(normalized)

        # Compute hash
        sha256 = self.compute_sha256(file_bytes)

        return {
            "tensor": tensor,
            "original_image": img,
            "resized_image": img_resized,
            "original_size": metadata["original_size"],
            "resized_size": img_resized.size,
            "sha256": sha256,
            "metadata": metadata,
            "preprocessing_version": self.version,
            "model_config": {
                "target_size": target_size,
                "patch_size": patch_size,
            },
        }

    def process_batch(
        self,
        images: List[Image.Image],
        model_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Process a batch of PIL images."""
        return [self.process_for_model(
            # Convert PIL to bytes for hashing
            __import__("io").BytesIO().write(img.convert("RGB").tobytes()) or b"",
            f"view_{i}.jpg",
            model_config,
        ) for i, img in enumerate(images)]

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Return current preprocessing configuration for reproducibility."""
        return {
            "version": self.version,
            "max_dimension": self.MAX_DIMENSION,
            "max_pixels": self.MAX_PIXELS,
            "max_file_size": self.MAX_FILE_SIZE,
            "supported_formats": list(self.SUPPORTED_FORMATS),
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }


# Singleton
_preprocessing_pipeline: Optional[PreprocessingPipeline] = None


def get_preprocessing_pipeline() -> PreprocessingPipeline:
    global _preprocessing_pipeline
    if _preprocessing_pipeline is None:
        _preprocessing_pipeline = PreprocessingPipeline()
    return _preprocessing_pipeline