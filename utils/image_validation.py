"""Image validation and sanitization utilities for RoomAI."""

from dataclasses import dataclass
import io
import os
from typing import BinaryIO, Optional, Union
from PIL import Image, ImageOps

from utils.logging_config import get_logger

logger = get_logger("ImageValidation")

DEFAULT_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MAX_FILE_SIZE_MB = 15.0
DEFAULT_MAX_DIMENSION = 4096


@dataclass
class ImageValidationResult:
    """Encapsulates the result of image validation."""

    is_valid: bool
    image: Optional[Image.Image] = None
    width: int = 0
    height: int = 0
    format: str = ""
    channels: int = 0
    file_size_mb: float = 0.0
    error_message: Optional[str] = None


def validate_image_file(
    file_input: Union[bytes, BinaryIO, str],
    filename: Optional[str] = None,
    max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    allowed_extensions: Optional[set] = None,
) -> ImageValidationResult:
    """Validates an uploaded image file for size, integrity, format, and dimensions.

    Args:
        file_input: Raw image bytes, a file-like object, or file path.
        filename: Optional original filename to validate extension.
        max_file_size_mb: Maximum allowed file size in megabytes.
        max_dimension: Maximum allowed pixel width or height.
        allowed_extensions: Set of permitted file extensions (e.g. {'.jpg', '.png'}).

    Returns:
        ImageValidationResult containing the parsed PIL image or an error message.
    """
    if allowed_extensions is None:
        allowed_extensions = DEFAULT_ALLOWED_EXTENSIONS

    # 1. Check filename extension if provided
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_extensions:
            return ImageValidationResult(
                is_valid=False,
                error_message=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
            )

    # 2. Extract raw bytes
    try:
        if isinstance(file_input, (str, os.PathLike)):
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
        elif hasattr(file_input, "read"):
            current_pos = file_input.tell() if hasattr(file_input, "tell") else None
            raw_bytes = file_input.read()
            if current_pos is not None and hasattr(file_input, "seek"):
                file_input.seek(current_pos)
        else:
            return ImageValidationResult(
                is_valid=False,
                error_message="Invalid input type: expected bytes, file-like object, or path.",
            )
    except Exception as exc:
        logger.error(f"Failed to read image bytes: {exc}")
        return ImageValidationResult(
            is_valid=False, error_message=f"Could not read image data: {str(exc)}"
        )

    # 3. Check file size
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb == 0:
        return ImageValidationResult(
            is_valid=False, error_message="Uploaded image is empty (0 bytes)."
        )
    if size_mb > max_file_size_mb:
        return ImageValidationResult(
            is_valid=False,
            file_size_mb=round(size_mb, 2),
            error_message=f"File size ({size_mb:.2f} MB) exceeds maximum limit ({max_file_size_mb:.2f} MB).",
        )

    # 4. Verify byte integrity
    try:
        verify_stream = io.BytesIO(raw_bytes)
        with Image.open(verify_stream) as img_verify:
            img_format = img_verify.format or "UNKNOWN"
            img_verify.verify()
    except Image.DecompressionBombError as exc:
        logger.warning(f"Decompression bomb rejected during verify: {exc}")
        return ImageValidationResult(
            is_valid=False,
            file_size_mb=round(size_mb, 2),
            error_message=f"Image rejected: potential decompression bomb DOS attack ({str(exc)}).",
        )
    except Exception as exc:
        logger.warning(f"Image integrity verification failed: {exc}")
        return ImageValidationResult(
            is_valid=False,
            file_size_mb=round(size_mb, 2),
            error_message="Image file appears corrupted or unreadable.",
        )

    # 5. Open image safely and convert to standard RGB
    try:
        open_stream = io.BytesIO(raw_bytes)
        img = Image.open(open_stream)

        # Handle EXIF orientation tags automatically
        img = ImageOps.exif_transpose(img)

        # Convert to RGB if needed (handling RGBA, grayscale, palletized)
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size

        # 6. Check dimension bounds
        if width > max_dimension or height > max_dimension:
            # Rescale preserving aspect ratio to avoid OOM in ML pipelines
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            width, height = img.size
            logger.info(
                f"Image exceeded max dimension {max_dimension}px; safely downsampled to {width}x{height}px."
            )

        return ImageValidationResult(
            is_valid=True,
            image=img,
            width=width,
            height=height,
            format=img_format,
            channels=3,
            file_size_mb=round(size_mb, 2),
        )

    except Image.DecompressionBombError as exc:
        logger.warning(f"Decompression bomb rejected during decode: {exc}")
        return ImageValidationResult(
            is_valid=False,
            file_size_mb=round(size_mb, 2),
            error_message=f"Image rejected: potential decompression bomb DOS attack ({str(exc)}).",
        )
    except Exception as exc:
        logger.error(f"Error decoding image stream: {exc}")
        return ImageValidationResult(
            is_valid=False,
            file_size_mb=round(size_mb, 2),
            error_message=f"Error decoding image: {str(exc)}",
        )
