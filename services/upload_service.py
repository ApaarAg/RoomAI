"""Upload and Image Ingestion Service with Security and Dimension Validation."""

import hashlib
import io
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from utils.image_validation import validate_image_file
from utils.logging_config import get_logger

logger = get_logger("UploadService")
REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = REPO_ROOT / "artifacts" / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB ceiling
Image.MAX_IMAGE_PIXELS = 100_000_000  # Decompression bomb DOS guard


class UploadService:
    """Manages file upload validation, cryptographic deduplication, and safe disk storage."""

    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or UPLOADS_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """Strips path traversal sequences and disallows unsafe characters."""
        base = os.path.basename(filename)
        # Keep alphanumeric, dot, underscore, dash
        clean = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
        return clean or f"upload_{int(time.time())}.jpg"

    def save_upload(self, file_bytes: bytes, original_filename: str) -> Dict[str, Any]:
        """Validates and saves uploaded image bytes to disk.

        Args:
            file_bytes: Raw bytes from HTTP upload.
            original_filename: Client-supplied filename.

        Returns:
            Dict containing upload_id, filepath, dimensions, file_size, and sha256.

        Raises:
            ValueError: If file exceeds size, is corrupted, or fails dimension bounds.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size ({len(file_bytes) / (1024*1024):.2f} MB) exceeds maximum allowable limit of 20 MB."
            )

        clean_name = self.sanitize_filename(original_filename)
        ext = Path(clean_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Allowed image extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )

        # Decode image to verify content integrity
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            pil_img.verify()  # Verifies image header integrity
        except Exception as exc:
            raise ValueError(f"Uploaded file is not a valid or readable image: {exc}")

        # Reopen after verify() (Pillow requirement)
        pil_img = Image.open(io.BytesIO(file_bytes))
        w, h = pil_img.size

        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}.")

        if w > 8192 or h > 8192:
            raise ValueError(f"Image dimensions ({w}x{h}) exceed maximum allowable bound (8192x8192).")

        # Compute SHA-256 for integrity and deduplication
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        upload_id = f"upl_{sha256[:16]}"
        target_filename = f"{upload_id}{ext}"
        target_path = self.upload_dir / target_filename

        # Write safely if not already present
        if not target_path.exists():
            target_path.write_bytes(file_bytes)
            logger.info(f"Saved uploaded image {upload_id} ({w}x{h}, {len(file_bytes)} bytes) to {target_path}")
        else:
            logger.debug(f"Deduplicated existing upload {upload_id}")

        return {
            "upload_id": upload_id,
            "filename": target_filename,
            "original_filename": clean_name,
            "filepath": str(target_path),
            "width": w,
            "height": h,
            "format": pil_img.format or ext.replace(".", "").upper(),
            "size_bytes": len(file_bytes),
            "sha256": sha256,
        }

    def get_upload_path(self, upload_id: str) -> Optional[Path]:
        """Resolves upload ID to existing disk path with path traversal protection."""
        clean_id = os.path.basename(upload_id)
        for ext in ALLOWED_EXTENSIONS:
            p = self.upload_dir / f"{clean_id}{ext}"
            if p.exists():
                return p
        return None

    def load_upload_image(self, upload_id: str) -> Optional[Image.Image]:
        """Loads and returns decoded PIL Image for a given upload ID."""
        path = self.get_upload_path(upload_id)
        if not path or not path.exists():
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            logger.error(f"Failed to load image for upload {upload_id}: {exc}")
            return None


# Global singleton instance
GLOBAL_UPLOAD_SERVICE = UploadService()
