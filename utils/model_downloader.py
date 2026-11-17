"""Automated and verified model checkpoint downloader for RoomAI.

Ensures that the primary Model A3 detector checkpoint is present locally,
downloading it from Hugging Face if missing, and verifying SHA-256 integrity
before allowing weights to load.
"""

import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import requests

logger = logging.getLogger("ModelDownloader")

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Apaarag/roomai-a3-indoor-detector/resolve/main/"
    "indoor_detector_a3_yolo11s_800px_best.pt?download=true"
)
DEFAULT_MODEL_SHA256 = "6CBC9E58F87156E07DD0E4E712B8AA96097AEB514A473621139A54CB83CEC576"
DEFAULT_MODEL_PATH = "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt"


class ModelDownloadError(FileNotFoundError, RuntimeError):
    """Raised when model checkpoint download fails due to network or server issues."""
    pass


class ChecksumMismatchError(FileNotFoundError, ValueError):
    """Raised when checkpoint SHA-256 checksum does not match expected value."""
    pass


def get_model_config() -> Tuple[str, str, Path]:
    """Retrieves target model url, expected sha256, and canonical path from environment or defaults."""
    url = os.getenv("ROOMAI_MODEL_URL", DEFAULT_MODEL_URL).strip()
    expected_sha256 = os.getenv("ROOMAI_MODEL_SHA256", DEFAULT_MODEL_SHA256).strip().lower()
    path_str = os.getenv("ROOMAI_MODEL_PATH", DEFAULT_MODEL_PATH).strip()

    target_path = Path(path_str)
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path

    return url, expected_sha256, target_path


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculates the SHA-256 hex digest of a file in streaming chunks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest().lower()


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verifies that the file at file_path has the expected SHA-256 checksum (case-insensitive)."""
    if not file_path.is_file():
        return False
    actual_hash = calculate_sha256(file_path)
    return actual_hash == expected_sha256.strip().lower()


def download_file(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    timeout: int = 120,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Downloads a file from a URL to dest_path using a temporary file and verifies checksum.

    If expected_sha256 is provided, verifies hash before replacing dest_path.
    If corrupted or download fails, the temporary file is deleted and an actionable exception is raised.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = dest_path.parent / f".tmp_{dest_path.name}_{uuid.uuid4().hex[:8]}"

    logger.info(f"Downloading checkpoint from {url} to temporary stage {temp_file}...")
    sha256 = hashlib.sha256()

    try:
        response = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
        response.raise_for_status()

        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    sha256.update(chunk)
    except Exception as exc:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Failed to download model checkpoint from '{url}': {exc}. "
            "Please check network connectivity or ROOMAI_MODEL_URL."
        ) from exc

    computed_hash = sha256.hexdigest().lower()
    if expected_sha256:
        clean_expected = expected_sha256.strip().lower()
        if computed_hash != clean_expected:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"Downloaded checkpoint failed SHA-256 verification. "
                f"Expected: '{clean_expected}', Got: '{computed_hash}'. "
                "Corrupted download rejected and discarded."
            )

    # Atomic move into destination
    if dest_path.exists():
        dest_path.unlink()
    shutil.move(str(temp_file), str(dest_path))
    logger.info(f"Successfully staged model checkpoint to {dest_path} (SHA-256: {computed_hash})")
    return dest_path


def ensure_model_checkpoint(
    target_path: Optional[Union[Path, str]] = None,
    url: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Ensures model checkpoint exists locally and passes SHA-256 verification.

    1. If target_path exists and not force_download:
       - Computes SHA-256.
       - If valid: returns target_path immediately without re-downloading.
       - If invalid: removes corrupted file and raises ChecksumMismatchError.
    2. If target_path does not exist (or was removed):
       - Downloads automatically from url.
       - Verifies SHA-256.
       - Saves to target_path and returns it.
    """
    default_url, default_sha256, default_path = get_model_config()

    url = (url or default_url).strip()
    expected_sha256 = (expected_sha256 or default_sha256).strip().lower()

    if target_path is None:
        resolved_path = default_path
    else:
        resolved_path = Path(target_path)
        if not resolved_path.is_absolute():
            resolved_path = REPO_ROOT / resolved_path

    if resolved_path.is_file() and not force_download:
        actual_hash = calculate_sha256(resolved_path)
        if actual_hash == expected_sha256:
            logger.info(f"Using verified existing model checkpoint: {resolved_path}")
            return resolved_path
        else:
            logger.error(
                f"Existing model checkpoint corrupted at '{resolved_path}'. "
                f"Expected SHA-256 '{expected_sha256}', found '{actual_hash}'. Removing corrupted file."
            )
            resolved_path.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"Corrupted model checkpoint at '{resolved_path}'. "
                f"Expected SHA-256: '{expected_sha256}', Got: '{actual_hash}'. "
                "Corrupted file was removed. Restart or re-run to trigger automatic clean download."
            )

    # Checkpoint is missing or force_download requested
    logger.info(f"Checkpoint not present at {resolved_path}. Initiating automatic download from {url}...")
    return download_file(
        url=url,
        dest_path=resolved_path,
        expected_sha256=expected_sha256,
    )
