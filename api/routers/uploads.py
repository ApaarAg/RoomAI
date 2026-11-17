"""Image Upload and Ingestion Endpoints."""

from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas import UploadResponse
from services.upload_service import GLOBAL_UPLOAD_SERVICE

router = APIRouter(prefix="/api/v1/uploads", tags=["Uploads"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """Uploads, validates, and stores an indoor scene photograph.

    Enforces:
    - Supported extensions (.jpg, .jpeg, .png, .webp).
    - File size <= 20 MB.
    - Dimensions <= 8192px and <= 100M total pixels (Decompression Bomb protection).
    - Cryptographic SHA-256 deduplication and sanitized filenames.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    try:
        content = await file.read()
        res = GLOBAL_UPLOAD_SERVICE.save_upload(file_bytes=content, original_filename=file.filename)
        return UploadResponse(
            status="success",
            upload_id=res["upload_id"],
            filename=res["filename"],
            original_filename=res["original_filename"],
            width=res["width"],
            height=res["height"],
            format=res["format"],
            size_bytes=res["size_bytes"],
            sha256=res["sha256"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process upload: {exc}",
        )


@router.get("/{upload_id}", response_model=UploadResponse)
def get_upload_metadata(upload_id: str) -> UploadResponse:
    """Retrieves metadata for a previously uploaded image."""
    path = GLOBAL_UPLOAD_SERVICE.get_upload_path(upload_id)
    if not path or not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Upload '{upload_id}' not found.")

    try:
        from PIL import Image
        img = Image.open(path)
        w, h = img.size
        size_bytes = path.stat().st_size
        return UploadResponse(
            status="success",
            upload_id=upload_id,
            filename=path.name,
            original_filename=path.name,
            width=w,
            height=h,
            format=img.format or path.suffix.replace(".", "").upper(),
            size_bytes=size_bytes,
            sha256=upload_id.replace("upl_", ""),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to read image: {exc}")

