"""Upload Endpoints."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status, Depends
from pathlib import Path
from PIL import Image
import io

from app.api.dependencies import get_artifact_store
from app.storage.artifacts import ArtifactStore
from app.api.schemas import UploadResponse


router = APIRouter(prefix="/api/v1/uploads", tags=["Uploads"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    artifact_store: ArtifactStore = Depends(get_artifact_store),
):
    """Upload an indoor scene photograph."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    # Read and validate
    content = await file.read()

    # Validate with PIL
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
        width, height = img.size
        format_str = img.format or Path(file.filename).suffix.lstrip(".").upper()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {exc}")

    # Store
    try:
        result = artifact_store.store_upload(content, file.filename)
        result.update({
            "width": width,
            "height": height,
            "format": format_str,
        })
        return UploadResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {exc}")


@router.get("/{upload_id}", response_model=UploadResponse)
def get_upload_metadata(
    upload_id: str,
    artifact_store: ArtifactStore = Depends(get_artifact_store),
):
    """Get upload metadata."""
    path = artifact_store.get_upload_path(upload_id)
    if not path or not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Upload '{upload_id}' not found.")

    try:
        img = Image.open(path)
        width, height = img.size
        size_bytes = path.stat().st_size

        return UploadResponse(
            status="success",
            upload_id=upload_id,
            filename=path.name,
            original_filename=path.name,
            width=width,
            height=height,
            format=img.format or path.suffix.lstrip(".").upper(),
            size_bytes=size_bytes,
            sha256=upload_id.replace("upl_", ""),
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to read image: {exc}")