"""Export Endpoints."""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pathlib import Path
from typing import List

from app.api.dependencies import get_job_store, get_artifact_store
from app.services.job_store import JobStore
from app.storage.artifacts import ArtifactStore


router = APIRouter(prefix="/api/v1/exports", tags=["Exports"])


@router.get("/{job_id}")
def list_job_exports(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
):
    """List available exports for a job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "COMPLETED":
        return {
            "job_id": job_id,
            "status": job["status"],
            "exports": {},
            "message": f"Job is {job['status']}, exports not yet available.",
        }

    artifacts = job.get("artifacts", {})
    return {
        "job_id": job_id,
        "status": "COMPLETED",
        "exports": {
            fmt: {
                "format": fmt,
                "path": info["path"],
                "size_bytes": info["size_bytes"],
                "download_url": f"/api/v1/exports/{job_id}/{fmt}",
            }
            for fmt, info in artifacts.items()
        },
    }


@router.get("/{job_id}/{fmt}")
def download_export(
    job_id: str,
    fmt: str,
    job_store: JobStore = Depends(get_job_store),
):
    """Download specific export format."""
    fmt_clean = fmt.lower().strip()
    if fmt_clean not in ("glb", "ply"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Supported: glb, ply.",
        )

    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "COMPLETED":
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is {job['status']}. Exports only available when COMPLETED.",
        )

    artifacts = job.get("artifacts", {})
    if fmt_clean not in artifacts:
        raise HTTPException(status_code=404, detail=f"Export '{fmt_clean}' not found.")

    file_path = Path(artifacts[fmt_clean]["path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file missing.")

    from fastapi.responses import FileResponse
    media_type = "model/gltf-binary" if fmt_clean == "glb" else "application/octet-stream"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=f"roomai_{job_id}_{fmt_clean}.{fmt_clean}",
    )


@router.get("/{job_id}/scene-graph")
def get_scene_graph(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
):
    """Get scene graph JSON for a completed job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Scene graph would be stored in job artifacts or separate file
    # For now return results_summary
    return {
        "job_id": job_id,
        "status": job["status"],
        "scene_graph": job.get("results_summary"),
        "model_id": job.get("model_id"),
    }