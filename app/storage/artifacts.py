"""RoomAI Artifact Storage.

Filesystem-based artifact storage with S3-compatible interface.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, BinaryIO
from dataclasses import dataclass, asdict
import uuid


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""
    artifact_id: str
    job_id: str
    file_type: str  # "input", "intermediate", "output", "export"
    filename: str
    storage_path: str
    size_bytes: int
    sha256: str
    created_at: str
    model_version: Optional[str] = None
    validity_status: str = "valid"  # "valid", "corrupted", "missing"
    mime_type: Optional[str] = None
    description: Optional[str] = None


class ArtifactStore:
    """Local filesystem artifact storage with metadata tracking."""

    def __init__(self, root_path: str = "artifacts"):
        self.root = Path(root_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.uploads_dir = self.root / "uploads"
        self.jobs_dir = self.root / "jobs"
        self.metadata_file = self.root / "artifact_metadata.json"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_index: Dict[str, ArtifactMetadata] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata index from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                for item in data:
                    meta = ArtifactMetadata(**item)
                    self._metadata_index[meta.artifact_id] = meta
            except Exception:
                pass  # Start fresh if corrupted

    def _save_metadata(self) -> None:
        """Save metadata index to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump([asdict(m) for m in self._metadata_index.values()], f, indent=2)
        except Exception:
            pass

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 of file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _compute_sha256_bytes(self, data: bytes) -> str:
        """Compute SHA-256 of bytes."""
        return hashlib.sha256(data).hexdigest()

    def store_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store an uploaded image and return metadata."""
        # Generate upload ID
        upload_id = f"upl_{uuid.uuid4().hex[:12]}"
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = ".jpg"
        filename = f"{upload_id}{ext}"

        # Save file
        file_path = self.uploads_dir / filename
        file_path.write_bytes(file_bytes)

        # Metadata
        sha256 = self._compute_sha256_bytes(file_bytes)
        size = len(file_bytes)

        metadata = ArtifactMetadata(
            artifact_id=upload_id,
            job_id=job_id or "unassigned",
            file_type="input",
            filename=filename,
            storage_path=str(file_path.relative_to(self.root)),
            size_bytes=size,
            sha256=sha256,
            created_at=datetime.utcnow().isoformat() + "Z",
            mime_type=self._guess_mime(ext),
            description=f"Uploaded image: {original_filename}",
        )

        self._metadata_index[upload_id] = metadata
        self._save_metadata()

        return {
            "upload_id": upload_id,
            "filename": filename,
            "original_filename": original_filename,
            "width": 0,  # Would need PIL to get dimensions
            "height": 0,
            "format": ext.lstrip(".").upper(),
            "size_bytes": size,
            "sha256": sha256,
        }

    def _guess_mime(self, ext: str) -> str:
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        return mime_map.get(ext.lower(), "application/octet-stream")

    def load_upload(self, upload_id: str) -> Optional[Path]:
        """Get path to uploaded file."""
        meta = self._metadata_index.get(upload_id)
        if not meta or meta.file_type != "input":
            return None
        path = self.root / meta.storage_path
        return path if path.exists() else None

    def get_upload_path(self, upload_id: str) -> Optional[Path]:
        """Get absolute path to upload."""
        return self.load_upload(upload_id)

    def get_job_dir(self, job_id: str) -> Path:
        """Get job artifact directory."""
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def store_job_artifact(
        self,
        job_id: str,
        file_path: Path,
        file_type: str,
        model_version: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Register a job output artifact."""
        artifact_id = f"art_{job_id}_{uuid.uuid4().hex[:8]}"

        # Copy to job directory if not already there
        dest_path = self.get_job_dir(job_id) / file_path.name
        if file_path != dest_path:
            shutil.copy2(file_path, dest_path)

        sha256 = self._compute_sha256(dest_path)
        size = dest_path.stat().st_size
        ext = dest_path.suffix.lower()

        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            job_id=job_id,
            file_type=file_type,
            filename=dest_path.name,
            storage_path=str(dest_path.relative_to(self.root)),
            size_bytes=size,
            sha256=sha256,
            created_at=datetime.utcnow().isoformat() + "Z",
            model_version=model_version,
            mime_type=self._guess_mime(ext),
            description=description,
        )

        self._metadata_index[artifact_id] = metadata
        self._save_metadata()

        return metadata

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """Get artifact metadata."""
        return self._metadata_index.get(artifact_id)

    def get_artifact_path(self, artifact_id: str) -> Optional[Path]:
        """Get absolute path to artifact file."""
        meta = self._metadata_index.get(artifact_id)
        if not meta:
            return None
        path = self.root / meta.storage_path
        return path if path.exists() else None

    def list_job_artifacts(self, job_id: str) -> List[ArtifactMetadata]:
        """List all artifacts for a job."""
        return [m for m in self._metadata_index.values() if m.job_id == job_id]

    def list_uploads(self, limit: int = 100) -> List[ArtifactMetadata]:
        """List recent uploads."""
        uploads = [m for m in self._metadata_index.values() if m.file_type == "input"]
        uploads.sort(key=lambda m: m.created_at, reverse=True)
        return uploads[:limit]

    def validate_artifact(self, artifact_id: str) -> bool:
        """Verify artifact file exists and hash matches."""
        meta = self._metadata_index.get(artifact_id)
        if not meta:
            return False

        path = self.root / meta.storage_path
        if not path.exists():
            meta.validity_status = "missing"
            self._save_metadata()
            return False

        actual_sha = self._compute_sha256(path)
        if actual_sha != meta.sha256:
            meta.validity_status = "corrupted"
            self._save_metadata()
            return False

        meta.validity_status = "valid"
        self._save_metadata()
        return True

    def cleanup_job(self, job_id: str) -> int:
        """Remove all artifacts for a job (cleanup)."""
        artifacts = self.list_job_artifacts(job_id)
        count = 0
        for meta in artifacts:
            path = self.root / meta.storage_path
            if path.exists():
                path.unlink()
            del self._metadata_index[meta.artifact_id]
            count += 1
        self._save_metadata()

        # Remove job directory if empty
        job_dir = self.jobs_dir / job_id
        if job_dir.exists() and not any(job_dir.iterdir()):
            job_dir.rmdir()

        return count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        total_size = 0
        total_files = 0
        by_type = {}

        for meta in self._metadata_index.values():
            total_size += meta.size_bytes
            total_files += 1
            by_type[meta.file_type] = by_type.get(meta.file_type, 0) + 1

        return {
            "root": str(self.root),
            "total_artifacts": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_type": by_type,
        }


# Singleton
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store