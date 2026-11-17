"""RoomAI Job Store (SQLite + SQLAlchemy).

Persistent job storage with full lifecycle tracking.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, String, Text, Float, Integer, DateTime,
    Index, select, update, delete
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.logging import get_logger


logger = get_logger("JobStore")

Base = declarative_base()


class JobModel(Base):
    """SQLAlchemy model for reconstruction jobs."""
    __tablename__ = "jobs"

    job_id = Column(String(64), primary_key=True)
    status = Column(String(32), nullable=False, default="QUEUED", index=True)
    stage = Column(String(256), nullable=False, default="Job queued for processing")
    progress = Column(Float, nullable=False, default=0.0)
    model_id = Column(String(64), nullable=False)
    view_count = Column(Integer, nullable=False, default=1)
    image_paths = Column(Text, nullable=False, default="[]")
    options = Column(Text, nullable=False, default="{}")
    error = Column(Text, nullable=True)
    warnings = Column(Text, nullable=False, default="[]")
    timing = Column(Text, nullable=False, default="{}")
    results_summary = Column(Text, nullable=True)
    artifacts = Column(Text, nullable=False, default="{}")
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)

    __table_args__ = (
        Index("idx_jobs_created_at", "created_at"),
        Index("idx_jobs_model_id", "model_id"),
    )


class JobStore:
    """Thread-safe persistent job store using SQLite/SQLAlchemy."""

    def __init__(self, db_path: Optional[str] = None):
        db_path = db_path or settings.database_url
        if db_path.startswith("sqlite:///"):
            file_path = db_path[10:]  # Remove "sqlite:///"
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            db_path,
            connect_args={"check_same_thread": False} if "sqlite" in db_path else {},
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables
        Base.metadata.create_all(bind=self.engine)

        logger.info(f"Job store initialized at {db_path}")

    @contextmanager
    def session(self) -> Session:
        """Provide a transactional scope."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _row_to_dict(self, row: JobModel) -> Dict[str, Any]:
        """Convert SQLAlchemy row to dictionary."""
        return {
            "job_id": row.job_id,
            "status": row.status,
            "stage": row.stage,
            "progress": float(row.progress),
            "model_id": row.model_id,
            "view_count": int(row.view_count),
            "image_paths": json.loads(row.image_paths) if row.image_paths else [],
            "options": json.loads(row.options) if row.options else {},
            "error": json.loads(row.error) if row.error else None,
            "warnings": json.loads(row.warnings) if row.warnings else [],
            "timing": json.loads(row.timing) if row.timing else {},
            "results_summary": json.loads(row.results_summary) if row.results_summary else None,
            "artifacts": json.loads(row.artifacts) if row.artifacts else {},
            "created_at": float(row.created_at),
            "updated_at": float(row.updated_at),
        }

    def create_job(
        self,
        job_id: str,
        model_id: str,
        image_paths: List[str],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new job record."""
        now = time.time()
        timing = {
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "elapsed_ms": None,
        }

        job = JobModel(
            job_id=job_id,
            status="QUEUED",
            stage="Job queued for processing",
            progress=0.0,
            model_id=model_id,
            view_count=len(image_paths),
            image_paths=json.dumps(image_paths),
            options=json.dumps(options or {}),
            timing=json.dumps(timing),
            created_at=now,
            updated_at=now,
        )

        with self.session() as session:
            session.add(job)
            session.flush()
            return self._row_to_dict(job)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        error: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        timing_updates: Optional[Dict[str, Any]] = None,
        results_summary: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update job fields atomically."""
        now = time.time()

        with self.session() as session:
            job = session.get(JobModel, job_id)
            if not job:
                return None

            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = float(progress)

            # Merge timing
            if timing_updates:
                current_timing = json.loads(job.timing) if job.timing else {}
                current_timing.update(timing_updates)
                job.timing = json.dumps(current_timing)

            # Merge warnings
            if warnings is not None:
                job.warnings = json.dumps(warnings)

            if error is not None:
                job.error = json.dumps(error)

            if results_summary is not None:
                job.results_summary = json.dumps(results_summary)

            if artifacts is not None:
                current_artifacts = json.loads(job.artifacts) if job.artifacts else {}
                current_artifacts.update(artifacts)
                job.artifacts = json.dumps(current_artifacts)

            job.updated_at = now
            session.flush()
            return self._row_to_dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        with self.session() as session:
            job = session.get(JobModel, job_id)
            if not job:
                return None
            return self._row_to_dict(job)

    def list_jobs(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        with self.session() as session:
            query = select(JobModel).order_by(JobModel.created_at.desc())
            if status:
                query = query.where(JobModel.status == status)
            query = query.offset(offset).limit(limit)

            jobs = session.execute(query).scalars().all()
            return [self._row_to_dict(job) for job in jobs]

    def count_jobs(self, status: Optional[str] = None) -> int:
        """Count jobs."""
        with self.session() as session:
            query = select(JobModel)
            if status:
                query = query.where(JobModel.status == status)
            return session.execute(query).scalar() or 0

    def delete_job(self, job_id: str) -> bool:
        """Delete a job record."""
        with self.session() as session:
            job = session.get(JobModel, job_id)
            if not job:
                return False
            session.delete(job)
            return True

    def recover_interrupted_jobs(self) -> int:
        """Mark running jobs from previous session as FAILED."""
        interrupted_statuses = ("QUEUED", "VALIDATING", "PROCESSING", "INFERENCE",
                               "GEOMETRY_POSTPROCESSING", "EXPORTING")

        with self.session() as session:
            jobs = session.execute(
                select(JobModel).where(JobModel.status.in_(interrupted_statuses))
            ).scalars().all()

            count = 0
            now = time.time()
            error_json = json.dumps({
                "code": "SERVER_RESTARTED",
                "message": "Job interrupted by server restart",
                "remediation": "Resubmit the reconstruction job",
            })

            for job in jobs:
                job.status = "FAILED"
                job.stage = "Interrupted by server restart"
                job.error = error_json
                job.updated_at = now
                count += 1

            if count > 0:
                logger.warning(f"Recovered {count} interrupted jobs")

            return count

    def get_job_artifacts(self, job_id: str) -> Dict[str, Any]:
        """Get artifact paths for a completed job."""
        job = self.get_job(job_id)
        if not job:
            return {}
        return job.get("artifacts", {})


# Singleton
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store