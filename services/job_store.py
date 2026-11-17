"""Persistent SQLite Job Store for RoomAI 3D Reconstruction Jobs."""

import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional
from utils.logging_config import get_logger

logger = get_logger("JobStore")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "artifacts" / "jobs.db"


class JobStore:
    """Thread-safe persistent SQLite store for reconstruction job metadata and state."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Opens SQLite connection with WAL mode and 10s timeout."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        """Initializes database schema if not present."""
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0.0,
                    model_id TEXT NOT NULL,
                    view_count INTEGER NOT NULL DEFAULT 1,
                    image_paths TEXT NOT NULL DEFAULT '[]',
                    options TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    warnings TEXT NOT NULL DEFAULT '[]',
                    timing TEXT NOT NULL DEFAULT '{}',
                    results_summary TEXT,
                    artifacts TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
            conn.commit()

    def create_job(
        self,
        job_id: str,
        model_id: str,
        image_paths: List[str],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates a new job record with initial state 'QUEUED'."""
        now = time.time()
        opts_json = json.dumps(options or {})
        paths_json = json.dumps(image_paths)
        timing_json = json.dumps({"created_at": now, "started_at": None, "completed_at": None, "elapsed_ms": None})

        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, stage, progress, model_id, view_count,
                    image_paths, options, error, warnings, timing, results_summary, artifacts,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "QUEUED",
                    "Job queued for processing",
                    0.0,
                    model_id,
                    len(image_paths),
                    paths_json,
                    opts_json,
                    None,
                    "[]",
                    timing_json,
                    None,
                    "{}",
                    now,
                    now,
                ),
            )
            conn.commit()

        return self.get_job(job_id) or {}

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
        """Updates job state fields atomically."""
        now = time.time()
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None

            # Merge timing
            current_timing = json.loads(row["timing"]) if row["timing"] else {}
            if timing_updates:
                current_timing.update(timing_updates)

            # Merge warnings
            current_warnings = json.loads(row["warnings"]) if row["warnings"] else []
            if warnings is not None:
                current_warnings = warnings

            # Merge artifacts
            current_artifacts = json.loads(row["artifacts"]) if row["artifacts"] else {}
            if artifacts is not None:
                current_artifacts.update(artifacts)

            new_status = status if status is not None else row["status"]
            new_stage = stage if stage is not None else row["stage"]
            new_progress = float(progress) if progress is not None else float(row["progress"])
            new_error = json.dumps(error) if error is not None else row["error"]
            new_results = json.dumps(results_summary) if results_summary is not None else row["results_summary"]

            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    stage = ?,
                    progress = ?,
                    error = ?,
                    warnings = ?,
                    timing = ?,
                    results_summary = ?,
                    artifacts = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    new_status,
                    new_stage,
                    new_progress,
                    new_error,
                    json.dumps(current_warnings),
                    json.dumps(current_timing),
                    new_results,
                    json.dumps(current_artifacts),
                    now,
                    job_id,
                ),
            )
            conn.commit()

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetches parsed job dictionary by ID."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None

            return {
                "job_id": row["job_id"],
                "status": row["status"],
                "stage": row["stage"],
                "progress": float(row["progress"]),
                "model_id": row["model_id"],
                "view_count": int(row["view_count"]),
                "image_paths": json.loads(row["image_paths"]),
                "options": json.loads(row["options"]),
                "error": json.loads(row["error"]) if row["error"] else None,
                "warnings": json.loads(row["warnings"]),
                "timing": json.loads(row["timing"]),
                "results_summary": json.loads(row["results_summary"]) if row["results_summary"] else None,
                "artifacts": json.loads(row["artifacts"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent jobs sorted by creation timestamp descending."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            jobs = []
            for row in rows:
                jobs.append({
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "stage": row["stage"],
                    "progress": float(row["progress"]),
                    "model_id": row["model_id"],
                    "view_count": int(row["view_count"]),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                    "error": json.loads(row["error"]) if row["error"] else None,
                })
            return jobs

    def recover_interrupted_jobs(self) -> int:
        """Marks any running jobs from a previous server session as FAILED."""
        with self._lock, self._get_connection() as conn:
            interrupted_statuses = ("PROCESSING", "RECONSTRUCTING", "FUSING", "EXPORTING", "VALIDATING")
            now = time.time()
            error_json = json.dumps({
                "code": "SERVER_RESTARTED",
                "message": "Reconstruction job was interrupted by a server restart or process termination.",
                "remediation": "Resubmit the reconstruction job.",
            })
            cur = conn.execute(
                f"""
                UPDATE jobs SET
                    status = 'FAILED',
                    stage = 'Interrupted by server restart',
                    error = ?,
                    updated_at = ?
                WHERE status IN ({','.join(['?']*len(interrupted_statuses))})
                """,
                (error_json, now, *interrupted_statuses),
            )
            count = cur.rowcount
            conn.commit()
            if count > 0:
                logger.warning(f"Recovered {count} interrupted jobs and transitioned to FAILED.")
            return count


# Global persistent job store singleton
GLOBAL_JOB_STORE = JobStore()

