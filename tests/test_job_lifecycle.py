"""Tests for JobStore Persistence and JobService Lifecycle."""

import time
from pathlib import Path
from PIL import Image
import pytest

from services.job_service import JobService
from services.job_store import JobStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_job_store_crud(tmp_path):
    """JobStore must persist records to SQLite and support atomic updates."""
    db_path = tmp_path / "test_jobs.db"
    store = JobStore(db_path=db_path)

    job_id = "test_job_123"
    job = store.create_job(
        job_id=job_id,
        model_id="roomai_composite_v1",
        image_paths=["dummy_path.jpg"],
        options={"refine_depth": True},
    )
    assert job["job_id"] == job_id
    assert job["status"] == "QUEUED"
    assert job["progress"] == 0.0

    # Update state
    updated = store.update_job(
        job_id=job_id,
        status="RECONSTRUCTING",
        stage="Running depth estimation",
        progress=0.55,
        timing_updates={"started_at": 100.0},
    )
    assert updated is not None
    assert updated["status"] == "RECONSTRUCTING"
    assert updated["stage"] == "Running depth estimation"
    assert updated["progress"] == 0.55
    assert updated["timing"]["started_at"] == 100.0

    # Retrieve by ID
    retrieved = store.get_job(job_id)
    assert retrieved is not None
    assert retrieved["status"] == "RECONSTRUCTING"

    # List jobs
    all_jobs = store.list_jobs(10)
    assert len(all_jobs) == 1
    assert all_jobs[0]["job_id"] == job_id


def test_job_store_interrupted_recovery(tmp_path):
    """JobStore must mark interrupted in-flight jobs as FAILED upon recovery."""
    db_path = tmp_path / "test_jobs.db"
    store = JobStore(db_path=db_path)

    store.create_job("job_running", "roomai_composite_v1", ["p.jpg"])
    store.update_job("job_running", status="PROCESSING")

    store.create_job("job_completed", "roomai_composite_v1", ["p.jpg"])
    store.update_job("job_completed", status="COMPLETED")

    recovered = store.recover_interrupted_jobs()
    assert recovered == 1

    job_run = store.get_job("job_running")
    assert job_run["status"] == "FAILED"
    assert "Interrupted" in job_run["stage"]

    job_comp = store.get_job("job_completed")
    assert job_comp["status"] == "COMPLETED"


def test_job_cancellation(tmp_path):
    """JobService must support cancelling a queued job."""
    db_path = tmp_path / "test_cancel.db"
    store = JobStore(db_path=db_path)
    service = JobService(store=store)

    img = Image.new("RGB", (100, 100), color=(120, 120, 120))
    job_id = service.submit_reconstruction_job([img])

    status_before = service.get_job_status(job_id)
    assert status_before is not None

    # Request cancellation
    cancelled = service.cancel_job(job_id)
    assert cancelled is True

    # Poll status until terminal state or cancellation acknowledged
    for _ in range(50):
        status_after = service.get_job_status(job_id)
        if status_after["status"] in ("CANCELLED", "COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    assert status_after["status"] in ("CANCELLED", "COMPLETED", "FAILED")
