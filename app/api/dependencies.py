"""RoomAI API Dependencies.

FastAPI dependency injection for services, configuration, and authentication.
"""

from typing import Optional, Generator
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings, get_model_config
from app.core.exceptions import UnavailableError, ValidationError
from app.services.model_registry.registry import ModelRegistryService
from app.services.inference.engine import InferenceEngine
from app.services.preprocessing.pipeline import PreprocessingPipeline
from app.storage.artifacts import ArtifactStore
from app.monitoring.metrics import MetricsCollector
from app.workers.gpu_worker import get_gpu_worker, GPUWorker
from app.services.job_store import get_job_store, JobStore


# Database
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


# Services (singletons)
_model_registry: Optional[ModelRegistryService] = None
_inference_engine: Optional[InferenceEngine] = None
_preprocessing_pipeline: Optional[PreprocessingPipeline] = None
_artifact_store: Optional[ArtifactStore] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_model_registry() -> ModelRegistryService:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistryService(settings.model_registry_path)
    return _model_registry


def get_inference_engine() -> InferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine(get_model_registry())
    return _inference_engine


def get_preprocessing_pipeline() -> PreprocessingPipeline:
    global _preprocessing_pipeline
    if _preprocessing_pipeline is None:
        _preprocessing_pipeline = PreprocessingPipeline()
    return _preprocessing_pipeline


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(settings.artifacts_root)
    return _artifact_store


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Authentication (placeholder for future)
async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Extract user from Authorization header (placeholder)."""
    if not authorization:
        return None
    # TODO: Implement JWT/API key validation
    return {"user_id": "anonymous", "scopes": []}


def require_model_available(model_id: str, registry: ModelRegistryService = Depends(get_model_registry)):
    """Dependency that verifies model is available before endpoint execution."""
    available, reason = registry.is_model_available(model_id)
    if not available:
        raise UnavailableError(
            f"Model '{model_id}' is not available",
            reason=reason,
        )
    return model_id


def validate_pagination(skip: int = 0, limit: int = 20) -> tuple[int, int]:
    """Validate pagination parameters."""
    if skip < 0:
        raise ValidationError("skip must be non-negative", field="skip")
    if limit < 1 or limit > 100:
        raise ValidationError("limit must be between 1 and 100", field="limit")
    return skip, limit