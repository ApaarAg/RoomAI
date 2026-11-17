"""RoomAI Backend Core Configuration.

Centralized settings loaded from YAML with environment override support.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # App
    app_name: str = "RoomAI Backend"
    app_version: str = "1.4.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # Database
    database_url: str = Field(default="sqlite:///artifacts/jobs.db", alias="DATABASE_URL")

    # GPU/Inference
    device: str = Field(default="cuda", alias="DEVICE")
    max_concurrent_jobs: int = Field(default=1, alias="MAX_CONCURRENT_JOBS")
    gpu_memory_fraction: float = Field(default=0.85, alias="GPU_MEMORY_FRACTION")

    # Storage
    artifacts_root: str = Field(default="artifacts", alias="ARTIFACTS_ROOT")
    uploads_dir: str = Field(default="uploads", alias="UPLOADS_DIR")
    jobs_dir: str = Field(default="jobs", alias="JOBS_DIR")

    # Model Registry
    model_registry_path: str = Field(default="models/registry/model_registry.yaml", alias="MODEL_REGISTRY_PATH")
    default_model_id: str = Field(default="roomai_composite_v1", alias="DEFAULT_MODEL_ID")

    # Timeouts (seconds)
    job_timeout: int = Field(default=3600, alias="JOB_TIMEOUT")
    inference_timeout: int = Field(default=300, alias="INFERENCE_TIMEOUT")
    upload_timeout: int = Field(default=60, alias="UPLOAD_TIMEOUT")

    # Monitoring
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file safely."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to load config from {path}: {exc}")


# Global settings instance
settings = Settings()

# Load legacy model config for backwards compatibility
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG = load_yaml_config(REPO_ROOT / "config" / "model_config.yaml")
APP_CONFIG = load_yaml_config(REPO_ROOT / "config" / "app.yaml")


def get_model_config(section: str) -> Dict[str, Any]:
    """Get model configuration section with defaults."""
    return MODEL_CONFIG.get(section, {})


def get_app_config(section: str) -> Dict[str, Any]:
    """Get app configuration section with defaults."""
    return APP_CONFIG.get(section, {})