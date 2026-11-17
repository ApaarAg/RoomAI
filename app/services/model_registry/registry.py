"""RoomAI Model Registry Service.

Loads and manages the immutable model registry from YAML.
Provides model lookup, availability checking, and version resolution.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import torch

from app.core.config import settings
from app.core.exceptions import ModelError, NotFoundError
from app.core.logging import get_logger


logger = get_logger("ModelRegistryService")


class ModelRegistryService:
    """Service for accessing the immutable model registry."""

    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path or settings.model_registry_path)
        self._registry: Dict[str, Any] = {}
        self._aliases: Dict[str, str] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from YAML file."""
        if not self.registry_path.exists():
            raise ModelError(
                f"Model registry not found at {self.registry_path}",
                details={"path": str(self.registry_path)},
            )

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            raise ModelError(
                f"Failed to parse model registry YAML",
                details={"path": str(self.registry_path), "error": str(exc)},
            )

        self._registry = data.get("models", {})
        self._aliases = data.get("active_aliases", {})
        self._registry_version = data.get("registry_version", "1.0.0")

        logger.info(
            "Model registry loaded",
            extra={
                "version": self._registry_version,
                "models": list(self._registry.keys()),
                "aliases": self._aliases,
            },
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models with key metadata."""
        return [
            {
                "model_id": model_id,
                "name": info.get("name"),
                "version": info.get("version"),
                "status": info.get("status"),
                "role": info.get("role"),
                "checkpoint_path": info.get("checkpoint_path"),
                "sha256": info.get("sha256"),
                "file_size_bytes": info.get("file_size_bytes"),
                "parameter_count": info.get("parameter_count"),
                "capabilities": info.get("capabilities"),
            }
            for model_id, info in self._registry.items()
        ]

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get full model specification by ID."""
        if model_id not in self._registry:
            # Try alias resolution
            resolved = self._aliases.get(model_id)
            if resolved and resolved in self._registry:
                model_id = resolved
            else:
                raise NotFoundError("Model", model_id)

        return self._registry[model_id]

    def resolve_model_id(self, model_id: str) -> str:
        """Resolve alias to concrete model ID."""
        return self._aliases.get(model_id, model_id)

    def get_checkpoint_path(self, model_id: str) -> Path:
        """Get absolute path to model checkpoint."""
        model = self.get_model(model_id)
        rel_path = model.get("checkpoint_path")
        if not rel_path:
            raise ModelError(f"Model '{model_id}' has no checkpoint path")

        abs_path = Path(rel_path)
        if not abs_path.is_absolute():
            # Resolve relative to repo root
            repo_root = Path(__file__).resolve().parents[3]
            abs_path = repo_root / rel_path

        return abs_path

    def verify_checkpoint(self, model_id: str) -> Tuple[bool, Optional[str]]:
        """Verify checkpoint exists and matches SHA-256."""
        model = self.get_model(model_id)
        expected_sha = model.get("sha256")
        checkpoint_path = self.get_checkpoint_path(model_id)

        if not checkpoint_path.exists():
            return False, f"Checkpoint not found: {checkpoint_path}"

        if not expected_sha:
            return True, None  # No hash to verify

        # Compute SHA-256
        import hashlib
        hasher = hashlib.sha256()
        try:
            with open(checkpoint_path, "rb") as f:
                for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    hasher.update(chunk)
        except Exception as exc:
            return False, f"Failed to read checkpoint: {exc}"

        actual_sha = hasher.hexdigest()
        if actual_sha != expected_sha:
            return False, f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"

        return True, None

    def is_model_available(self, model_id: str) -> Tuple[bool, Optional[str]]:
        """Check if model checkpoint exists and hardware requirements are met."""
        model = self.get_model(model_id)

        # Verify checkpoint
        ok, reason = self.verify_checkpoint(model_id)
        if not ok:
            return False, reason

        # Check GPU availability if required
        caps = model.get("capabilities", {})
        device = caps.get("device", "cuda")
        memory_required = caps.get("memory_required_mb", 0)

        if device == "cuda" and not torch.cuda.is_available():
            return False, "CUDA not available but model requires GPU"

        if device == "cuda" and torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            if free_mb < memory_required:
                return False, f"Insufficient VRAM: {free_mb:.0f} MB free, {memory_required} MB required"

        return True, None

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model status including availability."""
        model = self.get_model(model_id)
        available, reason = self.is_model_available(model_id)
        verified, verify_reason = self.verify_checkpoint(model_id)

        return {
            "model_id": model_id,
            "name": model.get("name"),
            "version": model.get("version"),
            "status": model.get("status"),
            "available": available,
            "checkpoint_verified": verified,
            "unavailable_reason": reason,
            "verification_error": verify_reason,
            "checkpoint_path": str(self.get_checkpoint_path(model_id)),
            "capabilities": model.get("capabilities"),
        }

    def get_alias_target(self, alias: str) -> Optional[str]:
        """Get the model ID an alias points to."""
        return self._aliases.get(alias)

    def get_all_aliases(self) -> Dict[str, str]:
        """Get all alias mappings."""
        return dict(self._aliases)

    def get_registry_version(self) -> str:
        """Get registry schema version."""
        return self._registry_version

    def reload(self) -> None:
        """Reload registry from disk (for development)."""
        self._load_registry()


# Singleton instance
_model_registry_service: Optional[ModelRegistryService] = None


def get_model_registry_service() -> ModelRegistryService:
    """Get or create the global model registry service."""
    global _model_registry_service
    if _model_registry_service is None:
        _model_registry_service = ModelRegistryService()
    return _model_registry_service