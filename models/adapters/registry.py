"""Dynamic Model Registry for 3D Reconstruction Adapters."""

from typing import Dict, List, Optional
from models.adapters.base import ModelCapabilities, ReconstructionModelAdapter
from models.adapters.hunyuan_mirror_adapter import HunyuanMirrorAdapter
from models.adapters.roomai_composite import RoomAICompositeAdapter
from utils.logging_config import get_logger

logger = get_logger("ReconstructionRegistry")


class ReconstructionModelRegistry:
    """Registry maintaining available 3D reconstruction model adapters."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self._adapters: Dict[str, ReconstructionModelAdapter] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Registers system default adapters."""
        # 1. Primary verified production composite model
        composite = RoomAICompositeAdapter(mock_mode=self.mock_mode)
        self._adapters["roomai_composite_v1"] = composite

        # 2. Next-generation foundation model (staged)
        hunyuan = HunyuanMirrorAdapter(mock_mode=self.mock_mode)
        self._adapters["hunyuan_world_mirror_1b"] = hunyuan

    def list_models(self) -> List[ModelCapabilities]:
        """Returns the capabilities of all registered models."""
        return [adapter.get_capabilities() for adapter in self._adapters.values()]

    def get_adapter(self, model_id: str) -> ReconstructionModelAdapter:
        """Retrieves adapter by model ID or raises KeyError."""
        if model_id not in self._adapters:
            available = list(self._adapters.keys())
            raise KeyError(f"Unknown reconstruction model '{model_id}'. Available models: {available}")
        return self._adapters[model_id]

    def get_default_model_id(self) -> str:
        """Returns the default production model identifier."""
        return "roomai_composite_v1"

    def register_adapter(self, model_id: str, adapter: ReconstructionModelAdapter) -> None:
        """Dynamically registers an adapter."""
        self._adapters[model_id] = adapter
        logger.info(f"Registered model adapter '{model_id}'.")


# Global singleton instance
RECONSTRUCTION_REGISTRY = ReconstructionModelRegistry(mock_mode=False)

