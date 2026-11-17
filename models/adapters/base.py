"""Abstract Base Class and Capabilities Schema for 3D Reconstruction Model Adapters."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from pydantic import BaseModel, Field

from schemas.scene import SceneGraph


class ModelCapabilities(BaseModel):
    """Formal capabilities declaration for a 3D reconstruction model adapter."""
    model_id: str = Field(description="Unique machine-readable identifier (e.g. 'roomai_composite_v1')")
    name: str = Field(description="Human-readable model name")
    version: str = Field(description="Semantic version string")
    status: str = Field(description="Operational status: 'PRODUCTION_READY', 'EXPERIMENTAL', 'STAGED', 'UNAVAILABLE'")
    
    # Supported modalities
    supports_single_view: bool = Field(default=True, description="Whether single-image reconstruction is supported")
    supports_multi_view: bool = Field(default=False, description="Whether multi-image sequence reconstruction is supported")
    min_views: int = Field(default=1, description="Minimum number of input images")
    max_views: int = Field(default=1, description="Maximum number of input images")
    
    # Feature outputs
    supports_metric_depth: bool = Field(default=True, description="Whether outputs are calibrated in physical meters")
    supports_surface_normals: bool = Field(default=False, description="Whether dense surface normal maps are produced")
    supports_3dgs: bool = Field(default=False, description="Whether 3D Gaussian Splatting parameters are generated")
    supports_mesh_glb: bool = Field(default=True, description="Whether binary glTF 2.0 GLB export is supported")
    supports_point_cloud_ply: bool = Field(default=True, description="Whether ASCII PLY point cloud export is supported")
    
    # Operational constraints
    device: str = Field(default="cuda", description="Target hardware execution device")
    memory_required_mb: int = Field(default=1500, description="Minimum free VRAM required in MB")
    description: str = Field(default="", description="Detailed architectural summary and intended usage")


class ReconstructionModelAdapter(ABC):
    """Abstract interface governing all 3D reconstruction models in RoomAI."""

    @abstractmethod
    def initialize(self) -> None:
        """Loads weights, validates checkpoints, and compiles inference pipelines."""
        pass

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Returns the formal capabilities and operational limits of this adapter."""
        pass

    @abstractmethod
    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Checks if checkpoint files exist and required hardware resources are present.
        
        Returns:
            Tuple of (is_available: bool, reason_if_unavailable: Optional[str])
        """
        pass

    @abstractmethod
    def validate_inputs(self, images: List[Image.Image]) -> Tuple[bool, Optional[str]]:
        """Validates that input images satisfy model view count, format, and resolution rules.
        
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        pass

    @abstractmethod
    def reconstruct(
        self,
        images: List[Image.Image],
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> SceneGraph:
        """Executes 3D reconstruction and returns a schema-compliant SceneGraph.

        Args:
            images: List of validated PIL Images (RGB).
            options: Optional inference parameters (e.g. refine_depth, downsample_points).
            progress_callback: Optional callable accepting (stage_name: str, progress_fraction: float).

        Returns:
            Fully populated, schema-compliant SceneGraph instance.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Frees model tensors and flushes GPU cache."""
        pass

