"""Experimental 3D Scene Understanding service."""

from typing import Any, Dict, Optional
from PIL import Image

from models.interfaces import BaseScene3D, ModelNotImplementedError
from models.scene3d_model import CustomScene3DModel, MockScene3DModel
from services.image_service import create_depth_heatmap
from utils.logging_config import get_logger

logger = get_logger("Scene3DService")


class Scene3DService:
    """Coordinates experimental 3D scene estimation and depth heatmap rendering."""

    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        self.mock_mode = mock_mode
        self.config = config or {} if mock_mode else config
        # For mock mode, normalize config; for real mode, keep as-is
        mock_config = self.config if mock_mode else None

        # Check if explicitly disabled in config
        self._disabled = False
        if config is not None:
            if config.get("enabled", True) is False:
                self._disabled = True
                logger.info("Scene3DService initialized with enabled=false")

        if self._disabled:
            # Create a placeholder that won't attempt to load
            self.model: BaseScene3D = MockScene3DModel(mock_config)
        elif mock_mode:
            self.model = MockScene3DModel(mock_config)
        else:
            self.model = CustomScene3DModel(self.config)

    def run(self, image: Image.Image) -> Dict[str, Any]:
        """Executes 3D estimation and renders a depth heatmap."""
        if image is None:
            return {
                "success": False,
                "response": None,
                "depth_map": None,
                "heatmap_image": None,
                "point_cloud": None,
                "error_message": "Input image cannot be None.",
                "is_implemented": True,
                "is_disabled": False,
            }

        # If disabled, return appropriate message
        if self._disabled:
            return {
                "success": False,
                "response": None,
                "depth_map": None,
                "heatmap_image": None,
                "error_message": "Scene3D inference is disabled (enabled: false in config).",
                "is_implemented": True,
                "is_disabled": True,
            }

        try:
            depth_map, response = self.model.reconstruct(image)
            heatmap_img = None
            if depth_map is not None:
                heatmap_img = create_depth_heatmap(depth_map)

            # Extract point cloud from response
            point_cloud = response.point_cloud if hasattr(response, 'point_cloud') else None

            return {
                "success": True,
                "response": response,
                "depth_map": depth_map,
                "heatmap_image": heatmap_img,
                "point_cloud": point_cloud,
                "error_message": None,
                "is_implemented": True,
                "is_disabled": False,
                "timing": {
                    "inference_time_ms": getattr(response, 'inference_time_ms', 0.0),
                }
            }
        except ModelNotImplementedError as exc:
            logger.warning(f"3D model not implemented: {exc}")
            return {
                "success": False,
                "response": None,
                "depth_map": None,
                "heatmap_image": None,
                "error_message": str(exc),
                "is_implemented": False,
                "is_disabled": False,
            }
        except Exception as exc:
            logger.error(f"Error executing 3D service: {exc}", exc_info=True)
            return {
                "success": False,
                "response": None,
                "depth_map": None,
                "heatmap_image": None,
                "error_message": f"3D reconstruction failed: {str(exc)}",
                "is_implemented": True,
                "is_disabled": False,
            }