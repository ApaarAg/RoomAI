"""Semantic Segmentation service coordinating model execution and mask overlay rendering."""

from typing import Any, Dict, Optional
from PIL import Image

from models.interfaces import BaseSegmenter, ModelNotImplementedError
from models.segmentation_model import CustomRoomSegmenter, MockRoomSegmenter
from services.image_service import create_segmentation_overlay
from utils.logging_config import get_logger

logger = get_logger("SegmentationService")


class SegmentationService:
    """Coordinates room segmentation model inference and overlay blending."""

    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        self.mock_mode = mock_mode
        self.config = config or {} if mock_mode else config
        self.model: BaseSegmenter = (
            MockRoomSegmenter(self.config) if mock_mode else CustomRoomSegmenter(self.config)
        )

    def run(self, image: Image.Image) -> Dict[str, Any]:
        """Executes segmentation and blends the output overlay on the original image."""
        if image is None:
            return {
                "success": False,
                "response": None,
                "overlay_image": None,
                "mask_array": None,
                "error_message": "Input image cannot be None.",
                "is_implemented": True,
            }

        try:
            response, mask_array = self.model.predict(image)
            overlay_img = None
            if mask_array is not None:
                overlay_img = create_segmentation_overlay(image, mask_array)

            return {
                "success": True,
                "response": response,
                "overlay_image": overlay_img,
                "mask_array": mask_array,
                "error_message": None,
                "is_implemented": True,
            }
        except ModelNotImplementedError as exc:
            logger.warning(f"Segmentation model not implemented: {exc}")
            return {
                "success": False,
                "response": None,
                "overlay_image": None,
                "mask_array": None,
                "error_message": str(exc),
                "is_implemented": False,
            }
        except Exception as exc:
            logger.error(f"Error executing segmentation service: {exc}", exc_info=True)
            return {
                "success": False,
                "response": None,
                "overlay_image": None,
                "mask_array": None,
                "error_message": f"Segmentation failed: {str(exc)}",
                "is_implemented": True,
            }
