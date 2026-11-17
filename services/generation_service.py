"""Generative Design Redesign service."""

from typing import Any, Dict, Optional
from PIL import Image

from models.generation_model import CustomDesignGenerator, MockDesignGenerator
from models.interfaces import BaseGenerator, ModelNotImplementedError
from schemas.requests import DesignRequest
from services.image_service import create_side_by_side_comparison
from utils.logging_config import get_logger

logger = get_logger("GenerationService")


class GenerationService:
    """Coordinates generative AI interior redesign workflows and side-by-side rendering."""

    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        self.mock_mode = mock_mode
        self.config = config or {}
        self.model: BaseGenerator = (
            MockDesignGenerator(self.config) if mock_mode else CustomDesignGenerator(self.config)
        )

    def run(self, image: Image.Image, request: DesignRequest) -> Dict[str, Any]:
        """Executes design generation and produces side-by-side comparison image."""
        try:
            generated_img, response = self.model.generate(image, request)
            comparison_img = create_side_by_side_comparison(
                image, generated_img, "Original Room", f"Redesign ({request.target_style})"
            )

            return {
                "success": True,
                "response": response,
                "generated_image": generated_img,
                "comparison_image": comparison_img,
                "error_message": None,
                "is_implemented": True,
            }
        except ModelNotImplementedError as exc:
            logger.warning(f"Generation model not implemented: {exc}")
            return {
                "success": False,
                "response": None,
                "generated_image": None,
                "comparison_image": None,
                "error_message": str(exc),
                "is_implemented": False,
            }
        except Exception as exc:
            logger.error(f"Error executing generation service: {exc}", exc_info=True)
            return {
                "success": False,
                "response": None,
                "generated_image": None,
                "comparison_image": None,
                "error_message": f"Design generation failed: {str(exc)}",
                "is_implemented": True,
            }
