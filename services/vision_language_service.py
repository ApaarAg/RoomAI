"""Vision-Language Model service for room Q&A."""

from typing import Any, Dict, Optional
from PIL import Image

from models.interfaces import BaseVisionLanguage, ModelNotImplementedError
from models.vision_language_model import (
    CustomVisionLanguageModel,
    MockVisionLanguageModel,
)
from utils.logging_config import get_logger

logger = get_logger("VisionLanguageService")


class VisionLanguageService:
    """Coordinates visual question answering on room images."""

    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        self.mock_mode = mock_mode
        self.config = config or {}
        self.model: BaseVisionLanguage = (
            MockVisionLanguageModel(self.config)
            if mock_mode
            else CustomVisionLanguageModel(self.config)
        )

    def run(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Queries the VLM with a natural language question about the room image."""
        try:
            if not question or not question.strip():
                return {
                    "success": False,
                    "response": None,
                    "error_message": "Question cannot be empty.",
                    "is_implemented": True,
                }

            response = self.model.ask(image, question.strip())
            return {
                "success": True,
                "response": response,
                "error_message": None,
                "is_implemented": True,
            }
        except ModelNotImplementedError as exc:
            logger.warning(f"VLM model not implemented: {exc}")
            return {
                "success": False,
                "response": None,
                "error_message": str(exc),
                "is_implemented": False,
            }
        except Exception as exc:
            logger.error(f"Error executing VLM service: {exc}", exc_info=True)
            return {
                "success": False,
                "response": None,
                "error_message": f"Vision-Language query failed: {str(exc)}",
                "is_implemented": True,
            }
