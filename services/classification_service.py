"""Room Classification and Style Recognition service."""

from typing import Any, Dict, Optional
from PIL import Image

from models.classification_model import CustomRoomClassifier, MockRoomClassifier
from models.interfaces import BaseClassifier, ModelNotImplementedError
from utils.logging_config import get_logger

logger = get_logger("ClassificationService")


class ClassificationService:
    """Coordinates room style/category classification model inference."""

    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        self.mock_mode = mock_mode
        self.config = config or {}
        self.model: BaseClassifier = (
            MockRoomClassifier(self.config) if mock_mode else CustomRoomClassifier(self.config)
        )

    def run(self, image: Image.Image) -> Dict[str, Any]:
        """Executes room style/type classification."""
        try:
            response = self.model.predict(image)
            return {
                "success": True,
                "response": response,
                "error_message": None,
                "is_implemented": True,
            }
        except ModelNotImplementedError as exc:
            logger.warning(f"Classification model not implemented: {exc}")
            return {
                "success": False,
                "response": None,
                "error_message": str(exc),
                "is_implemented": False,
            }
        except Exception as exc:
            logger.error(f"Error executing classification service: {exc}", exc_info=True)
            return {
                "success": False,
                "response": None,
                "error_message": f"Classification failed: {str(exc)}",
                "is_implemented": True,
            }
