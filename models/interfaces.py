"""Abstract base interfaces for RoomAI machine learning components.

CRITICAL RULE:
These interfaces specify the standard input/output contracts for models.
The actual neural network architectures, layers, training procedures, and
hyperparameter choices are to be implemented by the student.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from schemas.requests import DesignRequest
from schemas.responses import (
    ClassificationResponse,
    DetectionResponse,
    DesignGenerationResponse,
    Scene3DResponse,
    SegmentationResponse,
    VLMAnswerResponse,
)


class ModelNotImplementedError(NotImplementedError):
    """Raised when a real model component has not yet been implemented or loaded."""

    def __init__(self, model_name: str, module_path: str):
        super().__init__(
            f"'{model_name}' is not yet implemented or weights are missing.\n"
            f"Please implement your model architecture and loading logic in: {module_path}\n"
            f"(Or enable 'mock_mode: true' in config/app.yaml to test the UI with simulated outputs)."
        )
        self.model_name = model_name
        self.module_path = module_path


class BaseMLModel(ABC):
    """Base class for all RoomAI model wrappers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cpu")
        self.checkpoint_path = self.config.get("checkpoint_path", "")

    @abstractmethod
    def load_model(self) -> None:
        """Loads model architecture and checkpoint weights."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the model weights are loaded and ready for inference."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns a human-readable identifier for the model."""
        pass


class BaseDetector(BaseMLModel):
    """Interface for Object Detection models."""

    @abstractmethod
    def predict(
        self, image: Image.Image, confidence_threshold: float = 0.40
    ) -> DetectionResponse:
        """Performs object detection on a room image.

        Args:
            image: Validated PIL Image in RGB mode.
            confidence_threshold: Minimum detection confidence score.

        Returns:
            DetectionResponse containing detected furniture items and bounding boxes.
        """
        pass


class BaseSegmenter(BaseMLModel):
    """Interface for Semantic or Instance Segmentation models."""

    segmentation_mode: str = "instance"

    @abstractmethod
    def predict(
        self, image: Image.Image
    ) -> Tuple[SegmentationResponse, Optional[np.ndarray]]:
        """Segments room regions and furniture objects.

        Args:
            image: Validated PIL Image in RGB mode.

        Returns:
            Tuple of (SegmentationResponse metadata, 2D numpy mask array matching image size).
        """
        pass


class BaseClassifier(BaseMLModel):
    """Interface for Room Classification / Style Recognition models."""

    @abstractmethod
    def predict(self, image: Image.Image) -> ClassificationResponse:
        """Classifies room category or architectural style.

        Args:
            image: Validated PIL Image in RGB mode.

        Returns:
            ClassificationResponse with predicted class and probability distribution.
        """
        pass


class BaseVisionLanguage(BaseMLModel):
    """Interface for Vision-Language Models (VLM)."""

    @abstractmethod
    def ask(self, image: Image.Image, question: str) -> VLMAnswerResponse:
        """Answers a visual query regarding the room scene.

        Args:
            image: Validated PIL Image in RGB mode.
            question: User natural language question.

        Returns:
            VLMAnswerResponse with contextual visual explanation.
        """
        pass


class BaseGenerator(BaseMLModel):
    """Interface for Generative AI Redesign models."""

    @abstractmethod
    def generate(
        self, image: Image.Image, request: DesignRequest
    ) -> Tuple[Image.Image, DesignGenerationResponse]:
        """Generates a redesigned interior concept based on image and design instructions.

        Args:
            image: Source room PIL Image.
            request: DesignRequest containing target style and instructions.

        Returns:
            Tuple of (Redesigned PIL Image, DesignGenerationResponse metadata).
        """
        pass


class BaseScene3D(BaseMLModel):
    """Interface for Experimental 3D Scene Reconstruction models."""

    @abstractmethod
    def reconstruct(
        self, image: Image.Image
    ) -> Tuple[Optional[np.ndarray], Scene3DResponse]:
        """Performs 3D scene estimation (e.g. depth map or point cloud).

        Args:
            image: Source room PIL Image.

        Returns:
            Tuple of (Depth map numpy array, Scene3DResponse metadata).
        """
        pass
