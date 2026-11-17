"""Models package for RoomAI."""

from models.interfaces import (
    BaseClassifier,
    BaseDetector,
    BaseGenerator,
    BaseMLModel,
    BaseScene3D,
    BaseSegmenter,
    BaseVisionLanguage,
    ModelNotImplementedError,
)
from models.classification_model import CustomRoomClassifier, MockRoomClassifier
from models.detection_model import CustomObjectDetector, MockObjectDetector
from models.generation_model import CustomDesignGenerator, MockDesignGenerator
from models.scene3d_model import CustomScene3DModel, MockScene3DModel
from models.segmentation_model import CustomRoomSegmenter, MockRoomSegmenter
from models.vision_language_model import (
    CustomVisionLanguageModel,
    MockVisionLanguageModel,
)

__all__ = [
    "BaseMLModel",
    "BaseClassifier",
    "BaseDetector",
    "BaseSegmenter",
    "BaseVisionLanguage",
    "BaseGenerator",
    "BaseScene3D",
    "ModelNotImplementedError",
    "CustomRoomClassifier",
    "MockRoomClassifier",
    "CustomObjectDetector",
    "MockObjectDetector",
    "CustomRoomSegmenter",
    "MockRoomSegmenter",
    "CustomVisionLanguageModel",
    "MockVisionLanguageModel",
    "CustomDesignGenerator",
    "MockDesignGenerator",
    "CustomScene3DModel",
    "MockScene3DModel",
]
