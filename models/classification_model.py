"""Room Style and Scene Classification model stubs and mock implementations."""

from typing import Any, Dict, Optional
from PIL import Image

from models.interfaces import BaseClassifier, ModelNotImplementedError
from schemas.responses import ClassificationItem, ClassificationResponse


class CustomRoomClassifier(BaseClassifier):
    """Student's Room Classification Model.

    STUDENT LEARNING TASK:
    1. Implement your Convolutional Neural Network (CNN) or Vision Transformer (ViT).
    2. Define your training pipeline, dataset loaders, loss function, and evaluation metrics.
    3. Save the trained weights to 'checkpoints/classification_best.pt'.
    4. Implement load_model() and predict() below.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None

    def load_model(self) -> None:
        # TODO (Student): Load your trained PyTorch / TensorFlow weights here
        # Example:
        #   self.model = MyRoomCNN(num_classes=6)
        #   self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        #   self.model.eval()
        raise ModelNotImplementedError("CustomRoomClassifier", __file__)

    def is_ready(self) -> bool:
        return self.model is not None

    def get_model_name(self) -> str:
        return "Custom Room Classifier (Student Model)"

    def predict(self, image: Image.Image) -> ClassificationResponse:
        if not self.is_ready():
            raise ModelNotImplementedError("CustomRoomClassifier", __file__)

        # TODO (Student): Preprocess image, run forward pass, softmax, and format response
        raise ModelNotImplementedError("CustomRoomClassifier", __file__)


class MockRoomClassifier(BaseClassifier):
    """Deterministic mock classifier for testing the UI and application flow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._ready = True

    def load_model(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "Deterministic Mock Classifier (Simulated)"

    def predict(self, image: Image.Image) -> ClassificationResponse:
        # Deterministic dummy output for UI demonstration
        top_classes = [
            ClassificationItem(class_name="Living Room", score=0.88),
            ClassificationItem(class_name="Dining Room", score=0.08),
            ClassificationItem(class_name="Home Office", score=0.04),
        ]
        return ClassificationResponse(
            predicted_class="Living Room",
            confidence=0.88,
            top_classes=top_classes,
            is_simulated=True,
            model_name=self.get_model_name(),
        )
