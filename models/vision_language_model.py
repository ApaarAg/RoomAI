"""Vision-Language Model (VLM) stubs and mock implementations."""

from typing import Any, Dict, Optional
from PIL import Image

from models.interfaces import BaseVisionLanguage, ModelNotImplementedError
from schemas.responses import VLMAnswerResponse


class CustomVisionLanguageModel(BaseVisionLanguage):
    """Student's Vision-Language Model interface.

    STUDENT LEARNING TASK:
    1. Connect an open-source vision-language model (e.g. BLIP-2, MiniGPT-4, LLaVA-1.5, or SmolVLM).
    2. Manage prompt templates, visual token projection, and autoregressive generation.
    3. Implement load_model() and ask() below.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        # TODO (Student): Load your VLM processor and model weights here
        raise ModelNotImplementedError("CustomVisionLanguageModel", __file__)

    def is_ready(self) -> bool:
        return self.model is not None

    def get_model_name(self) -> str:
        return "Custom Vision-Language Model (Student VLM)"

    def ask(self, image: Image.Image, question: str) -> VLMAnswerResponse:
        if not self.is_ready():
            raise ModelNotImplementedError("CustomVisionLanguageModel", __file__)

        # TODO (Student): Tokenize prompt, extract vision features, generate tokens
        raise ModelNotImplementedError("CustomVisionLanguageModel", __file__)


class MockVisionLanguageModel(BaseVisionLanguage):
    """Deterministic mock VLM for testing the interactive Q&A interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._ready = True

    def load_model(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "Deterministic Mock VLM (Simulated)"

    def ask(self, image: Image.Image, question: str) -> VLMAnswerResponse:
        q_lower = question.lower()
        if "light" in q_lower or "window" in q_lower:
            answer = (
                "The room appears to benefit from natural diffused daylight, likely through side windows. "
                "Adding warm accent lighting (2700K floor lamps or recessed dimmable LEDs) would enhance evening ambiance."
            )
        elif "furniture" in q_lower or "layout" in q_lower:
            answer = (
                "The current spatial arrangement centers around the seating zone. "
                "To optimize flow, maintain at least 3 feet of clear walking pathway between the sofa and side tables."
            )
        elif "color" in q_lower or "wall" in q_lower:
            answer = (
                "The walls present a neutral palette. Pairing warm beige or muted sage with natural oak furniture "
                "will impart an inviting organic feel."
            )
        else:
            answer = (
                f"Regarding your inquiry ('{question}'): The room demonstrates balanced proportion and volume. "
                "Consider decluttering surfaces and incorporating textured textiles (wool rugs, linen drapes) for depth."
            )

        return VLMAnswerResponse(
            question=question,
            answer=answer,
            is_simulated=True,
            model_name=self.get_model_name(),
        )
