"""Generative Redesign model stubs and mock implementations."""

from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter

from models.interfaces import BaseGenerator, ModelNotImplementedError
from schemas.requests import DesignRequest
from schemas.responses import DesignGenerationResponse


class CustomDesignGenerator(BaseGenerator):
    """Student's Generative AI Redesign Model.

    STUDENT LEARNING TASK:
    1. Integrate an Image-to-Image or ControlNet pipeline (e.g. Stable Diffusion 1.5/SDXL ControlNet MLSD or Depth).
    2. Configure conditioning, guidance scales, prompt engineering, and latent sampling.
    3. Implement load_model() and generate() below.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pipeline = None

    def load_model(self) -> None:
        # TODO (Student): Load your diffusion / generation pipeline here
        raise ModelNotImplementedError("CustomDesignGenerator", __file__)

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def get_model_name(self) -> str:
        return "Custom Design Generator (Student Model)"

    def generate(
        self, image: Image.Image, request: DesignRequest
    ) -> Tuple[Image.Image, DesignGenerationResponse]:
        if not self.is_ready():
            raise ModelNotImplementedError("CustomDesignGenerator", __file__)

        # TODO (Student): Execute img2img / ControlNet diffusion generation
        raise ModelNotImplementedError("CustomDesignGenerator", __file__)


class MockDesignGenerator(BaseGenerator):
    """Deterministic mock generator for testing the redesign UI and before/after view."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._ready = True

    def load_model(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "Deterministic Mock Generator (Simulated Concept)"

    def generate(
        self, image: Image.Image, request: DesignRequest
    ) -> Tuple[Image.Image, DesignGenerationResponse]:
        # Synthesize an aesthetic redesign preview for UI workflow testing
        # Adjust brightness, warmth, and sharpness
        enhanced = ImageEnhance.Brightness(image).enhance(1.08)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.12)
        enhanced = ImageEnhance.Color(enhanced).enhance(0.95)  # Muted elegant tones
        enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)

        prompt_used = request.custom_prompt or f"A clean, elegant {request.target_style} style room."

        response = DesignGenerationResponse(
            success=True,
            style=request.target_style,
            prompt_applied=prompt_used,
            is_simulated=True,
        )

        return enhanced, response
