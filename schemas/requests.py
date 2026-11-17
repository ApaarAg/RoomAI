"""Input request schemas for RoomAI services."""

from typing import Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Configuration for running scene analysis components."""

    run_detection: bool = Field(default=True, description="Execute object detection")
    run_segmentation: bool = Field(default=True, description="Execute semantic segmentation")
    run_classification: bool = Field(default=True, description="Execute room style classification")
    run_scene3d: bool = Field(default=False, description="Execute experimental 3D reconstruction")
    confidence_threshold: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Minimum detection/classification threshold"
    )


class DesignRequest(BaseModel):
    """Configuration for AI interior redesign generation."""

    target_style: str = Field(
        default="Scandinavian",
        min_length=1,
        max_length=64,
        description="Target interior design aesthetic",
    )
    custom_prompt: str = Field(
        default="",
        max_length=500,
        description="User custom instructions for the redesign",
    )
    guidance_scale: float = Field(
        default=7.5, ge=1.0, le=20.0, description="Guidance scale for diffusion generation"
    )
    num_inference_steps: int = Field(
        default=30, ge=10, le=100, description="Sampling steps for generation"
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class VLMQuestionRequest(BaseModel):
    """Request schema for asking visual questions about the room."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Natural language question regarding the room image",
    )
