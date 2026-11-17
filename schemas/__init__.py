"""Pydantic schemas for RoomAI requests and responses."""

from schemas.requests import AnalysisRequest, DesignRequest, VLMQuestionRequest
from schemas.responses import (
    BoundingBox,
    ClassificationItem,
    ClassificationResponse,
    DetectionItem,
    DetectionResponse,
    DesignGenerationResponse,
    Scene3DResponse,
    SegmentationMask,
    SegmentationResponse,
    VLMAnswerResponse,
)

__all__ = [
    "AnalysisRequest",
    "DesignRequest",
    "VLMQuestionRequest",
    "BoundingBox",
    "DetectionItem",
    "DetectionResponse",
    "SegmentationMask",
    "SegmentationResponse",
    "ClassificationItem",
    "ClassificationResponse",
    "VLMAnswerResponse",
    "DesignGenerationResponse",
    "Scene3DResponse",
]
