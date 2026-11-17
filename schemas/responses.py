"""Output response schemas for RoomAI services and model outputs."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates [0.0, 1.0]."""

    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized left edge")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized top edge")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Normalized right edge")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom edge")

    @field_validator("x_max")
    @classmethod
    def validate_x_bounds(cls, v: float, info) -> float:
        if "x_min" in info.data and v < info.data["x_min"]:
            raise ValueError(f"x_max ({v}) cannot be less than x_min ({info.data['x_min']})")
        return v

    @field_validator("y_max")
    @classmethod
    def validate_y_bounds(cls, v: float, info) -> float:
        if "y_min" in info.data and v < info.data["y_min"]:
            raise ValueError(f"y_max ({v}) cannot be less than y_min ({info.data['y_min']})")
        return v

    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Converts normalized coordinates to integer pixel coordinates (left, top, right, bottom)."""
        left = int(self.x_min * width)
        top = int(self.y_min * height)
        right = int(self.x_max * width)
        bottom = int(self.y_max * height)
        return left, top, right, bottom


class DetectionItem(BaseModel):
    """Individual detected object with class, confidence, and bounding box."""

    class_id: int
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox
    model_provenance: Optional[str] = None


class DetectionResponse(BaseModel):
    """Structured response from the object detection service."""

    detections: List[DetectionItem] = Field(default_factory=list)
    total_objects: int = 0
    is_simulated: bool = False
    model_name: str = "Unspecified"
    checkpoint: str = "yolo11s.pt"
    inference_time_ms: float = 0.0
    device: str = "cpu"


class SegmentationMask(BaseModel):
    """Metadata for a segmented region in the room."""

    category: str
    coverage_percentage: float = Field(..., ge=0.0, le=100.0)
    mask_shape: Tuple[int, int]
    color_hex: str = "#3B82F6"
    instance_id: Optional[int] = None
    confidence: Optional[float] = None
    mask_points_3d: Optional[List[List[float]]] = None
    mask_points_colors: Optional[List[List[float]]] = None


class SegmentationResponse(BaseModel):
    """Structured response from semantic segmentation service."""

    masks: List[SegmentationMask] = Field(default_factory=list)
    categories_found: List[str] = Field(default_factory=list)
    is_simulated: bool = False
    model_name: str = "Unspecified"


class ClassificationItem(BaseModel):
    """Individual class candidate with prediction score."""

    class_name: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    """Structured response from scene/style classification service."""

    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_classes: List[ClassificationItem] = Field(default_factory=list)
    is_simulated: bool = False
    model_name: str = "Unspecified"


class VLMAnswerResponse(BaseModel):
    """Structured response from the Vision-Language Model interface."""

    question: str
    answer: str
    is_simulated: bool = False
    model_name: str = "Unspecified"


class DesignGenerationResponse(BaseModel):
    """Structured response from the Generative AI redesign service."""

    success: bool
    style: str
    prompt_applied: str
    is_simulated: bool = False
    honesty_disclaimer: str = (
        "RoomAI design concept generation does not preserve exact room dimensions, "
        "structural load-bearing walls, or physical manufacturing constraints. "
        "Generated visuals are creative concepts for aesthetic inspiration only."
    )
    error_message: Optional[str] = None


class Scene3DResponse(BaseModel):
    """Structured response for experimental 3D scene understanding."""

    status: str
    depth_available: bool = False
    point_cloud_available: bool = False
    is_simulated: bool = False
    message: str = "Experimental 3D module"
    model_name: str = "Unspecified"
    device: str = "cpu"
    inference_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    relative_depth_only: bool = False
    depth_unit: str = "relative (inverse depth, NOT metric)"
    disclaimer: str = ""
    depth_min: float = 0.0
    depth_max: float = 0.0
    camera_intrinsics: Optional[Dict[str, float]] = None
    point_cloud: Optional[Any] = None  # Dict[str, List[List[float]]] for xyz, rgb, depth arrays
    raw_depth_map: Optional[List[List[float]]] = None
