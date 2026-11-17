"""Pydantic Request and Response Schemas for RoomAI REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from models.adapters.base import ModelCapabilities


class ErrorDetail(BaseModel):
    """Structured error details."""
    code: str = Field(description="Machine-readable error code (e.g. 'VALIDATION_ERROR')")
    message: str = Field(description="Human-readable explanation of error")
    remediation: Optional[str] = Field(default=None, description="Suggested action to resolve error")


class ErrorResponse(BaseModel):
    """Standardized error envelope."""
    status: str = Field(default="error", description="'error'")
    error: ErrorDetail
    timestamp: float = Field(description="Unix timestamp of failure")


class UploadResponse(BaseModel):
    """Response returned upon successful image upload."""
    status: str = Field(default="success")
    upload_id: str = Field(description="Cryptographically secure unique upload identifier")
    filename: str = Field(description="Sanitized storage filename")
    original_filename: str = Field(description="Client-supplied filename")
    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    format: str = Field(description="Decoded image format (JPEG, PNG, WEBP)")
    size_bytes: int = Field(description="File size in bytes")
    sha256: str = Field(description="SHA-256 digest of image content")


class ReconstructionJobRequest(BaseModel):
    """Payload to initiate a 3D room reconstruction job."""
    upload_ids: List[str] = Field(
        min_length=1,
        max_length=10,
        description="List of 1 to 10 previously uploaded image identifiers depicting the room.",
    )
    model_id: Optional[str] = Field(
        default="roomai_composite_v1",
        description="Target 3D reconstruction model adapter (defaults to 'roomai_composite_v1')",
    )
    refine_depth: bool = Field(
        default=True,
        description="Whether to apply bilateral edge-preserving depth refinement",
    )
    downsample_points: int = Field(
        default=40000,
        description="Point cloud vertex ceiling for 3D exports",
    )
    camera_intrinsics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional calibrated pinhole intrinsics fx/fy/cx/cy in pixel units for the submitted image dimensions.",
    )


class JobSubmissionResponse(BaseModel):
    """Response returned immediately upon successful job submission."""
    status: str = Field(default="accepted")
    job_id: str = Field(description="Unique tracking ID for the queued reconstruction job")
    job_status: str = Field(default="QUEUED")
    model_id: str = Field(description="Selected reconstruction model adapter")
    view_count: int = Field(description="Number of views to be reconstructed")
    poll_url: str = Field(description="Relative URL to poll job status")


class JobStatusResponse(BaseModel):
    """Response returned when polling reconstruction job status."""
    job_id: str
    status: str = Field(description="Current status: QUEUED, VALIDATING, PROCESSING, RECONSTRUCTING, FUSING, EXPORTING, COMPLETED, FAILED, CANCELLED")
    stage: str = Field(description="Human-readable description of current processing stage")
    progress: float = Field(ge=0.0, le=1.0, description="Progress fraction between 0.0 and 1.0")
    model_id: str
    view_count: int
    timing: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    asset_acceptance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Reconstruction validation status, separate from inference job completion.",
    )
    created_at: float
    updated_at: float


class JobResultResponse(BaseModel):
    """Complete results payload for a completed reconstruction job."""
    job_id: str
    status: str
    is_completed: bool
    model_id: Optional[str] = None
    timing: Optional[Dict[str, Any]] = None
    results_summary: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    scene_graph: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class JobCancellationResponse(BaseModel):
    """Response returned upon job cancellation request."""
    job_id: str
    status: str
    cancelled: bool
    message: str


class ModelListResponse(BaseModel):
    """List of registered models and capabilities."""
    status: str = Field(default="success")
    default_model_id: str
    models: List[ModelCapabilities]


class HealthResponse(BaseModel):
    """Basic service liveness response."""
    status: str = Field(default="healthy")
    version: str
    service: str = "RoomAI Backend API"
    timestamp: float


class ReadinessResponse(BaseModel):
    """Hardware and model readiness response."""
    status: str = Field(description="'ready' or 'degraded'")
    cuda_available: bool
    device_name: str
    free_vram_gb: float
    total_vram_gb: float
    checkpoints_ready: bool
    models_ready: List[str]
    startup_error: Optional[str] = Field(default=None, description="Diagnostic startup error message if degraded")


# ---------------------------------------------------------------------------
# Canonical Object Detection API Contract (Model A3)
# ---------------------------------------------------------------------------

class ImageDimensions(BaseModel):
    """Input image dimensions in pixels."""
    width: int = Field(gt=0, description="Image width in pixels")
    height: int = Field(gt=0, description="Image height in pixels")
    channels: int = Field(default=3, description="Color channel count")


class BoundingBox2D(BaseModel):
    """2D Bounding box with both absolute pixel coordinates and normalized [0, 1] coordinates."""
    x1: float = Field(description="Top-left corner X (pixels)")
    y1: float = Field(description="Top-left corner Y (pixels)")
    x2: float = Field(description="Bottom-right corner X (pixels)")
    y2: float = Field(description="Bottom-right corner Y (pixels)")
    norm_x1: float = Field(ge=0.0, le=1.0, description="Normalized top-left corner X [0.0, 1.0]")
    norm_y1: float = Field(ge=0.0, le=1.0, description="Normalized top-left corner Y [0.0, 1.0]")
    norm_x2: float = Field(ge=0.0, le=1.0, description="Normalized bottom-right corner X [0.0, 1.0]")
    norm_y2: float = Field(ge=0.0, le=1.0, description="Normalized bottom-right corner Y [0.0, 1.0]")


class DetectionItem(BaseModel):
    """Single detected indoor scene object."""
    class_id: int = Field(ge=0, description="Model class index")
    class_name: str = Field(description="Indoor semantic category name (e.g. chair, bed, table)")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox2D = Field(description="2D spatial bounding box")


class InferenceMetadata(BaseModel):
    """Inference execution metadata."""
    model: str = Field(description="Active model identifier")
    latency_ms: float = Field(ge=0.0, description="Core GPU/CPU inference duration in milliseconds")
    device: str = Field(description="Execution device (e.g. 'cuda:0' or 'cpu')")
    cached: bool = Field(default=False, description="Whether this response was served from content cache")


class TimingBreakdown(BaseModel):
    """Granular latency profile on the server side."""
    preprocess_ms: float = Field(ge=0.0, description="Image decode and preprocessing time")
    inference_ms: float = Field(ge=0.0, description="Forward pass time")
    postprocess_ms: float = Field(ge=0.0, description="NMS, box unscaling, and serialization time")
    total_server_ms: float = Field(ge=0.0, description="Total backend handling time")
    queue_wait_ms: float = Field(default=0.0, ge=0.0, description="Time spent waiting for GPU worker lock")


class DetectionResponse(BaseModel):
    """Canonical response for RoomAI object detection."""
    status: str = Field(default="success", description="'success' or 'error'")
    request_id: str = Field(description="Unique request tracking identifier")
    image: ImageDimensions = Field(description="Image dimension metadata")
    detections: List[DetectionItem] = Field(default_factory=list, description="List of detected scene objects")
    inference: InferenceMetadata = Field(description="Inference provenance and latency")
    timing: TimingBreakdown = Field(description="Detailed timing breakdown")
    timestamp: float = Field(description="Unix timestamp of completion")

