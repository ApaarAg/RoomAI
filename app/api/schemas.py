"""API Request/Response Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable explanation")
    remediation: Optional[str] = Field(default=None, description="Suggested action")


class ErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail
    timestamp: float


class UploadResponse(BaseModel):
    status: str = "success"
    upload_id: str
    filename: str
    original_filename: str
    width: int
    height: int
    format: str
    size_bytes: int
    sha256: str


class ReconstructionJobRequest(BaseModel):
    upload_ids: List[str] = Field(min_length=1, max_length=10)
    model_id: Optional[str] = "roomai_composite_v1"
    refine_depth: bool = True
    downsample_points: int = 40000


class JobSubmissionResponse(BaseModel):
    status: str = "accepted"
    job_id: str
    job_status: str = "QUEUED"
    model_id: str
    view_count: int
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    model_id: str
    view_count: int
    timing: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    created_at: float
    updated_at: float


class JobResultResponse(BaseModel):
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
    job_id: str
    status: str
    cancelled: bool
    message: str


class ModelCapabilities(BaseModel):
    model_id: str
    name: str
    version: str
    status: str
    supports_single_view: bool = True
    supports_multi_view: bool = False
    min_views: int = 1
    max_views: int = 1
    supports_metric_depth: bool = True
    supports_surface_normals: bool = False
    supports_3dgs: bool = False
    supports_mesh_glb: bool = True
    supports_point_cloud_ply: bool = True
    device: str = "cuda"
    memory_required_mb: int = 1500
    description: str = ""


class ModelListResponse(BaseModel):
    status: str = "success"
    default_model_id: str
    models: List[ModelCapabilities]


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    service: str = "RoomAI Backend API"
    timestamp: float


class ReadinessResponse(BaseModel):
    status: str
    cuda_available: bool
    device_name: str
    free_vram_gb: float
    total_vram_gb: float
    checkpoints_ready: bool
    models_ready: List[str]