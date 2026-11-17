"""Scene graph and 3D representation schemas for RoomAI."""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict


class Object3DPointStatistics(BaseModel):
    """Statistical summary of 3D points belonging to an object."""
    n_points: int = Field(description="Number of valid 3D points")
    mean_depth_m: float = Field(description="Mean depth in meters")
    median_depth_m: float = Field(description="Median depth in meters")
    depth_std_m: float = Field(description="Depth standard deviation")
    depth_min_m: float = Field(description="Minimum depth")
    depth_max_m: float = Field(description="Maximum depth")


class Object3DGeometry(BaseModel):
    """3D geometric representation of an object."""
    centroid: Tuple[float, float, float] = Field(description="3D centroid [x, y, z] in meters")
    extent: Tuple[float, float, float] = Field(description="Approximate extent [width, height, depth] in meters")
    orientation_yaw: Optional[float] = Field(None, description="Yaw rotation in radians (estimated)")
    point_stats: Optional[Object3DPointStatistics] = None
    is_dimension_reliable: bool = Field(default=True, description="Whether estimated physical dimensions are within plausible bounds")
    dimension_warning: Optional[str] = Field(default=None, description="Warning if dimensions exceed plausible physical envelopes")
    raw_extent: Optional[Tuple[float, float, float]] = Field(default=None, description="Raw unprojected extent [width, height, depth] in meters")
    raw_floor_contact_delta_m: Optional[float] = Field(default=None, description="Raw unprojected bottom elevation discrepancy from floor in meters")
    post_floor_contact_delta_m: Optional[float] = Field(default=None, description="Post-alignment bottom elevation discrepancy from floor in meters")


class SpatialRelationship(BaseModel):
    """Geometric relationship between two objects in 3D space."""
    source_object_id: str = Field(description="ID of the source object")
    relationship_type: str = Field(description="Type: 'near', 'against', 'in_front_of', 'between', etc.")
    target_object_id: str = Field(description="ID of the target object")
    evidence: str = Field(description="Geometric condition that supports this relationship")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this relationship")


class RoomSurface3D(BaseModel):
    """3D representation of a room surface (wall, floor, ceiling)."""
    semantic_category: str = Field(description="RoomAI taxonomy category (wall, floor, ceiling, etc.)")
    pixel_region: Optional[Any] = Field(default=None, description="2D mask or pixel indices")
    n_points: int = Field(default=0, description="Number of 3D points on this surface")
    mean_depth_m: float = Field(default=0.0, description="Mean depth of surface points")
    centroid: Optional[Tuple[float, float, float]] = None


class SceneObject(BaseModel):
    """Complete representation of a detected object in 3D space."""
    object_id: str = Field(description="Unique identifier (e.g., 'bed_001')")
    semantic_class: str = Field(description="Semantic class from SegFormer/ADE20K")
    instance_id: int = Field(description="Instance ID from detector")
    detection_confidence: float = Field(ge=0.0, le=1.0)
    mask_provenance: str = Field(description="'a3_fastsam' or 'segformer'")

    # 2D properties
    bounding_box: Tuple[float, float, float, float] = Field(
        description="Normalized bbox [x_min, y_min, x_max, y_max]"
    )
    coverage_percentage: float = Field(ge=0.0, le=100.0, description="Pixel coverage within image")

    # 3D properties
    geometry: Optional[Object3DGeometry] = None
    point_cloud_available: bool = False
    xyz_coords: Optional[List[List[float]]] = Field(
        default=None, description="List of [x, y, z] coordinates, max 10k points"
    )
    rgb_colors: Optional[List[List[float]]] = Field(
        default=None, description="List of [r, g, b] colors (0.0-1.0 normalized), max 10k points"
    )


class SceneGraph(BaseModel):
    """Unified 3D scene representation combining all modalities."""
    image_id: Optional[str] = None
    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    model_name: str = Field(description="Depth model used")
    depth_mode: str = Field(default="metric", description="'metric' or 'relative'")
    camera_intrinsics: Optional[Dict[str, float]] = None

    objects: List[SceneObject] = Field(default_factory=list, description="Detected objects")
    surfaces: List[RoomSurface3D] = Field(default_factory=list, description="Semantic surfaces")
    relationships: List[SpatialRelationship] = Field(default_factory=list, description="Spatial relations")
    room_layout: Optional[Dict[str, Any]] = Field(default=None, description="Estimated room layout and bounding envelope")
    point_cloud: Optional[Dict[str, Any]] = Field(default=None, description="Dense 3D point cloud dictionary")

    n_total_points: int = 0
    n_valid_points: int = 0
    depth_min_m: float = 0.0
    depth_max_m: float = 0.0

    # Metric depth and visualization maps
    depth_map: Optional[Any] = Field(default=None, description="2D metric depth map array in meters")
    heatmap_image: Optional[Any] = Field(default=None, description="Rendered PIL Image of depth heatmap")

    # Multi-view registration and telemetry
    multiview_telemetry: Optional[List[Dict[str, Any]]] = Field(default=None, description="Pairwise multi-view registration telemetry")
    multiview_per_view_provenance: Optional[List[Dict[str, Any]]] = Field(default=None, description="Per-view registration and contribution details")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class Scene3DEvaluation(BaseModel):
    """Validation results for 3D reconstruction."""
    pixel_depth_alignment_error_px: float = 0.0
    coordinate_system_verified: bool = False
    color_alignment_verified: bool = False
    invalid_depth_handled: bool = False
    n_invalid_points_excluded: int = 0
    n_duplicate_points_removed: int = 0
    outliers_removed: int = 0
    verification_notes: str = ""