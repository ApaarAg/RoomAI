"""RoomAI Geometry Postprocessing Service.

Coordinate transformations, depth conversion, point filtering, normal validation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


# Coordinate system conventions
# Model (OpenGL/Blender): +X right, +Y up, +Z back
# RoomAI (OpenCV Pinhole): +X right, +Y down, +Z forward
C_MODEL_TO_ROOMAI = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
C_ROOMAI_TO_MODEL = C_MODEL_TO_ROOMAI.T  # Inverse = transpose for diagonal


@dataclass
class PointCloud:
    """Structured point cloud with metadata."""
    xyz: np.ndarray           # (N, 3) float32
    rgb: Optional[np.ndarray] = None  # (N, 3) float32 [0,1]
    normals: Optional[np.ndarray] = None  # (N, 3) float32
    confidence: Optional[np.ndarray] = None  # (N,) float32
    view_indices: Optional[np.ndarray] = None  # (N,) int

    def __len__(self) -> int:
        return len(self.xyz)

    def is_empty(self) -> bool:
        return len(self.xyz) == 0


class GeometryService:
    """Postprocesses raw model predictions into validated geometry."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_depth = self.config.get("min_depth", 0.1)
        self.max_depth = self.config.get("max_depth", 25.0)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.voxel_size = self.config.get("voxel_size", 0.01)
        self.normal_edge_threshold = self.config.get("normal_edge_threshold", 30.0)  # degrees

    def transform_coordinates(
        self,
        points: np.ndarray,
        from_space: str = "model",
        to_space: str = "roomai",
    ) -> np.ndarray:
        """Transform points between coordinate spaces."""
        if from_space == to_space:
            return points

        if from_space == "model" and to_space == "roomai":
            return points @ C_MODEL_TO_ROOMAI.T
        elif from_space == "roomai" and to_space == "model":
            return points @ C_ROOMAI_TO_MODEL.T
        else:
            raise ValueError(f"Unknown coordinate spaces: {from_space} -> {to_space}")

    def transform_normals(
        self,
        normals: np.ndarray,
        from_space: str = "model",
        to_space: str = "roomai",
    ) -> np.ndarray:
        """Transform normals between coordinate spaces (rotation only, no translation)."""
        # Normals transform with the rotation part only (no translation, no scale)
        # For diagonal matrix, same as points
        return self.transform_coordinates(normals, from_space, to_space)

    def filter_valid_points(
        self,
        points: np.ndarray,
        depth: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Filter points for validity."""
        # Finite check
        valid = np.isfinite(points).all(axis=-1)

        # Depth range check
        if depth is not None:
            valid = valid & np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)

        # Confidence threshold
        if confidence is not None:
            valid = valid & (confidence >= self.confidence_threshold)

        return valid

    def voxel_downsample(
        self,
        points: np.ndarray,
        attributes: Optional[Dict[str, np.ndarray]] = None,
        voxel_size: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Voxel grid downsampling with attribute averaging."""
        vs = voxel_size or self.voxel_size

        if len(points) == 0:
            return points, attributes or {}

        # Compute voxel indices
        voxel_indices = np.floor(points / vs).astype(np.int64)

        # Unique voxels
        _, unique_indices, inverse_indices = np.unique(
            voxel_indices, axis=0, return_index=True, return_inverse=True
        )

        # Average points in each voxel
        downsampled_points = np.zeros((len(unique_indices), 3), dtype=points.dtype)
        for i, idx in enumerate(unique_indices):
            voxel_mask = (inverse_indices == i)
            downsampled_points[i] = points[voxel_mask].mean(axis=0)

        # Downsample attributes
        downsampled_attrs = {}
        if attributes:
            for key, attr in attributes.items():
                if attr is not None and len(attr) == len(points):
                    downsampled = np.zeros((len(unique_indices),) + attr.shape[1:], dtype=attr.dtype)
                    for i in range(len(unique_indices)):
                        voxel_mask = (inverse_indices == i)
                        downsampled[i] = attr[voxel_mask].mean(axis=0)
                    downsampled_attrs[key] = downsampled

        return downsampled_points, downsampled_attrs

    def remove_duplicates(
        self,
        points: np.ndarray,
        attributes: Optional[Dict[str, np.ndarray]] = None,
        tolerance: float = 1e-6,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Remove duplicate points within tolerance."""
        if len(points) == 0:
            return points, attributes or {}

        # Round to tolerance
        rounded = np.round(points / tolerance).astype(np.int64)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)

        unique_points = points[unique_indices]
        unique_attrs = {}
        if attributes:
            for key, attr in attributes.items():
                if attr is not None and len(attr) == len(points):
                    unique_attrs[key] = attr[unique_indices]

        return unique_points, unique_attrs

    def validate_normals(
        self,
        normals: np.ndarray,
        points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Validate and clean normals."""
        # Normalize
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        valid_norm = norms > 1e-6
        normalized = np.where(valid_norm, normals / np.maximum(norms, 1e-6), 0.0)

        # If points available, check normal orientation (pointing toward camera)
        if points is not None:
            # View direction from origin to point
            view_dirs = -points
            view_norms = np.linalg.norm(view_dirs, axis=-1, keepdims=True)
            view_dirs = view_dirs / np.maximum(view_norms, 1e-6)

            # Dot product with normals
            dots = (normalized * view_dirs).sum(axis=-1)

            # Flip normals that point away from camera
            flip_mask = dots < 0
            normalized[flip_mask] = -normalized[flip_mask]

        return normalized

    def filter_normal_edges(
        self,
        normals: np.ndarray,
        points: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Filter normals at depth discontinuities."""
        # Project points to image coordinates (simplified)
        h, w = image_shape
        # This is a placeholder - full implementation needs camera intrinsics
        return np.ones(len(normals), dtype=bool)

    def consolidate_multiview(
        self,
        point_clouds: List[PointCloud],
        poses: Optional[List[np.ndarray]] = None,
    ) -> PointCloud:
        """Consolidate multiple view point clouds into single cloud."""
        if not point_clouds:
            return PointCloud(xyz=np.zeros((0, 3)))

        # For now, simple concatenation
        # Full implementation would use poses for alignment
        all_xyz = []
        all_rgb = []
        all_normals = []
        all_conf = []
        all_views = []

        for i, pc in enumerate(point_clouds):
            if pc.is_empty():
                continue
            all_xyz.append(pc.xyz)
            if pc.rgb is not None:
                all_rgb.append(pc.rgb)
            if pc.normals is not None:
                all_normals.append(pc.normals)
            if pc.confidence is not None:
                all_conf.append(pc.confidence)
            all_views.append(np.full(len(pc.xyz), i, dtype=np.int32))

        if not all_xyz:
            return PointCloud(xyz=np.zeros((0, 3)))

        consolidated = PointCloud(
            xyz=np.concatenate(all_xyz, axis=0),
            rgb=np.concatenate(all_rgb, axis=0) if all_rgb else None,
            normals=np.concatenate(all_normals, axis=0) if all_normals else None,
            confidence=np.concatenate(all_conf, axis=0) if all_conf else None,
            view_indices=np.concatenate(all_views, axis=0) if all_views else None,
        )

        # Downsample if too many points
        max_points = self.config.get("max_consolidated_points", 100000)
        if len(consolidated) > max_points:
            step = len(consolidated) // max_points
            idx = np.arange(0, len(consolidated), step)[:max_points]
            consolidated = PointCloud(
                xyz=consolidated.xyz[idx],
                rgb=consolidated.rgb[idx] if consolidated.rgb is not None else None,
                normals=consolidated.normals[idx] if consolidated.normals is not None else None,
                confidence=consolidated.confidence[idx] if consolidated.confidence is not None else None,
                view_indices=consolidated.view_indices[idx] if consolidated.view_indices is not None else None,
            )

        return consolidated

    def compute_bounding_boxes(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Compute 3D bounding boxes for point clusters."""
        if labels is None:
            # Single box for all points
            if len(points) == 0:
                return []
            min_bounds = points.min(axis=0)
            max_bounds = points.max(axis=0)
            centroid = (min_bounds + max_bounds) / 2
            extent = max_bounds - min_bounds
            return [{
                "centroid": centroid.tolist(),
                "extent": extent.tolist(),
                "min_bounds": min_bounds.tolist(),
                "max_bounds": max_bounds.tolist(),
                "num_points": len(points),
            }]

        # Per-label boxes
        boxes = []
        for label in np.unique(labels):
            mask = labels == label
            pts = points[mask]
            if len(pts) < 10:  # Too few points
                continue
            min_bounds = pts.min(axis=0)
            max_bounds = pts.max(axis=0)
            centroid = (min_bounds + max_bounds) / 2
            extent = max_bounds - min_bounds
            boxes.append({
                "label": int(label),
                "centroid": centroid.tolist(),
                "extent": extent.tolist(),
                "min_bounds": min_bounds.tolist(),
                "max_bounds": max_bounds.tolist(),
                "num_points": int(mask.sum()),
            })
        return boxes


# Singleton
_geometry_service: Optional[GeometryService] = None


def get_geometry_service() -> GeometryService:
    global _geometry_service
    if _geometry_service is None:
        _geometry_service = GeometryService()
    return _geometry_service
