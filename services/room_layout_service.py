"""Room Layout and 3D Structural Geometry Estimation.

Estimates:
1. Floor plane normal and camera elevation via RANSAC planar fitting on floor points.
2. Room bounding envelope (width, length, ceiling height in meters).
3. Ground-aligned 3D object bounding boxes and spatial relationships.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from schemas.scene import Object3DGeometry, Object3DPointStatistics, RoomSurface3D, SceneObject, SpatialRelationship
from utils.logging_config import get_logger

logger = get_logger("RoomLayoutService")


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    max_iterations: int = 500,
) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
    """Fits a 3D plane (Ax + By + Cz + D = 0) to points using RANSAC.

    Args:
        points: (N, 3) array of XYZ points.
        distance_threshold: Inlier distance threshold in meters.
        max_iterations: Max RANSAC trials.

    Returns:
        Tuple of (normal: [3] float, D: float, inlier_mask: [N] bool).
    """
    if len(points) < 10:
        return None, 0.0, np.zeros(len(points), dtype=bool)

    best_inliers = np.zeros(len(points), dtype=bool)
    best_normal = None
    best_d = 0.0
    max_inlier_count = 0

    n_pts = len(points)

    for _ in range(max_iterations):
        sample_idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = points[sample_idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-6:
            continue
        normal = normal / norm_mag

        d = -float(np.dot(normal, p1))

        # Calculate distances of all points to plane: |Ax + By + Cz + D|
        distances = np.abs(np.dot(points, normal) + d)
        inliers = distances < distance_threshold
        inlier_count = int(np.count_nonzero(inliers))

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if max_inlier_count < 10 or best_normal is None:
        # Fallback to PCA plane fit
        centroid = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - centroid)
        best_normal = vh[2]
        best_d = -float(np.dot(best_normal, centroid))
        best_inliers = np.abs(np.dot(points, best_normal) + best_d) < (distance_threshold * 2.0)

    # Ensure normal points upward (towards negative Y in camera frame where down is +Y)
    if best_normal[1] > 0:
        best_normal = -best_normal
        best_d = -best_d

    return best_normal, best_d, best_inliers


class RoomLayoutService:
    """Recovers metric room structural layout, floor planes, and spatial bounds."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def estimate_layout(
        self,
        point_cloud: Dict[str, Any],
        objects: List[SceneObject],
        surfaces: List[RoomSurface3D],
    ) -> Dict[str, Any]:
        """Analyzes 3D scene point cloud and object geometry to extract room layout.

        Args:
            point_cloud: Dict containing 'xyz' coordinates.
            objects: List of detected SceneObject instances.
            surfaces: List of extracted RoomSurface3D instances.

        Returns:
            Dict containing room bounding box, camera height, floor plane, and spatial metrics.
        """
        xyz_raw = point_cloud.get("xyz", [])
        if not xyz_raw:
            return {
                "floor_plane_detected": False,
                "camera_height_m": 1.4,
                "room_width_m": 4.0,
                "room_length_m": 5.0,
                "room_height_m": 2.6,
                "floor_elevation_m": -1.4,
                "floor_inlier_ratio": 0.0,
                "floor_residual_m": 0.0,
                "gravity_alignment_score": 0.0,
                "competing_wall_planes_found": 0,
                "calibrated": False,
            }

        xyz_arr = np.array(xyz_raw, dtype=np.float32)
        valid = np.isfinite(xyz_arr).all(axis=1) & (xyz_arr[:, 2] > 0.1) & (xyz_arr[:, 2] < 25.0)
        xyz_arr = xyz_arr[valid]

        if len(xyz_arr) < 100:
            return {
                "floor_plane_detected": False,
                "camera_height_m": 1.4,
                "room_width_m": 4.0,
                "room_length_m": 5.0,
                "room_height_m": 2.6,
                "floor_elevation_m": -1.4,
                "floor_inlier_ratio": 0.0,
                "floor_residual_m": 0.0,
                "gravity_alignment_score": 0.0,
                "competing_wall_planes_found": 0,
                "calibrated": False,
            }

        # Candidate floor selection: combine lower vertical quadrant with semantic floor if available
        y_threshold = float(np.percentile(xyz_arr[:, 1], 65))
        floor_candidate_points = xyz_arr[xyz_arr[:, 1] >= y_threshold]

        # Multi-hypothesis RANSAC plane search
        distance_threshold = float(self.config.get("distance_threshold", 0.08))
        normal, d_val, inliers = fit_plane_ransac(floor_candidate_points, distance_threshold=distance_threshold)

        competing_wall_planes = 0
        inlier_ratio = 0.0
        mean_residual = 0.0
        gravity_score = 0.0

        if normal is not None:
            gravity_score = round(float(abs(normal[1])), 3)
            n_inliers = int(np.count_nonzero(inliers))
            inlier_ratio = round(float(n_inliers / max(1, len(floor_candidate_points))), 3)
            if n_inliers > 0:
                dists = np.abs(np.dot(floor_candidate_points[inliers], normal) + d_val)
                mean_residual = round(float(np.mean(dists)), 4)

            # Check if this plane is predominantly a vertical wall rather than horizontal floor
            if abs(normal[1]) < 0.35 and (abs(normal[0]) >= 0.70 or abs(normal[2]) >= 0.70):
                competing_wall_planes += 1

        if normal is not None and abs(normal[1]) >= 0.70 and inlier_ratio >= 0.20:
            # Floor normal points predominantly along the vertical gravity axis (|n_y| >= 0.70)
            cam_height = float(abs(d_val) / (np.linalg.norm(normal) + 1e-6))
            # Clip to realistic indoor residential heights (0.8m to 2.2m)
            cam_height = float(np.clip(cam_height, 0.8, 2.2))
            floor_elevation = float(np.percentile(floor_candidate_points[:, 1], 90))
            floor_detected = True
        else:
            if normal is not None:
                logger.warning(
                    "Fitted plane normal [%.3f, %.3f, %.3f] rejected (gravity score: %.2f, inlier ratio: %.2f).",
                    *normal, gravity_score, inlier_ratio
                )
            normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            cam_height = 1.4
            floor_elevation = float(np.percentile(floor_candidate_points[:, 1], 90)) if len(floor_candidate_points) > 0 else 1.4
            floor_detected = False

        # Compute metric room bounding dimensions from 2nd to 98th percentiles to exclude outliers
        x_min, x_max = float(np.percentile(xyz_arr[:, 0], 2)), float(np.percentile(xyz_arr[:, 0], 98))
        y_min, y_max = float(np.percentile(xyz_arr[:, 1], 2)), float(np.percentile(xyz_arr[:, 1], 98))
        z_min, z_max = float(np.percentile(xyz_arr[:, 2], 2)), float(np.percentile(xyz_arr[:, 2], 98))

        room_width = max(2.0, round(x_max - x_min, 2))
        room_length = max(2.5, round(z_max - z_min, 2))
        room_height = max(2.2, round(y_max - y_min, 2))

        return {
            "floor_plane_detected": floor_detected,
            "floor_normal": normal.tolist() if normal is not None else [0.0, -1.0, 0.0],
            "camera_height_m": round(cam_height, 2),
            "room_width_m": room_width,
            "room_length_m": room_length,
            "room_height_m": room_height,
            "floor_elevation_m": round(floor_elevation, 2),
            "floor_inlier_ratio": inlier_ratio,
            "floor_residual_m": mean_residual,
            "gravity_alignment_score": gravity_score,
            "competing_wall_planes_found": competing_wall_planes,
            "assumptions": "Camera upright (+Y downward, +Z forward), floor is planar horizontal",
            "calibrated": False,
            "room_bounds": {
                "x_min": round(x_min, 2),
                "x_max": round(x_max, 2),
                "y_min": round(y_min, 2),
                "y_max": round(y_max, 2),
                "z_min": round(z_min, 2),
                "z_max": round(z_max, 2),
            }
        }

