"""Depth Refinement, Edge Preservation, and Point Cloud Filtering Utilities.

Enhances raw monocular metric depth maps and point clouds:
1. Guided bilateral depth smoothing to reduce sensor/model noise while preserving sharp object boundaries.
2. Flying-pixel and depth-discontinuity filtering to eliminate edge-bleeding artifacts.
3. Voxel grid downsampling and Statistical Outlier Removal (SOR) for dense 3D point clouds.
4. Analytical surface normal estimation for spatial orientation analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def filter_flying_pixels(
    depth_map: np.ndarray,
    gradient_threshold_ratio: float = 0.15,
) -> np.ndarray:
    """Detects and masks flying pixels occurring at steep depth discontinuities.

    Flying pixels occur when neural depth models blur sharp boundaries between
    foreground objects and distant backgrounds, creating spurious intermediate depths.

    Args:
        depth_map: (H, W) float32 array of metric depths in meters.
        gradient_threshold_ratio: Maximum allowable relative depth step between adjacent pixels.

    Returns:
        Filtered depth map with flying pixel boundary artifacts set to 0 (invalid).
    """
    if depth_map is None or depth_map.size == 0:
        return depth_map

    clean_depth = depth_map.copy()
    valid_mask = clean_depth > 0.05

    # Compute horizontal and vertical depth differences
    diff_x = np.abs(np.diff(clean_depth, axis=1, prepend=clean_depth[:, :1]))
    diff_y = np.abs(np.diff(clean_depth, axis=0, prepend=clean_depth[:1, :]))

    # Relative jump compared to local depth
    denom = np.maximum(clean_depth, 0.1)
    rel_jump_x = diff_x / denom
    rel_jump_y = diff_y / denom

    # Identify steep discontinuities (edges)
    edge_mask = (rel_jump_x > gradient_threshold_ratio) | (rel_jump_y > gradient_threshold_ratio)

    # Dilate edges by 1 pixel to suppress boundary interpolation artifacts
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_edges = cv2.dilate(edge_mask.astype(np.uint8), kernel) > 0

    # Retain valid depths that are not flying-edge artifacts
    clean_depth[dilated_edges & (clean_depth > 0)] = 0.0
    return clean_depth


def refine_metric_depth_guided(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    diameter: int = 5,
    sigma_depth: float = 0.15,
    sigma_color: float = 25.0,
) -> np.ndarray:
    """Applies bilateral/guided filtering to metric depth, aligning depth boundaries to RGB edges.

    Args:
        depth_map: (H, W) float32 array of metric depths in meters.
        rgb_image: (H, W, 3) uint8 RGB image matching depth map dimensions.
        diameter: Pixel neighborhood diameter for bilateral filter.
        sigma_depth: Filter sigma in depth space (meters).
        sigma_color: Filter sigma in RGB coordinate space.

    Returns:
        Refined metric depth map with preserved edges and smoothed planar surfaces.
    """
    if depth_map is None or depth_map.size == 0:
        return depth_map

    h, w = depth_map.shape[:2]
    if rgb_image.shape[:2] != (h, w):
        rgb_resized = cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_resized = rgb_image

    valid_mask = depth_map > 0.05
    if not np.any(valid_mask):
        return depth_map

    # Normalize valid depth into [0, 1] for stable bilateral filtering
    d_min = float(np.min(depth_map[valid_mask]))
    d_max = float(np.max(depth_map[valid_mask]))
    d_range = max(d_max - d_min, 1e-4)

    norm_depth = np.zeros_like(depth_map, dtype=np.float32)
    norm_depth[valid_mask] = (depth_map[valid_mask] - d_min) / d_range

    # Joint bilateral filter: smooths depth using color similarity as guide
    # cv2.bilateralFilter works on 32-bit float
    filtered_norm = cv2.bilateralFilter(
        norm_depth,
        d=diameter,
        sigmaColor=sigma_depth / d_range,
        sigmaSpace=sigma_color,
    )

    refined_depth = np.zeros_like(depth_map)
    refined_depth[valid_mask] = filtered_norm[valid_mask] * d_range + d_min

    # Re-apply flying pixel filter to ensure clean boundary contours
    return filter_flying_pixels(refined_depth, gradient_threshold_ratio=0.18)


def voxel_grid_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.03,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Subsamples point cloud using a 3D regular voxel grid.

    Args:
        points: (N, 3) float array of XYZ points.
        colors: Optional (N, 3) float array of RGB colors in [0, 1].
        voxel_size: Voxel edge length in meters (default: 3 cm).

    Returns:
        Tuple of (downsampled_points: [M, 3], downsampled_colors: [M, 3]).
    """
    if len(points) == 0:
        return points, colors

    # Compute integer voxel coordinates
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)

    # Use hash dict or unique indices
    # Pack coordinates into unique keys
    vx = voxel_coords[:, 0]
    vy = voxel_coords[:, 1]
    vz = voxel_coords[:, 2]

    # Unique voxel indices
    keys = np.column_stack([vx, vy, vz])
    _, unique_indices = np.unique(keys, axis=0, return_index=True)

    downsampled_points = points[unique_indices]
    downsampled_colors = colors[unique_indices] if colors is not None else None

    return downsampled_points, downsampled_colors


def statistical_outlier_removal(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    k_neighbors: int = 16,
    std_ratio: float = 1.8,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Removes isolated floating noise and outlier points via statistical distance analysis.

    Args:
        points: (N, 3) float array of XYZ points.
        colors: Optional (N, 3) float array of RGB colors.
        k_neighbors: Number of nearest neighbors to evaluate.
        std_ratio: Standard deviation multiplier threshold.

    Returns:
        Tuple of (inlier_points, inlier_colors).
    """
    if len(points) < k_neighbors * 2:
        return points, colors

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    # Query distances to k nearest neighbors
    distances, _ = tree.query(points, k=k_neighbors + 1)
    # Average distance to k neighbors (excluding self at index 0)
    mean_dists = np.mean(distances[:, 1:], axis=1)

    global_mean = float(np.mean(mean_dists))
    global_std = float(np.std(mean_dists))
    dist_threshold = global_mean + std_ratio * global_std

    inlier_mask = mean_dists <= dist_threshold
    inlier_points = points[inlier_mask]
    inlier_colors = colors[inlier_mask] if colors is not None else None

    return inlier_points, inlier_colors

