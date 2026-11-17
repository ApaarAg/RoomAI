"""
Official HunyuanWorld-Mirror Canonical Scene-Coordinate Normalization.

Faithful implementation of MultiViewDataset.apply_pointcloud_normalization from
HunyuanWorld-Mirror (training/data/multiview_dataset.py lines 176-238).

Mathematical Contract:
  Given N views belonging to a common scene with world point maps X_w,
  depth maps Z, and camera poses [R | t]:
  1. Valid world points: X_w[valid_mask == 1]
  2. Euclidean distance from world origin: dis = ||X_w||_2
  3. Scene-wide scale factor: s = nanmedian(dis) across all views in the scene
  4. Normalized world points: X'_w = X_w / s
  5. Normalized depth: Z' = Z / s
  6. Normalized camera translation: t' = t / s
  7. Invariants:
     - Camera rotation R remains UNCHANGED
     - Camera intrinsics K remain UNCHANGED
     - Unit surface normals n remain UNCHANGED (||n||_2 == 1.0)
  8. Invertibility:
     - X_w = X'_w * s
     - Z   = Z' * s
     - t   = t' * s
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def compute_world_coordinates(
    depthmap: np.ndarray,
    camera_intrs: np.ndarray,
    camera_poses: np.ndarray,
    c2w: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 3D world coordinates from depth, intrinsics, and camera pose.
    
    Args:
        depthmap: (H, W) array of depth values in meters
        camera_intrs: (3, 3) intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
        camera_poses: (4, 4) or (3, 4) camera matrix
        c2w: If True, camera_poses is cam2world; if False, world2cam
        
    Returns:
        X_world: (H, W, 3) 3D points in world coordinates
        valid_mask: (H, W) boolean mask of valid pixels (depth > 0 & finite)
    """
    H, W = depthmap.shape
    fu = float(camera_intrs[0, 0])
    fv = float(camera_intrs[1, 1])
    cu = float(camera_intrs[0, 2])
    cv = float(camera_intrs[1, 2])

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid_mask = (depthmap > 0.0) & np.isfinite(depthmap)

    z_cam = np.where(valid_mask, depthmap, 0.0).astype(np.float32)
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)  # (H, W, 3)

    if c2w:
        R_cam2world = camera_poses[:3, :3]
        t_cam2world = camera_poses[:3, 3]
    else:
        # world2cam: [R_w2c | t_w2c], inverse gives c2w
        R_w2c = camera_poses[:3, :3]
        t_w2c = camera_poses[:3, 3]
        R_cam2world = R_w2c.T
        t_cam2world = -R_cam2world @ t_w2c

    # X_world = R_cam2world @ X_cam + t_cam2world
    X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
    X_world = np.where(valid_mask[..., None], X_world, 0.0).astype(np.float32)
    return X_world, valid_mask


def compute_scene_scale_factor(
    views: List[Dict[str, np.ndarray]],
    norm_mode: str = "median_dis",
    min_scale: float = 1e-8,
) -> Tuple[float, bool]:
    """Compute the scene-wide canonical scale factor s from all views of a scene.
    
    Exact reproduction of HunyuanWorld-Mirror logic:
      norm_factor = np.nanmedian(dis, axis=-1)
      where dis = ||valid_pts||_2 across all concatenated views.
      
    Args:
        views: List of dicts, each with 'pts3d' (H, W, 3) and 'valid_mask' (H, W)
        norm_mode: Normalization strategy ('median_dis' or 'avg_dis')
        min_scale: Minimum numerical threshold for scale factor
        
    Returns:
        norm_factor: Float scale factor s
        bad_case: Boolean flag indicating if fallback was used
    """
    stat_mode, dis_mode = norm_mode.split("_")
    
    pts_list = [view["pts3d"] for view in views if "pts3d" in view]
    valid_list = [view["valid_mask"] for view in views if "valid_mask" in view]
    
    if len(pts_list) == 0:
        return 1.0, True

    all_pts = np.concatenate([p.reshape(-1, 3) for p in pts_list], axis=0).astype(np.float64)
    all_valid = np.concatenate([v.reshape(-1) for v in valid_list], axis=0) > 0

    if not np.any(all_valid):
        return 1.0, True

    valid_pts = all_pts[all_valid]
    dis = np.linalg.norm(valid_pts, ord=2, axis=-1)
    
    if dis_mode == "dis":
        pass
    elif dis_mode == "log1p":
        dis = np.log1p(dis)
    else:
        raise ValueError(f"Unsupported distance mode: {dis_mode}")

    bad_case = False
    if stat_mode == "median":
        norm_factor = float(np.nanmedian(dis))
    elif stat_mode == "avg":
        norm_factor = float(np.nanmean(dis))
    else:
        raise ValueError(f"Unsupported stat mode: {stat_mode}")

    if np.isnan(norm_factor) or np.isinf(norm_factor) or norm_factor <= 0.0:
        norm_factor = 1.0
        bad_case = True

    norm_factor = float(np.clip(norm_factor, a_min=min_scale, a_max=None))
    return norm_factor, bad_case


def normalize_scene_coordinates(
    views: List[Dict[str, np.ndarray]],
    norm_mode: str = "median_dis",
    inplace: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], float]:
    """Normalize 3D scene coordinates, depths, and camera translations across views.
    
    Applies exact WorldMirror normalization:
      pts3d /= norm_factor
      depthmap /= norm_factor
      camera_poses[:3, 3] /= norm_factor
      norm_factor stored in each view
      
    Args:
        views: List of view dictionaries containing:
               - 'pts3d': (H, W, 3) world point map
               - 'valid_mask': (H, W) boolean/binary mask
               - 'depthmap': (H, W) depth map
               - 'camera_poses': (4, 4) or (3, 4) cam2world or world2cam
               - 'camera_intrs': (3, 3) intrinsics (unscaled)
               - optional 'normals': (H, W, 3) unit surface normals (unscaled)
        norm_mode: Normalization strategy (default: 'median_dis')
        inplace: Whether to modify dicts in-place or make shallow copies
        
    Returns:
        normalized_views: List of modified view dicts
        norm_factor: The computed scene scale factor s
    """
    norm_factor, bad_case = compute_scene_scale_factor(views, norm_mode=norm_mode)

    out_views = []
    for v in views:
        view = v if inplace else dict(v)
        
        if "pts3d" in view:
            view["pts3d"] = (view["pts3d"] / norm_factor).astype(np.float32)
        if "depthmap" in view:
            view["depthmap"] = (view["depthmap"] / norm_factor).astype(np.float32)
        if "camera_poses" in view:
            cpose = view["camera_poses"].copy()
            cpose[:3, 3] = cpose[:3, 3] / norm_factor
            view["camera_poses"] = cpose
            
        view["norm_factor"] = norm_factor
        view["bad_case"] = bad_case
        out_views.append(view)

    return out_views, norm_factor


def denormalize_scene_coordinates(
    views: List[Dict[str, np.ndarray]],
    norm_factor: Optional[float] = None,
    inplace: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Denormalize normalized scene coordinates back to physical metric space.
    
    Invertibility:
      pts3d = pts3d * norm_factor
      depthmap = depthmap * norm_factor
      camera_poses[:3, 3] = camera_poses[:3, 3] * norm_factor
      
    Args:
        views: List of normalized view dicts
        norm_factor: Scale factor s. If None, extracted from view['norm_factor'].
        inplace: Whether to modify dicts in-place
        
    Returns:
        restored_views: List of metric-space view dicts
    """
    out_views = []
    for v in views:
        view = v if inplace else dict(v)
        s = norm_factor if norm_factor is not None else float(view.get("norm_factor", 1.0))
        
        if "pts3d" in view:
            view["pts3d"] = (view["pts3d"] * s).astype(np.float32)
        if "depthmap" in view:
            view["depthmap"] = (view["depthmap"] * s).astype(np.float32)
        if "camera_poses" in view:
            cpose = view["camera_poses"].copy()
            cpose[:3, 3] = cpose[:3, 3] * s
            view["camera_poses"] = cpose
            
        out_views.append(view)
    return out_views

