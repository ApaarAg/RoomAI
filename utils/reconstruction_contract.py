"""Canonical geometry, validation, and manifest helpers for reconstruction jobs.

The deployed API path stores evidence here without promoting model estimates to
calibrated measurements. Coordinates use a right/down/forward camera frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


CONTRACT_VERSION = "1.0.0"
CAMERA_FRAME = "opencv_camera_x_right_y_down_z_forward"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, reading it in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory tree in stable relative-path order."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def normalize_intrinsics(
    image_width: int,
    image_height: int,
    intrinsics: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], str]:
    """Validate pixel-unit pinhole intrinsics or provide an explicitly estimated K.

    `cx` and `cy` are pixel coordinates, never normalized fractions.
    """
    if intrinsics is None:
        focal = float(max(image_width, image_height))
        return {
            "fx": focal,
            "fy": focal,
            "cx": (image_width - 1) / 2.0,
            "cy": (image_height - 1) / 2.0,
        }, "estimated_centered_focal_max_dimension"

    try:
        values = {key: float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera_intrinsics must contain numeric fx, fy, cx, cy in pixel units") from exc
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("camera_intrinsics values must be finite")
    if values["fx"] <= 0 or values["fy"] <= 0:
        raise ValueError("camera focal lengths fx and fy must be positive")
    if not (-image_width <= values["cx"] <= 2 * image_width and -image_height <= values["cy"] <= 2 * image_height):
        raise ValueError("camera principal point appears outside plausible pixel coordinates")
    return values, "provided_pixel_intrinsics"


def transform_intrinsics(
    intrinsics: Dict[str, float],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    crop_left: float = 0.0,
    crop_top: float = 0.0,
    pad_left: float = 0.0,
    pad_top: float = 0.0,
) -> Dict[str, float]:
    """Apply resize, crop, then padding to pixel-unit camera intrinsics."""
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("intrinsic image scales must be positive")
    return {
        "fx": float(intrinsics["fx"] * scale_x),
        "fy": float(intrinsics["fy"] * scale_y),
        "cx": float((intrinsics["cx"] - crop_left) * scale_x + pad_left),
        "cy": float((intrinsics["cy"] - crop_top) * scale_y + pad_top),
    }


def depth_to_xyz(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Dict[str, float],
    validity_mask: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject valid optical-axis depth to deterministically sampled XYZ/RGB.

    Returns XYZ float32, RGB float32 in [0,1], and flattened source pixel indices.
    """
    depth_arr = np.asarray(depth, dtype=np.float32)
    rgb_arr = np.asarray(rgb)
    if depth_arr.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    height, width = depth_arr.shape
    if rgb_arr.shape != (height, width, 3):
        raise ValueError("RGB image dimensions must align exactly with depth")
    k, _ = normalize_intrinsics(width, height, intrinsics)
    valid = np.isfinite(depth_arr) & (depth_arr > 0)
    if validity_mask is not None:
        mask = np.asarray(validity_mask)
        if mask.shape != depth_arr.shape:
            raise ValueError("validity mask dimensions must match depth")
        valid &= mask.astype(bool)

    source_indices = np.flatnonzero(valid)
    if max_points is not None and max_points > 0 and len(source_indices) > max_points:
        rng = np.random.default_rng(seed)
        source_indices = np.sort(rng.choice(source_indices, size=max_points, replace=False))

    v, u = np.divmod(source_indices, width)
    z = depth_arr.reshape(-1)[source_indices]
    xyz = np.column_stack(((u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z)).astype(np.float32)
    rgb01 = rgb_arr.reshape(-1, 3)[source_indices].astype(np.float32) / 255.0
    return xyz, rgb01, source_indices.astype(np.int64)


def validate_reconstruction(
    rgb: np.ndarray,
    depth: np.ndarray,
    validity_mask: np.ndarray,
    xyz: np.ndarray,
    colors: np.ndarray,
    source_indices: np.ndarray,
    intrinsics: Dict[str, float],
    depth_semantics: str,
) -> Dict[str, Any]:
    """Run fail-closed numerical, registration, reprojection, and collapse checks."""
    rgb_arr = np.asarray(rgb)
    d = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(validity_mask, dtype=bool)
    points = np.asarray(xyz, dtype=np.float32)
    color_arr = np.asarray(colors, dtype=np.float32)
    indices = np.asarray(source_indices, dtype=np.int64)
    errors = []
    rgb_depth_aligned = rgb_arr.ndim == 3 and rgb_arr.shape[2] == 3 and d.ndim == 2 and rgb_arr.shape[:2] == d.shape
    if not rgb_depth_aligned:
        errors.append("rgb_depth_registration_mismatch")
    mask_aligned = mask.shape == d.shape
    if not mask_aligned:
        errors.append("validity_mask_shape_mismatch")
    if not rgb_depth_aligned or not mask_aligned:
        return {
            "is_valid": False,
            "errors": errors,
            "depth_semantics": depth_semantics,
            "finite_depth_ratio": float(np.isfinite(d).mean()) if d.size else 0.0,
            "valid_depth_ratio": 0.0,
            "invalid_depth_ratio": float(1.0 - np.isfinite(d).mean()) if d.size else 1.0,
            "point_count": int(len(points)) if points.ndim else 0,
            "bounds": None,
            "robust_extent_p01_p99": [],
            "covariance_eigenvalues": [],
            "collapsed": True,
            "source_pixel_mapping_valid": False,
            "xyz_reprojection_max_error_px": None,
        }
    if points.ndim != 2 or points.shape[1:] != (3,) or color_arr.shape != points.shape or len(indices) != len(points):
        errors.append("point_attribute_count_mismatch")
    finite_depth = np.isfinite(d)
    valid_pixels = mask & finite_depth & (d > 0) if mask_aligned else np.zeros_like(d, dtype=bool)
    if not finite_depth.any() or valid_pixels.sum() == 0:
        errors.append("no_valid_depth_pixels")
    invalid_fraction = float(1.0 - valid_pixels.mean()) if d.size else 1.0
    if not np.isfinite(points).all() or not np.isfinite(color_arr).all():
        errors.append("non_finite_point_or_color")
    if len(points) == 0:
        errors.append("empty_point_cloud")

    bounds = None
    covariance_eigenvalues = []
    robust_extent = []
    collapsed = True
    reprojection_max_error = None
    source_mapping_matches = False
    if len(points) and np.isfinite(points).all():
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        bounds = {"min": bounds_min.tolist(), "max": bounds_max.tolist(), "span": (bounds_max - bounds_min).tolist()}
        robust_extent = (np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)).tolist()
        centered = points - points.mean(axis=0, keepdims=True)
        eigen = np.linalg.eigvalsh(np.cov(centered, rowvar=False)) if len(points) > 3 else np.zeros(3)
        covariance_eigenvalues = np.maximum(eigen, 0).tolist()
        positive = np.asarray(robust_extent) > 1e-4
        collapsed = int(positive.sum()) < 2 or (eigen[-2] <= max(eigen[-1] * 1e-6, 1e-10))
        if collapsed:
            errors.append("collapsed_or_rank_deficient_geometry")
        if len(indices) and rgb_depth_aligned:
            h, w = d.shape
            source_mapping_matches = bool(np.all((indices >= 0) & (indices < h * w)))
            if source_mapping_matches:
                vv, uu = np.divmod(indices, w)
                k, _ = normalize_intrinsics(w, h, intrinsics)
                projected_u = points[:, 0] * k["fx"] / points[:, 2] + k["cx"]
                projected_v = points[:, 1] * k["fy"] / points[:, 2] + k["cy"]
                reprojection_max_error = float(max(np.max(np.abs(projected_u - uu)), np.max(np.abs(projected_v - vv))))
                if reprojection_max_error > 1e-3:
                    errors.append("xyz_pixel_reprojection_failed")
                expected_colors = rgb.reshape(-1, 3)[indices].astype(np.float32) / 255.0
                if not np.allclose(expected_colors, color_arr, atol=1 / 255 + 1e-6):
                    errors.append("rgb_source_pixel_mismatch")
                if not np.allclose(points[:, 2], d.reshape(-1)[indices], rtol=1e-5, atol=1e-5):
                    errors.append("point_depth_mismatch")
            else:
                errors.append("source_pixel_indices_out_of_range")
        elif len(points):
            errors.append("source_pixel_mapping_unavailable")
        if not np.isfinite(color_arr).all() or (len(color_arr) and (color_arr.min() < 0 or color_arr.max() > 1)):
            errors.append("invalid_rgb_range")

    if depth_semantics != "metric_model_estimate":
        errors.append(f"depth_semantics_not_accepted:{depth_semantics}")
    return {
        "is_valid": not errors,
        "errors": errors,
        "depth_semantics": depth_semantics,
        "finite_depth_ratio": float(finite_depth.mean()) if d.size else 0.0,
        "valid_depth_ratio": float(valid_pixels.mean()) if d.size else 0.0,
        "invalid_depth_ratio": invalid_fraction,
        "point_count": int(len(points)),
        "bounds": bounds,
        "robust_extent_p01_p99": robust_extent,
        "covariance_eigenvalues": covariance_eigenvalues,
        "collapsed": bool(collapsed),
        "source_pixel_mapping_valid": source_mapping_matches,
        "xyz_reprojection_max_error_px": reprojection_max_error,
    }


def save_rgb_png(image: Image.Image, path: Path) -> None:
    """Write decoded RGB as a lossless PNG without mutating the source image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=False)


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
