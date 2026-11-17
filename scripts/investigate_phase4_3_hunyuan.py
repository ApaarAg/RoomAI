"""Forensic-only capture of Hunyuan native output heads on an existing synthetic Hypersim frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters.hunyuan_mirror_adapter import resolve_hunyuan_checkpoint_path
from services.hunyuan_mirror_service import HunyuanWorldMirrorAdapter
from utils.reconstruction_contract import sha256_file

SOURCE = ROOT / "research/3d_reconstruction/data/ground_truth/standardized/hypersim/test/ai_013_010/frame_0000_rgb.png"


def summarize(value):
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(item) for item in value]
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else None
        finite_values = array[finite] if finite is not None and finite.any() else None
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite_ratio": float(finite.mean()) if finite is not None and array.size else None,
            "min": float(finite_values.min()) if finite_values is not None and finite_values.size else None,
            "median": float(np.median(finite_values)) if finite_values is not None and finite_values.size else None,
            "max": float(finite_values.max()) if finite_values is not None and finite_values.size else None,
            "mean": float(finite_values.mean()) if finite_values is not None and finite_values.size else None,
            "std": float(finite_values.std()) if finite_values is not None and finite_values.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_arrays(value, prefix=""):
    arrays = {}
    if isinstance(value, dict):
        for key, item in value.items():
            arrays.update(collect_arrays(item, f"{prefix}{key}__"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            arrays.update(collect_arrays(item, f"{prefix}{index}__"))
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        arrays[prefix.rstrip("_")] = value
    return arrays


def tensor_outputs_to_numpy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, dict):
        return {key: tensor_outputs_to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_outputs_to_numpy(item) for item in value]
    return value


def reprojection_summary(points, intrinsic, c2w=None):
    points = np.asarray(points, dtype=np.float64)
    height, width = points.shape[:2]
    if c2w is not None:
        world_to_camera = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
        points_h = np.concatenate([points.reshape(-1, 3), np.ones((height * width, 1))], axis=1)
        camera_points = (points_h @ world_to_camera.T)[:, :3].reshape(height, width, 3)
    else:
        camera_points = points
    z = camera_points[..., 2]
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    v, u = np.indices((height, width), dtype=np.float64)
    valid = np.isfinite(camera_points).all(axis=-1) & (z > 1e-8)
    if not valid.any():
        return {"valid_projected_pixel_ratio": 0.0, "median_reprojection_error_px": None, "p95_reprojection_error_px": None}
    u_proj = fx * camera_points[..., 0] / np.maximum(z, 1e-8) + cx
    v_proj = fy * camera_points[..., 1] / np.maximum(z, 1e-8) + cy
    error = np.sqrt((u_proj[valid] - u[valid]) ** 2 + (v_proj[valid] - v[valid]) ** 2)
    return {
        "valid_projected_pixel_ratio": float(valid.mean()),
        "median_reprojection_error_px": float(np.median(error)),
        "p95_reprojection_error_px": float(np.percentile(error, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty forensic directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    checkpoint = resolve_hunyuan_checkpoint_path()
    config = yaml.safe_load((ROOT / "config/model_config.yaml").read_text(encoding="utf-8"))
    expected_sha = str(config["hunyuan_world_mirror_1b"].get("expected_sha256", "")).lower()
    actual_sha = sha256(checkpoint) if checkpoint.is_file() else None
    image = Image.open(SOURCE).convert("RGB")
    adapter = HunyuanWorldMirrorAdapter(checkpoint_path=str(checkpoint), device="cuda", img_size=518)
    if not adapter.is_ready:
        report = {
            "status": "BLOCKED",
            "reason": "checkpoint/model adapter did not initialize; no inference outputs were fabricated",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": actual_sha,
            "expected_checkpoint_sha256": expected_sha or None,
            "checkpoint_hash_matches_expected": actual_sha == expected_sha if expected_sha else None,
        }
        (output / "hunyuan_forensic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    import torch
    tensor, _ = adapter.preprocess_image(image)
    batch = torch.stack([tensor], dim=0).unsqueeze(0)
    with torch.no_grad():
        raw_model_outputs = adapter.model(
            views={"img": batch}, cond_flags=[0, 0, 0], is_inference=True,
        )
    native = tensor_outputs_to_numpy(raw_model_outputs)
    transform = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    native["coordinate_transform_model_to_roomai"] = transform
    np.savez_compressed(output / "hunyuan_native_outputs.npz", **collect_arrays(native))

    result = adapter.reconstruct(image, confidence_threshold=0.5, max_points=40000)
    service_native = result.get("native_outputs") or {}
    points_raw = np.asarray(native.get("pts3d"))
    depth_raw = np.asarray(native.get("depth"))
    confidence_raw = native.get("pts3d_conf")
    camera_raw = {
        key: native.get(key)
        for key in ("camera_params", "camera_poses", "camera_intrs")
    }
    exported_points = np.asarray(result.get("point_cloud_xyz", []), dtype=np.float32)

    # Remove only known batch/view dimensions; never reshape a mismatched tensor to fit.
    point_map = points_raw[0, 0] if points_raw.ndim == 5 and points_raw.shape[0:2] == (1, 1) else None
    if point_map is None and points_raw.ndim == 5 and points_raw.shape[0:2] == (1, 1):
        point_map = points_raw[0, 0]
    if point_map is None and points_raw.ndim == 4 and points_raw.shape[0] == 1:
        point_map = points_raw[0]
    depth_map = depth_raw[0, 0] if depth_raw.ndim >= 4 and depth_raw.shape[0:2] == (1, 1) else None
    if depth_map is not None:
        depth_map = np.squeeze(depth_map)
    point_roomai = point_map @ transform.T if point_map is not None and transform.shape == (3, 3) else None

    geometry = {}
    if point_map is not None and point_roomai is not None:
        finite_points = np.isfinite(point_roomai).all(axis=-1)
        raw_points = point_roomai[finite_points]
        resized_rgb = np.asarray(image.resize((point_roomai.shape[1], point_roomai.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        raw_colors = resized_rgb[finite_points]
        if len(raw_points):
            bounds_min, bounds_max = raw_points.min(axis=0), raw_points.max(axis=0)
            cov = np.cov(raw_points, rowvar=False) if len(raw_points) > 3 else np.zeros((3, 3))
            eigen = np.maximum(np.linalg.eigvalsh(cov), 0.0)
            geometry = {
                "raw_point_map_finite_ratio": float(finite_points.mean()),
                "raw_point_map_count": int(len(raw_points)),
                "raw_point_map_bounds_min": bounds_min.tolist(),
                "raw_point_map_bounds_max": bounds_max.tolist(),
                "raw_point_map_span": (bounds_max - bounds_min).tolist(),
                "raw_point_map_covariance_eigenvalues": eigen.tolist(),
                "raw_point_map_pca_std": np.sqrt(eigen).tolist(),
                "raw_point_map_collapsed": bool(np.count_nonzero((bounds_max - bounds_min) > 1e-4) < 2 or eigen[1] <= max(eigen[2] * 1e-6, 1e-10)),
            }
            fig = plt.figure(figsize=(12, 4))
            for idx, (a, b, title) in enumerate(((0, 2, "Front"), (0, 1, "Top"), (2, 1, "Side")), 1):
                axis = fig.add_subplot(1, 3, idx)
                axis.scatter(raw_points[::max(1, len(raw_points) // 20000), a],
                             raw_points[::max(1, len(raw_points) // 20000), b],
                             c=raw_colors[::max(1, len(raw_points) // 20000)], s=0.2, linewidths=0)
                axis.set_title(title)
                axis.set_aspect("equal", adjustable="datalim")
            fig.tight_layout()
            fig.savefig(output / "pointcloud_orthographic_qa.png", dpi=160)
            plt.close(fig)
    depth_point_compare = None
    if depth_map is not None and point_map is not None and depth_map.shape == point_map.shape[:2]:
        z_point = point_roomai[..., 2]
        joint = np.isfinite(depth_map) & np.isfinite(z_point) & (depth_map > 0) & (np.abs(z_point) > 0)
        if joint.any():
            depth_point_compare = {
                "joint_pixel_count": int(joint.sum()),
                "median_depth_head": float(np.median(depth_map[joint])),
                "median_abs_point_z": float(np.median(np.abs(z_point[joint]))),
                "median_abs_z_over_depth": float(np.median(np.abs(z_point[joint]) / np.maximum(np.abs(depth_map[joint]), 1e-12))),
                "pearson_z_depth": float(np.corrcoef(np.abs(z_point[joint]).ravel(), np.abs(depth_map[joint]).ravel())[0, 1]),
                "scale_fit_applied": False,
            }
    camera_pose = native.get("camera_poses")
    camera_intrinsics = native.get("camera_intrs")
    camera_consistency = None
    if point_map is not None and camera_pose is not None and camera_intrinsics is not None:
        predicted_pose = np.asarray(camera_pose)[0, 0]
        predicted_k = np.asarray(camera_intrinsics)[0, 0]
        reference_camera = np.load(SOURCE.with_name("frame_0000_cam.npz"), allow_pickle=False)
        camera_consistency = {
            "predicted_intrinsics": predicted_k.tolist(),
            "reference_intrinsics": reference_camera["intrinsics"].astype(float).tolist(),
            "predicted_fx_over_reference_fx": float(predicted_k[0, 0] / reference_camera["intrinsics"][0, 0]),
            "direct_camera_frame_point_map": reprojection_summary(point_map, predicted_k),
            "predicted_world_frame_point_map": reprojection_summary(point_map, predicted_k, predicted_pose),
            "coordinate_convention_status": "unresolved; both model-point coordinate hypotheses are reported without selecting one",
        }
    report = {
        "status": "FORENSIC_ONLY",
        "evidence_class": "Hypersim V-Ray synthetic render; no real camera evidence",
        "source_rgb": str(SOURCE.relative_to(ROOT)),
        "source_rgb_sha256": sha256(SOURCE),
        "source_rgb_size": list(image.size),
        "rights": "Hypersim CC BY-SA 3.0; internal synthetic diagnostic only",
        "checkpoint_path": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "expected_checkpoint_sha256": expected_sha or None,
        "checkpoint_hash_matches_expected": actual_sha == expected_sha if expected_sha else None,
        "runtime": {"device": adapter.device, "dtype": str(adapter.dtype), "image_size": adapter.img_size},
        "preprocessing": {
            "source_width_height": list(image.size),
            "center_square_crop": True,
            "crop_xywh": [0, 0, image.width, image.height],
            "resize": "PIL bilinear to 518x518",
            "tensor": "RGB float32/255, CHW, converted to model dtype/device; batch B=1,V=1",
        },
        "raw_model_prediction_heads": summarize(native),
        "adapter_preserved_native_heads": summarize(service_native),
        "adapter_dropped_or_mis-keyed_heads": [
            key for key in ("camera_params", "camera_poses", "camera_intrs", "depth_conf", "normals_conf")
            if key in native and key not in service_native
        ],
        "camera_predictions": summarize(camera_raw),
        "point_confidence": summarize(confidence_raw),
        "native_geometry": geometry,
        "depth_vs_point_z": depth_point_compare,
        "camera_geometry_consistency": camera_consistency,
        "returned_filtered_point_cloud": summarize(exported_points),
        "native_depth_matches_primary_depth_shape": bool(depth_map is not None and result.get("primary_depth") is not None and depth_map.shape == np.asarray(result["primary_depth"]).shape),
        "confidence_threshold_parameter": 0.5,
        "confidence_threshold_applied_by_current_adapter": False,
        "production_enabled": False,
        "visual_qa": "pointcloud_orthographic_qa.png" if geometry else None,
        "production_disposition": "REVIEW_REQUIRED: scene-normalized semantics, geometry acceptance, model/head semantics and license applicability still require review",
    }
    (output / "hunyuan_forensic_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
