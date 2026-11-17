"""Evaluate the deployed Depth Anything V2 Metric Indoor Small model on multiple standardized Hypersim scenes.

This is synthetic-render validation only. The runner never fits scale and never
labels monocular predictions as calibrated ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.scene3d_model import CustomScene3DModel
from utils.reconstruction_contract import depth_to_xyz, sha256_path, validate_reconstruction

DATA = ROOT / "research/3d_reconstruction/data/ground_truth/standardized/hypersim"
CHECKPOINT = ROOT / "checkpoints/depth_anything_v2_metric_indoor_small"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_frame(model: CustomScene3DModel, split: str, scene_id: str, rgb_path: Path, output: Path) -> dict:
    stem = rgb_path.name.removesuffix("_rgb.png")
    depth_path = rgb_path.with_name(f"{stem}_depth.npy")
    camera_path = rgb_path.with_name(f"{stem}_cam.npz")
    mask_path = rgb_path.with_name(f"{stem}_mask.png")
    if not all(path.is_file() for path in (depth_path, camera_path, mask_path)):
        raise FileNotFoundError(f"Incomplete standardized frame: {rgb_path}")

    image = Image.open(rgb_path).convert("RGB")
    rgb = np.asarray(image)
    gt = np.load(depth_path, allow_pickle=False).astype(np.float32)
    cam = np.load(camera_path, allow_pickle=False)
    k_matrix = cam["intrinsics"].astype(np.float64)
    k = {"fx": float(k_matrix[0, 0]), "fy": float(k_matrix[1, 1]),
         "cx": float(k_matrix[0, 2]), "cy": float(k_matrix[1, 2])}
    source_mask = np.asarray(Image.open(mask_path).convert("L")) > 0

    _, response = model.reconstruct(image)
    pred = np.asarray(model.last_raw_depth, dtype=np.float32)
    if pred.shape != gt.shape or rgb.shape[:2] != gt.shape or source_mask.shape != gt.shape:
        raise ValueError(f"RGB/depth/mask/prediction alignment mismatch in {scene_id}/{stem}")

    gt_valid = source_mask & np.isfinite(gt) & (gt > 0.05) & (gt < 50.0)
    pred_valid = np.isfinite(pred) & (pred > 0.05) & (pred < 30.0)
    both = gt_valid & pred_valid
    if not both.any():
        raise ValueError(f"No jointly valid depth pixels in {scene_id}/{stem}")
    ratio = np.maximum(pred[both] / gt[both], gt[both] / pred[both])
    delta = pred[both] - gt[both]
    xyz, colors, pixels = depth_to_xyz(pred, rgb, k, validity_mask=pred_valid)
    recon = validate_reconstruction(rgb, pred, pred_valid, xyz, colors, pixels, k, "metric_model_estimate")
    covariance = np.asarray(recon["covariance_eigenvalues"], dtype=np.float64)
    pca = np.sqrt(np.maximum(covariance, 0.0)).tolist()

    result = {
        "split": split,
        "scene_id": scene_id,
        "frame": stem,
        "rgb_sha256": sha256(rgb_path),
        "ground_truth_depth_sha256": sha256(depth_path),
        "camera_intrinsics_pixel_units": k,
        "prediction_model": response.model_name,
        "prediction_device": response.device,
        "depth": {
            "ground_truth_valid_pixels": int(gt_valid.sum()),
            "prediction_valid_pixels": int((gt_valid & pred_valid).sum()),
            "coverage": float((gt_valid & pred_valid).sum() / max(1, gt_valid.sum())),
            "abs_rel": float(np.mean(np.abs(delta) / gt[both])),
            "rmse_m": float(np.sqrt(np.mean(delta ** 2))),
            "delta1": float(np.mean(ratio < 1.25)),
            "median_gt_over_prediction": float(np.median(gt[both]) / np.median(pred[both])),
            "median_prediction_m_estimate": float(np.median(pred[both])),
            "median_reference_m": float(np.median(gt[both])),
            "scale_fitted": False,
            "is_metric_calibrated": False,
        },
        "geometry": {
            "point_count": int(len(xyz)),
            "bounds": recon.get("bounds"),
            "robust_extent_p01_p99": recon.get("robust_extent_p01_p99"),
            "covariance_eigenvalues": covariance.tolist(),
            "pca_standard_deviation": pca,
            "pca_rank3_ratio": float(covariance[0] / max(covariance[-1], 1e-12)) if len(covariance) == 3 else 0.0,
            "reprojection_max_error_px": recon.get("xyz_reprojection_max_error_px"),
            "collapsed": recon.get("collapsed"),
            "validation_errors": recon.get("errors"),
        },
    }

    if stem.endswith("0000"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(image)
        axes[0].set_title(f"{split}/{scene_id} RGB")
        axes[1].imshow(np.where(pred_valid, pred, np.nan), cmap="magma")
        axes[1].set_title("DepthAnything V2 (meter estimate)")
        axes[2].imshow(np.where(gt_valid, gt, np.nan), cmap="magma")
        axes[2].set_title("Hypersim reference depth (m)")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / f"visual_qa_{split}_{scene_id}_{stem}.png", dpi=130)
        plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-scene", type=int, default=10)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty evaluation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    selected_scenes = [("test", "ai_013_010"), ("val", "ai_016_001")]
    frame_paths = []
    for split, scene_id in selected_scenes:
        scene_dir = DATA / split / scene_id
        frames = sorted(scene_dir.glob("frame_*_rgb.png"))
        if not frames:
            raise FileNotFoundError(f"No standardized RGB frames available for {split}/{scene_id}")
        n = min(args.frames_per_scene, len(frames))
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        frame_paths.extend((split, scene_id, frames[int(index)]) for index in indices)

    model = CustomScene3DModel(config={
        "checkpoint_path": str(CHECKPOINT),
        "device": "cuda:0",
        "depth_mode": "metric",
    })
    if not model.is_ready():
        raise RuntimeError(f"Depth model did not load: {model._init_error}")

    results = [eval_frame(model, split, scene_id, path, output) for split, scene_id, path in frame_paths]
    by_scene = {}
    for split, scene_id in selected_scenes:
        scene_results = [item for item in results if item["scene_id"] == scene_id]
        def stats(metric: str):
            values = np.asarray([item["depth"][metric] for item in scene_results], dtype=np.float64)
            return {"mean": float(values.mean()), "median": float(np.median(values)),
                    "min": float(values.min()), "max": float(values.max())}
        extents = np.asarray([item["geometry"]["robust_extent_p01_p99"] for item in scene_results], dtype=np.float64)
        bounds_spans = np.asarray([item["geometry"]["bounds"]["span"] for item in scene_results], dtype=np.float64)
        by_scene[f"{split}/{scene_id}"] = {
            "frame_count": len(scene_results),
            "abs_rel": stats("abs_rel"),
            "rmse_m": stats("rmse_m"),
            "delta1": stats("delta1"),
            "median_gt_over_prediction": stats("median_gt_over_prediction"),
            "valid_coverage": stats("coverage"),
            "point_count": {
                "mean": float(np.mean([item["geometry"]["point_count"] for item in scene_results])),
                "min": int(min(item["geometry"]["point_count"] for item in scene_results)),
                "max": int(max(item["geometry"]["point_count"] for item in scene_results)),
            },
            "median_robust_extent_p01_p99": np.median(extents, axis=0).tolist(),
            "median_aabb_span": np.median(bounds_spans, axis=0).tolist(),
            "all_clouds_noncollapsed": all(not item["geometry"]["collapsed"] for item in scene_results),
            "max_reprojection_error_px": max(item["geometry"]["reprojection_max_error_px"] for item in scene_results),
        }
    metric_names = ("abs_rel", "rmse_m", "delta1", "median_gt_over_prediction", "coverage")
    overall = {}
    for metric in metric_names:
        values = np.asarray([item["depth"][metric] for item in results], dtype=np.float64)
        overall[metric] = {"mean": float(values.mean()), "median": float(np.median(values)),
                           "min": float(values.min()), "max": float(values.max())}
    report = {
        "schema": "roomai.phase4_3_depth_benchmark",
        "dataset": "Hypersim standardized RGB + planar metric reference depth + camera intrinsics",
        "evidence_class": "synthetic_render_only; not real camera or production exhibit evidence",
        "model": "Depth Anything V2 Metric Indoor Small",
        "checkpoint_sha256": sha256_path(CHECKPOINT),
        "frame_count": len(results),
        "scene_count": len(by_scene),
        "scene_splits": ["test/ai_013_010", "val/ai_016_001"],
        "scale_fit_applied": False,
        "metric_model_output_is_calibrated_ground_truth": False,
        "provisional_gate_context": {
            "target_frames": 100,
            "target_scene_count": 5,
            "result_meets_sample_size": False,
            "reason": "Only two populated standardized Hypersim evaluation scenes are available locally; other test/validation scene directories contain no frames.",
        },
        "overall": overall,
        "per_scene": by_scene,
        "frames": results,
    }
    (output / "depth_benchmark_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "depth_benchmark_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("frame_count", "scene_count", "overall", "per_scene", "provisional_gate_context")}, indent=2))
    return 0


def render_markdown(report: dict) -> str:
    lines = [
        "# Phase 4.3 Held-Out Depth Evaluation",
        "",
        f"**Evidence class:** {report['evidence_class']}",
        "",
        f"Model: {report['model']}  ",
        f"Frames: {report['frame_count']} across {report['scene_count']} scenes  ",
        f"Checkpoint SHA-256: `{report['checkpoint_sha256']}`  ",
        "Scale fitting: none. A model output trained to predict metric depth is still an estimate, not camera-calibrated ground truth.",
        "",
        "## Aggregate frame metrics",
        "",
        "| Metric | Mean | Median | Range |",
        "|---|---:|---:|---:|",
    ]
    for key, value in report["overall"].items():
        lines.append(f"| {key} | {value['mean']:.6f} | {value['median']:.6f} | {value['min']:.6f}–{value['max']:.6f} |")
    lines += ["", "## Per-scene results", "", "| Split / scene | Frames | AbsRel mean | RMSE m mean | δ1 mean | Median GT/pred | Valid coverage | Median robust XYZ extent (m) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for scene, item in report["per_scene"].items():
        extent = ", ".join(f"{v:.3f}" for v in item["median_robust_extent_p01_p99"])
        lines.append(f"| {scene} | {item['frame_count']} | {item['abs_rel']['mean']:.5f} | {item['rmse_m']['mean']:.5f} | {item['delta1']['mean']:.5f} | {item['median_gt_over_prediction']['median']:.5f} | {item['valid_coverage']['mean']:.5f} | {extent} |")
    lines += ["", "## Gate status", "", "The sample is diagnostic only and does not satisfy the provisional 100-frame / five-scene evaluation minimum. Synthetic render performance cannot establish real-camera generalization. See the JSON for per-frame bounds, covariance/PCA, reprojection error, finite-depth coverage, hashes, and visual QA output names.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
