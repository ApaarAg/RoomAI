"""Run the deployed composite adapter plus canonical artifact persistence on one held-out Hypersim frame.

This diagnostic is synthetic-render evidence only and must never be presented as a real room exhibit.
"""

from __future__ import annotations

import json
import shutil
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adapters.roomai_composite import RoomAICompositeAdapter
from services.job_service import JobService
from utils.reconstruction_contract import sha256_file


SCENE = ROOT / "research/3d_reconstruction/data/ground_truth/standardized/hypersim/test/ai_013_010"
OUTPUT = ROOT / "asset-production/roomai/01-review/phase4_2-hypersim-ai_013_010-frame_0000-run08"


def main(output: Path = OUTPUT) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostic artifacts in {output}")
    output.mkdir(parents=True, exist_ok=True)
    rgb_path = SCENE / "frame_0000_rgb.png"
    depth_path = SCENE / "frame_0000_depth.npy"
    camera_path = SCENE / "frame_0000_cam.npz"
    image = Image.open(rgb_path).convert("RGB")
    camera = np.load(camera_path, allow_pickle=False)
    k_matrix = camera["intrinsics"].astype(np.float64)
    intrinsics = {"fx": float(k_matrix[0, 0]), "fy": float(k_matrix[1, 1]),
                  "cx": float(k_matrix[0, 2]), "cy": float(k_matrix[1, 2])}

    source_copy = output / "source" / "view_00.png"
    camera_copy = output / "source" / "view_00_cam.npz"
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    if not source_copy.exists():
        shutil.copy2(rgb_path, source_copy)
    if not camera_copy.exists():
        shutil.copy2(camera_path, camera_copy)
    options = {
        "camera_intrinsics": intrinsics,
        "source_artifacts": [{
            "view_index": 0,
            "source_original_path": str(source_copy.relative_to(output)),
            "source_original_sha256": sha256_file(source_copy),
            "source_original_bytes": source_copy.stat().st_size,
            "camera_calibration_evidence_path": str(camera_copy.relative_to(output)),
            "camera_calibration_evidence_sha256": sha256_file(camera_copy),
            "license": "Hypersim CC BY-SA 3.0; V-Ray synthetic render; internal diagnostic only",
        }],
        "camera_calibration_verified": True,
        "canonical_max_points": 200000,
        "desktop_max_points": 40000,
        "mobile_max_points": 15000,
        "sampling_seed": 0,
    }

    adapter = RoomAICompositeAdapter(mock_mode=False)
    scene = adapter.reconstruct([image], options=options)
    service = object.__new__(JobService)
    bundle = service._persist_and_validate_bundle(
        scene, [image], output, "roomai_composite_v1", options
    )

    ground_truth = np.load(depth_path, allow_pickle=False).astype(np.float32)
    predicted = np.asarray(scene.depth_map, dtype=np.float32).squeeze()
    valid = np.isfinite(ground_truth) & (ground_truth > 0) & np.isfinite(predicted) & (predicted > 0)
    delta = predicted[valid] - ground_truth[valid]
    abs_rel = float(np.mean(np.abs(delta) / ground_truth[valid]))
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    scale_ratio = float(np.median(ground_truth[valid]) / np.median(predicted[valid]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(image)
    axes[0].set_title("Held-out synthetic RGB")
    axes[1].imshow(predicted, cmap="magma")
    axes[1].set_title("Raw model depth (m estimate)")
    axes[2].imshow(ground_truth, cmap="magma")
    axes[2].set_title("Hypersim reference depth (m)")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "depth_comparison_qa.png", dpi=140)
    plt.close(fig)

    cloud = np.load(output / "point_cloud_canonical.npz", allow_pickle=False)
    xyz = cloud["xyz"]
    colors = cloud["rgb"]
    view = output / "pointcloud_orthographic_qa.png"
    fig = plt.figure(figsize=(12, 4))
    for idx, (a, b, title) in enumerate(((0, 2, "Front"), (0, 1, "Top"), (2, 1, "Side")), 1):
        ax = fig.add_subplot(1, 3, idx)
        ax.scatter(xyz[::max(1, len(xyz) // 20000), a], xyz[::max(1, len(xyz) // 20000), b],
                   c=colors[::max(1, len(xyz) // 20000)], s=0.2, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(view, dpi=160)
    plt.close(fig)

    report = {
        "test_kind": "held-out Hypersim V-Ray synthetic render; diagnostic only, not a photograph or product evidence",
        "license": options["source_artifacts"][0]["license"],
        "scene_id": "ai_013_010",
        "frame_id": 0,
        "rgb_sha256": sha256_file(rgb_path),
        "ground_truth_depth_sha256": sha256_file(depth_path),
        "camera_intrinsics_pixel_units": intrinsics,
        "depth": {
            "shape": list(predicted.shape), "finite_ratio": float(np.isfinite(predicted).mean()),
            "median_prediction_m": float(np.median(predicted[valid])),
            "median_ground_truth_m": float(np.median(ground_truth[valid])),
            "unscaled_abs_rel": abs_rel, "unscaled_rmse_m": rmse,
            "ground_truth_to_prediction_median_ratio": scale_ratio,
            "scale_fitted": False,
        },
        "bundle_acceptance": bundle["acceptance"],
        "bundle_artifacts": bundle["artifacts"],
        "visual_qa": ["depth_comparison_qa.png", "pointcloud_orthographic_qa.png"],
    }
    (output / "pipeline_test_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="New, empty diagnostic output directory")
    main(parser.parse_args().output.resolve())
