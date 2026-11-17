"""Phase 7.4.2: Authentic Residential Indoor Reconstruction Benchmark.

Evaluates 40 authentic residential indoor scenes from ADE20K:
- 8 Living Rooms
- 8 Bedrooms
- 8 Kitchens
- 6 Bathrooms
- 6 Dining Rooms
- 4 Home Offices

Measures:
1. Object detection & 3D bounding-box unprojection.
2. Raw unprojected physical dimensions and reliability flags.
3. Raw predicted floor-contact error vs post-alignment floor-contact error.
4. RANSAC floor plane fitting (gravity score, inlier ratio, camera elevation).
5. VRAM consumption and end-to-end inference latency.

Outputs results to artifacts/phase_7_4_2_evaluation_results.json.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from services.scene_fusion_service import SceneFusionService
from utils.device import get_device_info


def run_benchmark():
    print("=" * 80)
    print("ROOMAI PHASE 7.4.2: AUTHENTIC RESIDENTIAL INDOOR RECONSTRUCTION BENCHMARK")
    print("=" * 80)

    manifest_p = Path("artifacts/phase_7_4_2_indoor_dataset_manifest.json")
    if not manifest_p.exists():
        from scripts.curate_phase_7_4_2_indoor_dataset import curate_indoor_dataset
        manifest = curate_indoor_dataset()
    else:
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    val_dir = Path("archive (2)/ADEChallengeData2016/images/validation")
    print(f"Loaded {len(manifest)} authentic residential scenes from {manifest_p}.")

    # Initialize service with production weights
    service = SceneFusionService(mock_mode=False)
    service.load_models()
    print("Production models loaded on CUDA successfully.")

    results: List[Dict[str, Any]] = []

    total_t0 = time.perf_counter()
    raw_floor_errors = []
    post_floor_errors = []
    dimension_reliability_flags = []
    detected_object_counts = []
    floor_detected_count = 0
    gravity_scores = []

    for item in manifest:
        idx = item["index"]
        fn = item["filename"]
        cat = item["scene_category"]
        img_path = val_dir / fn

        if not img_path.exists():
            print(f"Warning: Image {img_path} not found; skipping.")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        t0 = time.perf_counter()
        scene = service.fuse(img)
        latency_ms = (time.perf_counter() - t0) * 1000

        dev_info = get_device_info()
        vram_mb = dev_info.get("allocated_vram_mb", 0.0)

        # Extract objects and geometry metrics
        scene_objs = []
        for obj in scene.objects:
            g = obj.geometry
            if g and g.centroid and g.extent:
                raw_floor_err = getattr(g, "raw_floor_contact_delta_m", None)
                post_floor_err = getattr(g, "post_floor_contact_delta_m", None)
                is_rel = getattr(g, "is_dimension_reliable", True)
                warn = getattr(g, "dimension_warning", None)

                if raw_floor_err is not None:
                    raw_floor_errors.append(raw_floor_err)
                if post_floor_err is not None:
                    post_floor_errors.append(post_floor_err)
                dimension_reliability_flags.append(is_rel)

                scene_objs.append({
                    "object_id": obj.object_id,
                    "class": obj.semantic_class,
                    "confidence": round(float(obj.detection_confidence), 3),
                    "centroid_m": [round(c, 2) for c in g.centroid],
                    "raw_extent_m": [round(e, 2) for e in g.extent],
                    "raw_floor_delta_m": raw_floor_err,
                    "post_floor_delta_m": post_floor_err,
                    "is_dimension_reliable": is_rel,
                    "dimension_warning": warn,
                })

        detected_object_counts.append(len(scene_objs))

        # Extract layout metrics
        layout = scene.room_layout or {}
        floor_det = layout.get("floor_plane_detected", False)
        if floor_det:
            floor_detected_count += 1
        grav_score = layout.get("gravity_alignment_score", 0.0)
        gravity_scores.append(grav_score)

        record = {
            "index": idx,
            "filename": fn,
            "category": cat,
            "resolution": [w, h],
            "latency_ms": round(latency_ms, 1),
            "vram_mb": round(vram_mb, 1),
            "n_valid_points": scene.n_valid_points,
            "depth_range_m": [round(scene.depth_min_m, 2), round(scene.depth_max_m, 2)],
            "floor_plane_detected": floor_det,
            "floor_normal": layout.get("floor_normal", []),
            "gravity_score": grav_score,
            "floor_inlier_ratio": layout.get("floor_inlier_ratio", 0.0),
            "floor_residual_m": layout.get("floor_residual_m", 0.0),
            "camera_height_m": layout.get("camera_height_m", 1.4),
            "room_dimensions_m": [
                layout.get("room_width_m", 0.0),
                layout.get("room_length_m", 0.0),
                layout.get("room_height_m", 0.0),
            ],
            "objects_count": len(scene_objs),
            "objects": scene_objs,
        }
        results.append(record)
        print(f"[{idx:02d}/40] {fn} ({cat:<12}) -> {len(scene_objs)} objs | floor: {floor_det} (grav: {grav_score:.2f}) | {latency_ms:.0f}ms")

    total_time_s = time.perf_counter() - total_t0

    # Summary statistics
    summary = {
        "total_scenes_evaluated": len(results),
        "total_objects_detected": sum(detected_object_counts),
        "mean_objects_per_scene": round(float(np.mean(detected_object_counts)), 2),
        "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in results])), 1),
        "peak_vram_mb": round(float(np.max([r["vram_mb"] for r in results])), 1),
        "floor_plane_detection_rate": round(floor_detected_count / max(1, len(results)) * 100, 1),
        "mean_gravity_alignment_score": round(float(np.mean(gravity_scores)), 3),
        "mean_raw_floor_contact_delta_m": round(float(np.mean(raw_floor_errors)), 3) if raw_floor_errors else 0.0,
        "mean_post_floor_contact_delta_m": round(float(np.mean(post_floor_errors)), 3) if post_floor_errors else 0.0,
        "dimension_reliability_rate": round(float(np.mean(dimension_reliability_flags)) * 100, 1) if dimension_reliability_flags else 100.0,
        "total_benchmark_time_s": round(total_time_s, 2),
    }

    final_payload = {
        "summary": summary,
        "per_scene_results": results,
    }

    out_p = Path("artifacts/phase_7_4_2_evaluation_results.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 7.4.2 BENCHMARK COMPLETE")
    print(f"  Total Scenes: {summary['total_scenes_evaluated']}")
    print(f"  Total Objects Detected: {summary['total_objects_detected']} (Avg: {summary['mean_objects_per_scene']} / scene)")
    print(f"  Floor Plane Detection Rate: {summary['floor_plane_detection_rate']}%")
    print(f"  Mean Gravity Score: {summary['mean_gravity_alignment_score']}")
    print(f"  Mean RAW Floor Contact Delta: {summary['mean_raw_floor_contact_delta_m']} m")
    print(f"  Mean POST Floor Contact Delta: {summary['mean_post_floor_contact_delta_m']} m (snapped)")
    print(f"  Dimension Reliability Rate: {summary['dimension_reliability_rate']}%")
    print(f"  Mean Latency: {summary['mean_latency_ms']} ms")
    print(f"  Peak VRAM: {summary['peak_vram_mb']} MB")
    print(f"  Results saved to: {out_p}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

