"""Comprehensive Benchmark and Validation Script for RoomAI Phase 7.3 Multi-View 3D Reconstruction.

Evaluates:
1. SIFT + FLANN feature extraction, matching, and 3D-3D Kabsch/Umeyama rigid registration.
2. Single-view guided edge-preserving depth refinement vs raw metric depth.
3. Multi-view point cloud fusion, voxel downsampling (3cm), and statistical outlier removal.
4. Multi-view 3D object association and bounding envelope expansion.
5. Export of consolidated multi-view scenes to GLB and PLY.
6. Execution latency, VRAM consumption, and registration inlier ratios.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.multiview_fusion_service import MultiViewFusionService
from utils.export_3d import export_point_cloud_ply, export_scene_glb


def run_multiview_benchmark():
    print("=" * 75)
    print("ROOMAI PHASE 7.3: MULTI-VIEW 3D RECONSTRUCTION BENCHMARK")
    print("=" * 75)

    with open("config/model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    val_dir = Path("archive (2)/ADEChallengeData2016/images/validation")
    img_files = ["ADE_val_00000001.jpg", "ADE_val_00000002.jpg", "ADE_val_00000005.jpg"]

    images = [Image.open(val_dir / f).convert("RGB") for f in img_files]
    print(f"Loaded {len(images)} room validation perspectives from ADE20K.")

    mv_service = MultiViewFusionService(mock_mode=False, config=model_cfg)
    mv_service.load_models()
    print("Loaded consolidated models on CUDA successfully.")

    out_dir = Path("artifacts/phase_7_3_multiview")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Single View Baseline vs. Guided Depth Refinement
    print("\n[Step 1] Benchmarking Single-View Raw vs Guided Depth Refinement...")
    t0 = time.perf_counter()
    raw_scene = mv_service.fuse_multiview([images[0]], refine_depth=False)
    t_raw = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    refined_scene = mv_service.fuse_multiview([images[0]], refine_depth=True)
    t_refined = (time.perf_counter() - t0) * 1000

    print(f"  Raw Scene: {t_raw:.1f} ms | Points: {raw_scene.n_valid_points:,}")
    print(f"  Refined Scene: {t_refined:.1f} ms | Points: {refined_scene.n_valid_points:,}")

    # 2. Multi-View Fusion (2 Views)
    print("\n[Step 2] Benchmarking 2-View Room Scene Fusion...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    fused_2v = mv_service.fuse_multiview([images[0], images[1]], refine_depth=True, voxel_size=0.03)
    t_2v = (time.perf_counter() - t0) * 1000
    vram_2v = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"  2-View Fused Scene: {t_2v:.1f} ms | Fused Points: {fused_2v.n_valid_points:,} | Objects: {len(fused_2v.objects)}")
    print(f"  Peak Allocated VRAM: {vram_2v:.1f} MB")

    tel_2v = getattr(fused_2v, "multiview_telemetry", [])
    if tel_2v:
        t_info = tel_2v[0]
        print(f"  Alignment: Baseline {t_info['baseline_m']} m | Rotation: {t_info['rotation_deg']}° | 3D Inliers: {t_info['inliers_count']} | RMSE: {t_info['rmse_m']} m")

    # 3. Multi-View Fusion (3 Views)
    print("\n[Step 3] Benchmarking 3-View Room Scene Fusion...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    fused_3v = mv_service.fuse_multiview([images[0], images[1], images[2]], refine_depth=True, voxel_size=0.03)
    t_3v = (time.perf_counter() - t0) * 1000
    vram_3v = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"  3-View Fused Scene: {t_3v:.1f} ms | Fused Points: {fused_3v.n_valid_points:,} | Objects: {len(fused_3v.objects)}")
    print(f"  Peak Allocated VRAM: {vram_3v:.1f} MB")

    # 4. Export Consolidated Multi-View Scene
    print("\n[Step 4] Exporting Consolidated Multi-View 3D Scene...")
    glb_bytes = export_scene_glb(fused_2v.point_cloud, fused_2v.objects, output_path=out_dir / "multiview_fused_scene.glb")
    ply_bytes = export_point_cloud_ply(fused_2v.point_cloud.get("xyz", []), fused_2v.point_cloud.get("rgb", []), output_path=out_dir / "multiview_fused_scene.ply")
    print(f"  Exported GLB ({len(glb_bytes):,} bytes) and PLY ({len(ply_bytes):,} bytes).")

    # Save empirical record
    def sanitize_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    results = sanitize_for_json({
        "single_view_raw_ms": round(t_raw, 1),
        "single_view_refined_ms": round(t_refined, 1),
        "two_view_fusion_ms": round(t_2v, 1),
        "three_view_fusion_ms": round(t_3v, 1),
        "peak_vram_mb": round(vram_3v, 1),
        "single_view_points": raw_scene.n_valid_points,
        "two_view_points": fused_2v.n_valid_points,
        "three_view_points": fused_3v.n_valid_points,
        "fused_objects_count": len(fused_2v.objects),
        "registration_telemetry": tel_2v,
        "glb_bytes": len(glb_bytes),
        "ply_bytes": len(ply_bytes),
    })

    res_path = Path("artifacts/phase_7_3_multiview_results.json")
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEmpirical benchmark results written to {res_path}")
    print("=" * 75)
    print("MULTI-VIEW RECONSTRUCTION BENCHMARK COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_multiview_benchmark()

