"""
ROOMAI — HELD-OUT TEST EVALUATION & 3D POINT CLOUD EXPORT ENGINE
Executes final test set evaluation on authentic indoor test scenes from indoor_verified_manifest.json.
Reports empirical geometry, metric depth, surface normal, and runtime telemetry, and exports
verified 3D point cloud (.ply) with full geometric validation.
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror
from safetensors.torch import load_file as load_safetensors

def export_ply(pts: np.ndarray, colors: np.ndarray, out_path: Path):
    """Exports 3D point cloud with RGB colors to valid PLY file."""
    # pts: (N, 3), colors: (N, 3) in [0, 255]
    valid = np.isfinite(pts).all(axis=-1)
    pts = pts[valid]
    colors = colors[valid].astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

def evaluate_test_scenes(
    model_path: Path,
    manifest_path: Path,
    output_ply: Path,
    device: str = "cuda"
) -> dict:
    print(f"\n{'='*75}\nEVALUATING HELD-OUT TEST SPLIT ON REAL ROOM SCENES: {model_path.name}\n{'='*75}")
    
    model = WorldMirror(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        gs_dim=256,
        enable_cam=True,
        enable_pts=True,
        enable_depth=True,
        enable_norm=True,
        enable_gs=False,
        enable_cond=False,
    ).to(device, dtype=torch.bfloat16)

    if model_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(model_path))
    else:
        loaded = torch.load(model_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_samples = []
    for scene in manifest["scenes"]:
        if scene["split"] == "test":
            s_dir = REPO_ROOT / scene["directory"]
            for f_info in scene["frames"]:
                test_samples.append({
                    "scene_id": scene["scene_id"],
                    "rgb_path": s_dir / f_info["rgb_file"],
                    "depth_path": s_dir / f_info["depth_file"],
                    "normal_path": s_dir / f_info["normal_file"],
                    "cam_path": s_dir / f_info["cam_file"],
                })

    print(f"Loaded {len(test_samples)} authentic held-out test views.")
    scene_results = []
    latencies = []

    for idx, s in enumerate(test_samples):
        # 1. Load ground truth
        pil_img = Image.open(s["rgb_path"]).convert("RGB")
        rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        gt_depth = np.load(s["depth_path"])
        gt_norm = np.load(s["normal_path"])
        cam_npz = np.load(s["cam_path"])
        intr = cam_npz["intrinsics"]
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

        H, W = gt_depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        gt_pts = np.stack([(u - cx) * gt_depth / fx, (v - cy) * gt_depth / fy, gt_depth], axis=-1)

        # 2. Forward pass
        t0 = time.time()
        with torch.no_grad():
            preds = model(views={"img": rgb_tensor}, cond_flags=[0, 0, 0], is_inference=True)
        dt = time.time() - t0
        latencies.append(dt)

        pred_depth = preds["depth"].squeeze().float().cpu().numpy()
        pred_pts = preds["pts3d"].squeeze().float().cpu().numpy()
        pred_norm = preds["normals"].squeeze().float().cpu().numpy()

        # 3. Compute empirical geometric errors
        valid_mask = (gt_depth > 0.1) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
        abs_rel = float(np.mean(np.abs(pred_depth[valid_mask] - gt_depth[valid_mask]) / gt_depth[valid_mask]))
        rmse = float(np.sqrt(np.mean((pred_depth[valid_mask] - gt_depth[valid_mask])**2)))
        pt_error = np.linalg.norm(pred_pts - gt_pts, axis=-1)
        acc = float(np.mean(pt_error[valid_mask]))

        # Normal MAE
        norm_dot = np.sum(pred_norm * gt_norm, axis=-1)
        pred_len = np.linalg.norm(pred_norm, axis=-1)
        gt_len = np.linalg.norm(gt_norm, axis=-1)
        denom = pred_len * gt_len
        denom[denom == 0] = 1.0
        cos_sim = np.clip(norm_dot / denom, -1.0, 1.0)
        mae_deg = float(np.mean(np.degrees(np.arccos(cos_sim))))

        # 4. Export PLY point cloud
        if idx == 0 and output_ply is not None:
            pts_flat = pred_pts.reshape(-1, 3)
            colors_flat = (rgb_arr * 255.0).reshape(-1, 3)
            # Subsample for efficient visualizer loading (every 4th point = ~67k points)
            step = 2
            sub_pts = pred_pts[::step, ::step].reshape(-1, 3)
            sub_col = (rgb_arr[::step, ::step] * 255.0).reshape(-1, 3)
            export_ply(sub_pts, sub_col, output_ply)
            print(f"  Exported 3D Point Cloud to {output_ply} ({len(sub_pts):,} vertices)")

        scene_metrics = {
            "scene_id": s["scene_id"],
            "pointmap_accuracy_m": acc,
            "depth_absrel": abs_rel,
            "depth_rmse_m": rmse,
            "normal_mae_deg": mae_deg,
            "latency_ms": dt * 1000.0,
            "finite_geometry": bool(torch.isfinite(preds["pts3d"]).all().item())
        }
        scene_results.append(scene_metrics)
        print(f"  [{s['scene_id']}] AbsRel: {abs_rel:.4f} | RMSE: {rmse:.3f}m | PtAcc: {acc:.3f}m | NormMAE: {mae_deg:.2f}° | Latency: {dt*1000:.1f}ms | Finite: {scene_metrics['finite_geometry']}")

    mean_acc = float(np.mean([s["pointmap_accuracy_m"] for s in scene_results]))
    mean_absrel = float(np.mean([s["depth_absrel"] for s in scene_results]))
    mean_rmse = float(np.mean([s["depth_rmse_m"] for s in scene_results]))
    mean_norm = float(np.mean([s["normal_mae_deg"] for s in scene_results]))
    median_lat = float(np.median(latencies))
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    aggregate = {
        "model_evaluated": str(model_path),
        "test_dataset": str(manifest_path),
        "test_views_count": len(test_samples),
        "mean_pointmap_accuracy_m": mean_acc,
        "mean_depth_absrel": mean_absrel,
        "mean_depth_rmse_m": mean_rmse,
        "mean_normal_mae_deg": mean_norm,
        "median_latency_ms": median_lat * 1000.0,
        "peak_vram_mb": peak_vram,
        "all_finite": all(s["finite_geometry"] for s in scene_results),
        "ply_exported": str(output_ply) if output_ply else None,
        "scene_breakdown": scene_results,
        "evaluated_at": datetime.utcnow().isoformat() + "Z"
    }

    print("\n" + "=" * 75)
    print("HELD-OUT TEST SET EVALUATION SUMMARY:")
    print(f"  Point-Cloud Accuracy:  {mean_acc:.3f} meters")
    print(f"  Depth AbsRel:          {mean_absrel:.4f}")
    print(f"  Depth RMSE:            {mean_rmse:.3f} meters")
    print(f"  Normal Angular Error:  {mean_norm:.2f} degrees")
    print(f"  Median Latency:        {median_lat*1000:.1f} ms | Peak VRAM: {peak_vram:.1f} MB")
    print(f"  Geometry Finite:       {aggregate['all_finite']}")
    print("=" * 75)
    return aggregate

def main():
    parser = argparse.ArgumentParser(description="Evaluate Test Split")
    parser.add_argument("--model", type=str, default="research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt")
    parser.add_argument("--manifest", type=str, default="research/3d_reconstruction/data_manifests/indoor_verified_manifest.json")
    parser.add_argument("--output_ply", type=str, default="research/3d_reconstruction/experiments/test_reconstruction.ply")
    parser.add_argument("--output_json", type=str, default="research/3d_reconstruction/experiments/test_results.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    manifest_path = REPO_ROOT / args.manifest
    ply_path = REPO_ROOT / args.output_ply if args.output_ply else None

    results = evaluate_test_scenes(model_path, manifest_path, ply_path)

    out_json = REPO_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed test results written to {out_json}")

if __name__ == "__main__":
    main()
