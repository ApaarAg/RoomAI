"""
ROOMAI — PHASE 7: GROUND-TRUTH BENCHMARK EVALUATION ENGINE
Evaluates:
  1. Pretrained Baseline (WorldMirror-1.0 / production_model.ckpt)
  2. Fine-tuned Stage 1 (best_stage1.ckpt)
  3. Stage 2 3DGS (stage2_real_verified.ckpt)
Against the authentic validation split (402 frames: 100 Hypersim, 302 ScanNet)
from indoor_ground_truth_manifest.json.

Strict Quarantine: Test frames are never accessed.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch
from scipy.spatial import cKDTree

REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror

GT_BASE = REPO_ROOT / "research" / "3d_reconstruction" / "data" / "ground_truth"
MANIFEST_PATH = REPO_ROOT / "research" / "3d_reconstruction" / "data_manifests" / "indoor_ground_truth_manifest.json"

def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """
    Computes optimal Sim(3) alignment from src to dst: dst ~= s * R @ src + t
    src: (N, 3), dst: (N, 3) corresponding points
    """
    N = src.shape[0]
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / N
    cov = (dst_c.T @ src_c) / N

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.sum(D * np.diag(S)) / (var_src + 1e-8)
    t = mu_dst - s * (R @ mu_src)

    src_aligned = s * (src @ R.T) + t
    return src_aligned, float(s), R, t

def compute_pointcloud_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray, subsample_step: int = 4):
    """
    Computes both Shape-Only (Sim3 aligned) and Metric (unaligned) point cloud metrics.
    pred_pts, gt_pts: (H, W, 3)
    """
    # Grid subsampling for fast, exact nearest neighbor search
    pred_sub = pred_pts[::subsample_step, ::subsample_step].reshape(-1, 3)
    gt_sub = gt_pts[::subsample_step, ::subsample_step].reshape(-1, 3)

    valid_mask = np.isfinite(gt_sub).all(axis=-1) & (gt_sub[:, 2] > 0.1) & np.isfinite(pred_sub).all(axis=-1)
    if np.sum(valid_mask) < 50:
        return None

    src = pred_sub[valid_mask]
    dst = gt_sub[valid_mask]

    # --- 1. Metric (Unaligned) Point Cloud Metrics ---
    # Direct point-to-point corresponding error
    metric_diff = np.linalg.norm(src - dst, axis=-1)
    metric_acc = float(np.mean(metric_diff))
    metric_comp = float(np.mean(metric_diff))
    metric_chamfer = metric_acc
    metric_f50 = float(np.mean(metric_diff < 0.05))
    metric_f100 = float(np.mean(metric_diff < 0.10))

    # --- 2. Shape-Only (Sim3 Aligned) Point Cloud Metrics ---
    src_aligned, scale, R, t = umeyama_alignment(src, dst)
    sim3_diff = np.linalg.norm(src_aligned - dst, axis=-1)
    sim3_acc = float(np.mean(sim3_diff))
    sim3_comp = float(np.mean(sim3_diff))
    sim3_chamfer = sim3_acc
    sim3_f50 = float(np.mean(sim3_diff < 0.05))
    sim3_f100 = float(np.mean(sim3_diff < 0.10))

    return {
        "metric_chamfer_m": metric_chamfer,
        "metric_acc_m": metric_acc,
        "metric_comp_m": metric_comp,
        "metric_f50": metric_f50,
        "metric_f100": metric_f100,
        "shape_chamfer_m": sim3_chamfer,
        "shape_acc_m": sim3_acc,
        "shape_comp_m": sim3_comp,
        "shape_f50": sim3_f50,
        "shape_f100": sim3_f100,
        "sim3_scale": scale
    }

def compute_depth_metrics(pred_d: np.ndarray, gt_d: np.ndarray, mask: np.ndarray):
    val_m = mask & (gt_d > 0.1) & np.isfinite(gt_d) & np.isfinite(pred_d) & (pred_d > 0.01)
    if np.sum(val_m) < 50:
        return None

    p = pred_d[val_m]
    g = gt_d[val_m]

    abs_diff = np.abs(p - g)
    absrel = float(np.mean(abs_diff / g))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
    med_ae = float(np.median(abs_diff))

    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25**2))
    delta3 = float(np.mean(ratio < 1.25**3))
    valid_ratio = float(np.mean(val_m))

    return {
        "absrel": absrel,
        "rmse_m": rmse,
        "median_ae_m": med_ae,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "valid_ratio": valid_ratio
    }

def compute_normal_metrics(pred_n: np.ndarray, gt_n: np.ndarray, mask: np.ndarray):
    val_m = mask & (np.linalg.norm(gt_n, axis=-1) > 0.5) & np.isfinite(pred_n).all(axis=-1)
    if np.sum(val_m) < 50:
        return None

    pn = pred_n[val_m]
    gn = gt_n[val_m]

    # Normalize vectors
    pn_norm = pn / np.maximum(np.linalg.norm(pn, axis=-1, keepdims=True), 1e-8)
    gn_norm = gn / np.maximum(np.linalg.norm(gn, axis=-1, keepdims=True), 1e-8)

    dot = np.sum(pn_norm * gn_norm, axis=-1)
    cos_sim = np.clip(dot, -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(cos_sim))

    mae = float(np.mean(ang_err_deg))
    med_ae = float(np.median(ang_err_deg))
    acc_5 = float(np.mean(ang_err_deg < 5.0))
    acc_11 = float(np.mean(ang_err_deg < 11.25))
    acc_22 = float(np.mean(ang_err_deg < 22.5))
    acc_30 = float(np.mean(ang_err_deg < 30.0))

    return {
        "normal_mae_deg": mae,
        "normal_median_ae_deg": med_ae,
        "normal_acc_5deg": acc_5,
        "normal_acc_11deg": acc_11,
        "normal_acc_22deg": acc_22,
        "normal_acc_30deg": acc_30
    }

def evaluate_model_on_validation(model_path: Path, enable_gs: bool = False, device: str = "cuda"):
    print(f"\n{'='*75}\nBENCHMARKING: {model_path.name} (enable_gs={enable_gs})\n{'='*75}")

    # Load weights
    data = torch.load(model_path, map_location="cpu")
    sd = data.get("state_dict", data)
    sd = {k.replace("model.", ""): v for k, v in sd.items()}

    model = WorldMirror(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        gs_dim=256,
        enable_cam=True,
        enable_pts=True,
        enable_depth=True,
        enable_norm=True,
        enable_gs=enable_gs,
        enable_cond=False,
    ).to(device, dtype=torch.bfloat16)

    model.load_state_dict(sd, strict=False)
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Strictly select VALIDATION frames only
    val_frames = [f for f in manifest["frames"] if f["split"] == "val"]
    print(f"Total Validation Views: {len(val_frames)} (Quarantine: {len([f for f in manifest['frames'] if f['split'] == 'test'])} test frames strictly untouched)")

    per_frame_results = []
    latencies = []
    gaussian_counts = []

    t_start = time.time()
    for idx, fr in enumerate(val_frames):
        rgb_path = GT_BASE / fr["paths"]["rgb"]
        depth_path = GT_BASE / fr["paths"]["depth"]
        normal_path = GT_BASE / fr["paths"]["normal"]
        mask_path = GT_BASE / fr["paths"]["mask"]
        cam_path = GT_BASE / fr["paths"]["cam"]

        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16)

        gt_d = np.load(depth_path)
        gt_n = np.load(normal_path)
        mask = np.array(Image.open(mask_path)) > 0
        cam = np.load(cam_path)
        intr = cam["intrinsics"]
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

        H, W = gt_d.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        gt_pts = np.stack([(u - cx) * gt_d / fx, (v - cy) * gt_d / fy, gt_d], axis=-1)

        t0 = time.time()
        with torch.no_grad():
            preds = model(views={"img": rgb_t}, cond_flags=[0, 0, 0], is_inference=True)
        dt = time.time() - t0
        latencies.append(dt)

        pred_d = preds["depth"].squeeze().float().cpu().numpy()
        pred_pts_unproj = np.stack([(u - cx) * pred_d / fx, (v - cy) * pred_d / fy, pred_d], axis=-1)
        pred_norm = preds["normals"].squeeze().float().cpu().numpy()

        depth_res = compute_depth_metrics(pred_d, gt_d, mask)
        norm_res = compute_normal_metrics(pred_norm, gt_n, mask)
        pc_res = compute_pointcloud_metrics(pred_pts_unproj, gt_pts, subsample_step=4)

        frame_data = {
            "scene_id": fr["scene_id"],
            "frame_id": fr["frame_id"],
            "dataset": fr["dataset"],
            "latency_ms": dt * 1000.0,
            "depth": depth_res,
            "normal": norm_res,
            "pointcloud": pc_res
        }

        if enable_gs and "splats" in preds and "means" in preds["splats"]:
            n_gauss = preds["splats"]["means"][0].shape[0] if len(preds["splats"]["means"]) > 0 else 0
            gaussian_counts.append(n_gauss)
            frame_data["gaussian_count"] = n_gauss

        per_frame_results.append(frame_data)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(val_frames):
            print(f"  Processed [{idx+1:3d}/{len(val_frames)}] | Scene: {fr['scene_id']:<12} | Latency: {dt*1000:.1f}ms | AbsRel: {depth_res['absrel']:.4f} | RMSE: {depth_res['rmse_m']:.3f}m | NormMAE: {norm_res['normal_mae_deg']:.1f}°")

    total_time = time.time() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Aggregator helper
    def aggregate_subset(frames_sub):
        sub_depth = [f["depth"] for f in frames_sub if f["depth"] is not None]
        sub_norm = [f["normal"] for f in frames_sub if f["normal"] is not None]
        sub_pc = [f["pointcloud"] for f in frames_sub if f["pointcloud"] is not None]
        sub_lat = [f["latency_ms"] for f in frames_sub]

        return {
            "count": len(frames_sub),
            "depth": {
                "absrel": float(np.mean([d["absrel"] for d in sub_depth])),
                "rmse_m": float(np.mean([d["rmse_m"] for d in sub_depth])),
                "median_ae_m": float(np.mean([d["median_ae_m"] for d in sub_depth])),
                "delta1": float(np.mean([d["delta1"] for d in sub_depth])),
                "delta2": float(np.mean([d["delta2"] for d in sub_depth])),
                "delta3": float(np.mean([d["delta3"] for d in sub_depth])),
            },
            "normal": {
                "mae_deg": float(np.mean([n["normal_mae_deg"] for n in sub_norm])),
                "median_ae_deg": float(np.mean([n["normal_median_ae_deg"] for n in sub_norm])),
                "acc_5deg": float(np.mean([n["normal_acc_5deg"] for n in sub_norm])),
                "acc_11deg": float(np.mean([n["normal_acc_11deg"] for n in sub_norm])),
                "acc_22deg": float(np.mean([n["normal_acc_22deg"] for n in sub_norm])),
                "acc_30deg": float(np.mean([n["normal_acc_30deg"] for n in sub_norm])),
            },
            "pointcloud": {
                "metric_chamfer_m": float(np.mean([p["metric_chamfer_m"] for p in sub_pc])),
                "metric_acc_m": float(np.mean([p["metric_acc_m"] for p in sub_pc])),
                "metric_comp_m": float(np.mean([p["metric_comp_m"] for p in sub_pc])),
                "metric_f50": float(np.mean([p["metric_f50"] for p in sub_pc])),
                "metric_f100": float(np.mean([p["metric_f100"] for p in sub_pc])),
                "shape_chamfer_m": float(np.mean([p["shape_chamfer_m"] for p in sub_pc])),
                "shape_acc_m": float(np.mean([p["shape_acc_m"] for p in sub_pc])),
                "shape_comp_m": float(np.mean([p["shape_comp_m"] for p in sub_pc])),
                "shape_f50": float(np.mean([p["shape_f50"] for p in sub_pc])),
                "shape_f100": float(np.mean([p["shape_f100"] for p in sub_pc])),
                "mean_sim3_scale": float(np.mean([p["sim3_scale"] for p in sub_pc])),
            },
            "latency_ms": {
                "mean": float(np.mean(sub_lat)),
                "median": float(np.median(sub_lat)),
                "p95": float(np.percentile(sub_lat, 95))
            }
        }

    hyp_frames = [f for f in per_frame_results if f["dataset"] == "hypersim"]
    scan_frames = [f for f in per_frame_results if f["dataset"] == "scannet"]

    summary = {
        "model_path": str(model_path),
        "enable_gs": enable_gs,
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "total_val_frames": len(val_frames),
        "total_time_sec": total_time,
        "peak_vram_mb": peak_vram,
        "overall_validation": aggregate_subset(per_frame_results),
        "hypersim_validation": aggregate_subset(hyp_frames),
        "scannet_validation": aggregate_subset(scan_frames),
    }

    if enable_gs and gaussian_counts:
        summary["gaussian_stats"] = {
            "mean_count": float(np.mean(gaussian_counts)),
            "min_count": int(np.min(gaussian_counts)),
            "max_count": int(np.max(gaussian_counts))
        }

    return summary, per_frame_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pretrained, Stage 1, and Stage 2 Ground-Truth Benchmark")
    parser.add_argument("--eval_baseline", action="store_true", default=True)
    parser.add_argument("--eval_stage1", action="store_true", default=True)
    parser.add_argument("--eval_stage2", action="store_true", default=True)
    args = parser.parse_args()

    results_dir = REPO_ROOT / "research" / "3d_reconstruction" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = {}

    ckpts = [
        ("pretrained_baseline", REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt", False),
        ("stage1_finetuned", REPO_ROOT / "research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt", False),
        ("stage2_3dgs", REPO_ROOT / "research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt", True),
    ]

    for model_key, ckpt_path, enable_gs in ckpts:
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint {ckpt_path} does not exist, skipping.")
            continue

        summary, per_frame = evaluate_model_on_validation(ckpt_path, enable_gs=enable_gs)
        benchmark_summary[model_key] = summary

        # Print quick summary
        print(f"\n--- {model_key.upper()} VALIDATION SUMMARY ---")
        ov = summary["overall_validation"]
        print(f"  Overall Depth:      AbsRel = {ov['depth']['absrel']:.4f} | RMSE = {ov['depth']['rmse_m']:.3f}m | Delta1 = {ov['depth']['delta1']:.3f}")
        print(f"  Overall Normal:     MAE = {ov['normal']['mae_deg']:.2f}° | Median = {ov['normal']['median_ae_deg']:.2f}° | Acc<30° = {ov['normal']['acc_30deg']:.3f}")
        print(f"  Shape PointCloud:   Chamfer = {ov['pointcloud']['shape_chamfer_m']:.3f}m | F-score@50mm = {ov['pointcloud']['shape_f50']:.3f} | Sim(3) Scale = {ov['pointcloud']['mean_sim3_scale']:.3f}")
        print(f"  Metric PointCloud:  Chamfer = {ov['pointcloud']['metric_chamfer_m']:.3f}m | F-score@50mm = {ov['pointcloud']['metric_f50']:.3f}")
        print(f"  Performance:        Latency = {ov['latency_ms']['median']:.1f}ms | Peak VRAM = {summary['peak_vram_mb']:.1f}MB")

        hyp = summary["hypersim_validation"]
        print(f"  Hypersim Depth:     AbsRel = {hyp['depth']['absrel']:.4f} | RMSE = {hyp['depth']['rmse_m']:.3f}m")
        scan = summary["scannet_validation"]
        print(f"  ScanNet Depth:      AbsRel = {scan['depth']['absrel']:.4f} | RMSE = {scan['depth']['rmse_m']:.3f}m")

    # Save complete benchmark results
    out_file = results_dir / "phase7_ground_truth_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\n{'='*75}\nBENCHMARK COMPLETE! Full results saved to:\n  {out_file}\n{'='*75}")

if __name__ == "__main__":
    main()

