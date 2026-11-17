"""
ROOMAI — PHASE 9: RIGOROUS PRETRAINED BASELINE BENCHMARK ENGINE
Evaluates the immutable pretrained WorldMirror-1.0 checkpoint on held-out validation data:
- Track A (Hypersim): scene ai_016_001 (100 validation frames).
- Track B (ScanNet): scenes scene0030_00, scene0030_01, scene0030_02 (held-out validation frames).

Computes:
1. Metric-scale (unaligned) depth metrics: AbsRel, RMSE, Delta1, Delta2.
2. Scale-aligned depth metrics: AbsRel, RMSE, Delta1, Delta2.
3. 3D Point-cloud metrics: Raw Euclidean L1/RMSE, Sim(3) Umeyama aligned L1.
4. Surface Normal metrics: MAE (deg), Median (deg), % < 11.25 deg, % < 22.5 deg, % < 30 deg.
5. Per-scene failure rate and valid pixel ratio.
6. Execution latency (ms) and peak VRAM allocation (MB).
Saves structured JSON results to research/3d_reconstruction/experiments/phase9/pretrained_baseline_evaluation.json.
"""

import sys
import os
import gc
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path("f:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror

DATA_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth"
MANIFEST_PATH = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json"
PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"
OUT_DIR = REPO_ROOT / "research/3d_reconstruction/experiments/phase9"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Computes optimal Sim(3) similarity transformation: dst = s * R @ src + t."""
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
    s = float(np.sum(D * np.diag(S)) / (var_src + 1e-8))
    t = mu_dst - s * (R @ mu_src)
    src_aligned = s * (src @ R.T) + t
    return src_aligned, s


def evaluate_frame(
    model: WorldMirror,
    f_info: Dict,
    device: str = "cuda"
) -> Dict:
    paths = f_info["paths"]
    
    # Load input RGB
    rgb_img = Image.open(DATA_ROOT / paths["rgb"]).convert("RGB")
    rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    # Load Ground Truths
    gt_d = np.load(DATA_ROOT / paths["depth"]).astype(np.float32)
    gt_n = np.load(DATA_ROOT / paths["normal"]).astype(np.float32)
    mask_raw = np.array(Image.open(DATA_ROOT / paths["mask"])) > 0
    valid_mask = mask_raw & (gt_d > 0.1) & np.isfinite(gt_d)

    # Load Camera
    cam_data = np.load(DATA_ROOT / paths["cam"])
    intr = cam_data["intrinsics"].astype(np.float32)
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    # Model inference with latency profiling
    t0 = time.time()
    with torch.no_grad():
        preds = model(views={"img": rgb_tensor}, cond_flags=[0, 0, 0], is_inference=True)
    dt_ms = (time.time() - t0) * 1000.0

    pred_d = preds["depth"].squeeze().float().cpu().numpy()
    pred_n = preds["normals"].squeeze().float().cpu().numpy()

    # Ground truth 3D points
    H, W = gt_d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    gt_pts = np.stack([(u - cx) * gt_d / fx, (v - cy) * gt_d / fy, gt_d], axis=-1)

    # Check validity
    finite_mask = valid_mask & np.isfinite(pred_d) & (pred_d > 0.01)
    n_valid = int(finite_mask.sum())

    if n_valid < 100:
        return {
            "failed": True,
            "scene_id": f_info["scene_id"],
            "frame_id": f_info["frame_id"],
            "dataset": f_info["dataset"],
            "valid_pixels": n_valid,
            "latency_ms": dt_ms,
        }

    p_val = pred_d[finite_mask]
    g_val = gt_d[finite_mask]

    # 1. Raw Metric-Scale Depth
    abs_raw = np.abs(p_val - g_val)
    raw_absrel = float(np.mean(abs_raw / g_val))
    raw_rmse = float(np.sqrt(np.mean(abs_raw ** 2)))
    raw_ratio = np.maximum(p_val / g_val, g_val / p_val)
    raw_delta1 = float(np.mean(raw_ratio < 1.25))
    raw_delta2 = float(np.mean(raw_ratio < 1.25 ** 2))

    # 2. Median Scale-Aligned Depth
    s_star = float(np.median(g_val) / np.clip(np.median(p_val), 1e-6, None))
    p_scaled = p_val * s_star
    abs_scaled = np.abs(p_scaled - g_val)
    scaled_absrel = float(np.mean(abs_scaled / g_val))
    scaled_rmse = float(np.sqrt(np.mean(abs_scaled ** 2)))
    scaled_ratio = np.maximum(p_scaled / g_val, g_val / p_scaled)
    scaled_delta1 = float(np.mean(scaled_ratio < 1.25))
    scaled_delta2 = float(np.mean(scaled_ratio < 1.25 ** 2))

    # 3. 3D Point-Map Metrics
    pred_pts_raw = np.stack([(u - cx) * pred_d / fx, (v - cy) * pred_d / fy, pred_d], axis=-1)
    
    # Subsample points for memory-efficient Sim(3) alignment
    sub_step = 4
    sub_mask = finite_mask[::sub_step, ::sub_step].reshape(-1)
    gt_pts_sub = gt_pts[::sub_step, ::sub_step].reshape(-1, 3)[sub_mask]
    pred_pts_sub = pred_pts_raw[::sub_step, ::sub_step].reshape(-1, 3)[sub_mask]

    raw_pts_diff = np.linalg.norm(pred_pts_sub - gt_pts_sub, axis=-1)
    pts_raw_mae = float(np.mean(raw_pts_diff))
    pts_raw_rmse = float(np.sqrt(np.mean(raw_pts_diff ** 2)))

    aligned_pts, sim3_s = umeyama_alignment(pred_pts_sub, gt_pts_sub)
    sim3_pts_diff = np.linalg.norm(aligned_pts - gt_pts_sub, axis=-1)
    pts_sim3_mae = float(np.mean(sim3_pts_diff))
    pts_sim3_rmse = float(np.sqrt(np.mean(sim3_pts_diff ** 2)))

    # 4. Surface Normal Orientation
    pn_norm = pred_n / np.clip(np.linalg.norm(pred_n, axis=-1, keepdims=True), 1e-8, None)
    gn_norm = gt_n / np.clip(np.linalg.norm(gt_n, axis=-1, keepdims=True), 1e-8, None)

    pn_v = pn_norm[finite_mask]
    gn_v = gn_norm[finite_mask]
    cos_sim = np.clip(np.sum(pn_v * gn_v, axis=-1), -1.0, 1.0)
    ang_err_deg = np.rad2deg(np.arccos(cos_sim))

    norm_mae = float(np.mean(ang_err_deg))
    norm_median = float(np.median(ang_err_deg))
    norm_pct_11 = float(np.mean(ang_err_deg < 11.25) * 100.0)
    norm_pct_22 = float(np.mean(ang_err_deg < 22.5) * 100.0)
    norm_pct_30 = float(np.mean(ang_err_deg < 30.0) * 100.0)

    return {
        "failed": False,
        "scene_id": f_info["scene_id"],
        "frame_id": f_info["frame_id"],
        "dataset": f_info["dataset"],
        "valid_pixels": n_valid,
        "valid_ratio": float(n_valid / (H * W)),
        "scale_factor_s_star": s_star,
        "raw_absrel": raw_absrel,
        "raw_rmse_m": raw_rmse,
        "raw_delta1": raw_delta1,
        "raw_delta2": raw_delta2,
        "scaled_absrel": scaled_absrel,
        "scaled_rmse_m": scaled_rmse,
        "scaled_delta1": scaled_delta1,
        "scaled_delta2": scaled_delta2,
        "pts_raw_mae_m": pts_raw_mae,
        "pts_raw_rmse_m": pts_raw_rmse,
        "pts_sim3_mae_m": pts_sim3_mae,
        "pts_sim3_rmse_m": pts_sim3_rmse,
        "sim3_scale": sim3_s,
        "norm_mae_deg": norm_mae,
        "norm_median_deg": norm_median,
        "norm_pct_11_25": norm_pct_11,
        "norm_pct_22_5": norm_pct_22,
        "norm_pct_30_0": norm_pct_30,
        "latency_ms": dt_ms,
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    valid_res = [r for r in results if not r.get("failed", False)]
    total = len(results)
    n_valid = len(valid_res)
    fail_count = total - n_valid

    if n_valid == 0:
        return {"total_frames": total, "failed_frames": fail_count, "failure_rate_pct": 100.0}

    keys = [
        "raw_absrel", "raw_rmse_m", "raw_delta1", "raw_delta2",
        "scaled_absrel", "scaled_rmse_m", "scaled_delta1", "scaled_delta2",
        "scale_factor_s_star",
        "pts_raw_mae_m", "pts_raw_rmse_m", "pts_sim3_mae_m", "pts_sim3_rmse_m", "sim3_scale",
        "norm_mae_deg", "norm_median_deg", "norm_pct_11_25", "norm_pct_22_5", "norm_pct_30_0",
        "latency_ms", "valid_ratio"
    ]

    agg = {
        "total_frames": total,
        "valid_frames": n_valid,
        "failed_frames": fail_count,
        "failure_rate_pct": float(fail_count / total * 100.0),
    }

    for k in keys:
        vals = [r[k] for r in valid_res]
        agg[f"mean_{k}"] = float(np.mean(vals))
        agg[f"median_{k}"] = float(np.median(vals))
        agg[f"std_{k}"] = float(np.std(vals))

    return agg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("ROOMAI — PHASE 9: PRETRAINED WORLDMIRROR-1.0 BASELINE BENCHMARK")
    print(f"Device: {device} | Checkpoint: {PRETRAINED_CKPT.name}")
    print("=" * 80)

    # 1. Checkpoint Verification
    h = hashlib.sha256()
    with open(PRETRAINED_CKPT, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    ckpt_sha = h.hexdigest()
    print(f"Checkpoint SHA-256: {ckpt_sha}")

    # 2. Instantiate Model
    print("Loading WorldMirror architecture...")
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
    ).to(device=device, dtype=torch.bfloat16)

    sd = torch.load(PRETRAINED_CKPT, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    del sd, clean_sd
    torch.cuda.empty_cache()

    # 3. Load Canonical Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter held-out validation frames
    val_frames = [f for f in manifest["frames"] if f["split"] == "val"]
    print(f"Total validation frames in manifest: {len(val_frames)}")

    # Segregate Track A (Hypersim) and Track B (ScanNet)
    hypersim_val = [f for f in val_frames if f["dataset"] == "hypersim"]
    scannet_val = [f for f in val_frames if f["dataset"] == "scannet"]

    print(f"  Track A (Hypersim): {len(hypersim_val)} frames (Scene: {sorted(list(set(f['scene_id'] for f in hypersim_val)))})")
    print(f"  Track B (ScanNet) : {len(scannet_val)} frames (Scenes: {sorted(list(set(f['scene_id'] for f in scannet_val)))})")

    torch.cuda.reset_peak_memory_stats(device=device)

    # 4. Evaluate Track A (Hypersim)
    print("\n" + "-" * 60)
    print("EVALUATING TRACK A: HYPERSIM HELD-OUT VALIDATION (ai_016_001)")
    print("-" * 60)
    hypersim_results = []
    for idx, fr in enumerate(hypersim_val):
        res = evaluate_frame(model, fr, device=device)
        hypersim_results.append(res)
        if (idx + 1) % 20 == 0 or idx == 0 or idx == len(hypersim_val) - 1:
            print(f"  [Hypersim {idx+1:03d}/{len(hypersim_val)}] Frame {fr['frame_id']} | Raw AbsRel: {res.get('raw_absrel', 0):.4f} | Scaled AbsRel: {res.get('scaled_absrel', 0):.4f} | Norm MAE: {res.get('norm_mae_deg', 0):.2f}° | Latency: {res.get('latency_ms', 0):.1f}ms")

    hypersim_agg = aggregate_metrics(hypersim_results)

    # 5. Evaluate Track B (ScanNet)
    print("\n" + "-" * 60)
    print("EVALUATING TRACK B: SCANNET HELD-OUT VALIDATION (scene0030_00/01/02)")
    print("-" * 60)
    # Evaluate 25 representative frames per ScanNet scene to strictly manage time while providing rich data
    scannet_scenes = sorted(list(set(f["scene_id"] for f in scannet_val)))
    scannet_selected = []
    for scn in scannet_scenes:
        scn_frames = [f for f in scannet_val if f["scene_id"] == scn]
        # select 25 evenly spaced frames
        step = max(1, len(scn_frames) // 25)
        scannet_selected.extend(scn_frames[::step][:25])

    print(f"Selected {len(scannet_selected)} representative frames across {len(scannet_scenes)} ScanNet scenes.")

    scannet_results = []
    scannet_by_scene = {scn: [] for scn in scannet_scenes}
    for idx, fr in enumerate(scannet_selected):
        res = evaluate_frame(model, fr, device=device)
        scannet_results.append(res)
        scannet_by_scene[fr["scene_id"]].append(res)
        if (idx + 1) % 20 == 0 or idx == 0 or idx == len(scannet_selected) - 1:
            print(f"  [ScanNet {idx+1:03d}/{len(scannet_selected)}] Scene {fr['scene_id']} #{fr['frame_id']} | Raw AbsRel: {res.get('raw_absrel', 0):.4f} | Scaled AbsRel: {res.get('scaled_absrel', 0):.4f} | Norm MAE: {res.get('norm_mae_deg', 0):.2f}°")

    scannet_agg = aggregate_metrics(scannet_results)
    scannet_scene_aggs = {scn: aggregate_metrics(res_list) for scn, res_list in scannet_by_scene.items()}

    peak_vram_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))

    # 6. Summary Presentation
    print("\n" + "=" * 80)
    print("PHASE 9 PRETRAINED BASELINE SUMMARY RESULTS")
    print("=" * 80)
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB | Average Frame Latency: {hypersim_agg['mean_latency_ms']:.1f} ms")
    print("\n--- TRACK A: HYPERSIM (Commercial Candidate) ---")
    print(f"  Total Frames Evaluated : {hypersim_agg['valid_frames']} / {hypersim_agg['total_frames']} (Failure Rate: {hypersim_agg['failure_rate_pct']:.1f}%)")
    print(f"  Raw Metric Depth AbsRel : {hypersim_agg['mean_raw_absrel']:.4f} ± {hypersim_agg['std_raw_absrel']:.4f}")
    print(f"  Raw Metric Depth RMSE   : {hypersim_agg['mean_raw_rmse_m']:.4f} m")
    print(f"  Raw Delta < 1.25        : {hypersim_agg['mean_raw_delta1']:.4f}")
    print(f"  Scaled Depth AbsRel     : {hypersim_agg['mean_scaled_absrel']:.4f} ± {hypersim_agg['std_scaled_absrel']:.4f}")
    print(f"  Scaled Depth RMSE       : {hypersim_agg['mean_scaled_rmse_m']:.4f} m")
    print(f"  Scaled Delta < 1.25     : {hypersim_agg['mean_scaled_delta1']:.4f}")
    print(f"  Point Map Raw MAE       : {hypersim_agg['mean_pts_raw_mae_m']:.4f} m")
    print(f"  Point Map Sim(3) MAE    : {hypersim_agg['mean_pts_sim3_mae_m']:.4f} m")
    print(f"  Surface Normal MAE      : {hypersim_agg['mean_norm_mae_deg']:.2f}° (Median: {hypersim_agg['mean_norm_median_deg']:.2f}°)")
    print(f"  Normals < 30° Acc       : {hypersim_agg['mean_norm_pct_30_0']:.1f}%")

    print("\n--- TRACK B: SCANNET (Non-Commercial Research) ---")
    print(f"  Total Frames Evaluated : {scannet_agg['valid_frames']} / {scannet_agg['total_frames']} (Failure Rate: {scannet_agg['failure_rate_pct']:.1f}%)")
    print(f"  Raw Metric Depth AbsRel : {scannet_agg['mean_raw_absrel']:.4f} ± {scannet_agg['std_raw_absrel']:.4f}")
    print(f"  Raw Metric Depth RMSE   : {scannet_agg['mean_raw_rmse_m']:.4f} m")
    print(f"  Scaled Depth AbsRel     : {scannet_agg['mean_scaled_absrel']:.4f} ± {scannet_agg['std_scaled_absrel']:.4f}")
    print(f"  Scaled Depth RMSE       : {scannet_agg['mean_scaled_rmse_m']:.4f} m")
    print(f"  Scaled Delta < 1.25     : {scannet_agg['mean_scaled_delta1']:.4f}")
    print(f"  Point Map Sim(3) MAE    : {scannet_agg['mean_pts_sim3_mae_m']:.4f} m")
    print(f"  Surface Normal MAE      : {scannet_agg['mean_norm_mae_deg']:.2f}°")

    # 7. Save JSON artifacts
    output_payload = {
        "checkpoint": PRETRAINED_CKPT.name,
        "checkpoint_sha256": ckpt_sha,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "peak_vram_mb": peak_vram_mb,
        "track_a_hypersim": {
            "aggregate": hypersim_agg,
            "per_frame": hypersim_results,
        },
        "track_b_scannet": {
            "aggregate": scannet_agg,
            "scene_aggregates": scannet_scene_aggs,
            "per_frame": scannet_results,
        },
    }

    out_file = OUT_DIR / "pretrained_baseline_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved structured evaluation report to: {out_file}")


if __name__ == "__main__":
    main()

