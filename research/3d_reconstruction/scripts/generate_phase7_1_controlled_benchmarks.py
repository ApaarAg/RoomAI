"""
ROOMAI — PHASE 7.1: CONTROLLED FORENSIC RE-BENCHMARK & VISUALIZATION ENGINE (MEMORY SAFE)
Executes:
1. Sequential GPU inference (one model at a time) to strictly obey RTX 4060 8GB limits.
2. Controlled evaluation on 10 representative validation frames (5 Hypersim, 5 ScanNet).
3. Side-by-side metric computation: Raw Metric vs. Scale-Aligned Depth.
4. Dual Point Cloud evaluation: Raw Euclidean vs. Umeyama Sim(3) Aligned.
5. Export of visual audit figures (RGB, GT Depth, Base Depth, Stage1 Depth, Error Maps, Normals).
6. Aggregates both Frame-Weighted and Scene-Weighted metrics.
"""
import os
import sys
import gc
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror

GT_BASE = REPO_ROOT / "research" / "3d_reconstruction" / "data" / "ground_truth"
MANIFEST_PATH = REPO_ROOT / "research" / "3d_reconstruction" / "data_manifests" / "indoor_ground_truth_manifest.json"
OUT_DIR = REPO_ROOT / "artifacts" / "phase7_1_forensic_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model(ckpt_path: Path):
    d = torch.load(ckpt_path, map_location='cpu')
    sd = d.get('state_dict', d)
    sd = {k.replace('model.', ''): v for k, v in sd.items()}
    m = WorldMirror(
        img_size=518, patch_size=14, embed_dim=1024, gs_dim=256,
        enable_cam=True, enable_pts=True, enable_depth=True, enable_norm=True,
        enable_gs=False, enable_cond=False
    ).to('cuda', dtype=torch.bfloat16)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
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

def save_visual_comparison(frame_info, rgb_arr, gt_d, pred_b, pred_s1, gt_n, norm_b, norm_s1, mask, out_path):
    valid = mask & (gt_d > 0.1) & np.isfinite(gt_d)
    
    # Scale align for fair visual comparison
    s_b = float(np.median(gt_d[valid])) / max(float(np.median(pred_b[valid])), 1e-6)
    s_s1 = float(np.median(gt_d[valid])) / max(float(np.median(pred_s1[valid])), 1e-6)
    pred_b_scaled = pred_b * s_b
    pred_s1_scaled = pred_s1 * s_s1

    err_b = np.abs(pred_b_scaled - gt_d)
    err_s1 = np.abs(pred_s1_scaled - gt_d)
    err_b[~valid] = 0
    err_s1[~valid] = 0

    norm_vis_gt = (gt_n + 1.0) / 2.0
    norm_vis_b = (norm_b + 1.0) / 2.0
    norm_vis_s1 = (norm_s1 + 1.0) / 2.0
    norm_vis_gt[~valid] = 0

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: RGB and Depth maps
    axes[0, 0].imshow(rgb_arr)
    axes[0, 0].set_title(f"RGB: {frame_info['scene_id']} #{frame_info['frame_id']}")
    axes[0, 0].axis('off')

    im_gtd = axes[0, 1].imshow(np.where(valid, gt_d, np.nan), cmap='turbo')
    axes[0, 1].set_title(f"GT Depth (m) [med: {np.median(gt_d[valid]):.2f}m]")
    axes[0, 1].axis('off')
    plt.colorbar(im_gtd, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_b = axes[0, 2].imshow(np.where(valid, pred_b, np.nan), cmap='turbo')
    axes[0, 2].set_title(f"Base RAW Depth (med: {np.median(pred_b[valid]):.2f})")
    axes[0, 2].axis('off')
    plt.colorbar(im_b, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_s1 = axes[0, 3].imshow(np.where(valid, pred_s1, np.nan), cmap='turbo')
    axes[0, 3].set_title(f"Stage1 RAW Depth (med: {np.median(pred_s1[valid]):.2f})")
    axes[0, 3].axis('off')
    plt.colorbar(im_s1, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 2: Scaled Depths and Error Maps
    im_bs = axes[1, 0].imshow(np.where(valid, pred_b_scaled, np.nan), cmap='turbo', vmin=gt_d[valid].min(), vmax=gt_d[valid].max())
    axes[1, 0].set_title(f"Base SCALED Depth (s={s_b:.2f})")
    axes[1, 0].axis('off')
    plt.colorbar(im_bs, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_s1s = axes[1, 1].imshow(np.where(valid, pred_s1_scaled, np.nan), cmap='turbo', vmin=gt_d[valid].min(), vmax=gt_d[valid].max())
    axes[1, 1].set_title(f"Stage1 SCALED Depth (s={s_s1:.2f})")
    axes[1, 1].axis('off')
    plt.colorbar(im_s1s, ax=axes[1, 1], fraction=0.046, pad=0.04)

    max_err = max(np.percentile(err_b[valid], 95), np.percentile(err_s1[valid], 95), 1.0)
    im_eb = axes[1, 2].imshow(np.where(valid, err_b, np.nan), cmap='magma', vmin=0, vmax=max_err)
    axes[1, 2].set_title(f"Base Error (MAE: {np.mean(err_b[valid]):.2f}m)")
    axes[1, 2].axis('off')
    plt.colorbar(im_eb, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im_es1 = axes[1, 3].imshow(np.where(valid, err_s1, np.nan), cmap='magma', vmin=0, vmax=max_err)
    axes[1, 3].set_title(f"Stage1 Error (MAE: {np.mean(err_s1[valid]):.2f}m)")
    axes[1, 3].axis('off')
    plt.colorbar(im_es1, ax=axes[1, 3], fraction=0.046, pad=0.04)

    # Row 3: Surface Normals
    axes[2, 0].imshow(np.clip(norm_vis_gt, 0, 1))
    axes[2, 0].set_title("GT Surface Normals")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(np.clip(norm_vis_b, 0, 1))
    axes[2, 1].set_title("Pretrained Surface Normals")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(np.clip(norm_vis_s1, 0, 1))
    axes[2, 2].set_title("Stage1 Surface Normals")
    axes[2, 2].axis('off')

    norm_dot_b = np.clip(np.sum(norm_b * gt_n, axis=-1), -1.0, 1.0)
    ang_err_b = np.degrees(np.arccos(norm_dot_b))
    norm_dot_s1 = np.clip(np.sum(norm_s1 * gt_n, axis=-1), -1.0, 1.0)
    ang_err_s1 = np.degrees(np.arccos(norm_dot_s1))

    diff_norm = ang_err_s1 - ang_err_b
    im_nd = axes[2, 3].imshow(np.where(valid, diff_norm, np.nan), cmap='bwr', vmin=-30, vmax=30)
    axes[2, 3].set_title("Norm Error Delta: S1 - Base (Red=S1 worse)")
    axes[2, 3].axis('off')
    plt.colorbar(im_nd, ax=axes[2, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def main():
    print("=" * 80)
    print("PHASE 7.1: EXECUTING SEQUENTIAL MEMORY-SAFE RE-BENCHMARK & VISUAL AUDIT")
    print("=" * 80)

    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    val_frames = [f for f in manifest['frames'] if f['split'] == 'val']
    
    # 5 Hypersim frames
    hyp_frames = [f for f in val_frames if f['dataset'] == 'hypersim'][:5]
    
    # 5 ScanNet frames across sequences
    scan_00 = [f for f in val_frames if f['scene_id'] == 'scene0030_00'][:2]
    scan_01 = [f for f in val_frames if f['scene_id'] == 'scene0030_01'][:2]
    scan_02 = [f for f in val_frames if f['scene_id'] == 'scene0030_02'][:1]
    scan_frames = scan_00 + scan_01 + scan_02

    control_frames = hyp_frames + scan_frames
    print(f"Loaded {len(control_frames)} representative views (5 Hypersim, 5 ScanNet).")

    # Pass 1: Pretrained Baseline
    print("\n--- INFERENCE PASS 1: PRETRAINED BASELINE ---")
    m_base = load_model(REPO_ROOT / 'research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt')
    base_preds = []
    
    for idx, fr in enumerate(control_frames):
        rgb_path = GT_BASE / fr['paths']['rgb']
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to('cuda', dtype=torch.bfloat16)
        
        with torch.no_grad():
            out = m_base(views={'img': rgb}, cond_flags=[0, 0, 0], is_inference=True)

        depth = out['depth'].squeeze().float().cpu().numpy()
        normals = out['normals'].squeeze().float().cpu().numpy()
        base_preds.append({"depth": depth, "normals": normals})
        print(f"  Base View [{idx+1:02d}/10] {fr['dataset']:<8} | Scene: {fr['scene_id']} f{fr['frame_id']} | Depth med: {np.median(depth):.3f}")

    del m_base
    gc.collect()
    torch.cuda.empty_cache()
    print("Pretrained Baseline unloaded, CUDA cache cleared.")

    # Pass 2: Stage 1 Fine-Tuned
    print("\n--- INFERENCE PASS 2: STAGE 1 FINE-TUNED ---")
    m_s1 = load_model(REPO_ROOT / 'research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt')
    s1_preds = []

    for idx, fr in enumerate(control_frames):
        rgb_path = GT_BASE / fr['paths']['rgb']
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to('cuda', dtype=torch.bfloat16)

        with torch.no_grad():
            out = m_s1(views={'img': rgb}, cond_flags=[0, 0, 0], is_inference=True)

        depth = out['depth'].squeeze().float().cpu().numpy()
        normals = out['normals'].squeeze().float().cpu().numpy()
        s1_preds.append({"depth": depth, "normals": normals})
        print(f"  Stage1 View [{idx+1:02d}/10] {fr['dataset']:<8} | Scene: {fr['scene_id']} f{fr['frame_id']} | Depth med: {np.median(depth):.3f}")

    del m_s1
    gc.collect()
    torch.cuda.empty_cache()
    print("Stage 1 model unloaded, CUDA cache cleared.")

    # Pass 3: Evaluation Metrics & Visual Exports on CPU
    print("\n--- COMPUTING METRICS & EXPORTING VISUAL DIAGNOSTICS ---")
    results = []

    for idx, fr in enumerate(control_frames):
        rgb_path = GT_BASE / fr['paths']['rgb']
        depth_path = GT_BASE / fr['paths']['depth']
        normal_path = GT_BASE / fr['paths']['normal']
        mask_path = GT_BASE / fr['paths']['mask']
        cam_path = GT_BASE / fr['paths']['cam']

        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        gt_d = np.load(depth_path)
        gt_n = np.load(normal_path)
        mask = np.array(Image.open(mask_path)) > 0
        valid = mask & (gt_d > 0.1) & np.isfinite(gt_d)

        cam = np.load(cam_path)
        intr = cam['intrinsics']
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

        H, W = gt_d.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        gt_pts = np.stack([(u - cx) * gt_d / fx, (v - cy) * gt_d / fy, gt_d], axis=-1)

        pred_d_b = base_preds[idx]["depth"]
        pred_d_s1 = s1_preds[idx]["depth"]
        norm_b = base_preds[idx]["normals"]
        norm_s1 = s1_preds[idx]["normals"]

        norm_b /= np.maximum(np.linalg.norm(norm_b, axis=-1, keepdims=True), 1e-8)
        norm_s1 /= np.maximum(np.linalg.norm(norm_s1, axis=-1, keepdims=True), 1e-8)

        # Scale factors
        s_b = float(np.median(gt_d[valid])) / max(float(np.median(pred_d_b[valid])), 1e-6)
        s_s1 = float(np.median(gt_d[valid])) / max(float(np.median(pred_d_s1[valid])), 1e-6)
        pred_d_b_scaled = pred_d_b * s_b
        pred_d_s1_scaled = pred_d_s1 * s_s1

        # Unprojected points
        pts_b_raw = np.stack([(u - cx) * pred_d_b / fx, (v - cy) * pred_d_b / fy, pred_d_b], axis=-1)
        pts_s1_raw = np.stack([(u - cx) * pred_d_s1 / fx, (v - cy) * pred_d_s1 / fy, pred_d_s1], axis=-1)

        # Sim3 point cloud alignment
        sub_step = 4
        sub_mask = valid[::sub_step, ::sub_step].reshape(-1)
        gt_sub = gt_pts[::sub_step, ::sub_step].reshape(-1, 3)[sub_mask]
        pts_b_sub = pts_b_raw[::sub_step, ::sub_step].reshape(-1, 3)[sub_mask]
        pts_s1_sub = pts_s1_raw[::sub_step, ::sub_step].reshape(-1, 3)[sub_mask]

        aligned_b, sim3_s_b, _, _ = umeyama_alignment(pts_b_sub, gt_sub)
        aligned_s1, sim3_s_s1, _, _ = umeyama_alignment(pts_s1_sub, gt_sub)

        # Delta accuracies
        def compute_deltas(p, g, m):
            ratio = np.maximum(p[m] / g[m], g[m] / p[m])
            return float(np.mean(ratio < 1.25)), float(np.mean(ratio < 1.25**2))

        d1_raw_b, d2_raw_b = compute_deltas(pred_d_b, gt_d, valid)
        d1_raw_s1, d2_raw_s1 = compute_deltas(pred_d_s1, gt_d, valid)
        d1_scaled_b, d2_scaled_b = compute_deltas(pred_d_b_scaled, gt_d, valid)
        d1_scaled_s1, d2_scaled_s1 = compute_deltas(pred_d_s1_scaled, gt_d, valid)

        res = {
            "scene_id": fr['scene_id'],
            "frame_id": int(fr['frame_id']),
            "dataset": fr['dataset'],
            "gt_median_depth_m": float(np.median(gt_d[valid])),
            "base": {
                "raw_median_depth": float(np.median(pred_d_b[valid])),
                "raw_absrel": float(np.mean(np.abs(pred_d_b[valid] - gt_d[valid]) / gt_d[valid])),
                "raw_rmse_m": float(np.sqrt(np.mean((pred_d_b[valid] - gt_d[valid])**2))),
                "raw_delta1": d1_raw_b,
                "raw_delta2": d2_raw_b,
                "scaled_absrel": float(np.mean(np.abs(pred_d_b_scaled[valid] - gt_d[valid]) / gt_d[valid])),
                "scaled_rmse_m": float(np.sqrt(np.mean((pred_d_b_scaled[valid] - gt_d[valid])**2))),
                "scaled_delta1": d1_scaled_b,
                "scaled_delta2": d2_scaled_b,
                "median_scale_s": s_b,
                "normal_mae_deg": float(np.mean(np.degrees(np.arccos(np.clip(np.sum(norm_b[valid] * gt_n[valid], axis=-1), -1.0, 1.0))))),
                "metric_chamfer_m": float(np.mean(np.linalg.norm(pts_b_sub - gt_sub, axis=-1))),
                "shape_chamfer_m": float(np.mean(np.linalg.norm(aligned_b - gt_sub, axis=-1))),
                "sim3_scale": sim3_s_b
            },
            "stage1": {
                "raw_median_depth": float(np.median(pred_d_s1[valid])),
                "raw_absrel": float(np.mean(np.abs(pred_d_s1[valid] - gt_d[valid]) / gt_d[valid])),
                "raw_rmse_m": float(np.sqrt(np.mean((pred_d_s1[valid] - gt_d[valid])**2))),
                "raw_delta1": d1_raw_s1,
                "raw_delta2": d2_raw_s1,
                "scaled_absrel": float(np.mean(np.abs(pred_d_s1_scaled[valid] - gt_d[valid]) / gt_d[valid])),
                "scaled_rmse_m": float(np.sqrt(np.mean((pred_d_s1_scaled[valid] - gt_d[valid])**2))),
                "scaled_delta1": d1_scaled_s1,
                "scaled_delta2": d2_scaled_s1,
                "median_scale_s": s_s1,
                "normal_mae_deg": float(np.mean(np.degrees(np.arccos(np.clip(np.sum(norm_s1[valid] * gt_n[valid], axis=-1), -1.0, 1.0))))),
                "metric_chamfer_m": float(np.mean(np.linalg.norm(pts_s1_sub - gt_sub, axis=-1))),
                "shape_chamfer_m": float(np.mean(np.linalg.norm(aligned_s1 - gt_sub, axis=-1))),
                "sim3_scale": sim3_s_s1
            }
        }
        results.append(res)

        fig_path = OUT_DIR / f"comparison_{fr['dataset']}_{fr['scene_id']}_f{int(fr['frame_id']):04d}.png"
        save_visual_comparison(fr, rgb_arr, gt_d, pred_d_b, pred_d_s1, gt_n, norm_b, norm_s1, mask, fig_path)
        print(f"  Processed [{idx+1:02d}/10] {fr['dataset']:<8} | Scene: {fr['scene_id']} f{fr['frame_id']} | Saved visual: {fig_path.name}")

    def aggregate_metrics(frames_list):
        def mean_field(model_key, field):
            return float(np.mean([f[model_key][field] for f in frames_list]))

        return {
            "count": len(frames_list),
            "base": {
                "raw_absrel": mean_field("base", "raw_absrel"),
                "raw_rmse_m": mean_field("base", "raw_rmse_m"),
                "raw_delta1": mean_field("base", "raw_delta1"),
                "scaled_absrel": mean_field("base", "scaled_absrel"),
                "scaled_rmse_m": mean_field("base", "scaled_rmse_m"),
                "scaled_delta1": mean_field("base", "scaled_delta1"),
                "normal_mae_deg": mean_field("base", "normal_mae_deg"),
                "metric_chamfer_m": mean_field("base", "metric_chamfer_m"),
                "shape_chamfer_m": mean_field("base", "shape_chamfer_m"),
                "median_scale_s": mean_field("base", "median_scale_s"),
                "sim3_scale": mean_field("base", "sim3_scale")
            },
            "stage1": {
                "raw_absrel": mean_field("stage1", "raw_absrel"),
                "raw_rmse_m": mean_field("stage1", "raw_rmse_m"),
                "raw_delta1": mean_field("stage1", "raw_delta1"),
                "scaled_absrel": mean_field("stage1", "scaled_absrel"),
                "scaled_rmse_m": mean_field("stage1", "scaled_rmse_m"),
                "scaled_delta1": mean_field("stage1", "scaled_delta1"),
                "normal_mae_deg": mean_field("stage1", "normal_mae_deg"),
                "metric_chamfer_m": mean_field("stage1", "metric_chamfer_m"),
                "shape_chamfer_m": mean_field("stage1", "shape_chamfer_m"),
                "median_scale_s": mean_field("stage1", "median_scale_s"),
                "sim3_scale": mean_field("stage1", "sim3_scale")
            }
        }

    hyp_res = [r for r in results if r["dataset"] == "hypersim"]
    scan_res = [r for r in results if r["dataset"] == "scannet"]

    frame_weighted = aggregate_metrics(results)
    hyp_agg = aggregate_metrics(hyp_res)
    scan_agg = aggregate_metrics(scan_res)

    scene_weighted = {
        "base": {
            k: 0.5 * (hyp_agg["base"][k] + scan_agg["base"][k]) for k in hyp_agg["base"]
        },
        "stage1": {
            k: 0.5 * (hyp_agg["stage1"][k] + scan_agg["stage1"][k]) for k in hyp_agg["stage1"]
        }
    }

    summary = {
        "per_frame": results,
        "frame_weighted": frame_weighted,
        "scene_weighted": scene_weighted,
        "hypersim_domain": hyp_agg,
        "scannet_domain": scan_agg
    }

    out_json = REPO_ROOT / "research" / "3d_reconstruction" / "experiments" / "phase7_1_controlled_rebenchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("CONTROLLED BENCHMARK COMPLETE!")
    print(f"Results saved to: {out_json}")
    print("=" * 80)

if __name__ == '__main__':
    main()
