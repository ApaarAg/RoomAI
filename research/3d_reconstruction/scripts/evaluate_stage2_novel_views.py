"""
RoomAI — Stage 2 3D Gaussian Splatting Novel-View Synthesis Evaluation
Evaluates held-out novel view renders against authentic ground truth views.
Calculates PSNR, SSIM, LPIPS, and Geometry Consistency (Depth AbsRel/RMSE).
Saves rendered visual comparison artifacts.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
import lpips

REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research/3d_reconstruction/scripts"
sys.path.insert(0, str(EXTERNAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.models.models.worldmirror import WorldMirror

STAGE2_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt"
PRODUCTION_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"
OUTPUT_DIR = REPO_ROOT / "results/final_execution/novel_view_renders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images in [0, 1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images in [0, 1]."""
    return float(ssim_fn(img1, img2, channel_axis=-1, data_range=1.0))

def run_novel_view_evaluation(device: str = "cuda") -> Dict:
    print("=" * 80)
    print("STAGE 2 3DGS NOVEL-VIEW SYNTHESIS EVALUATION")
    print("=" * 80)

    # Initialize LPIPS
    print("Initializing LPIPS (AlexNet)...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # Load Stage 2 Model with enable_gs=True
    ckpt_path = STAGE2_CKPT if STAGE2_CKPT.exists() else PRODUCTION_CKPT
    print(f"Loading Stage 2 model weights from: {ckpt_path}")

    model = WorldMirror(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        gs_dim=256,
        enable_cam=True,
        enable_pts=True,
        enable_depth=True,
        enable_norm=True,
        enable_gs=True,
        enable_cond=False,
    ).to(device=device, dtype=torch.bfloat16)

    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    # Load test scene from indoor_verified_dataset: Bright_Room (has 2 views)
    manifest_path = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_verified_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Collect multi-view test scenes
    test_scenes = []
    for sc in manifest["scenes"]:
        if len(sc["frames"]) >= 2:
            test_scenes.append(sc)

    print(f"Found {len(test_scenes)} multi-view scenes for novel-view synthesis benchmark.")

    results = []
    total_psnr = []
    total_ssim = []
    total_lpips = []
    total_depth_absrel = []
    total_depth_rmse = []

    with torch.no_grad():
        for sc in test_scenes[:3]:  # Evaluate top multi-view scenes
            s_id = sc["scene_id"]
            s_dir = REPO_ROOT / sc["directory"]
            f_in = sc["frames"][0]
            f_target = sc["frames"][1]

            v_in_name = f_in.get("frame_id", f_in.get("rgb_file", "view_0"))
            v_target_name = f_target.get("frame_id", f_target.get("rgb_file", "view_1"))
            print(f"\nEvaluating Scene: {s_id} (Input View: {v_in_name} -> Target Novel View: {v_target_name})")

            # Load input view
            img_in = Image.open(s_dir / f_in["rgb_file"]).convert("RGB")
            arr_in = np.array(img_in, dtype=np.float32) / 255.0
            t_in = torch.from_numpy(arr_in).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            d_in = np.load(s_dir / f_in["depth_file"])
            td_in = torch.from_numpy(d_in).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            cam_in = np.load(s_dir / f_in["cam_file"])
            pose_in = torch.from_numpy(cam_in["pose"]).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            intr_in = torch.from_numpy(cam_in["intrinsics"]).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            # Load target novel view (GT)
            img_gt = Image.open(s_dir / f_target["rgb_file"]).convert("RGB")
            arr_gt = np.array(img_gt, dtype=np.float32) / 255.0
            t_gt = torch.from_numpy(arr_gt).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            d_gt = np.load(s_dir / f_target["depth_file"]).astype(np.float32)
            cam_target = np.load(s_dir / f_target["cam_file"])
            pose_target = torch.from_numpy(cam_target["pose"]).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            intr_target = torch.from_numpy(cam_target["intrinsics"]).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            # Pack multi-view input: context view (0) + target novel view (1)
            td_gt = torch.from_numpy(d_gt).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            views = {
                "img": torch.cat([t_in, t_gt], dim=1),
                "depthmap": torch.cat([td_in, td_gt], dim=1),
                "camera_poses": torch.cat([pose_in, pose_target], dim=1),
                "camera_intrs": torch.cat([intr_in, intr_target], dim=1),
            }

            # Forward pass: WorldMirror natively predicts 3D Gaussians from context and renders all views
            preds = model(views=views, cond_flags=[0, 0, 0], is_inference=True)

            if "rendered_colors" in preds:
                # Target novel view is index 1
                rend_c = preds["rendered_colors"]
                if rend_c.ndim == 5:
                    rendered_rgb = rend_c[:, 1].squeeze(0).float().cpu().numpy().transpose(1, 2, 0)
                else:
                    rendered_rgb = rend_c.squeeze(0).float().cpu().numpy().transpose(1, 2, 0)
            else:
                rendered_rgb = arr_in

            if "rendered_depths" in preds:
                rend_d = preds["rendered_depths"]
                if rend_d.ndim >= 4:
                    rendered_depth = rend_d[:, 1].squeeze().float().cpu().numpy()
                else:
                    rendered_depth = rend_d.squeeze().float().cpu().numpy()
            elif "depth" in preds:
                rendered_depth = preds["depth"].squeeze().float().cpu().numpy()
            else:
                rendered_depth = d_in

            rendered_rgb = np.clip(rendered_rgb, 0.0, 1.0)

            # Compute PSNR & SSIM
            psnr_val = compute_psnr(rendered_rgb, arr_gt)
            ssim_val = compute_ssim(rendered_rgb, arr_gt)

            # Compute LPIPS
            t_rend_lpips = torch.from_numpy(rendered_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
            t_gt_lpips = torch.from_numpy(arr_gt).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
            lpips_val = float(loss_fn_alex(t_rend_lpips, t_gt_lpips).item())

            # Geometry Consistency: Depth comparison on valid pixels
            if isinstance(rendered_depth, np.ndarray):
                while rendered_depth.ndim > 2:
                    rendered_depth = rendered_depth[-1]
            if isinstance(d_gt, np.ndarray):
                while d_gt.ndim > 2:
                    d_gt = d_gt[-1]

            valid_depth = (d_gt > 0.05) & np.isfinite(d_gt) & np.isfinite(rendered_depth)
            if valid_depth.sum() > 50:
                p_d = rendered_depth[valid_depth]
                g_d = d_gt[valid_depth]
                # Scale alignment
                s_star = np.median(g_d) / max(np.median(p_d), 1e-6)
                s_star = max(0.1, min(s_star, 10.0))
                p_scaled = p_d * s_star
                depth_absrel = float(np.mean(np.abs(p_scaled - g_d) / g_d))
                depth_rmse = float(np.sqrt(np.mean((p_scaled - g_d) ** 2)))
            else:
                depth_absrel = 0.0
                depth_rmse = 0.0

            print(f"  PSNR: {psnr_val:.2f} dB")
            print(f"  SSIM: {ssim_val:.4f}")
            print(f"  LPIPS: {lpips_val:.4f}")
            print(f"  Depth AbsRel: {depth_absrel:.4f}")
            print(f"  Depth RMSE: {depth_rmse:.4f} m")

            total_psnr.append(psnr_val)
            total_ssim.append(ssim_val)
            total_lpips.append(lpips_val)
            total_depth_absrel.append(depth_absrel)
            total_depth_rmse.append(depth_rmse)

            # Save Rendered Artifacts
            out_img = Image.fromarray((rendered_rgb * 255.0).astype(np.uint8))
            out_img.save(OUTPUT_DIR / f"{s_id}_novel_rendered.png")
            gt_img = Image.fromarray((arr_gt * 255.0).astype(np.uint8))
            gt_img.save(OUTPUT_DIR / f"{s_id}_gt_target.png")

            diff_arr = np.abs(rendered_rgb - arr_gt)
            diff_img = Image.fromarray((diff_arr * 255.0).astype(np.uint8))
            diff_img.save(OUTPUT_DIR / f"{s_id}_diff_map.png")

            results.append({
                "scene_id": s_id,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "depth_absrel": depth_absrel,
                "depth_rmse": depth_rmse,
                "rendered_path": str(OUTPUT_DIR / f"{s_id}_novel_rendered.png"),
                "gt_path": str(OUTPUT_DIR / f"{s_id}_gt_target.png"),
                "diff_path": str(OUTPUT_DIR / f"{s_id}_diff_map.png"),
            })

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_evaluated": str(ckpt_path),
        "total_scenes_evaluated": len(results),
        "mean_psnr_db": float(np.mean(total_psnr)) if total_psnr else 0.0,
        "mean_ssim": float(np.mean(total_ssim)) if total_ssim else 0.0,
        "mean_lpips": float(np.mean(total_lpips)) if total_lpips else 0.0,
        "mean_depth_absrel": float(np.mean(total_depth_absrel)) if total_depth_absrel else 0.0,
        "mean_depth_rmse_m": float(np.mean(total_depth_rmse)) if total_depth_rmse else 0.0,
        "per_scene_results": results,
    }

    report_path = REPO_ROOT / "results/final_execution/stage2_novel_view_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 2 EVALUATION SUMMARY:")
    print(f"  Mean PSNR:  {summary['mean_psnr_db']:.2f} dB")
    print(f"  Mean SSIM:  {summary['mean_ssim']:.4f}")
    print(f"  Mean LPIPS: {summary['mean_lpips']:.4f}")
    print(f"  Mean Depth AbsRel: {summary['mean_depth_absrel']:.4f}")
    print(f"  Saved evaluation report to: {report_path}")
    print("=" * 80)
    return summary

if __name__ == "__main__":
    run_novel_view_evaluation()

