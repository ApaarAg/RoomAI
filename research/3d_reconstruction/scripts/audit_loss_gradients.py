"""
Loss Balancing and Gradient Audit for WorldMirror Fine-Tuning.

Audits:
1. Scalar loss values (Depth, Points, Normals).
2. Validity mask coverage and pixel counts.
3. Gradient vector L2 norms per task: ||grad_wrt_task||_2.
4. Relative gradient contribution to shared layers.
5. Finiteness of all gradients (0 NaN / 0 Inf).
"""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

REPO_ROOT = Path("f:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror
from scene_normalization import compute_world_coordinates, normalize_scene_coordinates
from corrected_losses import CanonicalMultiTaskLoss, MaskedNormalizedDepthLoss, MaskedNormalizedPointLoss, MaskedNormalLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing loss & gradient audit on {device}...")

    # Load 1 real Hypersim training sample
    hypersim_dir = REPO_ROOT / "research/3d_reconstruction/data/ground_truth/standardized/hypersim/train/ai_015_007"
    rgb_path = hypersim_dir / "frame_0000_rgb.png"
    depth_path = hypersim_dir / "frame_0000_depth.npy"
    norm_path = hypersim_dir / "frame_0000_normal.npy"
    cam_path = hypersim_dir / "frame_0000_cam.npz"

    rgb_pil = Image.open(rgb_path).convert("RGB")
    rgb_arr = np.array(rgb_pil, dtype=np.float32) / 255.0  # (518, 518, 3)
    rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    depth_arr = np.load(depth_path).astype(np.float32)  # (518, 518) in meters
    norm_arr = np.load(norm_path).astype(np.float32)    # (518, 518, 3)
    cam_data = np.load(cam_path)
    intr = cam_data["intrinsics"].astype(np.float32)
    c2w = cam_data["pose"].astype(np.float32)

    # Compute metric world coordinates and validity mask
    pts_metric, valid_mask = compute_world_coordinates(depth_arr, intr, c2w, c2w=True)

    # Apply canonical scene normalization
    views = [{
        "pts3d": pts_metric,
        "depthmap": depth_arr,
        "valid_mask": valid_mask,
        "camera_poses": c2w,
        "camera_intrs": intr,
        "normals": norm_arr,
    }]
    norm_views, scene_scale = normalize_scene_coordinates(views, norm_mode="median_dis")
    print(f"Real scene scale factor s: {scene_scale:.4f} meters")

    pts_norm = norm_views[0]["pts3d"]
    depth_norm = norm_views[0]["depthmap"]

    # Convert targets to tensors
    target_depth_norm = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    target_pts_norm = torch.from_numpy(pts_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    target_normals = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    target_mask = torch.from_numpy(valid_mask).unsqueeze(0).unsqueeze(0).to(device=device)

    # Also prepare UNNORMALIZED targets for direct comparison
    target_depth_unnorm = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    target_pts_unnorm = torch.from_numpy(pts_metric).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    # Load WorldMirror model
    ckpt_path = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"
    print(f"Loading pretrained WorldMirror from: {ckpt_path.name}")
    # Instantiate model
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

    # Load weights
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    del sd, clean_sd
    torch.cuda.empty_cache()

    # Freeze transformer backbone, train heads
    for p in model.visual_geometry_transformer.parameters():
        p.requires_grad = False
    
    trainable_heads = [model.depth_head, model.pts_head, model.norm_head]
    for head in trainable_heads:
        for p in head.parameters():
            p.requires_grad = True

    model.train()

    # Forward pass
    print("Running forward pass...")
    out = model(views={"img": rgb_t}, cond_flags=[0, 0, 0], is_inference=False)
    pred_depth = out["depth"]
    pred_pts = out["pts3d"]
    pred_norm = out["normals"]

    print("\n--- MODEL PREDICTION STATISTICS ---")
    print(f"Pred Depth : shape {pred_depth.shape}, min={pred_depth.min().item():.3f}, max={pred_depth.max().item():.3f}, median={pred_depth.median().item():.3f}")
    print(f"Pred Pts3D : shape {pred_pts.shape}, norm={torch.norm(pred_pts, dim=-1).median().item():.3f}")
    print(f"Pred Norm  : shape {pred_norm.shape}, unit norm={torch.norm(pred_norm, dim=-1).median().item():.3f}")

    print("\n--- TARGET STATISTICS ---")
    print(f"GT Metric Depth     : min={depth_arr[valid_mask].min():.3f}m, max={depth_arr[valid_mask].max():.3f}m, median={np.median(depth_arr[valid_mask]):.3f}m")
    print(f"GT Normalized Depth : min={depth_norm[valid_mask].min():.3f}, max={depth_norm[valid_mask].max():.3f}, median={np.median(depth_norm[valid_mask]):.3f}")
    print(f"GT Normalized Pts   : norm median={np.median(np.linalg.norm(pts_norm[valid_mask], axis=-1)):.3f}")

    # Evaluate individual losses
    depth_loss_fn = MaskedNormalizedDepthLoss(loss_type="l1")
    pts_loss_fn = MaskedNormalizedPointLoss(loss_type="l2")
    norm_loss_fn = MaskedNormalLoss(loss_type="cosine")

    # 1. Unnormalized vs Normalized Depth Loss
    loss_depth_unnorm, _ = depth_loss_fn(pred_depth, target_depth_unnorm, target_mask)
    loss_depth_norm, n_valid_d = depth_loss_fn(pred_depth, target_depth_norm, target_mask)

    # 2. Unnormalized vs Normalized Point Loss
    loss_pts_unnorm, _ = pts_loss_fn(pred_pts, target_pts_unnorm, target_mask)
    loss_pts_norm, n_valid_p = pts_loss_fn(pred_pts, target_pts_norm, target_mask)

    # 3. Normal Loss
    loss_norm, n_valid_n = norm_loss_fn(pred_norm, target_normals, target_mask)

    print("\n--- LOSS COMPARISON: UNNORMALIZED VS CORRECTED NORMALIZED ---")
    print(f"Depth Loss : Unnormalized = {loss_depth_unnorm.item():.4f}  -->  Corrected Normalized = {loss_depth_norm.item():.4f} (Valid pixels: {n_valid_d})")
    print(f"Pts Loss   : Unnormalized = {loss_pts_unnorm.item():.4f}  -->  Corrected Normalized = {loss_pts_norm.item():.4f} (Valid pixels: {n_valid_p})")
    print(f"Norm Loss  : {loss_norm.item():.4f} (Valid pixels: {n_valid_n})")

    # Compute gradient norms for each normalized loss
    def compute_grad_norm(loss_val):
        model.zero_grad()
        loss_val.backward(retain_graph=True)
        total_sq = 0.0
        finite = True
        for head in trainable_heads:
            for p in head.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        finite = False
                    total_sq += p.grad.norm(2).item() ** 2
        return np.sqrt(total_sq), finite

    gn_depth, fin_d = compute_grad_norm(loss_depth_norm)
    gn_pts, fin_p = compute_grad_norm(loss_pts_norm)
    gn_norm, fin_n = compute_grad_norm(loss_norm)

    # Multi-task combined loss
    multi_loss_fn = CanonicalMultiTaskLoss(weight_depth=1.0, weight_pts=0.5, weight_norm=0.2)
    targets_dict = {
        "depth_norm": target_depth_norm,
        "pts_norm": target_pts_norm,
        "normals": target_normals,
        "mask": target_mask,
    }
    total_loss, metrics = multi_loss_fn(out, targets_dict)
    gn_total, fin_t = compute_grad_norm(total_loss)

    print("\n--- GRADIENT NORM AUDIT (Trainable Heads) ---")
    print(f"Depth Grad Norm (w=1.0) : {gn_depth:.4f} (Finite: {fin_d})")
    print(f"Pts Grad Norm (w=0.5)   : {gn_pts * 0.5:.4f} (Raw: {gn_pts:.4f}, Finite: {fin_p})")
    print(f"Norm Grad Norm (w=0.2)  : {gn_norm * 0.2:.4f} (Raw: {gn_norm:.4f}, Finite: {fin_n})")
    print(f"Total Multi-Task Grad   : {gn_total:.4f} (Finite: {fin_t})")

    # Calculate gradient balance ratios
    sum_gn = (gn_depth) + (gn_pts * 0.5) + (gn_norm * 0.2)
    pct_depth = (gn_depth / sum_gn) * 100.0
    pct_pts = (gn_pts * 0.5 / sum_gn) * 100.0
    pct_norm = (gn_norm * 0.2 / sum_gn) * 100.0

    print(f"\n--- GRADIENT BALANCE PROPORTIONS ---")
    print(f"Depth Contribution  : {pct_depth:.1f}%")
    print(f"Points Contribution : {pct_pts:.1f}%")
    print(f"Normals Contribution: {pct_norm:.1f}%")

    results = {
        "scene_scale_factor": float(scene_scale),
        "unnormalized_losses": {
            "depth": float(loss_depth_unnorm.item()),
            "pts": float(loss_pts_unnorm.item()),
        },
        "corrected_losses": {
            "depth": float(loss_depth_norm.item()),
            "pts": float(loss_pts_norm.item()),
            "normals": float(loss_norm.item()),
            "total_weighted": float(total_loss.item()),
        },
        "valid_pixels": {
            "depth": n_valid_d,
            "pts": n_valid_p,
            "normals": n_valid_n,
        },
        "gradient_norms": {
            "depth_w1": float(gn_depth),
            "pts_w05": float(gn_pts * 0.5),
            "norm_w02": float(gn_norm * 0.2),
            "total": float(gn_total),
        },
        "gradient_percentages": {
            "depth": round(pct_depth, 2),
            "pts": round(pct_pts, 2),
            "normals": round(pct_norm, 2),
        },
        "all_gradients_finite": bool(fin_d and fin_p and fin_n and fin_t),
    }

    out_json = REPO_ROOT / "research/3d_reconstruction/experiments/phase8_corrected/loss_gradient_audit.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit results saved to: {out_json}")


if __name__ == "__main__":
    main()
