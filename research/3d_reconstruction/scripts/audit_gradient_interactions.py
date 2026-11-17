"""
Advanced Multi-Task Gradient Interaction and Conflict Audit for RoomAI.

Computes:
1. Per-task gradient norms for Depth, Points, and Normals.
2. Pairwise gradient cosine similarities: cos(g_i, g_j) = <g_i, g_j> / (||g_i|| * ||g_j||).
3. Conflict frequency: fraction of samples where cos(g_i, g_j) < 0 (destructive interference).
4. Shared-layer gradient interactions in the Visual Geometry Transformer backbone.
5. Combined gradient magnitude ||sum w_i g_i|| vs sum w_i ||g_i|| (cancellation ratio).
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("f:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror
from scene_normalization import compute_world_coordinates
from corrected_losses import MaskedNormalizedDepthLoss, MaskedNormalizedPointLoss, MaskedNormalLoss

DATA_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth"
MANIFEST_PATH = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json"
PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"


def flatten_grads(params: List[torch.Tensor]) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Multi-Task Gradient Interaction Audit on {device}...")

    # Load canonical manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_frames = [f for f in manifest["frames"] if f["split"] == "train" and f["dataset"] == "hypersim"]
    print(f"Auditing gradient interactions across {len(train_frames)} Hypersim training samples...")

    # Load model
    print("Instantiating WorldMirror architecture...")
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
    del sd, clean_sd
    torch.cuda.empty_cache()

    # Unfreeze prediction heads AND the last transformer block to audit shared-layer interactions
    for p in model.visual_geometry_transformer.parameters():
        p.requires_grad = False
    
    # Unfreeze the final transformer blocks (frame and global attention)
    # to evaluate gradient interactions in the shared representation
    last_frame_block = model.visual_geometry_transformer.frame_blocks[-1]
    last_global_block = model.visual_geometry_transformer.global_blocks[-1]
    shared_modules = [last_frame_block, last_global_block]
    shared_params = []
    for mod in shared_modules:
        for p in mod.parameters():
            p.requires_grad = True
            shared_params.append(p)
    head_params = []
    for h in [model.depth_head, model.pts_head, model.norm_head]:
        for p in h.parameters():
            p.requires_grad = True
            head_params.append(p)

    all_trainable = shared_params + head_params

    print(f"Trainable parameters: {sum(p.numel() for p in all_trainable):,} (Shared: {sum(p.numel() for p in shared_params):,}, Heads: {sum(p.numel() for p in head_params):,})")

    depth_loss_fn = MaskedNormalizedDepthLoss(loss_type="l1")
    pts_loss_fn = MaskedNormalizedPointLoss(loss_type="l2")
    norm_loss_fn = MaskedNormalLoss(loss_type="cosine")

    w_depth, w_pts, w_norm = 1.0, 0.5, 0.2

    # Metric accumulators
    gn_depth_list = []
    gn_pts_list = []
    gn_norm_list = []
    gn_comb_list = []
    sum_weighted_gn_list = []

    # Cosine similarities for shared backbone layer
    cos_dp_shared_list = []
    cos_dn_shared_list = []
    cos_pn_shared_list = []

    # Cosine similarities across all trainable parameters
    cos_dp_all_list = []
    cos_dn_all_list = []
    cos_pn_all_list = []

    model.train()
    sample_count = min(len(train_frames), 15)

    for i in range(sample_count):
        f_info = train_frames[i]
        paths = f_info["paths"]
        s_id = f_info["scene_id"]
        scale_s = float(f_info["canonical_normalization"]["scene_scale_factor"])

        # Load RGB
        rgb_img = Image.open(DATA_ROOT / paths["rgb"]).convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        # Load depth & mask
        depth_m = np.load(DATA_ROOT / paths["depth"]).astype(np.float32)
        depth_norm = depth_m / scale_s
        depth_norm_t = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        mask_raw = np.array(Image.open(DATA_ROOT / paths["mask"])) > 0
        valid_mask = mask_raw & np.isfinite(depth_m) & (depth_m > 0.0)
        mask_t = torch.from_numpy(valid_mask).unsqueeze(0).unsqueeze(0).to(device=device)

        # Camera & points
        cam_data = np.load(DATA_ROOT / paths["cam"])
        pts_w, _ = compute_world_coordinates(depth_m, cam_data["intrinsics"].astype(np.float32), cam_data["pose"].astype(np.float32), c2w=True)
        pts_norm = pts_w / scale_s
        pts_norm_t = torch.from_numpy(pts_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        # Normals
        norm_arr = np.load(DATA_ROOT / paths["normal"]).astype(np.float32)
        norm_t = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        # Forward pass
        preds = model(views={"img": rgb_t}, cond_flags=[0, 0, 0], is_inference=False)

        # 1. Depth Gradient
        model.zero_grad()
        loss_d, _ = depth_loss_fn(preds["depth"], depth_norm_t, mask_t)
        loss_d.backward(retain_graph=True)
        g_d_shared = flatten_grads(shared_params)
        g_d_all = flatten_grads(all_trainable)
        norm_d = g_d_all.norm(2).item()

        # 2. Points Gradient
        model.zero_grad()
        loss_p, _ = pts_loss_fn(preds["pts3d"], pts_norm_t, mask_t)
        loss_p.backward(retain_graph=True)
        g_p_shared = flatten_grads(shared_params)
        g_p_all = flatten_grads(all_trainable)
        norm_p = g_p_all.norm(2).item()

        # 3. Normals Gradient
        model.zero_grad()
        loss_n, _ = norm_loss_fn(preds["normals"], norm_t, mask_t)
        loss_n.backward(retain_graph=True)
        g_n_shared = flatten_grads(shared_params)
        g_n_all = flatten_grads(all_trainable)
        norm_n = g_n_all.norm(2).item()

        # 4. Combined Multi-Task Gradient
        model.zero_grad()
        loss_comb = w_depth * loss_d + w_pts * loss_p + w_norm * loss_n
        loss_comb.backward()
        g_comb = flatten_grads(all_trainable)
        norm_comb = g_comb.norm(2).item()

        # Cosine similarities for shared backbone representation
        def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
            dot = (a * b).sum().item()
            norm_prod = a.norm(2).item() * b.norm(2).item()
            return dot / max(norm_prod, 1e-8)

        cos_dp_shared = cosine_sim(g_d_shared, g_p_shared)
        cos_dn_shared = cosine_sim(g_d_shared, g_n_shared)
        cos_pn_shared = cosine_sim(g_p_shared, g_n_shared)

        cos_dp_all = cosine_sim(g_d_all, g_p_all)
        cos_dn_all = cosine_sim(g_d_all, g_n_all)
        cos_pn_all = cosine_sim(g_p_all, g_n_all)

        gn_depth_list.append(norm_d)
        gn_pts_list.append(norm_p)
        gn_norm_list.append(norm_n)
        gn_comb_list.append(norm_comb)
        sum_weighted_gn_list.append(w_depth * norm_d + w_pts * norm_p + w_norm * norm_n)

        cos_dp_shared_list.append(cos_dp_shared)
        cos_dn_shared_list.append(cos_dn_shared)
        cos_pn_shared_list.append(cos_pn_shared)

        cos_dp_all_list.append(cos_dp_all)
        cos_dn_all_list.append(cos_dn_all)
        cos_pn_all_list.append(cos_pn_all)

        print(f"Sample {i+1:02d} [{s_id} #{f_info['frame_id']}]: Shared CosSim(Depth, Pts)={cos_dp_shared:+.4f}, CosSim(Depth, Norm)={cos_dn_shared:+.4f}, CosSim(Pts, Norm)={cos_pn_shared:+.4f} | ||g_comb||={norm_comb:.4f}")

    # Aggregated Analysis
    mean_cos_dp_shared = float(np.mean(cos_dp_shared_list))
    mean_cos_dn_shared = float(np.mean(cos_dn_shared_list))
    mean_cos_pn_shared = float(np.mean(cos_pn_shared_list))

    conflict_freq_dp = float(np.mean(np.array(cos_dp_shared_list) < 0.0))
    conflict_freq_dn = float(np.mean(np.array(cos_dn_shared_list) < 0.0))
    conflict_freq_pn = float(np.mean(np.array(cos_pn_shared_list) < 0.0))

    mean_gn_depth = float(np.mean(gn_depth_list))
    mean_gn_pts = float(np.mean(gn_pts_list))
    mean_gn_norm = float(np.mean(gn_norm_list))
    mean_gn_comb = float(np.mean(gn_comb_list))
    mean_sum_weighted = float(np.mean(sum_weighted_gn_list))
    cancellation_ratio = 1.0 - (mean_gn_comb / max(mean_sum_weighted, 1e-8))

    print("\n" + "=" * 75)
    print("MULTI-TASK GRADIENT INTERACTION SUMMARY")
    print("=" * 75)
    print(f"Shared Layer Cosine Similarities (Mean across {sample_count} samples):")
    print(f"  Depth vs Points  : {mean_cos_dp_shared:+.4f} (Conflict Frequency: {conflict_freq_dp*100:.1f}%)")
    print(f"  Depth vs Normals : {mean_cos_dn_shared:+.4f} (Conflict Frequency: {conflict_freq_dn*100:.1f}%)")
    print(f"  Points vs Normals: {mean_cos_pn_shared:+.4f} (Conflict Frequency: {conflict_freq_pn*100:.1f}%)")
    print(f"\nGradient Norms (Mean):")
    print(f"  Depth Head Gradient Norm (w=1.0) : {mean_gn_depth:.4f}")
    print(f"  Points Head Gradient Norm (w=0.5): {mean_gn_pts * w_pts:.4f} (Raw: {mean_gn_pts:.4f})")
    print(f"  Normals Head Gradient Norm (w=0.2): {mean_gn_norm * w_norm:.4f} (Raw: {mean_gn_norm:.4f})")
    print(f"  Actual Combined Gradient Norm    : {mean_gn_comb:.4f}")
    print(f"  Linear Sum of Weighted Norms     : {mean_sum_weighted:.4f}")
    print(f"  Destructive Interference / Cancellation Ratio: {cancellation_ratio*100:.1f}%")

    out_data = {
        "samples_audited": sample_count,
        "shared_layer_cosine_similarity": {
            "depth_vs_points_mean": mean_cos_dp_shared,
            "depth_vs_normals_mean": mean_cos_dn_shared,
            "points_vs_normals_mean": mean_cos_pn_shared,
            "conflict_frequency_depth_points": conflict_freq_dp,
            "conflict_frequency_depth_normals": conflict_freq_dn,
            "conflict_frequency_points_normals": conflict_freq_pn,
        },
        "all_parameters_cosine_similarity": {
            "depth_vs_points_mean": float(np.mean(cos_dp_all_list)),
            "depth_vs_normals_mean": float(np.mean(cos_dn_all_list)),
            "points_vs_normals_mean": float(np.mean(cos_pn_all_list)),
        },
        "gradient_norms": {
            "depth_mean": mean_gn_depth,
            "points_mean": mean_gn_pts,
            "normals_mean": mean_gn_norm,
            "combined_actual_mean": mean_gn_comb,
            "linear_sum_weighted_mean": mean_sum_weighted,
            "cancellation_ratio": cancellation_ratio,
        },
        "pcgrad_recommendation": {
            "required": bool(conflict_freq_dp > 0.5 or conflict_freq_dn > 0.5 or conflict_freq_pn > 0.5),
            "reasoning": "PCGrad / gradient projection surgery is indicated only if conflicting gradient frequency exceeds 50% in shared layers. Otherwise, natural multi-task synergy dominates."
        }
    }

    out_json = REPO_ROOT / "research/3d_reconstruction/experiments/phase9/gradient_interaction_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nGradient interaction audit saved to: {out_json}")


if __name__ == "__main__":
    main()

