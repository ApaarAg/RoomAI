"""
RoomAI — Phase 12 Experiment 05: True LoRA Rank 8 Fine-Tuning
Target Architecture: Tencent HunyuanWorld-Mirror 1.0 (1.223B parameters)
Adapters: Low-Rank Adaptation (r=8, alpha=16) on Visual Geometry Transformer
Late Attention Blocks (frame_blocks[-2:], global_blocks[-2:]).
"""

import os
import sys
import gc
import json
import csv
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research/3d_reconstruction/scripts"
sys.path.insert(0, str(EXTERNAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.models.models.worldmirror import WorldMirror
from scene_normalization import compute_world_coordinates

DATA_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth"
EXPANDED_MANIFEST = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest_expanded.json"
RESULTS_ROOT = REPO_ROOT / "results/final_execution"
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"

# Certified Pretrained Baseline (Phase 3 Certification)
CANONICAL_BASELINE = {
    "track_a_hypersim": {
        "absrel": 0.54687,
        "rmse": 4.6727,
        "delta1": 0.2926,
        "pts_sim3_mae": 3.2401,
        "norm_mae": 61.531,
    },
    "track_b_scannet": {
        "absrel": 0.27296,
        "rmse": 0.6618,
        "delta1": 0.5959,
        "pts_sim3_mae": 0.4633,
        "norm_mae": 62.937,
    },
}

# --- LoRA Module Definition ---
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""
    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Freeze original linear layer
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Low rank matrices A and B: delta_W = (B @ A) * scaling
        self.lora_A = nn.Parameter(torch.zeros((r, in_features), dtype=original_linear.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), dtype=original_linear.weight.dtype))
        
        # Initialize: Kaiming uniform for A, zero for B (guarantees delta_W = 0 at start)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original_linear(x)
        lora = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return orig + lora

def inject_lora_into_transformer(model: WorldMirror, r: int = 8, lora_alpha: float = 16.0) -> List[nn.Parameter]:
    """Injects LoRALinear into the attention projections of late transformer blocks."""
    vgt = model.visual_geometry_transformer
    lora_params = []
    
    target_blocks = []
    if hasattr(vgt, "frame_blocks") and len(vgt.frame_blocks) >= 2:
        target_blocks.extend([("frame", idx, vgt.frame_blocks[idx]) for idx in [-2, -1]])
    if hasattr(vgt, "global_blocks") and len(vgt.global_blocks) >= 2:
        target_blocks.extend([("global", idx, vgt.global_blocks[idx]) for idx in [-2, -1]])
        
    for block_type, idx, blk in target_blocks:
        attn = blk.attn
        # Wrap qkv
        if isinstance(attn.qkv, nn.Linear) and not isinstance(attn.qkv, LoRALinear):
            attn.qkv = LoRALinear(attn.qkv, r=r, lora_alpha=lora_alpha).to(device=attn.qkv.weight.device, dtype=attn.qkv.weight.dtype)
            lora_params.extend([attn.qkv.lora_A, attn.qkv.lora_B])
        # Wrap proj
        if isinstance(attn.proj, nn.Linear) and not isinstance(attn.proj, LoRALinear):
            attn.proj = LoRALinear(attn.proj, r=r, lora_alpha=lora_alpha).to(device=attn.proj.weight.device, dtype=attn.proj.weight.dtype)
            lora_params.extend([attn.proj.lora_A, attn.proj.lora_B])
            
    print(f"Successfully injected LoRA (rank={r}, alpha={lora_alpha}) into {len(target_blocks)} attention blocks.")
    return lora_params

def merge_lora_weights(model: WorldMirror):
    """Merges LoRA adapter weights directly into linear layers for zero-overhead inference."""
    vgt = model.visual_geometry_transformer
    for block_list in [vgt.frame_blocks, vgt.global_blocks]:
        for blk in block_list:
            if isinstance(blk.attn.qkv, LoRALinear):
                delta = (blk.attn.qkv.lora_B @ blk.attn.qkv.lora_A) * blk.attn.qkv.scaling
                blk.attn.qkv.original_linear.weight.data += delta.to(blk.attn.qkv.original_linear.weight.dtype)
                blk.attn.qkv = blk.attn.qkv.original_linear
            if isinstance(blk.attn.proj, LoRALinear):
                delta = (blk.attn.proj.lora_B @ blk.attn.proj.lora_A) * blk.attn.proj.scaling
                blk.attn.proj.original_linear.weight.data += delta.to(blk.attn.proj.original_linear.weight.dtype)
                blk.attn.proj = blk.attn.proj.original_linear
    print("Merged all LoRA parameters directly into transformer linear weights.")

# --- Corrected Dataset with Camera-Space Geometry ---
class StandardizedExpandedDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str = "train", dataset_filter: Optional[str] = None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.frames = []
        for f in manifest["frames"]:
            if f["split"] == split:
                if dataset_filter is None or f["dataset"] == dataset_filter:
                    self.frames.append(f)

        print(f"[{split.upper()}] Loaded {len(self.frames)} canonical frames (filter: {dataset_filter}).")
        assert len(self.frames) > 0, f"No frames found for split {split}!"

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        f_info = self.frames[idx]
        paths = f_info["paths"]
        s_id = f_info["scene_id"]
        scale_s = float(f_info["canonical_normalization"]["scene_scale_factor"])

        # RGB
        rgb_img = Image.open(DATA_ROOT / paths["rgb"]).convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1)

        # Depth
        depth_m = np.load(DATA_ROOT / paths["depth"]).astype(np.float32)
        depth_norm_tensor = torch.from_numpy(depth_m / scale_s)
        depth_metric_tensor = torch.from_numpy(depth_m)

        # Mask
        mask_raw = np.array(Image.open(DATA_ROOT / paths["mask"])) > 0
        valid_mask = mask_raw & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 30.0)
        mask_tensor = torch.from_numpy(valid_mask)

        # Camera Intrinsics
        cam_data = np.load(DATA_ROOT / paths["cam"])
        intr = cam_data["intrinsics"].astype(np.float32)

        # Camera-frame 3D Points (Camera Origin at 0, 0, 0)
        H, W = depth_m.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        fu, fv = float(intr[0, 0]), float(intr[1, 1])
        cu, cv = float(intr[0, 2]), float(intr[1, 2])
        z_cam = np.where(valid_mask, depth_m, 0.0).astype(np.float32)
        x_cam = (u - cu) * z_cam / fu
        y_cam = (v - cv) * z_cam / fv
        pts_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
        pts_norm_tensor = torch.from_numpy(pts_cam / scale_s)

        # Normals (ensure coordinate alignment with OpenCV forward +Z)
        norm_arr = np.load(DATA_ROOT / paths["normal"]).astype(np.float32)
        if f_info["dataset"] == "hypersim":
            # Hypersim OpenGL to OpenCV forward frame: negate Y and Z
            norm_arr = norm_arr * np.array([1.0, -1.0, 1.0], dtype=np.float32)
            
        norm_tensor = torch.from_numpy(norm_arr)

        return {
            "rgb": rgb_tensor,
            "depth_norm": depth_norm_tensor,
            "depth_metric": depth_metric_tensor,
            "pts_norm": pts_norm_tensor,
            "normals": norm_tensor,
            "mask": mask_tensor,
            "scene_id": s_id,
            "frame_id": str(f_info["frame_id"]),
            "dataset": f_info["dataset"],
            "scale_factor": scale_s,
        }

def collate_single(batch: List[Dict]) -> Dict:
    return batch[0]

# --- Loss Function ---
class MaskedMultiTaskLoss(nn.Module):
    def __init__(self, weight_depth: float = 1.0, weight_pts: float = 0.2, weight_norm: float = 0.05):
        super().__init__()
        self.weight_depth = weight_depth
        self.weight_pts = weight_pts
        self.weight_norm = weight_norm

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        mask = targets["mask"].squeeze()
        valid = (mask > 0) & torch.isfinite(targets["depth_norm"].squeeze()) & (targets["depth_norm"].squeeze() > 0.0)
        
        if valid.sum() < 100:
            zero_loss = torch.tensor(0.0, device=preds["depth"].device, requires_grad=True)
            return zero_loss, {"loss_total": 0.0, "loss_depth": 0.0, "loss_pts": 0.0, "loss_norm": 0.0}

        # 1. Depth L1 loss
        p_d = preds["depth"].squeeze()
        g_d = targets["depth_norm"].squeeze()
        loss_depth = (p_d[valid] - g_d[valid]).abs().clamp(max=50.0).mean()

        # 2. Points L2 loss (camera coordinates)
        p_pts = preds["pts3d"].squeeze()
        g_pts = targets["pts_norm"].squeeze()
        loss_pts = torch.norm(p_pts[valid] - g_pts[valid], p=2, dim=-1).clamp(max=50.0).mean()

        # 3. Normal Cosine loss
        p_norm = preds["normals"].squeeze()
        g_norm = targets["normals"].squeeze()
        p_unit = F.normalize(p_norm[valid], p=2, dim=-1, eps=1e-8)
        g_unit = F.normalize(g_norm[valid], p=2, dim=-1, eps=1e-8)
        cos_sim = (p_unit * g_unit).sum(dim=-1).clamp(-1.0, 1.0)
        loss_norm = (1.0 - cos_sim).mean()

        total = self.weight_depth * loss_depth + self.weight_pts * loss_pts + self.weight_norm * loss_norm
        metrics = {
            "loss_total": float(total.item()),
            "loss_depth": float(loss_depth.item()),
            "loss_pts": float(loss_pts.item()),
            "loss_norm": float(loss_norm.item()),
        }
        return total, metrics

def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    N = src.shape[0]
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    if var_src < 1e-12:
        return src, 1.0
    cov = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(cov)
    V = Vt.T
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[2, 2] = -1
    R = U @ S @ V.T
    scale = np.sum(D * np.diag(S)) / var_src
    scale = max(1e-4, min(scale, 100.0))
    t = mu_dst - scale * (R @ mu_src)
    return scale * (src @ R.T) + t, float(scale)

def evaluate_validation_split(
    model: WorldMirror,
    dataset: StandardizedExpandedDataset,
    device: str = "cuda",
    max_frames: int = 100,
) -> Dict[str, float]:
    model.eval()
    absrel_list = []
    rmse_list = []
    delta1_list = []
    sim3_mae_list = []
    norm_mae_list = []

    eval_indices = np.linspace(0, len(dataset) - 1, min(len(dataset), max_frames), dtype=int)

    with torch.no_grad():
        for idx in eval_indices:
            batch = dataset[int(idx)]
            rgb = batch["rgb"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            gt_d = batch["depth_metric"].numpy()
            gt_pts = batch["pts_norm"].numpy()
            gt_norm = batch["normals"].numpy()
            mask = batch["mask"].numpy()
            scale_s = batch["scale_factor"]

            preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=True)
            pred_d_norm = preds["depth"].squeeze().float().cpu().numpy()
            pred_d_metric = pred_d_norm * scale_s
            pred_pts = preds["pts3d"].squeeze().float().cpu().numpy()
            pred_norm = preds["normals"].squeeze().float().cpu().numpy()

            valid = mask & (gt_d > 0.05) & np.isfinite(gt_d) & np.isfinite(pred_d_metric)
            if valid.sum() < 100:
                continue

            g_valid = gt_d[valid]
            p_valid = pred_d_metric[valid]

            s_star = np.median(g_valid) / max(np.median(p_valid), 1e-6)
            s_star = max(0.1, min(s_star, 10.0))
            p_scaled = p_valid * s_star

            absrel_list.append(float(np.mean(np.abs(p_scaled - g_valid) / g_valid)))
            rmse_list.append(float(np.sqrt(np.mean((p_scaled - g_valid) ** 2))))
            delta1_list.append(float(np.mean(np.maximum(p_scaled / g_valid, g_valid / p_scaled) < 1.25)))

            # Subsampled Sim(3)
            sub_step = 16
            valid_sub = valid[::sub_step, ::sub_step]
            pred_sub = (pred_pts * scale_s)[::sub_step, ::sub_step][valid_sub]
            gt_sub = (gt_pts * scale_s)[::sub_step, ::sub_step][valid_sub]
            if len(gt_sub) > 50:
                aligned_pts, _ = umeyama_alignment(pred_sub, gt_sub)
                sim3_mae_list.append(float(np.mean(np.linalg.norm(aligned_pts - gt_sub, axis=-1))))

            # Surface Normal MAE
            pn_valid = pred_norm[valid]
            gn_valid = gt_norm[valid]
            pn_unit = pn_valid / np.maximum(np.linalg.norm(pn_valid, axis=-1, keepdims=True), 1e-12)
            gn_unit = gn_valid / np.maximum(np.linalg.norm(gn_valid, axis=-1, keepdims=True), 1e-12)
            cos_sim = np.clip(np.sum(pn_unit * gn_unit, axis=-1), -1.0, 1.0)
            norm_mae_list.append(float(np.mean(np.degrees(np.arccos(cos_sim)))))

    return {
        "absrel": float(np.mean(absrel_list)) if absrel_list else 0.0,
        "rmse": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "delta1": float(np.mean(delta1_list)) if delta1_list else 0.0,
        "pts_sim3_mae": float(np.mean(sim3_mae_list)) if sim3_mae_list else 0.0,
        "norm_mae": float(np.mean(norm_mae_list)) if norm_mae_list else 0.0,
        "valid_frames": len(absrel_list),
    }

def main():
    print("=" * 80)
    print("STARTING EXPERIMENT 05: TRUE LORA RANK 8 ADAPTATION")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Load Pretrained Checkpoint
    print(f"Loading pretrained reference model: {PRETRAINED_CKPT}")
    model = WorldMirror(
        img_size=518, patch_size=14, embed_dim=1024, gs_dim=256,
        enable_cam=True, enable_pts=True, enable_depth=True, enable_norm=True,
        enable_gs=False, enable_cond=False,
    ).to(device=device, dtype=torch.bfloat16)

    sd = torch.load(PRETRAINED_CKPT, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    del sd, clean_sd
    torch.cuda.empty_cache()

    # Freeze entire model initially
    for p in model.parameters():
        p.requires_grad = False

    # 2. Inject LoRA Rank 8 into late transformer attention
    lora_params = inject_lora_into_transformer(model, r=8, lora_alpha=16.0)
    for p in lora_params:
        p.requires_grad = True

    # 3. Allow head parameters to adapt conservatively
    head_params = []
    for h in [model.depth_head, model.pts_head, model.norm_head, model.cam_head]:
        if h is not None:
            for p in h.parameters():
                p.requires_grad = True
                head_params.append(p)

    lora_count = sum(p.numel() for p in lora_params)
    head_count = sum(p.numel() for p in head_params)
    total_trainable = lora_count + head_count
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nArchitecture Parameter Audit:")
    print(f"  LoRA Rank 8 Parameters: {lora_count:,}")
    print(f"  Prediction Head Parameters: {head_count:,}")
    print(f"  Total Trainable Parameters: {total_trainable:,} ({(total_trainable/total_params)*100:.3f}% of total {total_params:,})")

    # Optimizer with differentiated learning rates
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": 5e-5},
        {"params": head_params, "lr": 2e-6},
    ], weight_decay=1e-4)

    # 4. Load Data
    train_dataset = StandardizedExpandedDataset(EXPANDED_MANIFEST, split="train", dataset_filter="hypersim")
    val_hypersim = StandardizedExpandedDataset(EXPANDED_MANIFEST, split="val", dataset_filter="hypersim")
    val_scannet = StandardizedExpandedDataset(EXPANDED_MANIFEST, split="val", dataset_filter="scannet")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=g,
        collate_fn=collate_single,
    )

    loss_fn = MaskedMultiTaskLoss(weight_depth=1.0, weight_pts=0.2, weight_norm=0.05)
    max_steps = 150

    print(f"\nBeginning training for {max_steps} steps...")
    model.train()
    step = 0
    t0 = time.time()
    loss_history = []

    for b_idx, batch in enumerate(train_loader):
        step += 1
        rgb = batch["rgb"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        gt_d = batch["depth_norm"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        gt_pts = batch["pts_norm"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        gt_norm = batch["normals"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        mask = batch["mask"].unsqueeze(0).unsqueeze(0).to(device=device)

        targets = {
            "depth_norm": gt_d,
            "pts_norm": gt_pts,
            "normals": gt_norm,
            "mask": mask,
        }

        optimizer.zero_grad()
        preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=False)
        total_loss, loss_dict = loss_fn(preds, targets)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        loss_val = float(total_loss.item())
        loss_history.append(loss_val)

        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  Step [{step:03d}/{max_steps}] Scene: {batch['scene_id']} | "
                  f"Loss: {loss_val:.4f} (depth: {loss_dict['loss_depth']:.4f}, pts: {loss_dict['loss_pts']:.4f}, norm: {loss_dict['loss_norm']:.4f}) | "
                  f"Time: {elapsed:.1f}s")

        if step >= max_steps:
            break

    # 5. Merge LoRA weights & Save Checkpoint
    merge_lora_weights(model)
    ckpt_filename = "exp_05_expanded_lora_rank8_final.ckpt"
    ckpt_path = CHECKPOINTS_DIR / ckpt_filename
    print(f"\nSaving merged checkpoint to: {ckpt_path}")
    torch.save({"state_dict": model.state_dict(), "exp_id": "exp_05_expanded_lora_rank8"}, str(ckpt_path))

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    ckpt_sha = hasher.hexdigest()
    print(f"Checkpoint SHA-256: {ckpt_sha}")

    # 6. Evaluate Validation Splits
    print("\n" + "-" * 80)
    print("EVALUATING TRACK A (HYPERSIM VALIDATION)...")
    track_a_res = evaluate_validation_split(model, val_hypersim, device=device, max_frames=100)
    print(f"  Hypersim AbsRel: {track_a_res['absrel']:.4f} (Baseline: {CANONICAL_BASELINE['track_a_hypersim']['absrel']:.4f})")
    print(f"  Hypersim RMSE:   {track_a_res['rmse']:.4f} m")
    print(f"  Hypersim Delta1: {track_a_res['delta1']:.4f}")
    print(f"  Hypersim Sim(3): {track_a_res['pts_sim3_mae']:.4f} m")
    print(f"  Hypersim Normal: {track_a_res['norm_mae']:.2f} deg")

    print("\nEVALUATING TRACK B (SCANNET VALIDATION)...")
    track_b_res = evaluate_validation_split(model, val_scannet, device=device, max_frames=100)
    print(f"  ScanNet AbsRel: {track_b_res['absrel']:.4f} (Baseline: {CANONICAL_BASELINE['track_b_scannet']['absrel']:.4f})")
    print(f"  ScanNet RMSE:   {track_b_res['rmse']:.4f} m")
    print(f"  ScanNet Delta1: {track_b_res['delta1']:.4f}")
    print(f"  ScanNet Sim(3): {track_b_res['pts_sim3_mae']:.4f} m")
    print(f"  ScanNet Normal: {track_b_res['norm_mae']:.2f} deg")

    # Gate verification
    hypersim_base = CANONICAL_BASELINE["track_a_hypersim"]["absrel"]
    scannet_base = CANONICAL_BASELINE["track_b_scannet"]["absrel"]

    hypersim_gain_pct = ((track_a_res["absrel"] - hypersim_base) / hypersim_base) * 100.0
    scannet_reg_pct = ((track_b_res["absrel"] - scannet_base) / scannet_base) * 100.0

    passed_hypersim = hypersim_gain_pct <= -3.0
    passed_scannet = scannet_reg_pct <= 5.0
    overall_pass = passed_hypersim and passed_scannet

    verdict = "PROMOTED TO PRODUCTION CANDIDATE" if overall_pass else "REJECTED (Pretrained baseline remains reference winner)"
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 05 RESULT: {verdict}")
    print(f"  Hypersim Delta: {hypersim_gain_pct:+.2f}% (Threshold: <= -3.0%, Passed: {passed_hypersim})")
    print(f"  ScanNet Delta:  {scannet_reg_pct:+.2f}% (Threshold: <= +5.0%, Passed: {passed_scannet})")
    print("=" * 80)

    # Save summary
    res_dict = {
        "exp_id": "exp_05_expanded_lora_rank8",
        "checkpoint": ckpt_filename,
        "checkpoint_sha256": ckpt_sha,
        "adaptation_mode": "lora_rank8",
        "trainable_parameters": total_trainable,
        "lora_parameters": lora_count,
        "steps_trained": step,
        "initial_loss": loss_history[0] if loss_history else 0.0,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "track_a_hypersim": track_a_res,
        "track_b_scannet": track_b_res,
        "baseline_comparison": {
            "hypersim_baseline_absrel": hypersim_base,
            "hypersim_trained_absrel": track_a_res["absrel"],
            "hypersim_relative_change_pct": round(hypersim_gain_pct, 2),
            "scannet_baseline_absrel": scannet_base,
            "scannet_trained_absrel": track_b_res["absrel"],
            "scannet_relative_change_pct": round(scannet_reg_pct, 2),
        },
        "passed_success_gate": overall_pass,
        "verdict": verdict,
    }

    out_file = RESULTS_ROOT / "phase12_exp05_lora_summary.json"
    with open(out_file, "w") as f:
        json.dump(res_dict, f, indent=2)
    print(f"Saved experiment summary to {out_file}")

    # Append to experiment_registry.csv
    registry_file = RESULTS_ROOT / "experiment_registry.csv"
    with open(registry_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_05_expanded_lora_rank8",
            ckpt_filename,
            "lora_rank8",
            "5e-5",
            step,
            round(track_a_res["absrel"], 4),
            f"{hypersim_gain_pct:+.2f}%",
            round(track_b_res["absrel"], 4),
            f"{scannet_reg_pct:+.2f}%",
            overall_pass,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])
    print(f"Updated registry: {registry_file}")

if __name__ == "__main__":
    main()
