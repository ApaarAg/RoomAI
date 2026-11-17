"""
ROOMAI — PHASE 9: ADVANCED CONTROLLED FINE-TUNING EXPERIMENT SUITE
Implements:
1. PyTorch Shuffled DataLoader with deterministic seed resolving Phase 8 sampling overlap.
2. Experiment A: Head-Only Adaptation (freeze backbone, train DPT heads, lr=2e-6).
3. Experiment B: Partial Backbone Adaptation (unfreeze frame_blocks[-1] & global_blocks[-1], lr=5e-7 / 2e-6).
4. Multi-task loss with conflict-mitigated normal weighting (w_depth=1.0, w_pts=0.5, w_norm=0.1).
5. Comprehensive dual-track validation evaluation (Track A Hypersim vs Track B ScanNet).
6. Peak VRAM & latency telemetry.
7. Atomic checkpoint serialization with SHA-256 digests.
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path("f:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror
from scene_normalization import compute_world_coordinates
from corrected_losses import MaskedNormalizedDepthLoss, MaskedNormalizedPointLoss, MaskedNormalLoss, CanonicalMultiTaskLoss

DATA_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth"
MANIFEST_PATH = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json"
PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"
EXP_DIR = REPO_ROOT / "research/3d_reconstruction/experiments/phase9"


class ShuffledCanonicalDataset(Dataset):
    """PyTorch Dataset delivering canonical scene-normalized geometric targets with validity masks."""
    def __init__(self, manifest_path: Path, split: str = "train", dataset_filter: str = None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.frames = []
        for f in manifest["frames"]:
            if f["split"] == split:
                if dataset_filter is None or f["dataset"] == dataset_filter:
                    self.frames.append(f)

        print(f"Loaded {len(self.frames)} frames for split '{split}' (filter: {dataset_filter}).")
        assert len(self.frames) > 0, f"No frames found for split {split}!"

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_info = self.frames[idx]
        paths = f_info["paths"]
        s_id = f_info["scene_id"]
        scale_s = float(f_info["canonical_normalization"]["scene_scale_factor"])

        # 1. RGB Image: (3, 518, 518) in [0, 1]
        rgb_img = Image.open(DATA_ROOT / paths["rgb"]).convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1)

        # 2. Metric Depth & Normalized Depth Target: (518, 518)
        depth_m = np.load(DATA_ROOT / paths["depth"]).astype(np.float32)
        depth_norm = depth_m / scale_s
        depth_norm_tensor = torch.from_numpy(depth_norm)
        depth_metric_tensor = torch.from_numpy(depth_m)

        # 3. Validity Mask: (518, 518)
        mask_raw = np.array(Image.open(DATA_ROOT / paths["mask"])) > 0
        valid_mask = mask_raw & np.isfinite(depth_m) & (depth_m > 0.0)
        mask_tensor = torch.from_numpy(valid_mask)

        # 4. Camera Intrinsics, Pose, and Normalized 3D World Points: (518, 518, 3)
        cam_data = np.load(DATA_ROOT / paths["cam"])
        intr = cam_data["intrinsics"].astype(np.float32)
        pose = cam_data["pose"].astype(np.float32)

        pts_w, _ = compute_world_coordinates(depth_m, intr, pose, c2w=True)
        pts_norm = pts_w / scale_s
        pts_norm_tensor = torch.from_numpy(pts_norm)

        # 5. Surface Normals: (518, 518, 3)
        norm_arr = np.load(DATA_ROOT / paths["normal"]).astype(np.float32)
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
    b = batch[0]
    return {
        "rgb": b["rgb"].unsqueeze(0).unsqueeze(0),                # (1, 1, 3, H, W)
        "depth_norm": b["depth_norm"].unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        "depth_metric": b["depth_metric"].unsqueeze(0).unsqueeze(0),
        "pts_norm": b["pts_norm"].unsqueeze(0).unsqueeze(0),      # (1, 1, H, W, 3)
        "normals": b["normals"].unsqueeze(0).unsqueeze(0),        # (1, 1, H, W, 3)
        "mask": b["mask"].unsqueeze(0).unsqueeze(0),              # (1, 1, H, W)
        "meta": {
            "scene_id": b["scene_id"],
            "frame_id": b["frame_id"],
            "dataset": b["dataset"],
            "scale_factor": b["scale_factor"],
            "raw_depth": b["depth_metric"].numpy(),
            "raw_mask": b["mask"].numpy(),
            "raw_norm": b["normals"].numpy(),
        }
    }


def load_model(ckpt_path: Path, device: str = "cuda", dtype=torch.bfloat16) -> WorldMirror:
    print(f"Instantiating WorldMirror architecture...")
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
    ).to(device=device, dtype=dtype)

    print(f"Loading weights from: {ckpt_path.name}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    del sd, clean_sd
    torch.cuda.empty_cache()
    return model


def evaluate_benchmark(model: WorldMirror, dataset: ShuffledCanonicalDataset, max_eval: int = 50, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    absrel_list = []
    rmse_list = []
    delta1_list = []
    norm_mae_list = []
    latencies = []

    eval_count = min(len(dataset), max_eval)
    torch.cuda.reset_peak_memory_stats(device=device)

    with torch.no_grad():
        for i in range(eval_count):
            item = dataset[i]
            rgb = item["rgb"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            gt_m = item["depth_metric"].numpy()
            mask = item["mask"].numpy()

            t0 = time.time()
            preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=True)
            dt = time.time() - t0
            latencies.append(dt)

            pred_depth = preds["depth"].squeeze().float().cpu().numpy()

            val_m = mask & (gt_m > 0.1) & np.isfinite(gt_m) & np.isfinite(pred_depth) & (pred_depth > 0.01)
            if val_m.sum() < 100:
                continue

            p_val = pred_depth[val_m]
            g_val = gt_m[val_m]
            s_star = float(np.median(g_val) / np.clip(np.median(p_val), 1e-6, None))
            p_scaled = p_val * s_star

            abs_diff = np.abs(p_scaled - g_val)
            absrel = float(np.mean(abs_diff / g_val))
            rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
            ratio = np.maximum(p_scaled / g_val, g_val / p_scaled)
            delta1 = float(np.mean(ratio < 1.25))

            absrel_list.append(absrel)
            rmse_list.append(rmse)
            delta1_list.append(delta1)

            if "normals" in preds:
                pred_norm = preds["normals"].squeeze().float().cpu().numpy()
                gt_norm = item["normals"].numpy()
                pn_unit = pred_norm[val_m] / np.clip(np.linalg.norm(pred_norm[val_m], axis=-1, keepdims=True), 1e-6, None)
                gn_unit = gt_norm[val_m] / np.clip(np.linalg.norm(gt_norm[val_m], axis=-1, keepdims=True), 1e-6, None)
                cos_sim = np.clip(np.sum(pn_unit * gn_unit, axis=-1), -1.0, 1.0)
                mae_deg = float(np.mean(np.rad2deg(np.arccos(cos_sim))))
                norm_mae_list.append(mae_deg)

    peak_vram_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

    return {
        "scaled_absrel": float(np.mean(absrel_list)),
        "scaled_rmse_m": float(np.mean(rmse_list)),
        "scaled_delta1": float(np.mean(delta1_list)),
        "normal_mae_deg": float(np.mean(norm_mae_list)) if norm_mae_list else 0.0,
        "mean_latency_ms": float(np.mean(latencies) * 1000.0),
        "peak_vram_mb": float(peak_vram_mb),
        "evaluated_frames": len(absrel_list),
    }


def execute_phase9_experiment(
    exp_name: str,
    adaptation_mode: str = "head_only",  # "head_only" or "partial_backbone"
    epochs: int = 2,
    device: str = "cuda",
) -> Dict:
    print("\n" + "=" * 75)
    print(f"PHASE 9 EXPERIMENT: {exp_name}")
    print(f"Adaptation Mode: {adaptation_mode} | Epochs: {epochs}")
    print("=" * 75)

    exp_sub_dir = EXP_DIR / exp_name
    exp_sub_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets & Shuffled DataLoader
    train_dataset = ShuffledCanonicalDataset(MANIFEST_PATH, split="train", dataset_filter="hypersim")
    val_hypersim = ShuffledCanonicalDataset(MANIFEST_PATH, split="val", dataset_filter="hypersim")
    val_scannet = ShuffledCanonicalDataset(MANIFEST_PATH, split="val", dataset_filter="scannet")

    # Deterministic shuffled loader with batch_size=1
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=g,
        collate_fn=collate_single,
    )

    # 2. Model Setup
    model = load_model(PRETRAINED_CKPT, device=device)

    # Freeze all parameters by default
    for p in model.parameters():
        p.requires_grad = False

    # Configure parameter groups
    trainable_heads = [model.depth_head, model.pts_head, model.norm_head]
    head_params = []
    for h in trainable_heads:
        for p in h.parameters():
            p.requires_grad = True
            head_params.append(p)

    if adaptation_mode == "partial_backbone":
        # Unfreeze final attention blocks of visual geometry transformer
        last_frame = model.visual_geometry_transformer.frame_blocks[-1]
        last_global = model.visual_geometry_transformer.global_blocks[-1]
        backbone_params = []
        for mod in [last_frame, last_global]:
            for p in mod.parameters():
                p.requires_grad = True
                backbone_params.append(p)

        param_groups = [
            {"params": backbone_params, "lr": 5e-7, "weight_decay": 0.1},
            {"params": head_params, "lr": 2e-6, "weight_decay": 0.1},
        ]
        all_trainable = backbone_params + head_params
        print(f"Partial Backbone Mode: {len(backbone_params)} backbone tensors (lr=5e-7) + {len(head_params)} head tensors (lr=2e-6).")
    else:
        # Head only
        param_groups = [
            {"params": head_params, "lr": 2e-6, "weight_decay": 0.1},
        ]
        all_trainable = head_params
        print(f"Head Only Mode: {len(head_params)} head tensors (lr=2e-6).")

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    loss_fn = CanonicalMultiTaskLoss(weight_depth=1.0, weight_pts=0.5, weight_norm=0.1)

    # 3. Initial Baseline Validation
    print("\nEvaluating initial baseline performance on validation split...")
    base_hyper = evaluate_benchmark(model, val_hypersim, max_eval=50, device=device)
    base_scannet = evaluate_benchmark(model, val_scannet, max_eval=50, device=device)
    print(f"  [Baseline] Hypersim: AbsRel = {base_hyper['scaled_absrel']:.4f}, RMSE = {base_hyper['scaled_rmse_m']:.4f}m, Delta1 = {base_hyper['scaled_delta1']:.4f}")
    print(f"  [Baseline] ScanNet : AbsRel = {base_scannet['scaled_absrel']:.4f}, RMSE = {base_scannet['scaled_rmse_m']:.4f}m, Delta1 = {base_scannet['scaled_delta1']:.4f}")

    # 4. Training Loop across Epochs
    step = 0
    history = []
    unique_scenes_seen = set()

    for epoch in range(1, epochs + 1):
        print(f"\n--- EPOCH {epoch}/{epochs} ---")
        model.train()
        for batch in train_loader:
            step += 1
            t0 = time.time()
            optimizer.zero_grad()

            rgb = batch["rgb"].to(device=device, dtype=torch.bfloat16)
            targets = {
                "depth_norm": batch["depth_norm"].to(device=device, dtype=torch.bfloat16),
                "pts_norm": batch["pts_norm"].to(device=device, dtype=torch.bfloat16),
                "normals": batch["normals"].to(device=device, dtype=torch.bfloat16),
                "mask": batch["mask"].to(device=device),
            }
            meta = batch["meta"]
            unique_scenes_seen.add(meta["scene_id"])

            preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=False)
            total_loss, metrics = loss_fn(preds, targets)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=0.5)
            optimizer.step()

            dt = time.time() - t0
            metrics["step"] = step
            metrics["epoch"] = epoch
            metrics["scene_id"] = meta["scene_id"]
            metrics["elapsed_s"] = round(dt, 2)
            history.append(metrics)

            if step % 5 == 0 or step == 1 or step == len(train_loader) * epochs:
                print(f"  Step [{step:02d}] Scene: {meta['scene_id']:<12} #{meta['frame_id']} | Loss: {total_loss.item():.4f} (d: {metrics['loss_depth']:.4f}, p: {metrics['loss_pts']:.4f}, n: {metrics['loss_norm']:.4f}) | {dt:.2f}s")

    print(f"\nTraining complete: {step} steps over {epochs} epochs. Unique scenes traversed: {sorted(list(unique_scenes_seen))}")

    # 5. Post-Training Validation Evaluation
    print("\nEvaluating post-training fine-tuned model...")
    post_hyper = evaluate_benchmark(model, val_hypersim, max_eval=50, device=device)
    post_scannet = evaluate_benchmark(model, val_scannet, max_eval=50, device=device)

    print(f"  [Post-Train] Hypersim: AbsRel = {post_hyper['scaled_absrel']:.4f}, RMSE = {post_hyper['scaled_rmse_m']:.4f}m, Delta1 = {post_hyper['scaled_delta1']:.4f}")
    print(f"  [Post-Train] ScanNet : AbsRel = {post_scannet['scaled_absrel']:.4f}, RMSE = {post_scannet['scaled_rmse_m']:.4f}m, Delta1 = {post_scannet['scaled_delta1']:.4f}")

    hyper_improved = post_hyper["scaled_absrel"] < base_hyper["scaled_absrel"]
    scannet_regression = post_scannet["scaled_absrel"] > (base_scannet["scaled_absrel"] * 1.10)

    print(f"\n  Result: Hypersim Improved = {hyper_improved} ({base_hyper['scaled_absrel']:.4f} -> {post_hyper['scaled_absrel']:.4f})")
    print(f"  Result: ScanNet Regression = {scannet_regression} ({base_scannet['scaled_absrel']:.4f} -> {post_scannet['scaled_absrel']:.4f})")

    # 6. Save Checkpoint & SHA-256
    ckpt_path = exp_sub_dir / f"{exp_name}_final.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "experiment": exp_name,
        "adaptation_mode": adaptation_mode,
        "epochs": epochs,
        "steps": step,
        "base_val_hyper": base_hyper,
        "post_val_hyper": post_hyper,
        "base_val_scannet": base_scannet,
        "post_val_scannet": post_scannet,
        "history": history,
    }, ckpt_path)

    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    ckpt_sha = h.hexdigest()
    print(f"Saved checkpoint: {ckpt_path.name} (SHA-256: {ckpt_sha})")

    summary = {
        "experiment": exp_name,
        "adaptation_mode": adaptation_mode,
        "epochs": epochs,
        "total_steps": step,
        "unique_scenes": list(unique_scenes_seen),
        "baseline_hypersim": base_hyper,
        "post_train_hypersim": post_hyper,
        "baseline_scannet": base_scannet,
        "post_train_scannet": post_scannet,
        "hypersim_improved": hyper_improved,
        "scannet_regression": scannet_regression,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
    }
    with open(exp_sub_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Phase 9 Experimental Suite on {device}...")

    # Experiment A: Head-Only Adaptation
    print("\n" + "#" * 75)
    print("# LAUNCHING PHASE 9 EXPERIMENT A: HEAD-ONLY ADAPTATION")
    print("#" * 75)
    res_a = execute_phase9_experiment(
        exp_name="phase9_exp_a_head_only",
        adaptation_mode="head_only",
        epochs=2,
        device=device,
    )
    torch.cuda.empty_cache()

    # Experiment B: Partial Backbone Adaptation
    print("\n" + "#" * 75)
    print("# LAUNCHING PHASE 9 EXPERIMENT B: PARTIAL BACKBONE ADAPTATION")
    print("#" * 75)
    res_b = execute_phase9_experiment(
        exp_name="phase9_exp_b_partial_backbone",
        adaptation_mode="partial_backbone",
        epochs=2,
        device=device,
    )
    torch.cuda.empty_cache()

    master_summary = {
        "experiment_a": res_a,
        "experiment_b": res_b,
    }
    with open(EXP_DIR / "phase9_experiments_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print("\n" + "=" * 75)
    print("PHASE 9 EXPERIMENTS COMPLETE. MASTER SUMMARY SAVED.")
    print("=" * 75)


if __name__ == "__main__":
    main()

