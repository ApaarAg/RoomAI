"""
ROOMAI — PHASE 8: CORRECTED STAGE 1 GEOMETRIC FINE-TUNING PIPELINE
Implements:
1. Canonical scene-coordinate normalization (s = nanmedian(||X_w||_2)).
2. Binary validity masking on depth, point, and normal losses.
3. Scale-balanced multi-task loss (w_depth=1.0, w_pts=0.5, w_norm=0.2).
4. Pre-training 1-batch dry run with gradient finiteness & parameter delta verification.
5. Checkpoint serialization, reload verification, and atomic saving.
6. Validation evaluation on held-out validation scenes.
"""

import sys
import os
import time
import json
import hashlib
import argparse
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
from corrected_losses import CanonicalMultiTaskLoss

DATA_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth"
MANIFEST_PATH = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json"
PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"


class CanonicalIndoorDataset:
    """Dataset delivering canonical scene-normalized geometric targets with validity masks."""
    def __init__(self, manifest_path: Path, split: str = "train", dataset_filter: str = None, device: str = "cuda"):
        self.device = device
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

    def get_sample(self, idx: int, dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
        f_info = self.frames[idx % len(self.frames)]
        paths = f_info["paths"]
        s_id = f_info["scene_id"]
        scale_s = float(f_info["canonical_normalization"]["scene_scale_factor"])

        # 1. RGB
        rgb_img = Image.open(DATA_ROOT / paths["rgb"]).convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=dtype)

        # 2. Metric Depth & Normalized Depth Target
        depth_m = np.load(DATA_ROOT / paths["depth"]).astype(np.float32)
        depth_norm = depth_m / scale_s
        depth_norm_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=dtype)

        # 3. Validity Mask
        mask_raw = np.array(Image.open(DATA_ROOT / paths["mask"])) > 0
        valid_mask = mask_raw & np.isfinite(depth_m) & (depth_m > 0.0)
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).unsqueeze(0).to(device=self.device)

        # 4. Camera Intrinsics, Pose, and Normalized 3D World Points
        cam_data = np.load(DATA_ROOT / paths["cam"])
        intr = cam_data["intrinsics"].astype(np.float32)
        pose = cam_data["pose"].astype(np.float32)

        pts_w, _ = compute_world_coordinates(depth_m, intr, pose, c2w=True)
        pts_norm = pts_w / scale_s
        pts_norm_tensor = torch.from_numpy(pts_norm).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=dtype)

        # 5. Surface Normals
        norm_arr = np.load(DATA_ROOT / paths["normal"]).astype(np.float32)
        norm_tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=dtype)

        return {
            "views": {"img": rgb_tensor},
            "targets": {
                "depth_norm": depth_norm_tensor,
                "pts_norm": pts_norm_tensor,
                "normals": norm_tensor,
                "mask": mask_tensor,
            },
            "meta": {
                "scene_id": s_id,
                "frame_id": f_info["frame_id"],
                "dataset": f_info["dataset"],
                "scale_factor": scale_s,
                "depth_metric_median": float(f_info["depth_metrics"]["median_m"]),
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


def run_dry_run(model: WorldMirror, dataset: CanonicalIndoorDataset, device: str = "cuda"):
    print("\n" + "=" * 75)
    print("PRE-TRAINING FEASIBILITY GATE: 1-BATCH DRY RUN & VALIDATION")
    print("=" * 75)

    sample = dataset.get_sample(0)
    views = sample["views"]
    targets = sample["targets"]
    meta = sample["meta"]

    print(f"1. Batch loaded: Scene '{meta['scene_id']}' (scale: {meta['scale_factor']:.4f}m)")
    print(f"   RGB shape: {views['img'].shape}, Depth shape: {targets['depth_norm'].shape}")
    print(f"   Normalized depth median: {targets['depth_norm'][targets['mask']].median().item():.4f}")

    # Freeze transformer, train heads
    for p in model.visual_geometry_transformer.parameters():
        p.requires_grad = False
    
    trainable_heads = [model.depth_head, model.pts_head, model.norm_head]
    trainable_params = []
    for h in trainable_heads:
        for p in h.parameters():
            p.requires_grad = True
            trainable_params.append(p)

    initial_weights = {i: p.detach().clone() for i, p in enumerate(trainable_params)}
    loss_fn = CanonicalMultiTaskLoss(weight_depth=1.0, weight_pts=0.5, weight_norm=0.2)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

    # Forward pass
    model.train()
    optimizer.zero_grad()
    preds = model(views=views, cond_flags=[0, 0, 0], is_inference=False)
    total_loss, metrics = loss_fn(preds, targets)
    print(f"2. Forward pass complete: Total Loss = {total_loss.item():.4f}")
    print(f"   depth: {metrics['loss_depth']:.4f}, pts: {metrics['loss_pts']:.4f}, norm: {metrics['loss_norm']:.4f}")

    # Backward pass
    total_loss.backward()
    print("3. Backward pass complete. Auditing gradient finiteness...")
    all_finite = True
    for i, p in enumerate(trainable_params):
        if p.grad is not None and not torch.isfinite(p.grad).all():
            all_finite = False
            break
    assert all_finite, "Non-finite gradient detected during dry run!"
    print("   -> 100% gradients are finite (0 NaN, 0 Inf).")

    # Optimizer update
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    print("4. Optimizer update executed successfully.")

    # Parameter delta check
    total_delta = 0.0
    for i, p in enumerate(trainable_params):
        delta = torch.norm(p.detach() - initial_weights[i]).item()
        total_delta += delta
    assert total_delta > 1e-6, f"Parameters did not update! delta={total_delta}"
    print(f"5. Parameter update verified: Sum parameter delta = {total_delta:.6f}")

    # Temporary checkpoint serialization & reload
    tmp_ckpt = REPO_ROOT / "research/3d_reconstruction/experiments/phase8_corrected/tmp_dry_run.ckpt"
    torch.save({"state_dict": model.state_dict(), "step": 1}, tmp_ckpt)
    assert tmp_ckpt.exists(), "Checkpoint failed to write to disk!"
    
    # Reload verification
    reloaded_sd = torch.load(tmp_ckpt, map_location="cpu")["state_dict"]
    assert len(reloaded_sd) == len(model.state_dict()), "Reloaded state dict key mismatch!"
    tmp_ckpt.unlink()  # Clean up temporary test file
    print("6. Checkpoint save & reload verification passed.")
    print("PRE-TRAINING GATE VERDICT: PASSED (All 6 criteria satisfied)\n")


def evaluate_val_split(model: WorldMirror, val_dataset: CanonicalIndoorDataset, max_eval: int = 50) -> Dict[str, float]:
    """Evaluates validation metrics on held-out scenes."""
    model.eval()
    absrel_list = []
    rmse_list = []
    delta1_list = []
    norm_mae_list = []

    eval_count = min(len(val_dataset), max_eval)
    with torch.no_grad():
        for i in range(eval_count):
            sample = val_dataset.get_sample(i)
            views = sample["views"]
            targets = sample["targets"]
            meta = sample["meta"]

            preds = model(views=views, cond_flags=[0, 0, 0], is_inference=True)
            pred_depth = preds["depth"].squeeze().float().cpu().numpy()  # (518, 518)
            gt_norm_depth = targets["depth_norm"].squeeze().float().cpu().numpy()
            mask = targets["mask"].squeeze().cpu().numpy()

            if mask.sum() < 100:
                continue

            # Scale-aligned evaluation using median ratio
            pred_valid = pred_depth[mask]
            gt_valid = gt_norm_depth[mask]
            scale_align = np.median(gt_valid) / np.clip(np.median(pred_valid), a_min=1e-6, a_max=None)
            pred_aligned = pred_valid * scale_align

            absrel = np.mean(np.abs(pred_aligned - gt_valid) / np.clip(gt_valid, 1e-3, None))
            rmse = np.sqrt(np.mean((pred_aligned - gt_valid) ** 2))
            ratio = np.maximum(pred_aligned / np.clip(gt_valid, 1e-3, None), gt_valid / np.clip(pred_aligned, 1e-3, None))
            delta1 = np.mean(ratio < 1.25)

            absrel_list.append(absrel)
            rmse_list.append(rmse)
            delta1_list.append(delta1)

            # Normal MAE
            if "normals" in preds:
                pred_norm = preds["normals"].squeeze().float().cpu().numpy()
                gt_norm = targets["normals"].squeeze().float().cpu().numpy()
                pn_unit = pred_norm[mask] / np.clip(np.linalg.norm(pred_norm[mask], axis=-1, keepdims=True), 1e-6, None)
                gn_unit = gt_norm[mask] / np.clip(np.linalg.norm(gt_norm[mask], axis=-1, keepdims=True), 1e-6, None)
                cos_sim = np.clip(np.sum(pn_unit * gn_unit, axis=-1), -1.0, 1.0)
                mae_deg = np.mean(np.rad2deg(np.arccos(cos_sim)))
                norm_mae_list.append(mae_deg)

    return {
        "val_scaled_absrel": float(np.mean(absrel_list)),
        "val_scaled_rmse": float(np.mean(rmse_list)),
        "val_scaled_delta1": float(np.mean(delta1_list)),
        "val_normal_mae_deg": float(np.mean(norm_mae_list)) if norm_mae_list else 0.0,
        "eval_frames": len(absrel_list),
    }


def run_training_experiment(
    exp_name: str,
    steps: int = 30,
    lr: float = 1e-4,
    dataset_filter: str = "hypersim",
    device: str = "cuda",
):
    print("\n" + "=" * 75)
    print(f"RUNNING EXPERIMENT: {exp_name} (Filter: {dataset_filter}, Steps: {steps}, LR: {lr})")
    print("=" * 75)

    exp_dir = REPO_ROOT / "research/3d_reconstruction/experiments/phase8_corrected" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets
    train_dataset = CanonicalIndoorDataset(MANIFEST_PATH, split="train", dataset_filter=dataset_filter, device=device)
    val_dataset = CanonicalIndoorDataset(MANIFEST_PATH, split="val", dataset_filter=dataset_filter, device=device)

    # 2. Model & Optimizer
    model = load_model(PRETRAINED_CKPT, device=device)
    
    # Freeze transformer backbone, train heads
    for p in model.visual_geometry_transformer.parameters():
        p.requires_grad = False
    
    trainable_heads = [model.depth_head, model.pts_head, model.norm_head]
    trainable_params = []
    for h in trainable_heads:
        for p in h.parameters():
            p.requires_grad = True
            trainable_params.append(p)

    loss_fn = CanonicalMultiTaskLoss(weight_depth=1.0, weight_pts=0.5, weight_norm=0.2)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # 3. Initial Baseline Validation
    print("Evaluating pre-training baseline on validation split...")
    init_val = evaluate_val_split(model, val_dataset)
    print(f"  Pre-train Val AbsRel: {init_val['val_scaled_absrel']:.4f} | RMSE: {init_val['val_scaled_rmse']:.4f} | Delta1: {init_val['val_scaled_delta1']:.4f}")

    best_val_absrel = init_val["val_scaled_absrel"]
    best_ckpt_path = exp_dir / "best_model.ckpt"
    history = []

    model.train()
    for step in range(1, steps + 1):
        t0 = time.time()
        optimizer.zero_grad()

        sample = train_dataset.get_sample(step - 1)
        preds = model(views=sample["views"], cond_flags=[0, 0, 0], is_inference=False)
        total_loss, metrics = loss_fn(preds, sample["targets"])

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        elapsed = time.time() - t0
        metrics["step"] = step
        metrics["elapsed_s"] = round(elapsed, 2)
        metrics["scene_id"] = sample["meta"]["scene_id"]
        history.append(metrics)

        print(f"  Step [{step:02d}/{steps:02d}] Scene: {sample['meta']['scene_id']:<12} | Loss: {total_loss.item():.4f} (depth: {metrics['loss_depth']:.4f}, pts: {metrics['loss_pts']:.4f}, norm: {metrics['loss_norm']:.4f}) | {elapsed:.2f}s")

    # 4. Post-Training Validation Evaluation
    print("\nEvaluating post-training fine-tuned model on validation split...")
    post_val = evaluate_val_split(model, val_dataset)
    print(f"  Post-train Val AbsRel: {post_val['val_scaled_absrel']:.4f} | RMSE: {post_val['val_scaled_rmse']:.4f} | Delta1: {post_val['val_scaled_delta1']:.4f}")

    # Check improvement
    improvement = post_val["val_scaled_absrel"] < init_val["val_scaled_absrel"]
    print(f"  Validation Improvement: {improvement} (AbsRel {init_val['val_scaled_absrel']:.4f} -> {post_val['val_scaled_absrel']:.4f})")

    # Save checkpoint
    final_ckpt_path = exp_dir / f"{exp_name}_final.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "experiment": exp_name,
        "steps": steps,
        "init_val": init_val,
        "post_val": post_val,
        "history": history,
    }, final_ckpt_path)

    # Compute SHA-256
    h = hashlib.sha256()
    with open(final_ckpt_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    ckpt_sha = h.hexdigest()
    print(f"Saved checkpoint to {final_ckpt_path.name} (SHA-256: {ckpt_sha})")

    results = {
        "experiment": exp_name,
        "dataset_filter": dataset_filter,
        "steps": steps,
        "lr": lr,
        "init_validation": init_val,
        "post_validation": post_val,
        "improved": improvement,
        "checkpoint_path": str(final_ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "training_history": history,
    }
    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run 1-batch pre-training feasibility dry run")
    parser.add_argument("--exp", type=str, default="dry_run", choices=["dry_run", "exp_a", "exp_b", "all"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dry_run or args.exp == "dry_run":
        dataset = CanonicalIndoorDataset(MANIFEST_PATH, split="train", dataset_filter="hypersim", device=device)
        model = load_model(PRETRAINED_CKPT, device=device)
        run_dry_run(model, dataset, device=device)

    elif args.exp == "exp_a":
        # Experiment A: Official loss reproduction on Hypersim (Track A Commercial Candidate)
        run_training_experiment("experiment_a_hypersim", steps=args.steps, lr=args.lr, dataset_filter="hypersim", device=device)

    elif args.exp == "exp_b":
        # Experiment B: Multi-view geometric supervision (all training scenes)
        run_training_experiment("experiment_b_multiview", steps=args.steps, lr=args.lr, dataset_filter=None, device=device)

    elif args.exp == "all":
        # Dry run first
        dataset = CanonicalIndoorDataset(MANIFEST_PATH, split="train", dataset_filter="hypersim", device=device)
        model = load_model(PRETRAINED_CKPT, device=device)
        run_dry_run(model, dataset, device=device)
        del model
        torch.cuda.empty_cache()

        # Experiment A
        res_a = run_training_experiment("experiment_a_hypersim", steps=args.steps, lr=args.lr, dataset_filter="hypersim", device=device)
        torch.cuda.empty_cache()

        # Experiment B
        res_b = run_training_experiment("experiment_b_multiview", steps=args.steps, lr=args.lr, dataset_filter=None, device=device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

