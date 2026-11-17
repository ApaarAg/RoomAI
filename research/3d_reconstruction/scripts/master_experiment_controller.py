"""
ROOMAI — MASTER EXPERIMENT CONTROLLER & AUTONOMOUS RESEARCH ENGINE
Coordinates the scientific execution cycle:
    TRAIN -> VALIDATE -> ANALYZE -> DIAGNOSE -> MODIFY -> RETRAIN

Maintains:
- results/master_training/controller_state.json
- results/master_training/experiment_registry.csv
- Checkpoints in results/master_training/checkpoints/
- Evaluations in results/master_training/evaluations/
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import gc
import csv
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path("f:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research/3d_reconstruction/scripts"
sys.path.insert(0, str(EXTERNAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.models.models.worldmirror import WorldMirror
from scene_normalization import compute_world_coordinates
from corrected_losses import CanonicalMultiTaskLoss

DATA_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth"
MANIFEST_PATH = REPO_ROOT / "data/manifests/indoor_canonical_manifest_v2.json"
if not MANIFEST_PATH.exists():
    MANIFEST_PATH = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest_v2.json"

RESULTS_ROOT = REPO_ROOT / "results/master_training"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATIONS_DIR = RESULTS_ROOT / "evaluations"
EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = RESULTS_ROOT / "controller_state.json"
REGISTRY_FILE = RESULTS_ROOT / "experiment_registry.csv"
PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"

# Canonical Pretrained Baseline Values (certified in Phase 3)
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


class MasterDataset(Dataset):
    """Loads standardized canonical indoor frames."""
    def __init__(self, manifest_path: Path, split: str = "train", dataset_filter: Optional[str] = None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.frames = []
        for f in manifest["frames"]:
            if f["split"] == split:
                if dataset_filter is None or f["dataset"] == dataset_filter:
                    self.frames.append(f)

        print(f"[{split.upper()}] Loaded {len(self.frames)} frames (filter: {dataset_filter}).")
        assert len(self.frames) > 0, f"No frames found for split {split}!"

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        f_info = self.frames[idx]
        paths = f_info["paths"]
        s_id = f_info["scene_id"]
        scale_s = float(f_info["canonical_normalization"]["scene_scale_factor"])

        # 1. RGB
        rgb_img = Image.open(DATA_ROOT / paths["rgb"]).convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1)

        # 2. Depth
        depth_m = np.load(DATA_ROOT / paths["depth"]).astype(np.float32)
        depth_norm_tensor = torch.from_numpy(depth_m / scale_s)
        depth_metric_tensor = torch.from_numpy(depth_m)

        # 3. Mask
        mask_raw = np.array(Image.open(DATA_ROOT / paths["mask"])) > 0
        valid_mask = mask_raw & np.isfinite(depth_m) & (depth_m > 0.0)
        mask_tensor = torch.from_numpy(valid_mask)

        # 4. Points
        cam_data = np.load(DATA_ROOT / paths["cam"])
        intr = cam_data["intrinsics"].astype(np.float32)
        pose = cam_data["pose"].astype(np.float32)

        pts_w, _ = compute_world_coordinates(depth_m, intr, pose, c2w=True)
        pts_norm_tensor = torch.from_numpy(pts_w / scale_s)

        # 5. Normals
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
    return batch[0]


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Computes optimal Sim(3) similarity transformation: dst = s * R @ src + t."""
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
    aligned_src = scale * (src @ R.T) + t
    return aligned_src, float(scale)


def load_model(ckpt_path: Path, device: str = "cuda", dtype=torch.bfloat16) -> WorldMirror:
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

    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    del sd, clean_sd
    torch.cuda.empty_cache()
    return model


def evaluate_dataset(
    model: WorldMirror,
    dataset: MasterDataset,
    device: str = "cuda",
) -> Dict[str, float]:
    model.eval()
    absrel_list = []
    rmse_list = []
    delta1_list = []
    sim3_mae_list = []
    norm_mae_list = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
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

            # Depth metrics
            absrel = np.mean(np.abs(p_scaled - g_valid) / g_valid)
            rmse = np.sqrt(np.mean((p_scaled - g_valid) ** 2))
            ratio = np.maximum(p_scaled / g_valid, g_valid / p_scaled)
            delta1 = np.mean(ratio < 1.25)

            absrel_list.append(float(absrel))
            rmse_list.append(float(rmse))
            delta1_list.append(float(delta1))

            # Point cloud Sim(3) metric
            sub_step = 16
            valid_sub = valid[::sub_step, ::sub_step]
            pred_sub = (pred_pts * scale_s)[::sub_step, ::sub_step][valid_sub]
            gt_sub = (gt_pts * scale_s)[::sub_step, ::sub_step][valid_sub]
            if len(gt_sub) > 50:
                aligned_pts, _ = umeyama_alignment(pred_sub, gt_sub)
                sim3_mae = np.mean(np.linalg.norm(aligned_pts - gt_sub, axis=-1))
                sim3_mae_list.append(float(sim3_mae))

            # Surface normal MAE
            pn_valid = pred_norm[valid]
            gn_valid = gt_norm[valid]
            pn_unit = pn_valid / np.maximum(np.linalg.norm(pn_valid, axis=-1, keepdims=True), 1e-12)
            gn_unit = gn_valid / np.maximum(np.linalg.norm(gn_valid, axis=-1, keepdims=True), 1e-12)
            cos_sim = np.clip(np.sum(pn_unit * gn_unit, axis=-1), -1.0, 1.0)
            ang_err = np.degrees(np.arccos(cos_sim))
            norm_mae_list.append(float(np.mean(ang_err)))

    return {
        "absrel": float(np.mean(absrel_list)) if absrel_list else 0.0,
        "rmse": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "delta1": float(np.mean(delta1_list)) if delta1_list else 0.0,
        "pts_sim3_mae": float(np.mean(sim3_mae_list)) if sim3_mae_list else 0.0,
        "norm_mae": float(np.mean(norm_mae_list)) if norm_mae_list else 0.0,
        "valid_frames": len(absrel_list),
    }


def execute_training_experiment(
    exp_id: str,
    adaptation_mode: str = "head_only",  # "head_only" or "partial_backbone"
    dataset_filter: Optional[str] = None, # None (multi-dataset) or "hypersim" or "scannet"
    lr_heads: float = 5e-5,
    lr_backbone: float = 1e-6,
    epochs: int = 2,
    weight_depth: float = 1.0,
    weight_pts: float = 0.5,
    weight_norm: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {exp_id}")
    print(f"Adaptation: {adaptation_mode} | Dataset: {dataset_filter or 'ALL_CANONICAL (Multi-Dataset)'}")
    print(f"LR Heads: {lr_heads} | LR Backbone: {lr_backbone} | Epochs: {epochs} | Seed: {seed}")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Datasets & Loaders
    train_dataset = MasterDataset(MANIFEST_PATH, split="train", dataset_filter=dataset_filter)
    val_hypersim = MasterDataset(MANIFEST_PATH, split="val", dataset_filter="hypersim")
    val_scannet = MasterDataset(MANIFEST_PATH, split="val", dataset_filter="scannet")

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=g,
        collate_fn=collate_single,
    )

    # 2. Instantiate Model from Untouched Pretrained Baseline
    model = load_model(PRETRAINED_CKPT, device=device, dtype=torch.bfloat16)

    # Freeze strategy
    for p in model.parameters():
        p.requires_grad = False

    head_params = []
    for h in [model.depth_head, model.pts_head, model.norm_head, model.cam_head]:
        if h is not None:
            for p in h.parameters():
                p.requires_grad = True
                head_params.append(p)

    param_groups = [{"params": head_params, "lr": lr_heads}]

    if adaptation_mode == "partial_backbone":
        vgt = model.visual_geometry_transformer
        backbone_params = []
        if hasattr(vgt, "frame_blocks") and len(vgt.frame_blocks) > 0:
            for p in vgt.frame_blocks[-1].parameters():
                p.requires_grad = True
                backbone_params.append(p)
        if hasattr(vgt, "global_blocks") and len(vgt.global_blocks) > 0:
            for p in vgt.global_blocks[-1].parameters():
                p.requires_grad = True
                backbone_params.append(p)
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
        print(f"Unfrozen late ViT blocks. Trainable backbone parameters: {sum(p.numel() for p in backbone_params):,}")

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_count:,} ({(trainable_count/1223250748)*100:.3f}% of total)")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    loss_fn = CanonicalMultiTaskLoss(
        weight_depth=weight_depth,
        weight_pts=weight_pts,
        weight_norm=weight_norm,
    )

    # 3. Training Loop
    model.train()
    history = []
    step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
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
                print(f"[WARN] NaN/Inf loss at step {step}, skipping update!")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = float(total_loss.item())
            epoch_loss += loss_val
            history.append({
                "step": step,
                "epoch": epoch,
                "loss": loss_val,
                "loss_depth": float(loss_dict["loss_depth"]),
                "loss_pts": float(loss_dict["loss_pts"]),
                "loss_norm": float(loss_dict["loss_norm"]),
                "dataset": batch["dataset"],
                "scene_id": batch["scene_id"],
            })

            if step % 50 == 0 or step == len(train_loader) * epochs or step == 1:
                print(f"Step {step:03d} | Ep {epoch}/{epochs} | Loss: {loss_val:.4f} (D: {loss_dict['loss_depth']:.4f}, P: {loss_dict['loss_pts']:.4f}, N: {loss_dict['loss_norm']:.4f}) | {batch['dataset']}:{batch['scene_id']}")

    train_duration = time.time() - t0
    print(f"\nTraining completed in {train_duration:.1f}s ({step} steps).")

    # 4. Save Checkpoint
    ckpt_out = CHECKPOINTS_DIR / f"{exp_id}_final.ckpt"
    torch.save({
        "exp_id": exp_id,
        "step": step,
        "epochs": epochs,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_final": loss_val,
        "adaptation_mode": adaptation_mode,
        "dataset_filter": dataset_filter,
    }, ckpt_out)
    ckpt_sha = hashlib.sha256(ckpt_out.read_bytes()).hexdigest()
    print(f"Saved checkpoint: {ckpt_out.name} (SHA-256: {ckpt_sha})")

    # 5. Full Validation on Both Tracks
    print("\nRunning comprehensive validation on held-out scenes...")
    t_val_start = time.time()
    val_res_hypersim = evaluate_dataset(model, val_hypersim, device=device)
    val_res_scannet = evaluate_dataset(model, val_scannet, device=device)
    val_duration = time.time() - t_val_start
    print(f"Validation completed in {val_duration:.1f}s.")

    # 6. Success Gate Analysis
    base_hyp = CANONICAL_BASELINE["track_a_hypersim"]
    base_sca = CANONICAL_BASELINE["track_b_scannet"]

    rel_d_hyp = ((val_res_hypersim["absrel"] - base_hyp["absrel"]) / base_hyp["absrel"]) * 100.0
    rel_d_sca = ((val_res_scannet["absrel"] - base_sca["absrel"]) / base_sca["absrel"]) * 100.0
    rel_p_hyp = ((val_res_hypersim["pts_sim3_mae"] - base_hyp["pts_sim3_mae"]) / base_hyp["pts_sim3_mae"]) * 100.0
    rel_p_sca = ((val_res_scannet["pts_sim3_mae"] - base_sca["pts_sim3_mae"]) / base_sca["pts_sim3_mae"]) * 100.0

    # Passed if at least 2 primary metrics improve by >= 3.0% (negative rel delta <= -3.0%) and no metric regresses > 1.0%
    primary_improvements = sum([
        rel_d_hyp <= -3.0,
        rel_d_sca <= -3.0,
        rel_p_hyp <= -3.0,
        rel_p_sca <= -3.0,
    ])
    regressions = sum([
        rel_d_hyp > 1.0,
        rel_d_sca > 1.0,
        rel_p_hyp > 1.0,
        rel_p_sca > 1.0,
    ])

    passed_success_gate = (primary_improvements >= 2) and (regressions == 0)

    print("\n--- VALIDATION COMPARISON VS PRETRAINED BASELINE ---")
    print(f"Track A (Hypersim): AbsRel {val_res_hypersim['absrel']:.4f} (Base: {base_hyp['absrel']:.4f}, Delta: {rel_d_hyp:+.2f}%)")
    print(f"Track A (Hypersim): Sim(3) {val_res_hypersim['pts_sim3_mae']:.4f}m (Base: {base_hyp['pts_sim3_mae']:.4f}m, Delta: {rel_p_hyp:+.2f}%)")
    print(f"Track B (ScanNet):  AbsRel {val_res_scannet['absrel']:.4f} (Base: {base_sca['absrel']:.4f}, Delta: {rel_d_sca:+.2f}%)")
    print(f"Track B (ScanNet):  Sim(3) {val_res_scannet['pts_sim3_mae']:.4f}m (Base: {base_sca['pts_sim3_mae']:.4f}m, Delta: {rel_p_sca:+.2f}%)")
    print(f"Primary Improvements (>= 3%): {primary_improvements}/4 | Critical Regressions (> 1%): {regressions}/4")
    print(f"SUCCESS GATE PASSED: {passed_success_gate}")

    eval_out = EVALUATIONS_DIR / f"{exp_id}_eval.json"
    result_data = {
        "exp_id": exp_id,
        "checkpoint": str(ckpt_out.name),
        "checkpoint_sha256": ckpt_sha,
        "adaptation_mode": adaptation_mode,
        "dataset_filter": dataset_filter,
        "lr_heads": lr_heads,
        "lr_backbone": lr_backbone,
        "epochs": epochs,
        "train_steps": step,
        "train_duration_s": train_duration,
        "val_duration_s": val_duration,
        "track_a_hypersim": val_res_hypersim,
        "track_b_scannet": val_res_scannet,
        "deltas": {
            "hypersim_absrel_rel_pct": rel_d_hyp,
            "hypersim_pts_sim3_rel_pct": rel_p_hyp,
            "scannet_absrel_rel_pct": rel_d_sca,
            "scannet_pts_sim3_rel_pct": rel_p_sca,
        },
        "passed_success_gate": passed_success_gate,
    }

    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def append_to_registry(res: Dict[str, Any]):
    file_exists = REGISTRY_FILE.exists()
    with open(REGISTRY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "exp_id",
                "checkpoint",
                "adaptation_mode",
                "dataset_filter",
                "lr_heads",
                "epochs",
                "hypersim_absrel",
                "hypersim_absrel_delta_pct",
                "hypersim_pts_sim3",
                "scannet_absrel",
                "scannet_absrel_delta_pct",
                "scannet_pts_sim3",
                "passed_gate",
                "timestamp",
            ])
        writer.writerow([
            res["exp_id"],
            res["checkpoint"],
            res["adaptation_mode"],
            res.get("dataset_filter") or "multi-dataset",
            res["lr_heads"],
            res["epochs"],
            f"{res['track_a_hypersim']['absrel']:.4f}",
            f"{res['deltas']['hypersim_absrel_rel_pct']:+.2f}%",
            f"{res['track_a_hypersim']['pts_sim3_mae']:.4f}",
            f"{res['track_b_scannet']['absrel']:.4f}",
            f"{res['deltas']['scannet_absrel_rel_pct']:+.2f}%",
            f"{res['track_b_scannet']['pts_sim3_mae']:.4f}",
            res["passed_success_gate"],
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])


def main():
    print("=" * 80)
    print("ROOMAI AUTONOMOUS EXPERIMENT CONTROLLER")
    print("=" * 80)

    # Initialize State
    state = {
        "active_experiment": None,
        "completed_experiments": [],
        "best_checkpoint": "production_model.ckpt",
        "best_checkpoint_sha256": "dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5",
        "best_metrics": {
            "hypersim_absrel": CANONICAL_BASELINE["track_a_hypersim"]["absrel"],
            "scannet_absrel": CANONICAL_BASELINE["track_b_scannet"]["absrel"],
        },
        "status": "RUNNING",
    }
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

    experiments_to_run = [
        {
            "exp_id": "exp_01_multidataset_head_only",
            "adaptation_mode": "head_only",
            "dataset_filter": None,  # Multi-dataset: all 204 train frames
            "lr_heads": 5e-5,
            "lr_backbone": 0.0,
            "epochs": 2,
        },
        {
            "exp_id": "exp_02_multidataset_conservative",
            "adaptation_mode": "head_only",
            "dataset_filter": None,
            "lr_heads": 2e-5,
            "lr_backbone": 0.0,
            "epochs": 2,
        },
        {
            "exp_id": "exp_03_multidataset_partial_backbone",
            "adaptation_mode": "partial_backbone",
            "dataset_filter": None,
            "lr_heads": 3e-5,
            "lr_backbone": 5e-7,
            "epochs": 2,
        },
    ]

    for exp_spec in experiments_to_run:
        exp_id = exp_spec["exp_id"]
        if exp_id in state["completed_experiments"]:
            print(f"Skipping already completed experiment: {exp_id}")
            continue

        state["active_experiment"] = exp_id
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        res = execute_training_experiment(**exp_spec)
        append_to_registry(res)

        state["completed_experiments"].append(exp_id)
        if res["passed_success_gate"]:
            print(f"\n[ALERT] EXPERIMENT {exp_id} PASSED SUCCESS GATE! Promoting as best checkpoint.")
            state["best_checkpoint"] = res["checkpoint"]
            state["best_checkpoint_sha256"] = res["checkpoint_sha256"]
            state["best_metrics"]["hypersim_absrel"] = res["track_a_hypersim"]["absrel"]
            state["best_metrics"]["scannet_absrel"] = res["track_b_scannet"]["absrel"]

        state["active_experiment"] = None
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    print("\n" + "=" * 80)
    print("MASTER EXPERIMENT SUITE COMPLETE")
    print(f"Best Certified Checkpoint: {state['best_checkpoint']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
