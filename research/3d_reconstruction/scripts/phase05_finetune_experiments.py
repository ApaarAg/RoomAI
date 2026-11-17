"""
ROOMAI — Phase 5: LoRA Fine-Tuning Experiment Controller

Implements and executes controlled fine-tuning experiments using the new 4,079-frame
Hypersim training dataset. This is the primary research task.

Experiments:
  E01 — Head-only fine-tuning (new data scale, 24 scenes)
  E02 — LoRA rank-4 on late attention blocks  
  E03 — LoRA rank-8 on late attention blocks
  E04 — LoRA rank-16 on late attention blocks (if VRAM allows)
  E05 — LoRA + geometric regularization
  E06 — Progressive unfreezing (if E01/E02 shows improvement)

Scientific protocol:
- Each experiment starts from production_model.ckpt
- Best checkpoint saved per experiment
- Validation evaluated after EVERY epoch
- Compare against CANONICAL baseline from Phase 3
- Never promote if validation regresses

Output:
  results/final_execution/experiment_registry.csv
  results/final_execution/controller_state.json
  results/final_execution/checkpoints/<exp_id>/
  reports/final_execution/phase_05_stage1_training.md
"""

import os
import gc
import sys
import csv
import json
import time
import math
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Paths ---
REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research/3d_reconstruction/scripts"
sys.path.insert(0, str(EXTERNAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from hypersim_dataset import HypersimDataset, MANIFEST_PATH, REPO_ROOT
from corrected_losses import CanonicalMultiTaskLoss

PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"
RESULTS_ROOT = REPO_ROOT / "results" / "final_execution"
CKPTS_DIR = RESULTS_ROOT / "checkpoints"
REPORTS_DIR = REPO_ROOT / "reports" / "final_execution"

for d in [RESULTS_ROOT, CKPTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STATE_FILE = RESULTS_ROOT / "controller_state.json"
REGISTRY_FILE = RESULTS_ROOT / "experiment_registry.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# CANONICAL BASELINE — from Phase 3 evaluation on new dataset
# (Will be updated when Phase 3 completes; initially set to Phase 9 values)
# ---------------------------------------------------------------------------
CANONICAL_BASELINE = None  # Loaded from Phase 3 results

def load_canonical_baseline() -> Dict:
    baseline_path = RESULTS_ROOT / "baseline" / "baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
        val = data.get("hypersim_val", {})
        return {
            "absrel": val.get("absrel", 0.5469),
            "delta1": val.get("delta1", 0.2926),
            "rmse": val.get("rmse", 4.6727),
            "norm_mean_ae": val.get("norm_mean_ae", 61.5),
        }
    # Fallback to Phase 9 canonical values
    return {
        "absrel": 0.5469,
        "delta1": 0.2926,
        "rmse": 4.6727,
        "norm_mean_ae": 61.531,
    }


# ---------------------------------------------------------------------------
# LoRA implementation
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for a linear layer.
    
    For a linear layer W (d_out × d_in), adds:
    h = W * x + (B * A) * x * alpha / r
    
    where A is (r × d_in) and B is (d_out × r).
    Only A and B are trainable. W remains frozen.
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original_linear  # kept frozen
        self.rank = rank
        self.alpha = alpha
        
        d_out, d_in = original_linear.weight.shape
        
        # LoRA matrices — initialized with kaiming/zero to start from identity
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        
        # Initialize A with kaiming, B with zeros → starts at zero adaptation
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return original_out + lora_out
    
    def get_trainable_params(self):
        return [self.lora_A, self.lora_B]
    
    def num_trainable_params(self):
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora_to_model(
    model: nn.Module,
    target_blocks: List[int],  # block indices to apply LoRA to
    rank: int = 8,
    alpha: float = 16.0,
    block_type: str = "frame",  # "frame" or "global"
) -> Tuple[nn.Module, List[nn.Parameter], int]:
    """
    Apply LoRA adapters to attention QKV and output projections in selected blocks.
    
    Returns: (model, trainable_params, num_trainable_params)
    """
    vgt = model.visual_geometry_transformer
    
    if block_type == "frame" and hasattr(vgt, "frame_blocks"):
        blocks = vgt.frame_blocks
    elif block_type == "global" and hasattr(vgt, "global_blocks"):
        blocks = vgt.global_blocks
    else:
        # Try to find any relevant block attribute
        for attr in ["frame_blocks", "global_blocks", "blocks", "transformer_blocks"]:
            if hasattr(vgt, attr):
                blocks = getattr(vgt, attr)
                break
        else:
            print(f"[WARN] No blocks found for block_type={block_type}")
            return model, [], 0
    
    trainable_params = []
    total_new_params = 0
    
    for block_idx in target_blocks:
        if block_idx >= len(blocks):
            print(f"[WARN] Block {block_idx} out of range (only {len(blocks)} blocks)")
            continue
        
        block = blocks[block_idx]
        attn = block.attn
        
        # Apply LoRA to QKV (combined)
        if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
            lora_qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
            attn.qkv = lora_qkv
            trainable_params.extend(lora_qkv.get_trainable_params())
            total_new_params += lora_qkv.num_trainable_params()
            print(f"  Applied LoRA rank-{rank} to block[{block_idx}].attn.qkv ({lora_qkv.num_trainable_params():,} params)")
        
        # Apply LoRA to output projection
        if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
            lora_proj = LoRALinear(attn.proj, rank=rank, alpha=alpha)
            attn.proj = lora_proj
            trainable_params.extend(lora_proj.get_trainable_params())
            total_new_params += lora_proj.num_trainable_params()
            print(f"  Applied LoRA rank-{rank} to block[{block_idx}].attn.proj ({lora_proj.num_trainable_params():,} params)")
    
    return model, trainable_params, total_new_params


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_worldmirror(ckpt_path: Path, device: str = "cuda", dtype=torch.bfloat16):
    from src.models.models.worldmirror import WorldMirror
    
    model = WorldMirror(
        img_size=518, patch_size=14, embed_dim=1024, gs_dim=256,
        enable_cam=True, enable_pts=True, enable_depth=True,
        enable_norm=True, enable_gs=False, enable_cond=False,
    ).to(device=device, dtype=dtype)
    
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    del sd, clean_sd
    torch.cuda.empty_cache()
    
    total = sum(p.numel() for p in model.parameters())
    print(f"[OK] WorldMirror loaded: {total:,} params, missing={len(missing)}, unexpected={len(unexpected)}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_on_val(
    model: nn.Module,
    val_dataset: HypersimDataset,
    device: str = "cuda",
    max_frames: int = 100,
) -> Dict[str, float]:
    """Quick validation evaluation for checkpoint selection."""
    model.eval()
    
    absrel_list = []
    delta1_list = []
    rmse_list = []
    
    with torch.no_grad():
        n = min(len(val_dataset), max_frames)
        for idx in range(n):
            sample = val_dataset[idx]
            if sample is None:
                continue
            
            mask = sample["mask"].numpy()
            if mask.sum() < 500:
                continue
            
            rgb = sample["rgb"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            gt_depth = sample["depth_metric"].numpy()
            scale_s = sample["scale_factor"]
            
            try:
                preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=True)
                pred_depth_norm = preds["depth"].squeeze().float().cpu().numpy()
                pred_depth_metric = pred_depth_norm * scale_s
            except Exception:
                continue
            
            valid = mask & (gt_depth > 0.05) & np.isfinite(gt_depth) & np.isfinite(pred_depth_metric) & (pred_depth_metric > 0.0)
            if valid.sum() < 100:
                continue
            
            g_v = gt_depth[valid]
            p_v = pred_depth_metric[valid]
            
            s_star = np.median(g_v) / max(np.median(p_v), 1e-6)
            s_star = max(0.1, min(s_star, 10.0))
            p_s = p_v * s_star
            
            absrel_list.append(float(np.mean(np.abs(p_s - g_v) / g_v)))
            rmse_list.append(float(np.sqrt(np.mean((p_s - g_v) ** 2))))
            ratio = np.maximum(p_s / g_v, g_v / p_s)
            delta1_list.append(float(np.mean(ratio < 1.25)))
    
    model.train()
    
    return {
        "absrel": float(np.mean(absrel_list)) if absrel_list else 999.0,
        "delta1": float(np.mean(delta1_list)) if delta1_list else 0.0,
        "rmse": float(np.mean(rmse_list)) if rmse_list else 999.0,
        "n_valid": len(absrel_list),
    }


# ---------------------------------------------------------------------------
# Checkpoint SHA-256
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Experiment State Management
# ---------------------------------------------------------------------------
def load_state() -> Dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "phase": "phase_05_stage1_training",
        "current_experiment": None,
        "completed_experiments": [],
        "best_checkpoint": str(PRETRAINED_CKPT),
        "best_absrel": None,  # Will be set from baseline
        "baseline_absrel": None,
        "timestamp": None,
    }


def save_state(state: Dict):
    state["timestamp"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_registry(row: Dict):
    is_new = not REGISTRY_FILE.exists()
    with open(REGISTRY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "exp_id", "timestamp", "adaptation_mode", "lora_rank",
            "lr_heads", "lr_lora", "lr_backbone", "epochs", "train_frames",
            "steps_completed", "best_val_absrel", "best_val_delta1",
            "baseline_absrel", "relative_improvement_pct",
            "outcome", "checkpoint_path", "checkpoint_sha256", "notes"
        ])
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Single Experiment
# ---------------------------------------------------------------------------
def run_experiment(
    exp_id: str,
    adaptation_mode: str,  # "head_only", "lora", "lora_regularized", "progressive"
    lora_rank: int = 0,
    lora_blocks_frame: List[int] = None,
    lora_blocks_global: List[int] = None,
    lr_heads: float = 2e-6,
    lr_lora: float = 5e-5,
    lr_backbone: float = 0.0,
    epochs: int = 3,
    weight_depth: float = 1.0,
    weight_pts: float = 0.5,
    weight_norm: float = 0.1,
    seed: int = 42,
    skip_frames: int = 5,
    baseline: Dict = None,
    grad_clip: float = 0.5,
) -> Dict[str, Any]:
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_id}")
    print(f"Mode: {adaptation_mode} | LoRA rank: {lora_rank} | Epochs: {epochs}")
    print(f"LR heads: {lr_heads} | LR LoRA: {lr_lora} | LR backbone: {lr_backbone}")
    print(f"Skip frames: {skip_frames} (reduces temporal correlation)")
    print("="*80)
    
    # Create checkpoint dir
    ckpt_dir = CKPTS_DIR / exp_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = {
        "exp_id": exp_id,
        "adaptation_mode": adaptation_mode,
        "lora_rank": lora_rank,
        "lora_blocks_frame": lora_blocks_frame,
        "lora_blocks_global": lora_blocks_global,
        "lr_heads": lr_heads,
        "lr_lora": lr_lora,
        "lr_backbone": lr_backbone,
        "epochs": epochs,
        "weight_depth": weight_depth,
        "weight_pts": weight_pts,
        "weight_norm": weight_norm,
        "seed": seed,
        "skip_frames": skip_frames,
        "started_at": datetime.now().isoformat(),
    }
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Load data
    print("\n[Data] Loading datasets...")
    train_ds = HypersimDataset(MANIFEST_PATH, split="train", skip_frames=skip_frames)
    val_ds = HypersimDataset(MANIFEST_PATH, split="val", skip_frames=2)
    
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, generator=g,
        num_workers=0,  # Windows: num_workers=0 to avoid multiprocessing issues
        collate_fn=lambda x: x[0],
    )
    
    n_train = len(train_ds)
    print(f"  Train: {n_train} frames | Val: {len(val_ds)} frames")
    
    # 2. Load model (always from pretrained checkpoint)
    print("\n[Model] Loading pretrained WorldMirror...")
    model = load_worldmirror(PRETRAINED_CKPT, device=DEVICE, dtype=torch.bfloat16)
    
    # Freeze all params first
    for p in model.parameters():
        p.requires_grad = False
    
    param_groups = []
    total_trainable = 0
    
    # 3. Setup trainable parameters based on adaptation mode
    if adaptation_mode in ["head_only", "lora", "lora_regularized", "progressive"]:
        # Always unfreeze prediction heads
        head_params = []
        for head_name in ["depth_head", "pts_head", "norm_head", "cam_head"]:
            head = getattr(model, head_name, None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True
                    head_params.append(p)
        
        head_param_count = sum(p.numel() for p in head_params)
        print(f"  Unfrozen prediction heads: {head_param_count:,} params")
        total_trainable += head_param_count
        param_groups.append({"params": head_params, "lr": lr_heads, "name": "heads"})
    
    if adaptation_mode in ["lora", "lora_regularized"] and lora_rank > 0:
        # Apply LoRA to frame blocks
        frame_lora_params = []
        global_lora_params = []
        
        if lora_blocks_frame:
            model, fp, fpc = apply_lora_to_model(
                model, lora_blocks_frame, rank=lora_rank, alpha=float(lora_rank),
                block_type="frame"
            )
            frame_lora_params = fp
            total_trainable += fpc
            if frame_lora_params:
                param_groups.append({"params": frame_lora_params, "lr": lr_lora, "name": "lora_frame"})
        
        if lora_blocks_global:
            model, gp, gpc = apply_lora_to_model(
                model, lora_blocks_global, rank=lora_rank, alpha=float(lora_rank),
                block_type="global"
            )
            global_lora_params = gp
            total_trainable += gpc
            if global_lora_params:
                param_groups.append({"params": global_lora_params, "lr": lr_lora, "name": "lora_global"})
    
    if adaptation_mode == "progressive" and lr_backbone > 0:
        # Unfreeze last 2 frame blocks
        vgt = model.visual_geometry_transformer
        backbone_params = []
        if hasattr(vgt, "frame_blocks"):
            for block in vgt.frame_blocks[-2:]:
                for p in block.parameters():
                    p.requires_grad = True
                    backbone_params.append(p)
        if hasattr(vgt, "global_blocks"):
            for block in vgt.global_blocks[-1:]:
                for p in block.parameters():
                    p.requires_grad = True
                    backbone_params.append(p)
        
        backbone_count = sum(p.numel() for p in backbone_params)
        total_trainable += backbone_count
        print(f"  Unfrozen backbone (last 2 frame + 1 global blocks): {backbone_count:,} params")
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
    
    print(f"\n  Total trainable parameters: {total_trainable:,} ({100*total_trainable/1223000000:.3f}% of total)")
    
    # 4. Optimizer and loss
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4, betas=(0.9, 0.95))
    
    # Cosine LR scheduler
    total_steps = epochs * n_train
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-8
    )
    
    loss_fn = CanonicalMultiTaskLoss(
        weight_depth=weight_depth,
        weight_pts=weight_pts,
        weight_norm=weight_norm,
    )
    
    # 5. Training loop
    model.train()
    
    training_history = []
    best_val_absrel = float("inf")
    best_ckpt_path = None
    step = 0
    t_start = time.time()
    
    # Evaluate before training (baseline for this experiment)
    print("\n[Eval] Pre-training validation...")
    pre_val = evaluate_on_val(model, val_ds, device=DEVICE, max_frames=50)
    print(f"  Pre-train val AbsRel={pre_val['absrel']:.4f}, Delta1={pre_val['delta1']:.4f}")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        
        print(f"\n[Epoch {epoch}/{epochs}] Training...")
        
        for b_idx, batch in enumerate(train_loader):
            if batch is None or not batch.get("mask", torch.zeros(1)).any():
                continue
            
            step += 1
            
            rgb = batch["rgb"].unsqueeze(0).unsqueeze(0).to(device=DEVICE, dtype=torch.bfloat16)
            gt_d = batch["depth_norm"].unsqueeze(0).unsqueeze(0).to(device=DEVICE, dtype=torch.bfloat16)
            gt_pts = batch["pts_norm"].unsqueeze(0).unsqueeze(0).to(device=DEVICE, dtype=torch.bfloat16)
            gt_norm = batch["normals"].unsqueeze(0).unsqueeze(0).to(device=DEVICE, dtype=torch.bfloat16)
            mask = batch["mask"].unsqueeze(0).unsqueeze(0).to(device=DEVICE)
            
            targets = {"depth_norm": gt_d, "pts_norm": gt_pts, "normals": gt_norm, "mask": mask}
            
            optimizer.zero_grad()
            
            try:
                preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=False)
                total_loss, loss_dict = loss_fn(preds, targets)
            except Exception as e:
                print(f"  [WARN] Forward pass failed at step {step}: {e}")
                continue
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  [WARN] NaN/Inf loss at step {step}, skipping")
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip
            )
            optimizer.step()
            scheduler.step()
            
            loss_val = float(total_loss.item())
            epoch_loss += loss_val
            epoch_steps += 1
            
            if step % 50 == 0 or step <= 5:
                vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                elapsed = time.time() - t_start
                print(f"  Step [{step:4d}] Scene={batch['scene_id']} "
                      f"Loss={loss_val:.4f} (d={loss_dict['loss_depth']:.4f}, "
                      f"p={loss_dict['loss_pts']:.4f}, n={loss_dict['loss_norm']:.4f}) "
                      f"VRAM={vram:.0f}MB t={elapsed:.0f}s")
            
            training_history.append({
                "step": step,
                "epoch": epoch,
                "loss": loss_val,
                "loss_depth": float(loss_dict["loss_depth"]),
                "loss_pts": float(loss_dict["loss_pts"]),
                "loss_norm": float(loss_dict["loss_norm"]),
                "scene_id": batch["scene_id"],
                "lr": optimizer.param_groups[0]["lr"],
            })
        
        # End of epoch — evaluate on validation
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f}, steps={epoch_steps}")
        
        print(f"[Eval] Epoch {epoch} validation...")
        val_metrics = evaluate_on_val(model, val_ds, device=DEVICE, max_frames=100)
        print(f"  Val AbsRel={val_metrics['absrel']:.4f}, Delta1={val_metrics['delta1']:.4f}, "
              f"RMSE={val_metrics['rmse']:.4f}, n={val_metrics['n_valid']}")
        
        # Compare to baseline
        if baseline:
            baseline_absrel = baseline.get("absrel", 0.5469)
            rel_change = (val_metrics["absrel"] - baseline_absrel) / baseline_absrel * 100
            print(f"  vs Baseline: {rel_change:+.2f}% ({'IMPROVED' if rel_change < 0 else 'REGRESSED'})")
        
        # Save checkpoint if best validation
        if val_metrics["absrel"] < best_val_absrel:
            best_val_absrel = val_metrics["absrel"]
            best_ckpt_path = ckpt_dir / f"best_{exp_id}.ckpt"
            torch.save(model.state_dict(), str(best_ckpt_path))
            print(f"  [CHECKPOINT] New best: {best_val_absrel:.4f} → saved to {best_ckpt_path}")
        
        # Always save epoch checkpoint
        epoch_ckpt = ckpt_dir / f"epoch_{epoch:02d}.ckpt"
        torch.save(model.state_dict(), str(epoch_ckpt))
        
        # Update history with val metrics
        training_history[-1]["val_absrel"] = val_metrics["absrel"]
        training_history[-1]["val_delta1"] = val_metrics["delta1"]
        training_history[-1]["val_rmse"] = val_metrics["rmse"]
    
    # Save training history
    hist_path = ckpt_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(training_history, f, indent=2)
    
    # Compute result
    baseline_absrel = baseline.get("absrel", 0.5469) if baseline else 0.5469
    rel_improvement = (baseline_absrel - best_val_absrel) / baseline_absrel * 100
    
    outcome = "IMPROVED" if best_val_absrel < baseline_absrel * 0.97 else (
        "MARGINAL" if best_val_absrel <= baseline_absrel else "REGRESSED"
    )
    
    # SHA-256 of best checkpoint
    ckpt_sha = sha256_file(best_ckpt_path) if best_ckpt_path and best_ckpt_path.exists() else "N/A"
    
    result = {
        "exp_id": exp_id,
        "adaptation_mode": adaptation_mode,
        "lora_rank": lora_rank,
        "total_trainable_params": total_trainable,
        "best_val_absrel": best_val_absrel,
        "pre_val_absrel": pre_val["absrel"],
        "baseline_absrel": baseline_absrel,
        "relative_improvement_pct": rel_improvement,
        "outcome": outcome,
        "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path else None,
        "checkpoint_sha256": ckpt_sha,
        "steps_completed": step,
        "train_frames_total": n_train,
        "training_history": training_history,
    }
    
    # Save to registry
    append_registry({
        "exp_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "adaptation_mode": adaptation_mode,
        "lora_rank": lora_rank,
        "lr_heads": lr_heads,
        "lr_lora": lr_lora,
        "lr_backbone": lr_backbone,
        "epochs": epochs,
        "train_frames": n_train,
        "steps_completed": step,
        "best_val_absrel": f"{best_val_absrel:.4f}",
        "best_val_delta1": f"{evaluate_on_val(model, val_ds, 'cuda', 10)['delta1']:.4f}",
        "baseline_absrel": f"{baseline_absrel:.4f}",
        "relative_improvement_pct": f"{rel_improvement:.2f}",
        "outcome": outcome,
        "checkpoint_path": str(best_ckpt_path) if best_ckpt_path else "",
        "checkpoint_sha256": ckpt_sha,
        "notes": f"LoRA rank={lora_rank} skip={skip_frames}",
    })
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_id} COMPLETE")
    print(f"Best Val AbsRel: {best_val_absrel:.4f} (baseline: {baseline_absrel:.4f})")
    print(f"Relative change: {rel_improvement:+.2f}%")
    print(f"Outcome: {outcome}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"{'='*70}\n")
    
    return result


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------
def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')} IST] Phase 5: Stage 1 Fine-Tuning Controller")
    print(f"Dataset: {MANIFEST_PATH}")
    print(f"Production checkpoint: {PRETRAINED_CKPT}")
    print()
    
    # Load canonical baseline
    baseline = load_canonical_baseline()
    print(f"Canonical baseline: AbsRel={baseline['absrel']:.4f}")
    
    # Load controller state (supports resume)
    state = load_state()
    state["baseline_absrel"] = baseline["absrel"]
    state["best_absrel"] = baseline["absrel"]
    
    completed = set(state.get("completed_experiments", []))
    
    # -----------------------------------------------------------------------
    # Inspect model structure to identify block counts
    # -----------------------------------------------------------------------
    print("\n[Model Inspection] Checking VGT block structure...")
    model = load_worldmirror(PRETRAINED_CKPT, device=DEVICE)
    vgt = model.visual_geometry_transformer
    
    n_frame_blocks = len(vgt.frame_blocks) if hasattr(vgt, "frame_blocks") else 0
    n_global_blocks = len(vgt.global_blocks) if hasattr(vgt, "global_blocks") else 0
    print(f"  frame_blocks: {n_frame_blocks}")
    print(f"  global_blocks: {n_global_blocks}")
    
    # Target last 4 frame blocks and last 2 global blocks for LoRA
    lora_frame_targets = list(range(max(0, n_frame_blocks - 4), n_frame_blocks))
    lora_global_targets = list(range(max(0, n_global_blocks - 2), n_global_blocks))
    print(f"  LoRA frame block targets: {lora_frame_targets}")
    print(f"  LoRA global block targets: {lora_global_targets}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # -----------------------------------------------------------------------
    # Experiment schedule
    # -----------------------------------------------------------------------
    experiments = [
        # E01: Head-only, new data scale (same as Phase 9 but 131× more data)
        {
            "exp_id": "E01_head_only_v2",
            "adaptation_mode": "head_only",
            "lora_rank": 0,
            "lr_heads": 2e-6,
            "lr_lora": 0.0,
            "lr_backbone": 0.0,
            "epochs": 3,
            "skip_frames": 5,  # ~820 frames per epoch
        },
        # E02: LoRA rank 4 — small rank, test if backbone adaptation helps
        {
            "exp_id": "E02_lora_rank4",
            "adaptation_mode": "lora",
            "lora_rank": 4,
            "lora_blocks_frame": lora_frame_targets,
            "lora_blocks_global": lora_global_targets,
            "lr_heads": 2e-6,
            "lr_lora": 1e-4,
            "lr_backbone": 0.0,
            "epochs": 3,
            "skip_frames": 5,
        },
        # E03: LoRA rank 8
        {
            "exp_id": "E03_lora_rank8",
            "adaptation_mode": "lora",
            "lora_rank": 8,
            "lora_blocks_frame": lora_frame_targets,
            "lora_blocks_global": lora_global_targets,
            "lr_heads": 2e-6,
            "lr_lora": 5e-5,
            "lr_backbone": 0.0,
            "epochs": 3,
            "skip_frames": 5,
        },
        # E04: LoRA rank 16
        {
            "exp_id": "E04_lora_rank16",
            "adaptation_mode": "lora",
            "lora_rank": 16,
            "lora_blocks_frame": lora_frame_targets,
            "lora_blocks_global": lora_global_targets,
            "lr_heads": 2e-6,
            "lr_lora": 2e-5,
            "lr_backbone": 0.0,
            "epochs": 3,
            "skip_frames": 5,
        },
        # E05: Progressive unfreezing (if any above shows improvement)
        {
            "exp_id": "E05_progressive_unfreeze",
            "adaptation_mode": "progressive",
            "lora_rank": 0,
            "lr_heads": 1e-6,
            "lr_lora": 0.0,
            "lr_backbone": 5e-8,
            "epochs": 2,
            "skip_frames": 5,
        },
    ]
    
    all_results = []
    
    for exp_config in experiments:
        exp_id = exp_config["exp_id"]
        
        if exp_id in completed:
            print(f"\n[SKIP] {exp_id} already completed")
            continue
        
        state["current_experiment"] = exp_id
        save_state(state)
        
        try:
            result = run_experiment(
                baseline=baseline,
                lora_blocks_frame=exp_config.get("lora_blocks_frame"),
                lora_blocks_global=exp_config.get("lora_blocks_global"),
                **{k: v for k, v in exp_config.items() 
                   if k not in ["lora_blocks_frame", "lora_blocks_global"]},
            )
            
            all_results.append(result)
            completed.add(exp_id)
            state["completed_experiments"] = list(completed)
            
            # Update best if improved
            if result["best_val_absrel"] < state["best_absrel"]:
                state["best_absrel"] = result["best_val_absrel"]
                state["best_checkpoint"] = result["best_ckpt_path"]
                print(f"\n[PROMOTION] New best checkpoint: {exp_id} AbsRel={result['best_val_absrel']:.4f}")
            
            save_state(state)
            
            # Clear GPU memory between experiments
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_id} failed: {e}")
            traceback.print_exc()
            state["current_experiment"] = None
            save_state(state)
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("PHASE 5 — EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Experiment':<25} {'Val AbsRel':>12} {'vs Baseline':>12} {'Outcome':<15}")
    print("-"*80)
    print(f"{'PRETRAINED BASELINE':<25} {baseline['absrel']:>12.4f} {'—':>12} {'REFERENCE':<15}")
    for r in all_results:
        rel = r["relative_improvement_pct"]
        sign = "+" if rel >= 0 else ""
        print(f"{r['exp_id']:<25} {r['best_val_absrel']:>12.4f} {sign}{rel:>11.2f}% {r['outcome']:<15}")
    print("="*80)
    
    best = min(all_results, key=lambda x: x["best_val_absrel"]) if all_results else None
    if best and best["best_val_absrel"] < baseline["absrel"] * 0.97:
        print(f"\n[SUCCESS] Best fine-tuned model: {best['exp_id']}")
        print(f"  AbsRel: {best['best_val_absrel']:.4f} (improvement: {best['relative_improvement_pct']:+.2f}%)")
        print(f"  Checkpoint: {best['best_ckpt_path']}")
    else:
        print(f"\n[RESULT] No fine-tuned model exceeded 3% improvement threshold.")
        print(f"  Best AbsRel achieved: {min(r['best_val_absrel'] for r in all_results):.4f}")
        print(f"  Pretrained checkpoint remains the production winner.")
    
    # Write final training report
    _write_phase5_report(all_results, baseline, state)
    
    print(f"\n[DONE] Phase 5 complete at {datetime.now().strftime('%H:%M:%S IST')}")


def _write_phase5_report(all_results, baseline, state):
    lines = [
        "# RoomAI — Phase 5: Stage 1 Training and Fine-Tuning",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  ",
        f"**Status:** {'COMPLETE' if all_results else 'IN PROGRESS'}  ",
        "",
        "## Experiment Results",
        "",
        "| Experiment | Adaptation | LoRA Rank | Val AbsRel | vs Baseline | Outcome |",
        "|---|---|---|---|---|---|",
        f"| PRETRAINED | — | — | {baseline['absrel']:.4f} | REFERENCE | PRODUCTION |",
    ]
    
    for r in all_results:
        rel = r["relative_improvement_pct"]
        sign = "+" if rel >= 0 else ""
        lines.append(
            f"| {r['exp_id']} | {r['adaptation_mode']} | {r['lora_rank']} | "
            f"{r['best_val_absrel']:.4f} | {sign}{rel:.2f}% | {r['outcome']} |"
        )
    
    if state.get("best_checkpoint") != str(PRETRAINED_CKPT):
        lines += [
            "",
            f"> [!IMPORTANT]",
            f"> Best checkpoint promoted: `{state['best_checkpoint']}`",
            f"> Val AbsRel: {state['best_absrel']:.4f}",
        ]
    else:
        lines += [
            "",
            "> [!NOTE]",
            "> Pretrained checkpoint retained as production winner.",
        ]
    
    report_path = REPORTS_DIR / "phase_05_stage1_training.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Phase 5 report: {report_path}")


if __name__ == "__main__":
    main()

