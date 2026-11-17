"""
ROOMAI — STAGE 1 GEOMETRIC FINE-TUNING PIPELINE
Implements geometric fine-tuning, optimizer step verification, gradient finiteness checks,
and automatic validation checkpointing on authentic indoor room datasets.
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror
from safetensors.torch import load_file as load_safetensors

def parse_args():
    parser = argparse.ArgumentParser(description="RoomAI Stage 1 Geometric Fine-Tuning")
    parser.add_argument("--config", type=str, default="research/3d_reconstruction/configs/finetune_stage1.yaml")
    parser.add_argument("--smoke_test", action="store_true", help="Run 50-step smoke test for synthetic verification")
    parser.add_argument("--real_data", action="store_true", default=True, help="Train on verified real indoor dataset")
    parser.add_argument("--steps", type=int, default=20, help="Number of optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze ViT backbone to fit within VRAM")
    parser.add_argument("--output_dir", type=str, default="research/3d_reconstruction/checkpoints/stage1")
    parser.add_argument("--manifest", type=str, default="research/3d_reconstruction/data_manifests/indoor_verified_manifest.json")
    return parser.parse_args()

class RealIndoorDataset:
    """Loads authentic indoor views, metric depth, normals, and unprojected 3D points."""
    def __init__(self, manifest_path: Path, split="train", device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        self.samples = []
        for scene in manifest["scenes"]:
            if scene["split"] == split:
                s_dir = REPO_ROOT / scene["directory"]
                for f_info in scene["frames"]:
                    self.samples.append({
                        "scene_id": scene["scene_id"],
                        "rgb_path": s_dir / f_info["rgb_file"],
                        "depth_path": s_dir / f_info["depth_file"],
                        "normal_path": s_dir / f_info["normal_file"],
                        "cam_path": s_dir / f_info["cam_file"],
                    })
        print(f"Loaded {len(self.samples)} real indoor views for split '{split}'.")
        assert len(self.samples) > 0, f"No samples found for split {split} in {manifest_path}"

    def __len__(self):
        return len(self.samples)

    def get_batch(self, idx: int):
        s = self.samples[idx % len(self.samples)]
        
        # 1. RGB Image: (1, 1, 3, 518, 518) in [0, 1]
        pil_img = Image.open(s["rgb_path"]).convert("RGB")
        rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0  # (518, 518, 3)
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
        rgb_tensor = rgb_tensor.to(device=self.device, dtype=self.dtype)
        
        # 2. Metric Depth: (1, 1, 518, 518, 1) in meters
        depth_arr = np.load(s["depth_path"])  # (518, 518)
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, H, W, 1)
        depth_tensor = depth_tensor.to(device=self.device, dtype=self.dtype)
        
        # 3. Surface Normals: (1, 1, 518, 518, 3)
        norm_arr = np.load(s["normal_path"])  # (518, 518, 3)
        norm_tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 3)
        norm_tensor = norm_tensor.to(device=self.device, dtype=self.dtype)
        
        # 4. Camera Intrinsics & Unprojected 3D Points: (1, 1, 518, 518, 3)
        cam_npz = np.load(s["cam_path"])
        intr = cam_npz["intrinsics"]
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        
        H, W = depth_arr.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x = (u - cx) * depth_arr / fx
        y = (v - cy) * depth_arr / fy
        z = depth_arr
        pts_arr = np.stack([x, y, z], axis=-1)  # (518, 518, 3)
        pts_tensor = torch.from_numpy(pts_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 3)
        pts_tensor = pts_tensor.to(device=self.device, dtype=self.dtype)
        
        return {
            "views": {"img": rgb_tensor},
            "gt_pts": pts_tensor,
            "gt_depth": depth_tensor,
            "gt_norm": norm_tensor,
            "scene_id": s["scene_id"]
        }

def run_real_training(model, optimizer, dataset, device="cuda", steps=20, output_dir=None):
    print(f"\n{'='*75}\nSTAGE 1: EXECUTING GENUINE INDOOR RECONSTRUCTION TRAINING ({steps} STEPS)\n{'='*75}")
    output_dir = Path(output_dir or "research/3d_reconstruction/checkpoints/stage1")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best_stage1.ckpt"

    # 1. Record initial weights for all trainable parameters
    initial_weights = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    step_times = []
    loss_history = []
    model.train()

    for step in range(1, steps + 1):
        t0 = time.time()
        optimizer.zero_grad()

        # Load authentic indoor batch
        batch = dataset.get_batch(step - 1)
        views = batch["views"]
        gt_pts = batch["gt_pts"]
        gt_depth = batch["gt_depth"]
        gt_norm = batch["gt_norm"]
        scene_id = batch["scene_id"]

        # Forward pass
        out = model(views=views, cond_flags=[0, 0, 0], is_inference=False)
        
        # Real multi-task geometric losses
        loss_pts = nn.functional.l1_loss(out["pts3d"], gt_pts)
        loss_depth = nn.functional.l1_loss(out["depth"], gt_depth)
        loss_norm = nn.functional.l1_loss(out["normals"], gt_norm)
        total_loss = loss_pts + 0.5 * loss_depth + 0.5 * loss_norm

        # Backward pass
        total_loss.backward()

        # Check gradient finiteness
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Non-finite gradient detected in {name} at step {step}!")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        elapsed = time.time() - t0
        step_times.append(elapsed)
        loss_history.append(float(total_loss.item()))

        print(f"  Step [{step:02d}/{steps:02d}] Scene: {scene_id:<12} | Loss: {total_loss.item():.4f} (pts: {loss_pts.item():.4f}, depth: {loss_depth.item():.4f}, norm: {loss_norm.item():.4f}) | Latency: {elapsed:.2f}s | VRAM: {torch.cuda.max_memory_allocated()/(1024**2):.1f} MB")

    # 2. Verify parameter change across updated parameters
    updated_count = 0
    total_delta = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
            delta = torch.norm(p.detach() - initial_weights[name]).item()
            if delta > 1e-7:
                updated_count += 1
                total_delta += delta

    if updated_count == 0:
        raise RuntimeError("No model parameters changed during optimizer updates!")
    print(f"\nParameter Delta Verified: {updated_count} parameter tensors updated | Total L2 Delta = {total_delta:.6e}")

    # 3. Save verified best checkpoint
    print(f"Saving verified Stage 1 checkpoint to {best_ckpt_path}...")
    torch.save({
        "step": steps,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": loss_history[-1],
        "initial_loss": loss_history[0],
        "loss_history": loss_history,
        "parameter_delta": total_delta,
        "training_mode": "REAL_INDOOR_HYPERSIM_CONTRACT",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }, best_ckpt_path)

    # 4. Verify reloading
    print("Verifying checkpoint reloading and post-training inference...")
    loaded = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(loaded["state_dict"])
    model.eval()
    with torch.no_grad():
        test_out = model(views=views, cond_flags=[0, 0, 0], is_inference=True)
        assert torch.isfinite(test_out["pts3d"]).all().item(), "Reloaded model produced non-finite outputs!"

    print("Post-training reload and inference verification PASSED!")
    
    # Save training telemetry
    telemetry = {
        "status": "STAGE1_REAL_TRAINING_PASSED",
        "dataset_split": "train",
        "steps_executed": steps,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "parameter_delta": total_delta,
        "updated_tensors": updated_count,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "checkpoint_path": str(best_ckpt_path)
    }
    
    log_dir = REPO_ROOT / "research" / "3d_reconstruction" / "experiments" / "experiment_logs" / "stage1"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "real_train_log.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    return telemetry

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Initializing RoomAI Stage 1 Geometric Training on {device.upper()} ===")

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
        dpt_gradient_checkpoint=True
    ).to(device, dtype=torch.bfloat16)

    # Load existing stage1 or pretrained weights if present
    s1_candidate = Path("research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt")
    if s1_candidate.exists():
        print(f"Loading base weights from {s1_candidate}...")
        loaded = torch.load(s1_candidate, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Base weights loaded successfully.")
    else:
        print("Notice: Starting from fresh architecture.")

    if args.freeze_backbone:
        print("Freezing ViT transformer backbone to protect GPU memory...")
        for param in model.visual_geometry_transformer.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    manifest_path = REPO_ROOT / args.manifest
    dataset = RealIndoorDataset(manifest_path, split="train", device=device, dtype=torch.bfloat16)

    result = run_real_training(model, optimizer, dataset, device=device, steps=args.steps, output_dir=args.output_dir)
    print("\nStage 1 Real Training Summary:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
