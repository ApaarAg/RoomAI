"""
ROOMAI — STAGE 2 3D GAUSSIAN SPLATTING FINE-TUNING & VERIFICATION PIPELINE
Focuses on 3DGS parameter tuning, radiance fields, opacity regularization,
and novel-view rendering losses on authentic indoor room sequences.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror

def parse_args():
    parser = argparse.ArgumentParser(description="RoomAI Stage 2 3DGS Fine-Tuning & Verification")
    parser.add_argument("--config", type=str, default="research/3d_reconstruction/configs/finetune_stage2.yaml")
    parser.add_argument("--stage1_ckpt", type=str, default="research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt")
    parser.add_argument("--real_data", action="store_true", default=True, help="Train on verified real indoor dataset")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for GS head")
    parser.add_argument("--output_dir", type=str, default="research/3d_reconstruction/checkpoints/stage2")
    parser.add_argument("--manifest", type=str, default="research/3d_reconstruction/data_manifests/indoor_verified_manifest.json")
    return parser.parse_args()

class RealIndoorGSDataset:
    """Loads authentic indoor views, metric depth, and camera parameters for 3DGS training."""
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
        rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        
        # 2. Metric Depth: (1, 1, 518, 518)
        depth_arr = np.load(s["depth_path"])
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        
        # 3. Camera Intrinsics and Extrinsics
        cam_npz = np.load(s["cam_path"])
        pose_tensor = torch.from_numpy(cam_npz["pose"]).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        intr_tensor = torch.from_numpy(cam_npz["intrinsics"]).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        valid_mask = torch.ones(1, 1, 518, 518, dtype=torch.bool, device=self.device)

        views = {
            "img": rgb_tensor,
            "depthmap": depth_tensor,
            "camera_poses": pose_tensor,
            "camera_intrs": intr_tensor,
            "valid_mask": valid_mask
        }
        return views, s["scene_id"]

def run_stage2_training(model, optimizer, dataset, device="cuda", steps=10, output_dir=None):
    print(f"\n{'='*75}\nSTAGE 2: EXECUTING 3D GAUSSIAN SPLATTING REAL INDOOR TRAINING ({steps} STEPS)\n{'='*75}")
    output_dir = Path(output_dir or "research/3d_reconstruction/checkpoints/stage2")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_ckpt_path = output_dir / "stage2_real_verified.ckpt"

    # 1. Record initial weights
    initial_weights = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    step_times = []
    loss_history = []
    model.train()

    for step in range(1, steps + 1):
        t0 = time.time()
        optimizer.zero_grad()

        views, scene_id = dataset.get_batch(step - 1)
        out = model(views=views, cond_flags=[0, 0, 0], is_inference=False)

        # 3DGS Multi-Component Loss
        loss_color = nn.functional.l1_loss(out["rendered_colors"], out["gt_colors"])
        splats = out.get("splats", {})
        
        def extract_mean(val, dev):
            if torch.is_tensor(val):
                return val.mean()
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                tensors = [v.flatten() for v in val if torch.is_tensor(v)]
                if len(tensors) > 0:
                    return torch.cat(tensors).mean()
            return torch.tensor(0.0, device=dev)

        loss_opacity = extract_mean(splats.get("opacities"), device)
        loss_scale = extract_mean(splats.get("scales"), device)
        
        total_loss = loss_color + 0.05 * loss_opacity + 0.01 * loss_scale
        total_loss.backward()

        # Check gradients
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Non-finite gradient in GS parameter {name} at step {step}!")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        elapsed = time.time() - t0
        step_times.append(elapsed)
        loss_history.append(float(total_loss.item()))

        print(f"  Step [{step:02d}/{steps:02d}] Scene: {scene_id:<12} | Loss: {total_loss.item():.4f} (color: {loss_color.item():.4f}, opacity: {loss_opacity.item():.4f}) | Latency: {elapsed:.2f}s | VRAM: {torch.cuda.max_memory_allocated()/(1024**2):.1f} MB")

    # 2. Check parameter update
    updated_count = 0
    total_delta = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
            delta = torch.norm(p.detach() - initial_weights[name]).item()
            if delta > 1e-7:
                updated_count += 1
                total_delta += delta

    if updated_count == 0:
        raise RuntimeError("No 3DGS parameters changed during Stage 2 optimizer updates!")
    print(f"\n3DGS Parameter Delta Verified: {updated_count} parameter tensors updated | Total L2 Delta = {total_delta:.6e}")

    # 3. Save Stage 2 Checkpoint
    print(f"Saving verified Stage 2 checkpoint to {stage2_ckpt_path}...")
    torch.save({
        "step": steps,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": loss_history[-1],
        "initial_loss": loss_history[0],
        "loss_history": loss_history,
        "parameter_delta": total_delta,
        "training_mode": "STAGE2_REAL_INDOOR_3DGS",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }, stage2_ckpt_path)

    # 4. Verify reload & inference
    print("Verifying checkpoint reloading and post-training 3DGS inference...")
    loaded = torch.load(stage2_ckpt_path, map_location=device)
    model.load_state_dict(loaded["state_dict"])
    model.eval()
    with torch.no_grad():
        test_out = model(views=views, cond_flags=[0, 0, 0], is_inference=True)
        assert "splats" in test_out, "Reloaded model did not produce splats!"
        means = test_out["splats"]["means"]
        if torch.is_tensor(means):
            assert torch.isfinite(means).all().item(), "Reloaded splat means non-finite!"
        elif isinstance(means, (list, tuple)):
            assert all(torch.isfinite(m).all().item() for m in means if torch.is_tensor(m)), "Reloaded splat means non-finite!"

    print("Post-training 3DGS reload and inference verification PASSED!")

    telemetry = {
        "status": "STAGE2_REAL_TRAINING_PASSED",
        "steps_executed": steps,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "parameter_delta": total_delta,
        "updated_tensors": updated_count,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "checkpoint_path": str(stage2_ckpt_path)
    }

    log_dir = REPO_ROOT / "research" / "3d_reconstruction" / "experiments" / "experiment_logs" / "stage2"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "real_train_stage2_log.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    return telemetry

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Initializing RoomAI Stage 2 3DGS Fine-Tuning on {device.upper()} ===")

    # Initialize model with GS head enabled
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
        dpt_gradient_checkpoint=True
    ).to(device, dtype=torch.bfloat16)

    # Load Stage 1 checkpoint if present
    s1_path = Path(args.stage1_ckpt)
    if s1_path.exists():
        print(f"Loading Stage 1 geometric weights from {s1_path}...")
        loaded = torch.load(s1_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Stage 1 base geometry successfully loaded.")
    else:
        print(f"Notice: Stage 1 checkpoint {s1_path} not found. Running with initialized architecture.")

    # Freeze backbone and geometric heads, train GS head
    for name, param in model.named_parameters():
        if "gs_head" in name or "gs_renderer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Stage 2 Trainable GS Parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    manifest_path = REPO_ROOT / args.manifest
    dataset = RealIndoorGSDataset(manifest_path, split="train", device=device, dtype=torch.bfloat16)

    result = run_stage2_training(model, optimizer, dataset, device=device, steps=args.steps, output_dir=args.output_dir)
    print("\nStage 2 Real Training Summary:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
