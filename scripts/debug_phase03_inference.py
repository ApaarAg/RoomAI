"""
Debug: Investigate why model inference fails on new HypersimDataset samples.
"""
import os, sys
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research/3d_reconstruction/scripts"
sys.path.insert(0, str(EXTERNAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from hypersim_dataset import HypersimDataset, MANIFEST_PATH, REPO_ROOT

PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"
DEVICE = "cuda"

print("Step 1: Load dataset sample...")
val_ds = HypersimDataset(MANIFEST_PATH, split="val", skip_frames=1)
print(f"Val dataset: {len(val_ds)} frames")

sample = val_ds[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"RGB shape: {sample['rgb'].shape}, range: [{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
print(f"Depth shape: {sample['depth_norm'].shape}")
print(f"Mask: {sample['mask'].sum()}/{sample['mask'].numel()} valid pixels")
print(f"scale_factor: {sample['scale_factor']}")

print("\nStep 2: Load WorldMirror model...")
from src.models.models.worldmirror import WorldMirror

model = WorldMirror(
    img_size=518, patch_size=14, embed_dim=1024, gs_dim=256,
    enable_cam=True, enable_pts=True, enable_depth=True,
    enable_norm=True, enable_gs=False, enable_cond=False,
).to(device=DEVICE, dtype=torch.bfloat16)

sd = torch.load(str(PRETRAINED_CKPT), map_location="cpu")
if "state_dict" in sd:
    sd = sd["state_dict"]
clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(clean_sd, strict=False)
print(f"Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
if missing[:5]:
    print(f"First missing: {missing[:5]}")
del sd, clean_sd
torch.cuda.empty_cache()

print("\nStep 3: Test inference...")
model.eval()
rgb = sample["rgb"].unsqueeze(0).unsqueeze(0).to(device=DEVICE, dtype=torch.bfloat16)
print(f"Input shape: {rgb.shape}")

try:
    with torch.no_grad():
        preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=True)
    print(f"Preds keys: {list(preds.keys())}")
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            print(f"  preds[{k}]: shape={v.shape}, dtype={v.dtype}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\nStep 4: Check mask and depth validity...")
gt_depth = sample["depth_metric"].numpy()
mask = sample["mask"].numpy()
scale_s = sample["scale_factor"]

print(f"GT depth (metric): range=[{gt_depth[mask].min():.3f}, {gt_depth[mask].max():.3f}]m")
print(f"Scale factor: {scale_s:.3f}")
print(f"Valid pixels: {mask.sum()}/{mask.numel()} = {100*mask.sum()/mask.numel():.1f}%")

print("\nStep 5: Try computing metrics manually...")
try:
    pred_depth_norm = preds["depth"].squeeze().float().cpu().numpy()
    pred_depth_metric = pred_depth_norm * scale_s
    print(f"Predicted depth (metric): range=[{pred_depth_metric.min():.4f}, {pred_depth_metric.max():.4f}]")
    print(f"Predicted depth NaN: {np.isnan(pred_depth_metric).sum()}, Inf: {np.isinf(pred_depth_metric).sum()}")
    
    valid = mask & (gt_depth > 0.05) & np.isfinite(gt_depth) & np.isfinite(pred_depth_metric) & (pred_depth_metric > 0.0)
    print(f"Joint valid pixels: {valid.sum()}")
    
    if valid.sum() > 100:
        g_v = gt_depth[valid]
        p_v = pred_depth_metric[valid]
        s_star = np.median(g_v) / max(np.median(p_v), 1e-6)
        p_s = p_v * s_star
        absrel = float(np.mean(np.abs(p_s - g_v) / g_v))
        print(f"Scale s*: {s_star:.4f}")
        print(f"AbsRel: {absrel:.4f}")
    else:
        print("Not enough valid pixels for metrics!")
        
except Exception as e:
    import traceback
    print(f"ERROR in metrics: {e}")
    traceback.print_exc()

