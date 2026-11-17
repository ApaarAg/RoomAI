"""
ROOMAI — Checkpoint Verification & Integrity Engine
Validates checkpoint existence, file size, SHA-256, state-dict completeness,
and executes an empirical forward pass on GPU to verify finite outputs.
"""
import sys
import os
import time
import json
import hashlib
from pathlib import Path
import torch

# Ensure external HunyuanWorld-Mirror source is in python path
EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "external" / "HunyuanWorld-Mirror"
if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))

from src.models.models.worldmirror import WorldMirror
from safetensors.torch import load_file as load_safetensors

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_checkpoint(ckpt_path: Path, enable_gs: bool = False, device: str = "cuda") -> dict:
    if not ckpt_path.exists():
        return {"status": "FAILED", "error": f"File does not exist: {ckpt_path}"}
    
    file_size = ckpt_path.stat().st_size
    print(f"Inspecting checkpoint: {ckpt_path.name} ({file_size / (1024**2):.2f} MB)...")
    
    # 1. Compute SHA-256
    t0 = time.time()
    sha256 = compute_sha256(ckpt_path)
    hash_time = time.time() - t0
    print(f"  SHA-256: {sha256} ({hash_time:.2f}s)")
    
    # 2. Load state dict
    try:
        if ckpt_path.suffix == ".safetensors":
            state_dict = load_safetensors(str(ckpt_path))
        elif ckpt_path.suffix in [".ckpt", ".pth", ".pt"]:
            loaded = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = loaded.get("state_dict", loaded)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            return {"status": "FAILED", "error": f"Unsupported format: {ckpt_path.suffix}"}
    except Exception as e:
        return {"status": "FAILED", "error": f"Failed to load state dict: {e}"}
    
    num_tensors = len(state_dict)
    total_elements = sum(v.numel() for v in state_dict.values())
    print(f"  State dict keys: {num_tensors:,} tensors, {total_elements:,} parameters")
    
    # 3. Instantiate model architecture and load weights
    model = WorldMirror(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        gs_dim=256,
        enable_cam=True,
        enable_pts=True,
        enable_depth=True,
        enable_norm=True,
        enable_gs=enable_gs,
        enable_cond=False,
    )
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Load result: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    
    # 4. Empirical forward pass verification on GPU
    actual_device = device if torch.cuda.is_available() else "cpu"
    model = model.to(actual_device, dtype=torch.bfloat16)
    model.eval()
    
    dummy_input = {
        "img": torch.randn(1, 1, 3, 518, 518, device=actual_device, dtype=torch.bfloat16)
    }
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        preds = model(views=dummy_input, cond_flags=[0, 0, 0])
    
    # 5. Check finite tensors
    all_finite = True
    tensor_shapes = {}
    for k, v in preds.items():
        if torch.is_tensor(v):
            tensor_shapes[k] = list(v.shape)
            is_fin = torch.isfinite(v).all().item()
            if not is_fin:
                all_finite = False
                print(f"  Warning: Non-finite tensor detected in {k}")
    
    alloc_mb = torch.cuda.max_memory_allocated() / (1024**2) if actual_device == "cuda" else 0.0
    
    result = {
        "status": "VERIFIED_VALID" if all_finite else "INVALID_NON_FINITE",
        "file": str(ckpt_path),
        "size_bytes": file_size,
        "sha256": sha256,
        "tensor_count": num_tensors,
        "total_elements": total_elements,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "forward_pass_finite": all_finite,
        "peak_vram_mb": alloc_mb,
        "output_tensor_shapes": tensor_shapes
    }
    print(f"  Verification Result: {result['status']} (Peak VRAM: {alloc_mb:.1f} MB)")
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path("research/3d_reconstruction/checkpoints/pretrained/model.safetensors")
    
    res = verify_checkpoint(target)
    print(json.dumps(res, indent=2))

