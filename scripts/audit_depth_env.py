"""Audit script for Depth Estimation dependencies and environment."""

import sys
import json
import torch

info = {
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "packages": {}
}

for pkg in ["transformers", "huggingface_hub", "timm", "cv2", "PIL", "scipy"]:
    try:
        mod = __import__(pkg)
        info["packages"][pkg] = getattr(mod, "__version__", "installed")
    except ImportError as e:
        info["packages"][pkg] = f"missing: {e}"

print(json.dumps(info, indent=2))
