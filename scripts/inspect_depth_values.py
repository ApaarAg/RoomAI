"""Inspect raw Depth Anything V2 Small output values on indoor and exterior images."""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

model_dir = "checkpoints/depth_anything_v2_small"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Loading local checkpoint from {model_dir} onto {device}...")
processor = AutoImageProcessor.from_pretrained(model_dir)
model = AutoModelForDepthEstimation.from_pretrained(model_dir)
model.to(device)
model.eval()

img_paths = [
    ("Indoor", "data/indoor_canonical/images/test/df_L3D121S21ENDIHOOVJYUI5L7SFOS3P3W2888.jpg"),
    ("Exterior", "data/exterior_canonical/images/test/cmp_test_0000.jpg"),
]

for label, p in img_paths:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model(**inputs)
        pred = out.predicted_depth
        
        # Bicubic interpolation to original size
        pred_resized = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
    depth = pred_resized.cpu().numpy()
    print(f"\n--- {label} ({p}) ---")
    print(f"Original size: {w}x{h}, Output shape: {depth.shape}")
    print(f"Min: {depth.min():.4f}, Max: {depth.max():.4f}, Mean: {depth.mean():.4f}, Std: {depth.std():.4f}")
    print(f"Any NaNs: {np.isnan(depth).any()}, Any Infs: {np.isinf(depth).any()}")
    
    # Check normalization: Min-Max to [0.0, 1.0]
    norm_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    print(f"Normalized Min: {norm_depth.min():.4f}, Normalized Max: {norm_depth.max():.4f}")
