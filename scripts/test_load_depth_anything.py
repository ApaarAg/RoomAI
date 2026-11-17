"""Test loading Depth-Anything-V2-Small-hf from transformers and running a dummy forward pass."""

import time
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np

def main():
    print("Testing Depth Anything V2 Small via HuggingFace transformers...")
    start_t = time.time()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading processor and model from {model_id}...")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device)
    model.eval()

    load_time = time.time() - start_t
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {load_time:.2f}s! Total parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Test dummy image
    dummy_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    inputs = processor(images=dummy_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_start = time.time()

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_time = (time.time() - inf_start) * 1000

    print(f"Output predicted_depth shape: {predicted_depth.shape}")
    print(f"Inference time: {inf_time:.2f} ms")

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(dummy_img.height, dummy_img.width),
        mode="bicubic",
        align_corners=False,
    )
    depth_np = prediction.squeeze().cpu().numpy()
    print(f"Resized depth map shape: {depth_np.shape}, min: {depth_np.min():.4f}, max: {depth_np.max():.4f}")

    # Check memory
    if torch.cuda.is_available():
        alloc_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        res_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
        print(f"GPU memory allocated: {alloc_mb:.1f} MB, reserved: {res_mb:.1f} MB")

    print("SUCCESS: Depth Anything V2 Small works cleanly!")


if __name__ == "__main__":
    main()
