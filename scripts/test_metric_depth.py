import torch
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
print(f"Testing loading: {model_id}")
try:
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    print("SUCCESS: Loaded processor and model!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config max_depth: {getattr(model.config, 'max_depth', None)}")
    print(f"Config depth_estimation_type: {getattr(model.config, 'depth_estimation_type', None)}")
except Exception as e:
    print(f"Failed to load: {e}")

