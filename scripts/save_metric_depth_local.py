import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
local_dir = Path("checkpoints/depth_anything_v2_metric_indoor_small")
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading and caching metric indoor checkpoint to {local_dir}...")
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)

processor.save_pretrained(local_dir)
model.save_pretrained(local_dir)
print(f"Saved locally to {local_dir}")
for p in local_dir.iterdir():
    print(f" - {p.name} ({p.stat().st_size:,} bytes)")

