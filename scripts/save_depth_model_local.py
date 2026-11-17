"""Save cached Depth Anything V2 Small model and processor to local checkpoints folder."""

from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

save_dir = Path("checkpoints/depth_anything_v2_small")
save_dir.mkdir(parents=True, exist_ok=True)

model_id = "depth-anything/Depth-Anything-V2-Small-hf"
print(f"Loading from HF cache ({model_id}) and saving to {save_dir}...")

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)

processor.save_pretrained(str(save_dir))
model.save_pretrained(str(save_dir))

print(f"Successfully saved to {save_dir}!")
for f in save_dir.iterdir():
    print(f" - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
