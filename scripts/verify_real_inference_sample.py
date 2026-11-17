"""Perform one real inference using a real room image to verify that the result is not simulated."""

import sys
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.detection_service import DetectionService
TEST_IMG = REPO_ROOT / "data" / "deepfurniture" / "images" / "test" / "L3D121S3ENDIHLIMMQUI5NZUR7Q3P3XI888.jpg"

if not TEST_IMG.exists():
    TEST_IMG = next((REPO_ROOT / "data" / "deepfurniture" / "images" / "test").glob("*.jpg"))

print("=" * 65)
print(f" REAL INFERENCE TEST ON SAMPLE ROOM IMAGE")
print(f" Image path: {TEST_IMG}")
print("=" * 65)

image = Image.open(TEST_IMG).convert("RGB")
print(f" Image dimensions: {image.size} (width, height)")

# Instantiate DetectionService in Real Mode (mock_mode=False) with default config
service = DetectionService(mock_mode=False)
result = service.run(image, confidence_threshold=0.35, save_annotated=True)

print(f"\nSuccess:            {result['success']}")
print(f"Is Implemented:     {result['is_implemented']}")
print(f"Error Message:      {result['error_message']}")
print(f"Saved Annotated Img:{result['saved_image_path']}")

resp = result["response"]
print(f"\nResponse Metadata:")
print(f"  Model Name:       {resp.model_name}")
print(f"  Checkpoint:       {resp.checkpoint}")
print(f"  Device:           {resp.device}")
print(f"  Inference Time:   {resp.inference_time_ms:.2f} ms")
print(f"  Is Simulated:     {resp.is_simulated} (VERIFIED NOT SIMULATED)")
print(f"  Total Detections: {resp.total_objects}")

print("\nDetected Objects:")
for idx, d in enumerate(resp.detections, 1):
    b = d.box
    clean_label = d.label.replace("#", " / ")
    print(f"  {idx}. {clean_label:<18} (conf: {d.confidence:.2f}) -> box: [{b.x_min:.3f}, {b.y_min:.3f}, {b.x_max:.3f}, {b.y_max:.3f}]")

assert resp.is_simulated is False, "CRITICAL ERROR: Response was simulated!"
assert resp.checkpoint == "deepfurniture_yolo11n_finetune_best.pt"
assert resp.total_objects > 0, "No objects detected in sample room!"
print("\nVERIFICATION COMPLETE: Real fine-tuned detector inference verified successfully!")
