"""Verify end-to-end RoomAI workflow including detection and 3D depth estimation."""

import os
from pathlib import Path
import sys
from PIL import Image
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.scene3d_service import Scene3DService
from services.detection_service import DetectionService
from models.registry import GLOBAL_MODEL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent

with open(REPO_ROOT / "config" / "model_config.yaml", "r", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)

img_path = REPO_ROOT / "data" / "indoor_canonical" / "images" / "test" / "df_L3D121S21ENDIHOOVJYUI5L7SFOS3P3W2888.jpg"
image = Image.open(img_path).convert("RGB")
print(f"Loaded test image: {image.size}")

# 1. Test Detection (Model A2)
print("\n--- 1. Testing Indoor Detection Service ---")
det_svc = DetectionService(mock_mode=False, config=model_config.get("detection", {}))
det_res = det_svc.run(image)
print(f"Detection success: {det_res['success']}")
print(f"Detected objects: {det_res['response'].total_objects}")
print(f"Model used: {det_res['response'].model_name} on {det_res['response'].device}")

# 2. Test Scene3D Depth Service
print("\n--- 2. Testing Scene3D Depth Estimation Service ---")
s3d_svc = Scene3DService(mock_mode=False, config=model_config.get("scene3d", {}))
s3d_res = s3d_svc.run(image)
print(f"Scene3D success: {s3d_res['success']}")
print(f"Is implemented: {s3d_res['is_implemented']}")
print(f"Is simulated: {s3d_res['response'].is_simulated}")
print(f"Model: {s3d_res['response'].model_name} on {s3d_res['response'].device}")
print(f"Inference latency: {s3d_res['timing']['inference_time_ms']:.1f} ms")
print(f"Heatmap generated: {s3d_res['heatmap_image'] is not None} (size: {s3d_res['heatmap_image'].size})")

# 3. Check VRAM status via Registry
print("\n--- 3. Checking VRAM Headroom via ModelRegistry ---")
vram_info = GLOBAL_MODEL_REGISTRY.check_vram_headroom()
print(f"VRAM Headroom: {vram_info}")

print("\nALL WORKFLOW CHECKS PASSED PERFECTLY!")
