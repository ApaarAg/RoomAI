"""Script to verify all checkpoint files: task, nc, class names, architecture, and size."""

from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"

print("=" * 80)
print("RoomAI Checkpoint Verification & Architecture Audit")
print("=" * 80)

checkpoints = sorted(list(CKPT_DIR.glob("*.pt")))
for ckpt in checkpoints:
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    print(f"\n--- Checkpoint: {ckpt.name} ({size_mb:.2f} MB) ---")
    try:
        yolo = YOLO(str(ckpt))
        task = getattr(yolo, "task", "unknown")
        names = getattr(yolo, "names", {})
        nc = len(names)
        # Check architecture if available
        arch = "YOLO"
        if hasattr(yolo, "model") and yolo.model is not None:
            arch = yolo.model.__class__.__name__
        print(f"  Task: {task}")
        print(f"  Architecture: {arch}")
        print(f"  Number of Classes (nc): {nc}")
        print(f"  Classes: {names}")
    except Exception as e:
        print(f"  Error loading checkpoint: {e}")

# Check Depth Anything directory
depth_dir = CKPT_DIR / "depth_anything_v2_small"
if depth_dir.exists():
    print(f"\n--- Checkpoint Directory: {depth_dir.name} ---")
    for f in depth_dir.iterdir():
        print(f"  File: {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
