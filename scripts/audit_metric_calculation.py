import os
import sys
from pathlib import Path

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")

def main():
    b4 = YOLO(str(REPO_ROOT / "checkpoints" / "exterior_detector_b4_yolo11s_curated_best.pt"))

    for yml_name in ["eval_canonical_original.yaml", "eval_canonical_corrected.yaml"]:
        res = b4.val(data=str(REPO_ROOT / "data" / yml_name), split="test", conf=0.001, iou=0.7, workers=0, verbose=False)
        p_idx = list(res.ap_class_index).index(6) if 6 in res.ap_class_index else -1
        print(f"\n==========================================")
        print(f"Dataset: {yml_name}")
        print(f"Overall mAP50: {res.box.map50:.4f}, mAP50-95: {res.box.map:.4f}")
        print(f"Overall mp: {res.box.mp:.4f}, mr: {res.box.mr:.4f}")
        if p_idx != -1:
            print(f"Porch AP50: {res.box.ap50[p_idx]:.4f}")
            print(f"Porch AP50-95: {res.box.ap[p_idx]:.4f}")
            print(f"Porch Precision: {res.box.p[p_idx]:.4f}")
            print(f"Porch Recall: {res.box.r[p_idx]:.4f}")
            print(f"Porch F1: {res.box.f1[p_idx]:.4f}")

if __name__ == "__main__":
    main()
