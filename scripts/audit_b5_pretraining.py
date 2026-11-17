"""Phase 1: Pre-training Audit for Exterior Detector B5."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import psutil
import torch
import yaml

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")

def check_env():
    print("=== Environment & Hardware Audit ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {vram_gb:.2f} GB")
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_avail_gb = psutil.virtual_memory().available / (1024**3)
    print(f"System RAM: {ram_gb:.2f} GB (Available: {ram_avail_gb:.2f} GB)")

    disk = psutil.disk_usage(str(REPO_ROOT))
    disk_total_gb = disk.total / (1024**3)
    disk_free_gb = disk.free / (1024**3)
    print(f"Disk Total: {disk_total_gb:.2f} GB, Free: {disk_free_gb:.2f} GB")

def check_checkpoints():
    print("\n=== Checkpoint Lineage & Hashes ===")
    expected = {
        "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
        "exterior_detector_b3_yolo11s_rfs_best.pt": "2c44291fa642aba3893db001520deba0",
        "exterior_detector_b4_yolo11s_curated_best.pt": "d2710fba760713e68ea306e029ab651c",
    }
    for name, exp_hash in expected.items():
        p = REPO_ROOT / "checkpoints" / name
        if p.exists():
            h = hashlib.md5(p.read_bytes()).hexdigest()
            status = "VERIFIED" if h == exp_hash else "MISMATCH"
            print(f"{name}: {h} ({status}, size: {p.stat().st_size:,} bytes)")
        else:
            print(f"{name}: MISSING")

def check_datasets():
    print("\n=== Existing Exterior Datasets & Splits ===")
    data_dir = REPO_ROOT / "data"
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and "exterior" in d.name]
    for d in subdirs:
        print(f"\nDataset: {d.name}")
        for split in ["train", "val", "test"]:
            img_dir = d / "images" / split
            lbl_dir = d / "labels" / split
            n_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
            n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            print(f"  {split}: {n_imgs} images, {n_lbls} label files")

def check_source_cache():
    print("\n=== Source & Cache Directories ===")
    src_dir = REPO_ROOT / "data" / "source"
    if src_dir.exists():
        for item in src_dir.iterdir():
            if item.is_dir():
                print(f"  data/source/{item.name}: {len(list(item.glob('*')))} items")
            else:
                print(f"  data/source/{item.name}: {item.stat().st_size:,} bytes")
    
    cache_oi = REPO_ROOT / "data" / "cache_oi_test_images"
    if cache_oi.exists():
        print(f"  data/cache_oi_test_images: {len(list(cache_oi.glob('*.jpg')))} cached images")

if __name__ == "__main__":
    check_env()
    check_checkpoints()
    check_datasets()
    check_source_cache()
