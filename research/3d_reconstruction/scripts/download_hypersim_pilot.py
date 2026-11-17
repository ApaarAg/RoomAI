"""
RoomAI — Phase 6: Hypersim Pilot Dataset Downloader
Downloads verified synthetic scenes from Apple's official CDN and metadata repositories.
"""

import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

BASE_CDN = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"
RAW_ROOT = Path(r"f:\RoomAI\research\3d_reconstruction\data\ground_truth\raw\hypersim")
DOWNLOADS_ROOT = Path(r"f:\RoomAI\research\3d_reconstruction\data\ground_truth\downloads")

METADATA_CAMERA_URL = "https://raw.githubusercontent.com/apple/ml-hypersim/master/contrib/mikeroberts3000/metadata_camera_parameters.csv"
METADATA_SPLITS_URL = "https://raw.githubusercontent.com/apple/ml-hypersim/master/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"

# Target scenes for pilot (Multi-scene train, dedicated val, dedicated test)
TARGET_SCENES = {
    "train": ["ai_015_007", "ai_011_010", "ai_047_008"],
    "val": ["ai_016_001"],
    "test": ["ai_013_010"]
}

def download_with_progress(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[HYPERSIM] Archive already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return True
        
    print(f"\n[HYPERSIM] Fetching {url} -> {dest_path.name}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            speed = mb_downloaded / max(1e-3, time.time() - start_time)
            sys.stdout.write(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) @ {speed:.2f} MB/s   ")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print(f"\n[HYPERSIM] Finished downloading {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB) in {time.time()-start_time:.1f}s")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {url}: {e}")
        return False

def extract_archive(zip_path: Path, extract_dir: Path):
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"[HYPERSIM] Already extracted: {extract_dir.name}")
        return True
    print(f"[HYPERSIM] Extracting {zip_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print(f"[HYPERSIM] Extracted {zip_path.name} successfully.")
    return True

def fetch_metadata():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    cam_meta = RAW_ROOT / "metadata_camera_parameters.csv"
    if not cam_meta.exists():
        print("[HYPERSIM] Downloading camera metadata CSV...")
        urllib.request.urlretrieve(METADATA_CAMERA_URL, cam_meta)
    
    split_meta = RAW_ROOT / "metadata_images_split_scene_v1.csv"
    if not split_meta.exists():
        print("[HYPERSIM] Downloading split partitions CSV...")
        urllib.request.urlretrieve(METADATA_SPLITS_URL, split_meta)
    print("[HYPERSIM] Metadata tables verified.")

def main():
    print("=" * 70)
    print("ROOMAI — HYPERSIM PILOT ACQUISITION PIPELINE")
    print("=" * 70)
    fetch_metadata()
    
    for split, scenes in TARGET_SCENES.items():
        print(f"\n--- Split: {split.upper()} ({len(scenes)} scenes) ---")
        for sc in scenes:
            zip_url = f"{BASE_CDN}/{sc}.zip"
            zip_path = DOWNLOADS_ROOT / f"{sc}.zip"
            extract_path = RAW_ROOT / sc
            
            ok = download_with_progress(zip_url, zip_path)
            if ok:
                extract_archive(zip_path, extract_path)
            else:
                print(f"[ERROR] Could not complete acquisition for {sc}")

    print("\n" + "=" * 70)
    print("[HYPERSIM] Pilot acquisition completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()

