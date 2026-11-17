"""
RoomAI — Phase 6: ScanNet Pilot Dataset Downloader
Fast concurrent downloader for ScanNet RGB-D sequences from verified open repository.
"""

import os
import sys
import time
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_API = "https://huggingface.co/api/datasets/fjd/scannet-processed-test"
RESOLVE_BASE = "https://huggingface.co/datasets/fjd/scannet-processed-test/resolve/main/test"
DEST_ROOT = Path(r"f:\RoomAI\research\3d_reconstruction\data\ground_truth\raw\scannet")

# Target scene sequences grouped by physical room (zero leakage across splits)
SCENES = {
    # Train: scene0019 (2 scans) and scene0015 (1 scan)
    "train": ["scene0019_00", "scene0019_01", "scene0015_00"],
    # Val: scene0030 (3 scans)
    "val": ["scene0030_00", "scene0030_01", "scene0030_02"],
    # Test: scene0011 (2 scans)
    "test": ["scene0011_00", "scene0011_01"]
}

def fetch_file(url: str, out_path: Path, max_retries: int = 3) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                with open(out_path, "wb") as f:
                    f.write(data)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] Failed {url} after {max_retries} attempts: {e}")
                return False
            time.sleep(1.0)
    return False

def get_scene_file_list(scene_id: str, all_siblings: list) -> list:
    prefix = f"test/{scene_id}/"
    scene_files = [s for s in all_siblings if s.startswith(prefix)]
    return scene_files

def download_scene(scene_id: str, scene_files: list, max_workers: int = 8):
    scene_dir = DEST_ROOT / scene_id
    tasks = []
    
    for rel_path in scene_files:
        # e.g. test/scene0019_00/color/0000.jpg -> relative to scene: color/0000.jpg
        local_rel = rel_path.replace(f"test/{scene_id}/", "")
        out_file = scene_dir / local_rel
        url = f"{RESOLVE_BASE}/{scene_id}/{local_rel}"
        tasks.append((url, out_file))
        
    print(f"\n[SCANNET] Starting download for {scene_id} ({len(tasks)} files)...")
    t0 = time.time()
    success = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetch_file, url, out): (url, out) for url, out in tasks}
        for future in as_completed(future_map):
            if future.result():
                success += 1
            else:
                failed += 1
                
    elapsed = time.time() - t0
    print(f"[SCANNET] {scene_id} complete: {success} succeeded, {failed} failed in {elapsed:.2f}s")
    return success, failed

def main():
    print("=" * 70)
    print("ROOMAI — SCANNET PILOT ACQUISITION PIPELINE")
    print("=" * 70)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("[SCANNET] Fetching dataset metadata from HuggingFace...")
    req = urllib.request.Request(REPO_API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        repo_data = json.loads(resp.read().decode("utf-8"))
    siblings = [s["rfilename"] for s in repo_data.get("siblings", [])]
    print(f"[SCANNET] Total repo files: {len(siblings)}")
    
    total_scenes = sum(len(sc_list) for sc_list in SCENES.values())
    print(f"[SCANNET] Target pilot: {total_scenes} sequences across 3 splits.")
    
    grand_success = 0
    grand_failed = 0
    start_time = time.time()
    
    for split, scene_list in SCENES.items():
        print(f"\n--- Split: {split.upper()} ({len(scene_list)} sequences) ---")
        for sc in scene_list:
            sc_files = get_scene_file_list(sc, siblings)
            s_ok, s_fail = download_scene(sc, sc_files, max_workers=8)
            grand_success += s_ok
            grand_failed += s_fail
            
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SCANNET] Grand Total: {grand_success} files downloaded, {grand_failed} failed in {total_time:.2f}s")
    print(f"[SCANNET] Storage location: {DEST_ROOT}")
    print("=" * 70)

if __name__ == "__main__":
    main()

