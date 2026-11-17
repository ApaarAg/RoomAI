"""Phase 14 Stage B: Targeted Dataset Expansion for Model B6.

Curates targeted training data from Open Images V7 Train split:
- Targets: Porch (/m/04m6gz), Window blind / Shutter (/m/031b6r), Door (/m/02dgv), House/Building (/m/03jm5, /m/0cgh4)
- Safeguards:
  1. Byte-for-byte MD5 verification on existing checkpoints.
  2. Zero MD5 collisions against train, val, and test splits.
  3. Strict 64-bit DCT perceptual hash (pHash) Hamming distance >= 5 against validation and locked test splits.
  4. Locked test split remains completely untouched.
- Output: data/exterior_b6_expanded/ with dataset.yaml and b6_manifest.json.
"""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
from typing import Any, Dict, List, Set, Tuple

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import yaml

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
TRAIN_DET_CSV = Path(r"C:\Users\apaar\fiftyone\open-images-v7\train\labels\detections.csv")
B5_UNIFIED_DIR = REPO_ROOT / "data" / "exterior_b5_unified"
OUTPUT_DIR = REPO_ROOT / "data" / "exterior_b6_expanded"
CACHE_DIR = REPO_ROOT / "data" / "cache_oi_b6_candidates"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}

OI_MID_TO_CANONICAL = {
    "/m/04m6gz": 6,  # Porch
    "/m/031b6r": 8,  # Window blind -> Shutter_blind
    "/m/02dgv": 3,   # Door
    "/m/0d4v4": 2,   # Window
    "/m/03jm5": 0,   # House -> Building
    "/m/0cgh4": 0,   # Building
    "/m/021sj1": 0,  # Office building -> Building
}

EXPECTED_CHECKPOINTS = {
    "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
    "exterior_detector_b3_yolo11s_rfs_best.pt": "2c44291fa642aba3893db001520deba0",
    "exterior_detector_b4_yolo11s_curated_best.pt": "d2710fba760713e68ea306e029ab651c",
    "exterior_detector_b5_transfer_best.pt": "e12b4181fee244a3569dfa782ba3c3c3",
}


def verify_checkpoints():
    print("=== Step 0: Safeguard Checkpoint Verification ===")
    for ckpt_name, expected_hash in EXPECTED_CHECKPOINTS.items():
        p = REPO_ROOT / "checkpoints" / ckpt_name
        assert p.exists(), f"Checkpoint missing: {p}"
        actual_hash = hashlib.md5(p.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {ckpt_name}: {actual_hash} != {expected_hash}"
        print(f"  [VERIFIED] {ckpt_name}: {actual_hash}")


def compute_phash(img: Image.Image, hash_size: int = 8) -> int:
    img_gray = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.Resampling.BILINEAR)
    pixels = np.array(img_gray, dtype=np.float32)
    dct = cv2.dct(pixels)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    diff = dct_low > med
    h = 0
    for b in diff.flatten():
        h = (h << 1) | int(b)
    return h


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def download_image(img_id: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    url = f"https://s3.amazonaws.com/open-images-dataset/train/{img_id}.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def main():
    print("=== Stage B: Data Expansion for Model B6 ===")
    verify_checkpoints()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gather all existing hashes to prevent cross-split leakage
    print("\n1. Indexing existing hashes from B5 unified splits...")
    existing_hashes = set()
    protected_phashes = {}

    for split in ["val", "test", "train"]:
        img_dir = B5_UNIFIED_DIR / "images" / split
        for img_p in img_dir.glob("*.jpg"):
            md5_val = hashlib.md5(img_p.read_bytes()).hexdigest()
            existing_hashes.add(md5_val)
            if split in ["val", "test"]:
                try:
                    with Image.open(img_p) as im:
                        protected_phashes[img_p.name] = compute_phash(im)
                except Exception:
                    pass

    print(f"  Indexed {len(existing_hashes)} existing MD5 hashes.")
    print(f"  Protected {len(protected_phashes)} validation & test pHashes for near-duplicate screening.")

    # 2. Extract candidate images from Train detections CSV
    print("\n2. Scanning Train detections CSV for Porch and Shutter/Blind candidates...")
    porch_target = 250
    blind_target = 150
    candidate_records = []
    seen_candidates = set()

    porch_count = 0
    blind_count = 0

    chunksize = 500000
    for chunk_idx, chunk in enumerate(pd.read_csv(TRAIN_DET_CSV, chunksize=chunksize, usecols=['ImageID', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax'])):
        chunk_filtered = chunk[chunk['LabelName'].isin(OI_MID_TO_CANONICAL.keys()) & (chunk['Confidence'] == 1)]
        
        # Group by image
        grouped = chunk_filtered.groupby('ImageID')
        for img_id, rows in grouped:
            if img_id in seen_candidates:
                continue

            labels = set(rows['LabelName'])
            has_porch = '/m/04m6gz' in labels
            has_blind = '/m/031b6r' in labels

            if not (has_porch or has_blind):
                continue

            if has_porch and porch_count < porch_target:
                candidate_records.append((img_id, rows))
                seen_candidates.add(img_id)
                porch_count += 1
            elif has_blind and blind_count < blind_target:
                candidate_records.append((img_id, rows))
                seen_candidates.add(img_id)
                blind_count += 1

            if porch_count >= porch_target and blind_count >= blind_target:
                break

        if porch_count >= porch_target and blind_count >= blind_target:
            break
        print(f"  Processed {(chunk_idx + 1) * chunksize:,} rows... Found {porch_count} porch, {blind_count} blind candidates.")

    print(f"\nTotal Candidates Selected for Ingestion: {len(candidate_records)}")
    print(f"  Porch scenes: {porch_count} | Shutter/Blind scenes: {blind_count}")

    # 3. Download and verify images
    print("\n3. Downloading candidate images from AWS Open Images S3...")
    download_tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for img_id, _ in candidate_records:
            dest = CACHE_DIR / f"{img_id}.jpg"
            download_tasks.append((img_id, executor.submit(download_image, img_id, dest), dest))

    downloaded_candidates = []
    for img_id, fut, dest in download_tasks:
        if fut.result() and dest.exists() and dest.stat().st_size > 1000:
            downloaded_candidates.append((img_id, dest))

    print(f"  Successfully downloaded: {len(downloaded_candidates)} images.")

    # 4. Strict Quality Gates: Deduplication & Near-Duplicate Screening
    print("\n4. Applying Quality Gates: MD5 collision & 64-bit DCT pHash screening...")
    accepted_new_images = []
    rejected_near_duplicates = 0
    rejected_collisions = 0

    record_map = {img_id: rows for img_id, rows in candidate_records}

    for img_id, dest in downloaded_candidates:
        img_bytes = dest.read_bytes()
        cand_md5 = hashlib.md5(img_bytes).hexdigest()

        if cand_md5 in existing_hashes:
            rejected_collisions += 1
            continue

        try:
            with Image.open(dest) as im:
                im.verify()
            with Image.open(dest) as im:
                w_orig, h_orig = im.size
                cand_phash = compute_phash(im)
        except Exception:
            continue

        # Check against protected validation & test sets
        is_leak = False
        for prot_name, prot_h in protected_phashes.items():
            if hamming_distance(cand_phash, prot_h) < 5:
                is_leak = True
                rejected_near_duplicates += 1
                break

        if is_leak:
            continue

        existing_hashes.add(cand_md5)
        accepted_new_images.append((img_id, dest, cand_md5, record_map[img_id]))

    print(f"  Quality Gate Summary:")
    print(f"    Accepted Clean Images: {len(accepted_new_images)}")
    print(f"    Rejected MD5 Collisions: {rejected_collisions}")
    print(f"    Rejected Near-Duplicates (d < 5): {rejected_near_duplicates}")

    # 5. Populate data/exterior_b6_expanded/
    print("\n5. Constructing data/exterior_b6_expanded/ dataset structure...")
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy existing 903 training images from B5
    b5_train_imgs = list((B5_UNIFIED_DIR / "images" / "train").glob("*.jpg"))
    for p in b5_train_imgs:
        shutil.copy2(p, OUTPUT_DIR / "images" / "train" / p.name)
        lbl_p = B5_UNIFIED_DIR / "labels" / "train" / f"{p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, OUTPUT_DIR / "labels" / "train" / lbl_p.name)

    # Copy val and test splits untouched
    for split in ["val", "test"]:
        for p in (B5_UNIFIED_DIR / "images" / split).glob("*.jpg"):
            shutil.copy2(p, OUTPUT_DIR / "images" / split / p.name)
            lbl_p = B5_UNIFIED_DIR / "labels" / split / f"{p.stem}.txt"
            if lbl_p.exists():
                shutil.copy2(lbl_p, OUTPUT_DIR / "labels" / split / lbl_p.name)

    # Ingest accepted new training images
    new_boxes_count = {cid: 0 for cid in range(10)}
    b6_manifest = []

    for idx, (img_id, dest, md5_val, rows) in enumerate(accepted_new_images):
        new_filename = f"oi_b6_train_{idx:04d}.jpg"
        target_img_p = OUTPUT_DIR / "images" / "train" / new_filename
        target_lbl_p = OUTPUT_DIR / "labels" / "train" / f"oi_b6_train_{idx:04d}.txt"

        shutil.copy2(dest, target_img_p)

        boxes = []
        for _, r in rows.iterrows():
            cid = OI_MID_TO_CANONICAL.get(r['LabelName'], None)
            if cid is None:
                continue
            xmin, xmax = float(r['XMin']), float(r['XMax'])
            ymin, ymax = float(r['YMin']), float(r['YMax'])

            bw = xmax - xmin
            bh = ymax - ymin
            xc = xmin + bw / 2.0
            yc = ymin + bh / 2.0

            if bw < 0.005 or bh < 0.005:
                continue

            boxes.append((cid, round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)))
            new_boxes_count[cid] += 1

        with open(target_lbl_p, "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n")

        b6_manifest.append({
            "assigned_filename": new_filename,
            "original_image_id": img_id,
            "source_dataset": "OpenImagesV7_Train",
            "md5": md5_val,
            "num_boxes": len(boxes),
            "boxes": boxes,
        })

    print(f"\n  Integrated {len(accepted_new_images)} new clean training images into B6 Train split.")
    print("  New annotations added:")
    for cid in range(10):
        cname = CANONICAL_CLASSES[cid]
        if new_boxes_count[cid] > 0:
            print(f"    {cname:<15}: +{new_boxes_count[cid]} boxes")

    # Final split verification
    final_train_imgs = len(list((OUTPUT_DIR / "images" / "train").glob("*.jpg")))
    final_val_imgs = len(list((OUTPUT_DIR / "images" / "val").glob("*.jpg")))
    final_test_imgs = len(list((OUTPUT_DIR / "images" / "test").glob("*.jpg")))
    total_imgs = final_train_imgs + final_val_imgs + final_test_imgs

    print(f"\nFinal B6 Dataset Counts:")
    print(f"  Train Images: {final_train_imgs} (was 903, +{final_train_imgs - 903})")
    print(f"  Val Images  : {final_val_imgs} (locked, unchanged)")
    print(f"  Test Images : {final_test_imgs} (locked, untouched)")
    print(f"  Total Images: {total_imgs}")

    # Write dataset.yaml
    yaml_content = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 10,
        "names": {i: CANONICAL_CLASSES[i] for i in range(10)},
    }
    with open(OUTPUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # Save manifest
    manifest_p = OUTPUT_DIR / "b6_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(b6_manifest, f, indent=2)

    print(f"  [SAVED] YAML written to {OUTPUT_DIR / 'dataset.yaml'}")
    print(f"  [SAVED] Manifest written to {manifest_p}")


if __name__ == "__main__":
    main()
