"""Phase 22 Data Curation and Quality Screening.

Curates 960 total external architectural images (300 pilot B7 + 260 local cache + 400 fresh porch/shutter)
combined with 1,302 baseline B6 images = 2,262 total training images.

Quality Gates:
1. Checkpoint cryptographic invariant verification (B2, B6, B7, Porch Specialist).
2. Header verification via PIL.Image.verify() and decode check.
3. Coordinate validity: 0 < x, y, w, h <= 1.0.
4. Exact MD5 deduplication against B6 train, B7 pilot train, val, and retired test.
5. 64-bit DCT pHash screening against all validation and retired test scenes (Hamming d >= 5).
6. Loss mask generation: verified classes = 1.0, unannotated classes = 0.0.
7. Produces data/phase22_dataset/ and reports/phase22_dataset_provenance.csv.
"""

import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import socket

socket.setdefaulttimeout(10.0)

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from PIL import Image
import yaml

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
sys.path.insert(0, str(REPO_ROOT))

from utils.multidataset_harmonization import (
    CANONICAL_EXTERIOR_CLASSES,
    CANONICAL_NAME_TO_ID,
    validate_yolo_box,
    compute_image_phash,
    compute_phash_hamming_distance,
)

TRAIN_DET_CSV = Path(r"C:\Users\apaar\fiftyone\open-images-v7\train\labels\detections.csv")
B6_EXPANDED_DIR = REPO_ROOT / "data" / "exterior_b6_expanded"
PILOT_B7_DIR = REPO_ROOT / "data" / "pilot_b7_dataset"
OUTPUT_DIR = REPO_ROOT / "data" / "phase22_dataset"
CACHE_FRESH_DIR = REPO_ROOT / "data" / "cache_phase22_fresh"
PROVENANCE_CSV = REPO_ROOT / "reports" / "phase22_dataset_provenance.csv"

CANDIDATE_DIRS = [
    REPO_ROOT / "data" / "cache_pilot_b7_candidates",
    REPO_ROOT / "data" / "cache_oi_b6_candidates",
    REPO_ROOT / "data" / "cache_oi_b5_images",
]

EXPECTED_CHECKPOINTS = {
    "exterior_detector_b2_yolo11s_best.pt": "14a931dd0adabc16cbd7b28ed1d17f66",
    "exterior_detector_b6_expanded_best.pt": "f5ceea56b0610ea14b647014e48152dc",
    "exterior_detector_b7_pilot_best.pt": "41433e929910eda4414ed6541bac2fe4",
    "exterior_specialist_porch_yolo11s_best.pt": "4261fc1ad5f6fd9ac29c0499e2adddfd",
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


def verify_checkpoints():
    print("=== Step 0: Preflight Checkpoint Verification ===")
    for ckpt_name, expected_hash in EXPECTED_CHECKPOINTS.items():
        p = REPO_ROOT / "checkpoints" / ckpt_name
        assert p.exists(), f"Checkpoint missing: {p}"
        actual_hash = hashlib.md5(p.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {ckpt_name}: {actual_hash} != {expected_hash}"
        print(f"  [VERIFIED] {ckpt_name}: {actual_hash}")


def download_image(img_id: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    url = f"https://s3.amazonaws.com/open-images-dataset/train/{img_id}.jpg"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = resp.read()
        if len(data) > 1000:
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        return False
    except Exception:
        return False


def main():
    start_time = time.time()
    print("=== Phase 22: Controlled Dataset Expansion & Curation ===")
    verify_checkpoints()

    CACHE_FRESH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_img_dir = OUTPUT_DIR / "images" / "train"
    train_lbl_dir = OUTPUT_DIR / "labels" / "train"
    val_img_dir = OUTPUT_DIR / "images" / "val"
    val_lbl_dir = OUTPUT_DIR / "labels" / "val"
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Index all protected MD5 hashes and perceptual hashes (val + test)
    print("\n1. Indexing existing MD5 hashes and protected pHashes...")
    existing_hashes = set()
    protected_phashes = {}

    for split in ["train", "val", "test"]:
        s_dir = B6_EXPANDED_DIR / "images" / split
        if s_dir.exists():
            for p in s_dir.glob("*.jpg"):
                h = hashlib.md5(p.read_bytes()).hexdigest()
                existing_hashes.add(h)
                if split in ["val", "test"]:
                    try:
                        with Image.open(p) as im:
                            protected_phashes[p.name] = compute_image_phash(im)
                    except Exception:
                        pass

    b7_train_dir = PILOT_B7_DIR / "images" / "train"
    if b7_train_dir.exists():
        for p in b7_train_dir.glob("*.jpg"):
            existing_hashes.add(hashlib.md5(p.read_bytes()).hexdigest())

    print(f"  Total protected MD5s indexed: {len(existing_hashes)}")
    print(f"  Total protected validation/test pHashes: {len(protected_phashes)}")

    # 2. Gather candidate images on disk
    print("\n2. Scanning local candidate pools...")
    candidate_map = {}
    for cd in CANDIDATE_DIRS:
        if cd.exists():
            for p in cd.glob("*.jpg"):
                if p.stat().st_size > 1000:
                    candidate_map[p.stem] = p

    b7_manifest_p = PILOT_B7_DIR / "pilot_manifest.json"
    used_ids = set()
    if b7_manifest_p.exists():
        with open(b7_manifest_p, "r") as f:
            b7_man = json.load(f)
        used_ids = {item["original_image_id"] for item in b7_man}

    available_candidates = {k: v for k, v in candidate_map.items() if k not in used_ids}
    print(f"  Unique candidates on disk: {len(candidate_map)}")
    print(f"  Already in B7 pilot: {len(used_ids)}")
    print(f"  Available new candidates on disk: {len(available_candidates)}")

    # 3. Scan Open Images annotations for available on-disk candidates AND search for fresh porch/shutter
    print("\n3. Scanning Open Images CSV for annotations & fresh porch/shutter candidates...")
    cand_annotations = {}
    fresh_porch_candidates = {}
    fresh_shutter_candidates = {}

    target_fresh_porch = 250
    target_fresh_shutter = 150

    seen_ids = set(used_ids)
    for p in B6_EXPANDED_DIR.glob("**/*.jpg"):
        seen_ids.add(p.stem)
    for k in available_candidates:
        seen_ids.add(k)

    chunksize = 1000000
    for chunk_idx, chunk in enumerate(pd.read_csv(TRAIN_DET_CSV, chunksize=chunksize, usecols=['ImageID', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax'])):
        chunk_conf1 = chunk[chunk['Confidence'] == 1]
        
        # Check available on-disk candidates
        chunk_avail = chunk_conf1[chunk_conf1['ImageID'].isin(available_candidates.keys()) & chunk_conf1['LabelName'].isin(OI_MID_TO_CANONICAL.keys())]
        for img_id, rows in chunk_avail.groupby('ImageID'):
            if img_id not in cand_annotations:
                cand_annotations[img_id] = []
            cand_annotations[img_id].extend(rows.to_dict('records'))

        # Check fresh porch
        if len(fresh_porch_candidates) < target_fresh_porch:
            p_chunk = chunk_conf1[chunk_conf1['LabelName'] == '/m/04m6gz']
            for img_id, rows in p_chunk.groupby('ImageID'):
                if img_id not in seen_ids and len(fresh_porch_candidates) < target_fresh_porch:
                    fresh_porch_candidates[img_id] = rows.to_dict('records')
                    seen_ids.add(img_id)

        # Check fresh shutter
        if len(fresh_shutter_candidates) < target_fresh_shutter:
            s_chunk = chunk_conf1[chunk_conf1['LabelName'] == '/m/031b6r']
            for img_id, rows in s_chunk.groupby('ImageID'):
                if img_id not in seen_ids and len(fresh_shutter_candidates) < target_fresh_shutter:
                    fresh_shutter_candidates[img_id] = rows.to_dict('records')
                    seen_ids.add(img_id)

    print(f"  Annotated on-disk candidates: {len(cand_annotations)}")
    print(f"  Fresh porch candidates identified: {len(fresh_porch_candidates)}")
    print(f"  Fresh shutter candidates identified: {len(fresh_shutter_candidates)}")

    # 4. Download fresh candidates in parallel
    fresh_to_download = []
    fresh_map = {}
    for img_id in list(fresh_porch_candidates.keys()) + list(fresh_shutter_candidates.keys()):
        dest = CACHE_FRESH_DIR / f"{img_id}.jpg"
        fresh_map[img_id] = dest
        if not (dest.exists() and dest.stat().st_size > 1000):
            fresh_to_download.append((img_id, dest))

    print(f"\n4. Downloading {len(fresh_to_download)} fresh images via 16 parallel threads...")
    if fresh_to_download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futs = [executor.submit(download_image, iid, dst) for iid, dst in fresh_to_download]
            done, _ = concurrent.futures.wait(futs, timeout=60.0)

    # 5. Apply Quality Gates to all candidates
    print("\n5. Applying Quality Gates (Header, MD5, pHash >= 5, Box Coordinates)...")
    accepted_records = []
    rej_corrupt = 0
    rej_md5 = 0
    rej_phash = 0
    rej_no_valid = 0

    all_candidate_batches = []
    # 1. On-disk candidates
    for img_id, rows in cand_annotations.items():
        all_candidate_batches.append((img_id, available_candidates[img_id], rows, "local_cache"))
    # 2. Fresh porch
    for img_id, rows in fresh_porch_candidates.items():
        dst = fresh_map[img_id]
        if dst.exists() and dst.stat().st_size > 1000:
            all_candidate_batches.append((img_id, dst, rows, "fresh_porch"))
    # 3. Fresh shutter
    for img_id, rows in fresh_shutter_candidates.items():
        dst = fresh_map[img_id]
        if dst.exists() and dst.stat().st_size > 1000:
            all_candidate_batches.append((img_id, dst, rows, "fresh_shutter"))

    print(f"  Total candidate items to evaluate through quality gates: {len(all_candidate_batches)}")

    for img_id, src_path, rows, group in all_candidate_batches:
        raw_bytes = src_path.read_bytes()
        md5_val = hashlib.md5(raw_bytes).hexdigest()

        if md5_val in existing_hashes:
            rej_md5 += 1
            continue

        try:
            with Image.open(src_path) as im:
                im.verify()
            with Image.open(src_path) as im:
                w, h = im.size
                cand_ph = compute_image_phash(im)
        except Exception:
            rej_corrupt += 1
            continue

        is_leak = False
        for prot_name, prot_h in protected_phashes.items():
            if compute_phash_hamming_distance(cand_ph, prot_h) < 5:
                is_leak = True
                rej_phash += 1
                break
        if is_leak:
            continue

        valid_boxes = []
        cids_present = set()
        for r in rows:
            cid = OI_MID_TO_CANONICAL.get(r['LabelName'], None)
            if cid is None:
                continue
            xmin, xmax = float(r['XMin']), float(r['XMax'])
            ymin, ymax = float(r['YMin']), float(r['YMax'])
            bw = xmax - xmin
            bh = ymax - ymin
            xc = xmin + bw / 2.0
            yc = ymin + bh / 2.0

            val_res = validate_yolo_box(cid, xc, yc, bw, bh)
            if val_res.is_valid:
                valid_boxes.append((cid, *val_res.box_coords))
                cids_present.add(cid)

        if not valid_boxes:
            rej_no_valid += 1
            continue

        existing_hashes.add(md5_val)
        accepted_records.append({
            "img_id": img_id,
            "src_path": src_path,
            "md5": md5_val,
            "group": group,
            "boxes": valid_boxes,
            "cids_present": cids_present,
            "width": w,
            "height": h,
        })

    print(f"  Quality Gate Summary:")
    print(f"    Accepted Clean New Images: {len(accepted_records)}")
    print(f"    Rejected MD5 Collisions: {rej_md5}")
    print(f"    Rejected Near-Duplicates (pHash d < 5): {rej_phash}")
    print(f"    Rejected Corrupt/Decode: {rej_corrupt}")
    print(f"    Rejected No Valid Boxes: {rej_no_valid}")

    # 6. Copy baseline training and validation images
    print("\n6. Assembling Phase 22 dataset...")
    # Clear and rebuild clean train/val
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        for f in d.glob("*"):
            f.unlink()

    # Copy 1,302 B6 baseline images
    b6_train_imgs = list((B6_EXPANDED_DIR / "images" / "train").glob("*.jpg"))
    print(f"  Copying {len(b6_train_imgs)} B6 baseline training images...")
    for p in b6_train_imgs:
        shutil.copy2(p, train_img_dir / p.name)
        lbl_p = B6_EXPANDED_DIR / "labels" / "train" / f"{p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, train_lbl_dir / lbl_p.name)

    # Copy 154 authorized validation images untouched
    val_imgs = list((PILOT_B7_DIR / "images" / "val").glob("*.jpg"))
    print(f"  Copying {len(val_imgs)} authorized validation images untouched...")
    for p in val_imgs:
        shutil.copy2(p, val_img_dir / p.name)
        lbl_p = PILOT_B7_DIR / "labels" / "val" / f"{p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, val_lbl_dir / lbl_p.name)

    # Initialize mask lookup: baseline 1,302 images are fully verified ([1.0] * 10)
    verified_class_masks = {}
    for p in b6_train_imgs:
        verified_class_masks[p.name] = [1.0] * 10

    provenance_rows = []

    for p in b6_train_imgs:
        provenance_rows.append({
            "assigned_filename": p.name,
            "original_id": p.stem,
            "source_dataset": "RoomAI_B6_Baseline",
            "group": "legacy_canonical",
            "license": "Internal/Academic",
            "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            "mask_type": "fully_annotated",
            "classes_verified": "0,1,2,3,4,5,6,7,8,9",
            "classes_masked_out": "",
        })

    # Ingest 300 B7 pilot images
    print("  Ingesting 300 B7 pilot images...")
    with open(b7_manifest_p, "r") as f:
        pilot_manifest = json.load(f)

    with open(PILOT_B7_DIR / "verified_class_masks.json", "r") as f:
        pilot_masks = json.load(f)

    for item in pilot_manifest:
        fname = item["assigned_filename"]
        src_img = PILOT_B7_DIR / "images" / "train" / fname
        src_lbl = PILOT_B7_DIR / "labels" / "train" / f"{Path(fname).stem}.txt"
        shutil.copy2(src_img, train_img_dir / fname)
        if src_lbl.exists():
            shutil.copy2(src_lbl, train_lbl_dir / src_lbl.name)

        mask = pilot_masks[fname]
        verified_class_masks[fname] = mask

        v_cids = [str(i) for i, m in enumerate(mask) if m > 0.5]
        m_cids = [str(i) for i, m in enumerate(mask) if m <= 0.5]
        provenance_rows.append({
            "assigned_filename": fname,
            "original_id": item["original_image_id"],
            "source_dataset": item["source_dataset"],
            "group": f"pilot_b7_{item['group']}",
            "license": item["license"],
            "md5": item["md5"],
            "mask_type": "partially_annotated_masked",
            "classes_verified": ",".join(v_cids),
            "classes_masked_out": ",".join(m_cids),
        })

    # Ingest accepted new records
    print(f"  Ingesting {len(accepted_records)} new curated candidate images...")
    total_new_boxes = {cid: 0 for cid in range(10)}

    for idx, cand in enumerate(accepted_records):
        fname = f"phase22_ext_{idx:04d}.jpg"
        dst_img = train_img_dir / fname
        dst_lbl = train_lbl_dir / f"phase22_ext_{idx:04d}.txt"

        shutil.copy2(cand["src_path"], dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            for b in cand["boxes"]:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                total_new_boxes[b[0]] += 1

        mask = [0.0] * 10
        for cid in cand["cids_present"]:
            mask[cid] = 1.0

        verified_class_masks[fname] = mask

        v_cids = [str(i) for i, m in enumerate(mask) if m > 0.5]
        m_cids = [str(i) for i, m in enumerate(mask) if m <= 0.5]
        provenance_rows.append({
            "assigned_filename": fname,
            "original_id": cand["img_id"],
            "source_dataset": "OpenImagesV7_Train",
            "group": f"phase22_{cand['group']}",
            "license": "CC BY 4.0 / CC BY 2.0",
            "md5": cand["md5"],
            "mask_type": "partially_annotated_masked",
            "classes_verified": ",".join(v_cids),
            "classes_masked_out": ",".join(m_cids),
        })

    # Save verified_class_masks.json
    masks_path = OUTPUT_DIR / "verified_class_masks.json"
    with open(masks_path, "w", encoding="utf-8") as f:
        json.dump(verified_class_masks, f, indent=2)
    print(f"  Saved verified_class_masks.json ({len(verified_class_masks)} training images).")

    # Save dataset.yaml
    dataset_yaml = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 10,
        "names": {i: name for i, name in enumerate(CANONICAL_EXTERIOR_CLASSES)},
    }
    with open(OUTPUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    print("  Saved dataset.yaml.")

    # Save reports/phase22_dataset_provenance.csv
    PROVENANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "assigned_filename", "original_id", "source_dataset", "group",
        "license", "md5", "mask_type", "classes_verified", "classes_masked_out"
    ]
    with open(PROVENANCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provenance_rows)
    print(f"  Saved {PROVENANCE_CSV} ({len(provenance_rows)} entries).")

    total_ext = 300 + len(accepted_records)
    print(f"\n=== Curation Complete in {time.time() - start_time:.2f}s ===")
    print(f"Total Training Images: {len(verified_class_masks)}")
    print(f"  - Baseline B6 Images: {len(b6_train_imgs)}")
    print(f"  - Pilot B7 Images: 300")
    print(f"  - New Curated Images: {len(accepted_records)}")
    print(f"  - Total External Curated Images: {total_ext} (Target: ~800-1000)")
    print(f"New Boxes Ingested:")
    for cid, cnt in total_new_boxes.items():
        if cnt > 0:
            print(f"  {CANONICAL_EXTERIOR_CLASSES[cid]} (id {cid}): {cnt} boxes")


if __name__ == "__main__":
    main()
