import os
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
TRAIN_DET_CSV = Path(r"C:\Users\apaar\fiftyone\open-images-v7\train\labels\detections.csv")

# Collect all candidate image IDs
candidate_dirs = [
    REPO_ROOT / "data" / "cache_pilot_b7_candidates",
    REPO_ROOT / "data" / "cache_oi_b6_candidates",
    REPO_ROOT / "data" / "cache_oi_b5_images",
]

candidate_map = {}
for d in candidate_dirs:
    if d.exists():
        for p in d.glob("*.jpg"):
            if p.stat().st_size > 1000:
                candidate_map[p.stem] = p

print(f"Total unique candidate images found on disk: {len(candidate_map)}")

# Exclude images already in pilot B7
b7_manifest_p = REPO_ROOT / "data" / "pilot_b7_dataset" / "pilot_manifest.json"
if b7_manifest_p.exists():
    import json
    with open(b7_manifest_p, "r") as f:
        pilot_manifest = json.load(f)
    used_ids = {item["original_image_id"] for item in pilot_manifest}
    print(f"Images already used in B7 pilot: {len(used_ids)}")
else:
    used_ids = set()

available_ids = set(candidate_map.keys()) - used_ids
print(f"Available unused candidate images on disk: {len(available_ids)}")

OI_MID_TO_CANONICAL = {
    "/m/04m6gz": (6, "porch"),
    "/m/031b6r": (8, "shutter_blind"),
    "/m/02dgv": (3, "door"),
    "/m/0d4v4": (2, "window"),
    "/m/03jm5": (0, "building"),
    "/m/0cgh4": (0, "building"),
    "/m/021sj1": (0, "building"),
}

# Scan detections.csv for these image IDs
found_annotations = {}
chunksize = 1000000

for chunk_idx, chunk in enumerate(pd.read_csv(TRAIN_DET_CSV, chunksize=chunksize, usecols=['ImageID', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax'])):
    chunk_avail = chunk[chunk['ImageID'].isin(available_ids) & chunk['LabelName'].isin(OI_MID_TO_CANONICAL.keys()) & (chunk['Confidence'] == 1)]
    for img_id, rows in chunk_avail.groupby('ImageID'):
        if img_id not in found_annotations:
            found_annotations[img_id] = []
        found_annotations[img_id].extend(rows.to_dict('records'))

print(f"Found annotations for {len(found_annotations)} unused candidate images.")
class_counts = {cid: 0 for cid in range(10)}
for img_id, rows in found_annotations.items():
    cids_in_img = set()
    for r in rows:
        cid = OI_MID_TO_CANONICAL[r['LabelName']][0]
        cids_in_img.add(cid)
    for cid in cids_in_img:
        class_counts[cid] += 1

print("Class presence in candidates:")
for cid, cnt in class_counts.items():
    print(f"  Class {cid}: {cnt} images")

