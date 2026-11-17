import os
import sys
import hashlib
from pathlib import Path
import pandas as pd

os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
VAL_CSV = Path(r"C:\Users\apaar\fiftyone\open-images-v7\validation\labels\detections.csv")
TEST_CSV = Path(r"C:\Users\apaar\fiftyone\open-images-v7\test\labels\detections.csv")
FO_VAL_DATA = Path(r"C:\Users\apaar\fiftyone\open-images-v7\validation\data")

# Compute hashes of all existing canonical images
canonical_hashes = {}
for p in (REPO_ROOT / "data" / "exterior_canonical" / "images").rglob("*.jpg"):
    with open(p, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()
        canonical_hashes[h] = p

print(f"Total existing canonical images: {len(canonical_hashes)}")

# Check FO validation data images
fo_val_files = {p.stem: p for p in FO_VAL_DATA.glob("*.jpg")}
print(f"Total images in FO validation data: {len(fo_val_files)}")

# Check val porch images
df_val = pd.read_csv(VAL_CSV)
clean_val_porch = df_val[(df_val["LabelName"] == "/m/04m6gz") & ((df_val["XMax"] - df_val["XMin"]) * (df_val["YMax"] - df_val["YMin"]) <= 0.85)]
val_porch_img_ids = clean_val_porch["ImageID"].unique()

val_in_fo = 0
val_in_canonical = 0
for img_id in val_porch_img_ids:
    if img_id in fo_val_files:
        val_in_fo += 1
        with open(fo_val_files[img_id], "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
            if h in canonical_hashes:
                val_in_canonical += 1
                print(f"Validation porch image {img_id} is in canonical: {canonical_hashes[h].name}")

print(f"Validation porch images: {len(val_porch_img_ids)} total | in FiftyOne local cache: {val_in_fo} | already in canonical: {val_in_canonical}")

# Check test porch images
df_test = pd.read_csv(TEST_CSV)
clean_test_porch = df_test[(df_test["LabelName"] == "/m/04m6gz") & ((df_test["XMax"] - df_test["XMin"]) * (df_test["YMax"] - df_test["YMin"]) <= 0.85)]
test_porch_img_ids = clean_test_porch["ImageID"].unique()
print(f"Test porch images: {len(test_porch_img_ids)} total")
