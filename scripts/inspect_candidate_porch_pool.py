import os
import sys
from pathlib import Path

# Fix DLL load on Windows
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Library\bin")
os.add_dll_directory(r"C:\Users\apaar\miniconda3\envs\tf-gpu\Scripts")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import hashlib

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
VAL_CSV = Path(r"C:\Users\apaar\fiftyone\open-images-v7\validation\labels\detections.csv")
TEST_CSV = Path(r"C:\Users\apaar\fiftyone\open-images-v7\test\labels\detections.csv")

# Existing canonical image hashes
canonical_dir = REPO_ROOT / "data" / "exterior_canonical"
existing_hashes = set()
for p in canonical_dir.rglob("*.jpg"):
    with open(p, "rb") as f:
        existing_hashes.add(hashlib.md5(f.read()).hexdigest())

print(f"Existing canonical unique image hashes: {len(existing_hashes)}")

# Inspect validation detections for Porch
df_val = pd.read_csv(VAL_CSV)
porch_val = df_val[df_val["LabelName"] == "/m/04m6gz"].copy()
porch_val["area"] = (porch_val["XMax"] - porch_val["XMin"]) * (porch_val["YMax"] - porch_val["YMin"])
clean_val = porch_val[porch_val["area"] <= 0.85]
val_imgs = set(clean_val["ImageID"].unique())
print(f"Validation clean porch boxes: {len(clean_val)} across {len(val_imgs)} images")

# Inspect test detections for Porch
df_test = pd.read_csv(TEST_CSV)
porch_test = df_test[df_test["LabelName"] == "/m/04m6gz"].copy()
porch_test["area"] = (porch_test["XMax"] - porch_test["XMin"]) * (porch_test["YMax"] - porch_test["YMin"])
clean_test = porch_test[porch_test["area"] <= 0.85]
test_imgs = set(clean_test["ImageID"].unique())
print(f"Test clean porch boxes: {len(clean_test)} across {len(test_imgs)} images")

# Check overlap between val and test
overlap = val_imgs.intersection(test_imgs)
print(f"Overlap between val and test porch images: {len(overlap)}")
total_distinct = val_imgs.union(test_imgs)
print(f"Total distinct candidate porch images from Open Images val+test: {len(total_distinct)}")
