"""Download, parse, and prepare Dalby123/DeepFurniture dataset for YOLO training and Hugging Face Dataset format."""

import io
import json
import os
import shutil
import ssl
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple
import certifi
from PIL import Image

# 1. Patch SSL certificate loading for Windows Python 3.10
orig_load = ssl.SSLContext.load_default_certs
def safe_load(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        orig_load(self, purpose)
    except Exception:
        self.load_verify_locations(cafile=certifi.where())
ssl.SSLContext.load_default_certs = safe_load

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

from huggingface_hub import hf_hub_download

# Define paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "deepfurniture"
YAML_PATH = DATA_DIR / "deepfurniture.yaml"

# DeepFurniture 11 classes (1-indexed in raw annotations -> 0-indexed for YOLO)
CATEGORY_NAMES = {
    1: "cabinet#shelf",
    2: "table",
    3: "chair#stool",
    4: "lamp",
    5: "door",
    6: "bed",
    7: "sofa",
    8: "plant",
    9: "decoration",
    10: "curtain",
    11: "home-appliance",
}


def download_deepfurniture_assets() -> Tuple[Path, Dict[int, str]]:
    """Downloads categories.json and scenes_0000.tar.gz from Hugging Face."""
    print("Fetching categories.json...")
    cat_path = hf_hub_download(
        repo_id="Dalby123/DeepFurniture",
        filename="data/metadata/categories.json",
        repo_type="dataset",
    )
    with open(cat_path, "r", encoding="utf-8") as f:
        raw_cats = json.load(f)
    categories = {int(k): v for k, v in raw_cats.items()}

    print("Fetching scenes_0000.tar.gz (cached if already downloaded)...")
    tar_path = hf_hub_download(
        repo_id="Dalby123/DeepFurniture",
        filename="data/scenes/scenes_0000.tar.gz",
        repo_type="dataset",
    )
    return Path(tar_path), categories


def prepare_yolo_dataset(tar_path: Path, categories: Dict[int, str]) -> Dict[str, int]:
    """Extracts scenes from tar, converts bboxes to YOLO format, and splits into train/val/test."""
    print(f"\nProcessing archive: {tar_path}")

    # Prepare output directories
    for split in ["train", "val", "test"]:
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    scenes: Dict[str, Dict[str, tarfile.TarInfo]] = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        print("Indexing files in archive...")
        for member in tar.getmembers():
            parts = member.name.split("/")
            if len(parts) >= 2:
                scene_id = parts[0]
                fname = parts[1]
                if scene_id not in scenes:
                    scenes[scene_id] = {}
                if fname in ("image.jpg", "annotation.json"):
                    scenes[scene_id][fname] = member

        scene_ids = sorted(list(scenes.keys()))
        total_scenes = len(scene_ids)
        print(f"Total valid scenes discovered: {total_scenes}")

        # Deterministic 70 / 15 / 15 split
        n_train = int(total_scenes * 0.70)
        n_val = int(total_scenes * 0.15)
        train_ids = set(scene_ids[:n_train])
        val_ids = set(scene_ids[n_train : n_train + n_val])
        test_ids = set(scene_ids[n_train + n_val :])

        print(f"Split distribution -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

        counts = {"train": 0, "val": 0, "test": 0, "total_instances": 0}
        class_instance_counts = {cls_name: 0 for cls_name in CATEGORY_NAMES.values()}

        for idx, scene_id in enumerate(scene_ids):
            files = scenes[scene_id]
            if "image.jpg" not in files or "annotation.json" not in files:
                continue

            if scene_id in train_ids:
                split = "train"
            elif scene_id in val_ids:
                split = "val"
            else:
                split = "test"

            # 1. Extract image bytes and get resolution
            img_file = tar.extractfile(files["image.jpg"])
            img_bytes = img_file.read()
            pil_img = Image.open(io.BytesIO(img_bytes))
            w, h = pil_img.size

            # Save image
            img_dest = DATA_DIR / "images" / split / f"{scene_id}.jpg"
            with open(img_dest, "wb") as f_out:
                f_out.write(img_bytes)

            # 2. Extract and parse annotation
            ann_file = tar.extractfile(files["annotation.json"])
            ann = json.load(ann_file)

            yolo_lines = []
            for inst in ann.get("instances", []):
                cat_id = inst.get("categoryID")
                if cat_id not in CATEGORY_NAMES:
                    continue

                cls_idx = cat_id - 1  # 0 to 10
                cls_name = CATEGORY_NAMES[cat_id]
                bbox = inst.get("boundingBox", {})
                x_min = float(bbox.get("xMin", 0))
                x_max = float(bbox.get("xMax", 0))
                y_min = float(bbox.get("yMin", 0))
                y_max = float(bbox.get("yMax", 0))

                # Sanity check
                if x_max <= x_min or y_max <= y_min or w <= 0 or h <= 0:
                    continue

                # Normalize to [0, 1]
                x_center = ((x_min + x_max) / 2.0) / w
                y_center = ((y_min + y_max) / 2.0) / h
                box_w = (x_max - x_min) / w
                box_h = (y_max - y_min) / h

                # Clamp bounds
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                box_w = max(0.001, min(1.0, box_w))
                box_h = max(0.001, min(1.0, box_h))

                yolo_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
                class_instance_counts[cls_name] += 1
                counts["total_instances"] += 1

            # Write YOLO label file
            label_dest = DATA_DIR / "labels" / split / f"{scene_id}.txt"
            with open(label_dest, "w", encoding="utf-8") as f_lbl:
                f_lbl.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            counts[split] += 1
            if (idx + 1) % 200 == 0 or (idx + 1) == total_scenes:
                print(f"Processed {idx + 1}/{total_scenes} scenes...")

    # Write deepfurniture.yaml
    yaml_content = f"""# Ultralytics YOLO configuration for DeepFurniture dataset
path: {DATA_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for i in range(11):
        yaml_content += f"  {i}: {CATEGORY_NAMES[i + 1]}\n"

    with open(YAML_PATH, "w", encoding="utf-8") as f_yaml:
        f_yaml.write(yaml_content)

    print(f"\nWrote YOLO configuration file to: {YAML_PATH}")
    print("\nExtraction Summary:")
    print(f"  Train images: {counts['train']}")
    print(f"  Val images:   {counts['val']}")
    print(f"  Test images:  {counts['test']}")
    print(f"  Total instances: {counts['total_instances']}")
    print("Instances per class:")
    for k, v in class_instance_counts.items():
        print(f"  {k:16s}: {v}")

    return counts


if __name__ == "__main__":
    tar_path, categories = download_deepfurniture_assets()
    prepare_yolo_dataset(tar_path, categories)
