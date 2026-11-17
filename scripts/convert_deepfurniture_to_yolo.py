"""
Convert DeepFurniture Dataset to YOLO Detection Format for RoomAI.
High-speed sequential streaming conversion with reproducible 80/10/10 scene-level splitting.
Extracts RGB images, normalizes bounding boxes, and generates data.yaml, class_mapping.json,
and dataset_manifest.csv.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tarfile
import time
from collections import Counter, defaultdict

from PIL import Image
import yaml

CLASS_MAPPING = {
    1: {"yolo_id": 0, "name": "cabinet_shelf", "orig": "cabinet#shelf", "desc": "Storage, shelves, display cabinets"},
    2: {"yolo_id": 1, "name": "table", "orig": "table", "desc": "Dining tables, coffee tables, desks"},
    3: {"yolo_id": 2, "name": "chair_stool", "orig": "chair#stool", "desc": "Chairs, armchairs, stools, dining seating"},
    4: {"yolo_id": 3, "name": "lamp", "orig": "lamp", "desc": "Floor lamps, table lamps, pendant lamps"},
    5: {"yolo_id": 4, "name": "door", "orig": "door", "desc": "Interior doors and room entryways"},
    6: {"yolo_id": 5, "name": "bed", "orig": "bed", "desc": "Beds and headboards"},
    7: {"yolo_id": 6, "name": "sofa", "orig": "sofa", "desc": "Couches, sofas, sectional seating"},
    8: {"yolo_id": 7, "name": "plant", "orig": "plant", "desc": "Indoor potted plants and greenery"},
    9: {"yolo_id": 8, "name": "decoration", "orig": "decoration", "desc": "Wall art, mirrors, vases, decorative accessories"},
    10: {"yolo_id": 9, "name": "curtain", "orig": "curtain", "desc": "Drapes, blinds, window curtains"},
    11: {"yolo_id": 10, "name": "home_appliance", "orig": "home-appliance", "desc": "TV, refrigerator, oven, microwaves"},
}


def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def convert_dataset(source_dir: Path, output_dir: Path, seed: int = 42, max_scenes: int = None):
    print("=" * 60)
    print("Converting DeepFurniture to YOLO Format for RoomAI")
    print("=" * 60)
    t0 = time.time()

    # 1. Prepare target directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Save class_mapping.json
    mapping_path = output_dir / "class_mapping.json"
    clean_mapping = {
        str(info["yolo_id"]): {
            "yolo_id": info["yolo_id"],
            "name": info["name"],
            "original_id": orig_id,
            "original_name": info["orig"],
            "description": info["desc"],
            "retained": True,
        }
        for orig_id, info in CLASS_MAPPING.items()
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(clean_mapping, f, indent=2)
    print(f"[Created]: Class mapping saved to: {mapping_path}")

    # 3. Locate source archives
    tar_files = sorted(source_dir.glob("scenes_*.tar.gz"))
    if not tar_files:
        tar_files = sorted((source_dir / "data" / "scenes").glob("scenes_*.tar.gz"))
    if not tar_files:
        raise FileNotFoundError(f"No scene archives found in {source_dir}")

    # Pass 1: Quick index of scene IDs for deterministic split
    print("\nPass 1: Fast indexing scene IDs for reproducible splitting...")
    all_scene_ids = []
    for tar_path in tar_files:
        with tarfile.open(tar_path, "r|gz") as tar:
            for member in tar:
                if member.name.endswith("annotation.json"):
                    scene_id = member.name.replace("\\", "/").split("/")[0]
                    all_scene_ids.append((tar_path, scene_id))
                    if max_scenes and len(all_scene_ids) >= max_scenes:
                        break

    print(f"Total indexed scenes: {len(all_scene_ids)}")

    # Reproducible 80 / 10 / 10 scene-level partition
    rng = random.Random(seed)
    shuffled = list(all_scene_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_ids = set(s[1] for s in shuffled[:n_train])
    val_ids = set(s[1] for s in shuffled[n_train:n_train + n_val])
    test_ids = set(s[1] for s in shuffled[n_train + n_val:])

    # Strict isolation verification
    assert train_ids.isdisjoint(val_ids), "Data leakage between train and val!"
    assert train_ids.isdisjoint(test_ids), "Data leakage between train and test!"
    assert val_ids.isdisjoint(test_ids), "Data leakage between val and test!"

    scene_to_split = {}
    for sid in train_ids:
        scene_to_split[sid] = "train"
    for sid in val_ids:
        scene_to_split[sid] = "val"
    for sid in test_ids:
        scene_to_split[sid] = "test"

    print("\nDataset Partition Plan (Zero Leakage):")
    print(f"  train: {len(train_ids):4d} scenes ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  val:   {len(val_ids):4d} scenes ({len(val_ids)/n_total*100:.1f}%)")
    print(f"  test:  {len(test_ids):4d} scenes ({len(test_ids)/n_total*100:.1f}%)")

    # Pass 2: Sequential extraction and conversion
    print("\nPass 2: Streaming extraction, YOLO label conversion, and manifest generation...")
    manifest_rows = []
    split_object_counts = Counter()
    category_split_counts = defaultdict(Counter)
    converted_count = 0

    for tar_path in tar_files:
        current_scene_id = None
        current_img_bytes = None
        current_width = 0
        current_height = 0
        current_md5 = ""
        current_split = None

        with tarfile.open(tar_path, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                parts = member.name.replace("\\", "/").split("/")
                scene_id = parts[0]
                file_name = parts[-1]

                if scene_id not in scene_to_split:
                    continue

                split_name = scene_to_split[scene_id]

                if file_name == "image.jpg":
                    current_scene_id = scene_id
                    current_split = split_name
                    f = tar.extractfile(member)
                    current_img_bytes = f.read()
                    current_md5 = compute_md5(current_img_bytes)

                    # Get dimensions via Pillow
                    with Image.open(io.BytesIO(current_img_bytes)) as img:
                        current_width, current_height = img.size

                    # Write image to disk
                    out_img_path = output_dir / "images" / split_name / f"{scene_id}.jpg"
                    with open(out_img_path, "wb") as f_out:
                        f_out.write(current_img_bytes)

                elif file_name == "annotation.json":
                    if scene_id != current_scene_id:
                        continue

                    f = tar.extractfile(member)
                    ann_data = json.load(f)

                    instances = ann_data.get("instances", [])
                    yolo_lines = []
                    categories_in_scene = set()

                    for inst in instances:
                        cid = inst.get("categoryID")
                        if cid not in CLASS_MAPPING:
                            continue

                        yolo_info = CLASS_MAPPING[cid]
                        yolo_cid = yolo_info["yolo_id"]
                        cname = yolo_info["name"]

                        bbox = inst.get("boundingBox", {})
                        xmin = float(bbox.get("xMin", 0))
                        xmax = float(bbox.get("xMax", 0))
                        ymin = float(bbox.get("yMin", 0))
                        ymax = float(bbox.get("yMax", 0))

                        # Clip to pixel dimensions
                        xmin = max(0.0, min(float(current_width), xmin))
                        xmax = max(0.0, min(float(current_width), xmax))
                        ymin = max(0.0, min(float(current_height), ymin))
                        ymax = max(0.0, min(float(current_height), ymax))

                        bw = xmax - xmin
                        bh = ymax - ymin

                        if bw <= 0.0 or bh <= 0.0:
                            continue

                        # Normalized coordinates
                        x_center = ((xmin + xmax) / 2.0) / current_width
                        y_center = ((ymin + ymax) / 2.0) / current_height
                        norm_w = bw / current_width
                        norm_h = bh / current_height

                        # Clamp to [0, 1]
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        norm_w = max(1e-6, min(1.0, norm_w))
                        norm_h = max(1e-6, min(1.0, norm_h))

                        yolo_lines.append(f"{yolo_cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                        categories_in_scene.add(cname)
                        split_object_counts[current_split] += 1
                        category_split_counts[current_split][cname] += 1

                    # Write label file (empty .txt for negative samples)
                    out_lbl_path = output_dir / "labels" / current_split / f"{scene_id}.txt"
                    with open(out_lbl_path, "w", encoding="utf-8") as f_lbl:
                        if yolo_lines:
                            f_lbl.write("\n".join(yolo_lines) + "\n")

                    # Record in manifest
                    manifest_rows.append({
                        "scene_id": scene_id,
                        "split": current_split,
                        "image_file": f"images/{current_split}/{scene_id}.jpg",
                        "label_file": f"labels/{current_split}/{scene_id}.txt",
                        "num_objects": len(yolo_lines),
                        "categories_present": ";".join(sorted(categories_in_scene)),
                        "width": current_width,
                        "height": current_height,
                        "md5": current_md5,
                    })

                    converted_count += 1
                    if converted_count % 100 == 0 or converted_count == n_total:
                        print(f"  Processed {converted_count}/{n_total} scenes in {time.time()-t0:.1f}s...", flush=True)

    # 4. Save dataset_manifest.csv
    manifest_path = output_dir / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scene_id", "split", "image_file", "label_file", "num_objects",
            "categories_present", "width", "height", "md5"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\n[Created]: Dataset manifest saved to: {manifest_path}")

    # 5. Save data.yaml
    names_dict = {info["yolo_id"]: info["name"] for info in CLASS_MAPPING.values()}
    data_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_dict,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=False)
    print(f"[Created]: YOLO data.yaml saved to: {yaml_path}")

    # Summary Output
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total Scenes Converted:  {len(manifest_rows):,}")
    print(f"Total Objects Annotated: {sum(split_object_counts.values()):,}")
    for split_name in ["train", "val", "test"]:
        sc_cnt = len([r for r in manifest_rows if r["split"] == split_name])
        obj_cnt = split_object_counts[split_name]
        print(f"  {split_name:<5}: {sc_cnt:4d} scenes, {obj_cnt:5d} objects (avg {obj_cnt/max(1, sc_cnt):.1f}/scene)")

    print(f"Elapsed time: {time.time()-t0:.1f}s")
    return manifest_path, yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepFurniture to YOLO format.")
    parser.add_argument("--source-dir", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\source\DeepFurniture")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\processed\deepfurniture_yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    convert_dataset(
        Path(args.source_dir),
        Path(args.output_dir),
        seed=args.seed,
        max_scenes=args.max_scenes,
    )
