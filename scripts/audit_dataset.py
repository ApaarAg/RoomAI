"""Audit script for val2017 dataset and COCO annotations."""

import json
import os
from collections import Counter

val_dir = r"c:\Users\apaar\Desktop\RoomAI\val2017"
val_images = set(os.listdir(val_dir))
print(f"val2017 directory: {val_dir}")
print(f"val2017 image count: {len(val_images)}")

ann_path = r"c:\Users\apaar\fiftyone\coco-2017\raw\instances_val2017.json"
train_ann_path = r"c:\Users\apaar\fiftyone\coco-2017\raw\instances_train2017.json"
train_img_dir = r"c:\Users\apaar\fiftyone\coco-2017\train\data"

print(f"\nChecking validation annotations at: {ann_path}")
if os.path.exists(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco_val = json.load(f)

    images = coco_val["images"]
    annotations = coco_val["annotations"]
    categories = coco_val["categories"]

    print(f"Total images in instances_val2017.json: {len(images)}")
    print(f"Total annotations in instances_val2017.json: {len(annotations)}")
    print(f"Total categories: {len(categories)}")

    # Check filename matching
    json_filenames = {img["file_name"]: img["id"] for img in images}
    matched = val_images.intersection(set(json_filenames.keys()))
    print(f"Filename matches with val2017: {len(matched)} / {len(val_images)}")

    cat_map = {c["id"]: c for c in categories}

    # Count occurrences of furniture categories in val2017
    furniture_supercats = {"furniture", "indoor", "appliance"}
    furniture_names = {
        "chair", "couch", "bed", "dining table", "toilet",
        "tv", "potted plant", "vase", "clock", "refrigerator",
        "microwave", "oven", "sink", "book"
    }

    relevant_cats = {
        cid: c["name"]
        for cid, c in cat_map.items()
        if c.get("supercategory") in furniture_supercats or c["name"] in furniture_names
    }

    print("\nRelevant Furniture / Room categories in COCO:")
    for cid, cname in sorted(relevant_cats.items(), key=lambda x: x[1]):
        scat = cat_map[cid].get("supercategory", "other")
        print(f"  ID {cid:2d}: {cname:<16} (supercategory: {scat})")

    # Annotations distribution in val2017
    cat_counts = Counter()
    images_with_cat = {cid: set() for cid in relevant_cats}

    for ann in annotations:
        cid = ann["category_id"]
        if cid in relevant_cats:
            cat_counts[cid] += 1
            images_with_cat[cid].add(ann["image_id"])

    print("\nInstances and Images in val2017:")
    for cid, cname in sorted(relevant_cats.items(), key=lambda x: cat_counts[x[0]], reverse=True):
        print(f"  {cname:<16}: {cat_counts[cid]:4d} instances across {len(images_with_cat[cid]):4d} images")

print(f"\nChecking training data at: {train_img_dir}")
if os.path.exists(train_img_dir):
    train_files = os.listdir(train_img_dir)
    print(f"Found {len(train_files)} images in {train_img_dir}")
if os.path.exists(train_ann_path):
    print(f"Found training annotations at: {train_ann_path} ({os.path.getsize(train_ann_path)/(1024**2):.1f} MB)")
