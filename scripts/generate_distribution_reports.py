"""
scripts/generate_distribution_reports.py
Generates machine-readable reports:
- reports/open_images_pilot_class_distribution.json
- reports/open_images_pilot_class_distribution.csv
Based on the verified manifest.json and Open Images V7 pilot ground truth.
"""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

# Paths
dataset_dir = Path("F:/RoomAI_Data/yolo_open_images_pilot")
manifest_path = dataset_dir / "manifest.json"
out_json_path = Path("F:/RoomAI/reports/open_images_pilot_class_distribution.json")
out_csv_path = Path("F:/RoomAI/reports/open_images_pilot_class_distribution.csv")

# Load manifest
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# Raw candidate classes metrics (20 classes) from comprehensive audit
RAW_CANDIDATE_CLASSES = {
    "Tree": {
        "open_images_id": "/m/07j7r",
        "total_boxes": 18499,
        "total_images": 4991,
        "mean_boxes_per_image": 3.71,
        "group_of_pct": 26.5,
        "occluded_pct": 76.8,
        "truncated_pct": 45.9,
        "depiction_pct": 2.3
    },
    "Window": {
        "open_images_id": "/m/0d4v4",
        "total_boxes": 9018,
        "total_images": 1017,
        "mean_boxes_per_image": 8.87,
        "group_of_pct": 3.2,
        "occluded_pct": 38.0,
        "truncated_pct": 11.8,
        "depiction_pct": 1.5
    },
    "Car": {
        "open_images_id": "/m/0k4j",
        "total_boxes": 4197,
        "total_images": 1539,
        "mean_boxes_per_image": 2.73,
        "group_of_pct": 5.7,
        "occluded_pct": 55.8,
        "truncated_pct": 32.9,
        "depiction_pct": 2.1
    },
    "Building": {
        "open_images_id": "/m/0cgh4",
        "total_boxes": 2997,
        "total_images": 1761,
        "mean_boxes_per_image": 1.70,
        "group_of_pct": 15.1,
        "occluded_pct": 70.3,
        "truncated_pct": 66.1,
        "depiction_pct": 2.1
    },
    "House": {
        "open_images_id": "/m/03jm5",
        "total_boxes": 2233,
        "total_images": 1010,
        "mean_boxes_per_image": 2.21,
        "group_of_pct": 3.7,
        "occluded_pct": 37.3,
        "truncated_pct": 20.6,
        "depiction_pct": 2.3
    },
    "Chair": {
        "open_images_id": "/m/01mzpv",
        "total_boxes": 2170,
        "total_images": 427,
        "mean_boxes_per_image": 5.08,
        "group_of_pct": 3.0,
        "occluded_pct": 86.9,
        "truncated_pct": 29.4,
        "depiction_pct": 0.7
    },
    "Table": {
        "open_images_id": "/m/04bcr3",
        "total_boxes": 1440,
        "total_images": 749,
        "mean_boxes_per_image": 1.92,
        "group_of_pct": 0.6,
        "occluded_pct": 72.5,
        "truncated_pct": 39.3,
        "depiction_pct": 0.1
    },
    "Bicycle": {
        "open_images_id": "/m/0199g",
        "total_boxes": 613,
        "total_images": 267,
        "mean_boxes_per_image": 2.30,
        "group_of_pct": 5.2,
        "occluded_pct": 82.9,
        "truncated_pct": 29.9,
        "depiction_pct": 2.9
    },
    "Door": {
        "open_images_id": "/m/02dgv",
        "total_boxes": 307,
        "total_images": 207,
        "mean_boxes_per_image": 1.48,
        "group_of_pct": 0.0,
        "occluded_pct": 49.8,
        "truncated_pct": 27.4,
        "depiction_pct": 1.6
    },
    "Motorcycle": {
        "open_images_id": "/m/04_sv",
        "total_boxes": 241,
        "total_images": 119,
        "mean_boxes_per_image": 2.03,
        "group_of_pct": 2.1,
        "occluded_pct": 51.5,
        "truncated_pct": 11.6,
        "depiction_pct": 0.0
    },
    "Bus": {
        "open_images_id": "/m/01bjv",
        "total_boxes": 212,
        "total_images": 135,
        "mean_boxes_per_image": 1.57,
        "group_of_pct": 0.9,
        "occluded_pct": 53.8,
        "truncated_pct": 36.3,
        "depiction_pct": 2.4
    },
    "Truck": {
        "open_images_id": "/m/07rnxq",
        "total_boxes": 187,
        "total_images": 129,
        "mean_boxes_per_image": 1.45,
        "group_of_pct": 0.5,
        "occluded_pct": 57.8,
        "truncated_pct": 32.1,
        "depiction_pct": 1.1
    },
    "Cabinetry": {
        "open_images_id": "/m/01s55n",
        "total_boxes": 120,
        "total_images": 40,
        "mean_boxes_per_image": 3.00,
        "group_of_pct": 5.0,
        "occluded_pct": 63.3,
        "truncated_pct": 33.3,
        "depiction_pct": 0.0
    },
    "Bench": {
        "open_images_id": "/m/01x3jk",
        "total_boxes": 118,
        "total_images": 62,
        "mean_boxes_per_image": 1.90,
        "group_of_pct": 1.7,
        "occluded_pct": 66.1,
        "truncated_pct": 23.7,
        "depiction_pct": 1.7
    },
    "Bookcase": {
        "open_images_id": "/m/03__z0",
        "total_boxes": 106,
        "total_images": 64,
        "mean_boxes_per_image": 1.66,
        "group_of_pct": 3.8,
        "occluded_pct": 63.2,
        "truncated_pct": 77.4,
        "depiction_pct": 0.9
    },
    "Stairs": {
        "open_images_id": "/m/01lynh",
        "total_boxes": 104,
        "total_images": 83,
        "mean_boxes_per_image": 1.25,
        "group_of_pct": 51.0,
        "occluded_pct": 58.6,
        "truncated_pct": 61.5,
        "depiction_pct": 1.9
    },
    "Bed": {
        "open_images_id": "/m/03ssj5",
        "total_boxes": 70,
        "total_images": 52,
        "mean_boxes_per_image": 1.35,
        "group_of_pct": 0.0,
        "occluded_pct": 67.1,
        "truncated_pct": 65.7,
        "depiction_pct": 2.9
    },
    "Couch": {
        "open_images_id": "/m/02crq1",
        "total_boxes": 64,
        "total_images": 55,
        "mean_boxes_per_image": 1.16,
        "group_of_pct": 0.0,
        "occluded_pct": 23.4,
        "truncated_pct": 23.4,
        "depiction_pct": 0.0
    },
    "Office building": {
        "open_images_id": "/m/021sj1",
        "total_boxes": 37,
        "total_images": 37,
        "mean_boxes_per_image": 1.00,
        "group_of_pct": 0.0,
        "occluded_pct": 24.3,
        "truncated_pct": 24.3,
        "depiction_pct": 0.0
    },
    "Kitchen & dining room table": {
        "open_images_id": "/m/0h8n5zk",
        "total_boxes": 26,
        "total_images": 23,
        "mean_boxes_per_image": 1.13,
        "group_of_pct": 0.0,
        "occluded_pct": 7.7,
        "truncated_pct": 0.0,
        "depiction_pct": 0.0
    }
}

# Harmonized 18-class taxonomy definitions
HARMONIZED_SOURCES = {
    0: ("Building", ["Building", "Office building", "Skyscraper"]),
    1: ("House", ["House"]),
    2: ("Door", ["Door"]),
    3: ("Window", ["Window"]),
    4: ("Stairs", ["Stairs"]),
    5: ("Chair", ["Chair"]),
    6: ("Couch", ["Couch", "Sofa bed", "Loveseat", "Studio couch"]),
    7: ("Table", ["Table", "Kitchen & dining room table", "Coffee table"]),
    8: ("Bed", ["Bed"]),
    9: ("Bookcase", ["Bookcase"]),
    10: ("Cabinetry", ["Cabinetry", "Cupboard", "Bathroom cabinet"]),
    11: ("Bench", ["Bench"]),
    12: ("Car", ["Car", "Van"]),
    13: ("Truck", ["Truck"]),
    14: ("Bus", ["Bus"]),
    15: ("Tree", ["Tree"]),
    16: ("Motorcycle", ["Motorcycle"]),
    17: ("Bicycle", ["Bicycle"]),
}

# Compute per-class statistics from manifest
class_boxes_split = {cid: {"train": 0, "val": 0, "test": 0, "total": 0} for cid in range(18)}
class_images_split = {cid: {"train": set(), "val": set(), "test": set(), "total": set()} for cid in range(18)}

cname_to_cid = {info[0]: cid for cid, info in HARMONIZED_SOURCES.items()}

for img_id, meta in manifest["images"].items():
    s = meta["split"]
    for cname, count in meta.get("class_box_counts", {}).items():
        if cname in cname_to_cid:
            cid = cname_to_cid[cname]
            class_boxes_split[cid][s] += count
            class_boxes_split[cid]["total"] += count
            class_images_split[cid][s].add(img_id)
            class_images_split[cid]["total"].add(img_id)

harmonized_data = {}
csv_rows = []

for cid in range(18):
    cname, sources = HARMONIZED_SOURCES[cid]
    tb = class_boxes_split[cid]["total"]
    tr_b = class_boxes_split[cid]["train"]
    va_b = class_boxes_split[cid]["val"]
    te_b = class_boxes_split[cid]["test"]

    ti = len(class_images_split[cid]["total"])
    tr_i = len(class_images_split[cid]["train"])
    va_i = len(class_images_split[cid]["val"])
    te_i = len(class_images_split[cid]["test"])

    mean_bpi = round(tb / ti, 2) if ti else 0.0

    harmonized_data[cname] = {
        "class_id": cid,
        "target_name": cname,
        "source_classes": sources,
        "total_boxes": tb,
        "train_boxes": tr_b,
        "val_boxes": va_b,
        "test_boxes": te_b,
        "total_images": ti,
        "train_images": tr_i,
        "val_images": va_i,
        "test_images": te_i,
        "mean_boxes_per_image": mean_bpi
    }

    # Attribute defaults from primary source or aggregated
    primary_source = sources[0]
    raw_info = RAW_CANDIDATE_CLASSES.get(primary_source, {})
    group_of_pct = raw_info.get("group_of_pct", 0.0)
    occluded_pct = raw_info.get("occluded_pct", 0.0)
    truncated_pct = raw_info.get("truncated_pct", 0.0)
    depiction_pct = raw_info.get("depiction_pct", 0.0)

    csv_rows.append({
        "class_id": cid,
        "class_name": cname,
        "source_classes": ";".join(sources),
        "total_boxes": tb,
        "train_boxes": tr_b,
        "val_boxes": va_b,
        "test_boxes": te_b,
        "total_images": ti,
        "train_images": tr_i,
        "val_images": va_i,
        "test_images": te_i,
        "mean_boxes_per_image": mean_bpi,
        "group_of_pct": group_of_pct,
        "occluded_pct": occluded_pct,
        "truncated_pct": truncated_pct,
        "depiction_pct": depiction_pct
    })

# Full JSON structure
report_json = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "dataset_metadata": {
        "dataset_name": "roomai_open_images_pilot",
        "source_split": "train",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images": 10000,
        "total_raw_boxes": 92953,
        "num_candidate_classes": 20,
        "num_harmonized_classes": 18,
        "seed": 42
    },
    "splits": {
        "train": {
            "image_count": manifest["split_statistics"]["train"]["images"],
            "label_count": manifest["split_statistics"]["train"]["labels"],
            "box_count": manifest["split_statistics"]["train"]["boxes"]
        },
        "val": {
            "image_count": manifest["split_statistics"]["val"]["images"],
            "label_count": manifest["split_statistics"]["val"]["labels"],
            "box_count": manifest["split_statistics"]["val"]["boxes"]
        },
        "test": {
            "image_count": manifest["split_statistics"]["test"]["images"],
            "label_count": manifest["split_statistics"]["test"]["labels"],
            "box_count": manifest["split_statistics"]["test"]["boxes"]
        }
    },
    "leakage_verification": {
        "is_disjoint": True,
        "image_id_leak_count": 0,
        "md5_leak_count": 0
    },
    "corpus_health": {
        "corrupted_images": 0,
        "zero_byte_files": 0,
        "duplicate_md5_pairs": 0,
        "color_modes": {
            "RGB": 9668,
            "L": 329,
            "CMYK": 3
        },
        "containers": {
            "JPEG": 9993,
            "MPO": 7
        }
    },
    "dimension_summary": {
        "width": {"min": 443, "max": 4896, "median": 1024, "mean": 975.8, "std": 184.21},
        "height": {"min": 270, "max": 4608, "median": 768, "mean": 777.37, "std": 172.45},
        "aspect_ratio": {"min": 0.4326, "max": 3.7926, "median": 1.3333, "mean": 1.2845, "std": 0.2814}
    },
    "raw_candidate_classes": RAW_CANDIDATE_CLASSES,
    "harmonized_classes": harmonized_data
}

# Write JSON
out_json_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(report_json, f, indent=2)
print(f"Generated: {out_json_path}")

# Write CSV
fieldnames = [
    "class_id", "class_name", "source_classes",
    "total_boxes", "train_boxes", "val_boxes", "test_boxes",
    "total_images", "train_images", "val_images", "test_images",
    "mean_boxes_per_image", "group_of_pct", "occluded_pct", "truncated_pct", "depiction_pct"
]
with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Generated: {out_csv_path}")
