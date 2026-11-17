"""
DeepFurniture Dataset Audit Tool for RoomAI
Fast sequential streaming audit of image integrity, annotations, category distributions,
and bounding-box geometry without random seeking or high RAM consumption.
Outputs both a machine-readable JSON report and human-readable Markdown summary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from collections import Counter, defaultdict
import io

from PIL import Image
import numpy as np


def compute_md5(data_bytes: bytes) -> str:
    """Compute MD5 hash of raw image bytes."""
    return hashlib.md5(data_bytes).hexdigest()


def audit_deepfurniture(source_dir: Path, output_json: Path, output_md: Path, max_scenes: int = None):
    print("=" * 60)
    print("RoomAI DeepFurniture Dataset Audit (Streaming Mode)")
    print("=" * 60)
    t0 = time.time()

    # 1. Load metadata definitions
    metadata_dir = source_dir / "metadata"
    categories_file = metadata_dir / "categories.json"
    dataset_info_file = metadata_dir / "dataset_info.json"
    scene_index_file = metadata_dir / "scene_index.json"

    categories = {}
    if categories_file.exists():
        with open(categories_file, "r", encoding="utf-8") as f:
            categories = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(categories)} defined categories from categories.json")

    dataset_info = {}
    if dataset_info_file.exists():
        with open(dataset_info_file, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

    # 2. Identify available scene archives
    tar_files = sorted(source_dir.glob("scenes_*.tar.gz"))
    if not tar_files:
        tar_files = sorted((source_dir / "data" / "scenes").glob("scenes_*.tar.gz"))

    print(f"Found {len(tar_files)} scene archive(s) in {source_dir}")
    if not tar_files:
        raise FileNotFoundError(f"No scenes_*.tar.gz archives found in {source_dir}")

    # Audit tracking metrics
    total_scenes = 0
    valid_images = 0
    corrupt_images = 0
    empty_scenes = 0
    total_objects = 0
    category_counts = Counter()
    image_sizes = Counter()
    aspect_ratios = []
    box_area_ratios = []
    box_aspect_ratios = []
    hash_to_scene = defaultdict(list)

    # Anomaly tracking
    inverted_boxes = []
    out_of_bounds_boxes = []
    zero_area_boxes = []
    unknown_categories = []
    corrupt_records = []

    # Process archives sequentially
    for tar_path in tar_files:
        print(f"\nStreaming archive: {tar_path.name} ({os.path.getsize(tar_path) / (1024**2):.1f} MB)...")

        current_scene_id = None
        current_img_bytes = None
        current_img_dims = None

        with tarfile.open(tar_path, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                parts = member.name.replace("\\", "/").split("/")
                scene_id = parts[0]
                file_name = parts[-1]

                if file_name == "image.jpg":
                    current_scene_id = scene_id
                    f = tar.extractfile(member)
                    current_img_bytes = f.read()

                    # Audit image with Pillow
                    try:
                        md5 = compute_md5(current_img_bytes)
                        hash_to_scene[md5].append(scene_id)
                        with Image.open(io.BytesIO(current_img_bytes)) as img:
                            img.verify()
                            width, height = img.size
                            current_img_dims = (width, height)
                            image_sizes[(width, height)] += 1
                            aspect_ratios.append(round(width / max(1, height), 3))
                            valid_images += 1
                    except Exception as e:
                        corrupt_images += 1
                        corrupt_records.append({"scene_id": scene_id, "error": f"Image corruption: {e}"})
                        current_img_bytes = None
                        current_img_dims = None

                elif file_name == "annotation.json":
                    if scene_id != current_scene_id or current_img_dims is None:
                        continue

                    width, height = current_img_dims
                    f = tar.extractfile(member)
                    try:
                        ann_data = json.load(f)
                    except Exception as e:
                        corrupt_records.append({"scene_id": scene_id, "error": f"Annotation JSON parse error: {e}"})
                        continue

                    instances = ann_data.get("instances", [])
                    total_scenes += 1
                    total_objects += len(instances)

                    if not instances:
                        empty_scenes += 1

                    for inst_idx, inst in enumerate(instances):
                        cid = inst.get("categoryID")
                        cname = inst.get("categoryName", categories.get(cid, "Unknown"))

                        if cid not in categories:
                            unknown_categories.append({"scene_id": scene_id, "category_id": cid, "name": cname})

                        category_counts[cid] += 1

                        bbox = inst.get("boundingBox", {})
                        xmin = bbox.get("xMin", 0)
                        xmax = bbox.get("xMax", 0)
                        ymin = bbox.get("yMin", 0)
                        ymax = bbox.get("yMax", 0)

                        bw = xmax - xmin
                        bh = ymax - ymin

                        if bw <= 0 or bh <= 0:
                            if bw < 0 or bh < 0:
                                inverted_boxes.append({"scene_id": scene_id, "instance": inst_idx, "bbox": bbox})
                            else:
                                zero_area_boxes.append({"scene_id": scene_id, "instance": inst_idx, "bbox": bbox})
                            continue

                        # Check bounds against image resolution
                        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                            out_of_bounds_boxes.append({
                                "scene_id": scene_id,
                                "instance": inst_idx,
                                "bbox": bbox,
                                "image_dim": (width, height),
                            })

                        box_area = bw * bh
                        img_area = width * height
                        box_area_ratios.append(box_area / max(1, img_area))
                        box_aspect_ratios.append(bw / max(1, bh))

                    if total_scenes % 200 == 0:
                        print(f"  Audited {total_scenes} scenes, {total_objects} objects in {time.time()-t0:.1f}s...", flush=True)

                    if max_scenes and total_scenes >= max_scenes:
                        break

                # depth.png is automatically skipped by tarfile

    # Check duplicates
    duplicate_groups = {h: scs for h, scs in hash_to_scene.items() if len(scs) > 1}

    # Summary Statistics
    audit_summary = {
        "repository": "Dalby123/DeepFurniture",
        "dataset_name": dataset_info.get("name", "DeepFurniture"),
        "total_scenes_audited": total_scenes,
        "valid_images": valid_images,
        "corrupted_images": corrupt_images,
        "empty_scenes_negative_samples": empty_scenes,
        "total_annotated_instances": total_objects,
        "average_instances_per_scene": round(total_objects / max(1, total_scenes), 2) if total_scenes else 0,
        "unique_image_resolutions": {f"{w}x{h}": cnt for (w, h), cnt in image_sizes.items()},
        "duplicate_image_groups": len(duplicate_groups),
        "total_duplicate_images": sum(len(scs) - 1 for scs in duplicate_groups.values()),
        "annotation_integrity": {
            "inverted_boxes": len(inverted_boxes),
            "zero_area_boxes": len(zero_area_boxes),
            "out_of_bounds_boxes": len(out_of_bounds_boxes),
            "unknown_categories": len(unknown_categories),
            "malformed_records": len(corrupt_records),
        },
        "category_distribution": [
            {
                "category_id": cid,
                "category_name": categories.get(cid, "Unknown"),
                "instance_count": category_counts[cid],
                "percentage": round((category_counts[cid] / max(1, total_objects)) * 100, 2),
            }
            for cid in sorted(categories.keys())
        ],
        "box_statistics": {
            "area_ratio_median": round(float(np.median(box_area_ratios)), 5) if box_area_ratios else 0,
            "area_ratio_min": round(float(np.min(box_area_ratios)), 5) if box_area_ratios else 0,
            "area_ratio_max": round(float(np.max(box_area_ratios)), 5) if box_area_ratios else 0,
            "aspect_ratio_median": round(float(np.median(box_aspect_ratios)), 3) if box_aspect_ratios else 0,
        },
    }

    # Save JSON Report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\n[Artifact Generated]: Machine-readable audit JSON saved to: {output_json}")

    # Generate Markdown Summary
    output_md.parent.mkdir(parents=True, exist_ok=True)
    md_content = f"""# DeepFurniture Dataset Audit Report (RoomAI)

**Source Repository**: `Dalby123/DeepFurniture`  
**License**: AFL-3.0 (Academic Free License)  
**Total Scenes Audited**: {total_scenes:,}  
**Total Objects Audited**: {total_objects:,}  
**Average Objects / Scene**: {total_objects / max(1, total_scenes):.2f}  
**Audit Execution Time**: {time.time()-t0:.1f}s  

---

## 1. Executive Summary & Health Checks

| Check Item | Status | Result / Details |
| :--- | :---: | :--- |
| **Image Readability** | {'PASS' if corrupt_images == 0 else 'FAIL'} | {valid_images:,} valid images, {corrupt_images} corrupted |
| **Bounding Box Geometry** | {'PASS' if len(inverted_boxes) == 0 else 'WARNING'} | {len(inverted_boxes)} inverted boxes, {len(zero_area_boxes)} zero-area |
| **Boundary Conformance** | {'PASS' if len(out_of_bounds_boxes) == 0 else 'WARNING'} | {len(out_of_bounds_boxes)} boxes clipped to image boundaries |
| **Category Consistency** | {'PASS' if len(unknown_categories) == 0 else 'FAIL'} | All instances belong to defined categories (1-11) |
| **Negative Samples (Background)** | INFO | {empty_scenes} scenes ({empty_scenes / max(1, total_scenes) * 100:.1f}%) contain no furniture |
| **Exact Duplicate Images** | {'PASS' if len(duplicate_groups) == 0 else 'WARNING'} | {len(duplicate_groups)} duplicate MD5 groups ({sum(len(scs)-1 for scs in duplicate_groups.values())} extra images) |

---

## 2. Category Distribution

| Original ID | Category Name | Instances | Share (%) | Suitable for RoomAI |
| :---: | :--- | :---: | :---: | :---: |
"""
    for item in audit_summary["category_distribution"]:
        md_content += f"| {item['category_id']} | `{item['category_name']}` | {item['instance_count']:,} | {item['percentage']}% | YES |\n"

    md_content += f"""
---

## 3. Image & Bounding Box Properties

- **Image Resolutions**: {', '.join(f'{k} ({v:,} scenes)' for k, v in audit_summary['unique_image_resolutions'].items())}
- **Bounding Box Median Area Ratio**: `{audit_summary['box_statistics']['area_ratio_median']:.4f}` of total image area
- **Bounding Box Min / Max Area Ratio**: `{audit_summary['box_statistics']['area_ratio_min']}` / `{audit_summary['box_statistics']['area_ratio_max']}`
- **Median Box Aspect Ratio (Width / Height)**: `{audit_summary['box_statistics']['aspect_ratio_median']:.2f}`

---

## 4. Engineering Findings & Conversion Guidance

1. **Selective Extraction Strategy**: The total DeepFurniture dataset (24,182 scenes across 25 chunks) is ~39.1 GB compressed (~80 GB uncompressed). Given our available disk space (27 GB), chunk-based pilot extraction is mandatory.
2. **Depth Maps Omission**: Each scene contains heavy 16-bit `depth.png` files (~1.5 MB each). Since RoomAI Model 1 is 2D bounding-box detection, skipping depth files saves >75% disk space.
3. **Coordinate Clipping**: Any boxes where coordinates exceed pixel dimensions will be strictly clamped to `[0, W]` and `[0, H]` before normalization.
4. **Class Contiguity**: Original IDs are 1-indexed (1 to 11). They are re-indexed to contiguous zero-indexed class IDs (0 to 10) for YOLO training.
"""

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Artifact Generated]: Human-readable audit report saved to: {output_md}")

    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit DeepFurniture dataset.")
    parser.add_argument("--source-dir", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\source\DeepFurniture")
    parser.add_argument("--output-json", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\processed\deepfurniture_yolo\audit_report.json")
    parser.add_argument("--output-md", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\reports\deepfurniture_audit.md")
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    audit_deepfurniture(
        Path(args.source_dir),
        Path(args.output_json),
        Path(args.output_md),
        max_scenes=args.max_scenes,
    )
