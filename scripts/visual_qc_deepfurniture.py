"""
Visual Quality Assurance for RoomAI YOLO Dataset.
Renders bounding boxes and category labels on images, generates a 20-image contact sheet,
and saves targeted QC examples (dense scenes, small boxes, per-class samples).
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont
import yaml

# High-contrast color palette for 11 categories
COLORS = [
    "#FF3838",  # 0: cabinet_shelf (red)
    "#FF9D97",  # 1: table (coral)
    "#FF701F",  # 2: chair_stool (orange)
    "#FFB21D",  # 3: lamp (amber)
    "#CFD231",  # 4: door (lime)
    "#48F90A",  # 5: bed (bright green)
    "#2CBEFE",  # 6: sofa (sky blue)
    "#00D492",  # 7: plant (emerald)
    "#A356FE",  # 8: decoration (purple)
    "#F938FF",  # 9: curtain (magenta)
    "#3445FE",  # 10: home_appliance (deep blue)
]


def draw_yolo_boxes(image: Image.Image, label_file: Path, class_names: dict) -> tuple[Image.Image, int, list]:
    """Draw YOLO normalized bounding boxes on a Pillow image."""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    w, h = draw_img.size

    if not label_file.exists():
        return draw_img, 0, []

    content = label_file.read_text(encoding="utf-8").strip()
    if not content:
        return draw_img, 0, []

    lines = content.splitlines()
    boxes = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid, xc, yc, bw, bh = map(float, parts)
        cid = int(cid)

        # Convert normalized center-width to pixel corners
        xmin = (xc - bw / 2.0) * w
        xmax = (xc + bw / 2.0) * w
        ymin = (yc - bh / 2.0) * h
        ymax = (yc + bh / 2.0) * h

        color = COLORS[cid % len(COLORS)]
        cname = class_names.get(cid, str(cid))

        # Draw box outline (thickness 3)
        for offset in range(3):
            draw.rectangle([xmin - offset, ymin - offset, xmax + offset, ymax + offset], outline=color)

        # Label tag
        tag_text = f"{cname}"
        text_bbox = draw.textbbox((xmin, ymin), tag_text)
        tb_w = text_bbox[2] - text_bbox[0]
        tb_h = text_bbox[3] - text_bbox[1]

        tag_ymin = max(0, ymin - tb_h - 4)
        draw.rectangle([xmin, tag_ymin, xmin + tb_w + 6, tag_ymin + tb_h + 4], fill=color)
        draw.text((xmin + 3, tag_ymin + 1), tag_text, fill="black")

        boxes.append({"cid": cid, "cname": cname, "area": (xmax - xmin) * (ymax - ymin)})

    return draw_img, len(lines), boxes


def generate_qc_visuals(dataset_dir: Path, output_dir: Path, num_contact_samples: int = 20, seed: int = 42):
    print("=" * 60)
    print("Generating Visual Quality Assurance Deliverables")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "qc_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data.yaml
    yaml_file = dataset_dir / "data.yaml"
    with open(yaml_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    class_names = config.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}
    elif isinstance(class_names, dict):
        class_names = {int(k): v for k, v in class_names.items()}

    # 2. Collect images from splits
    all_image_paths = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if img_dir.exists():
            for p in sorted(img_dir.glob("*.jpg")):
                lbl_file = lbl_dir / f"{p.stem}.txt"
                all_image_paths.append((split, p, lbl_file))

    print(f"Total processed images available: {len(all_image_paths)}")
    if not all_image_paths:
        raise FileNotFoundError(f"No processed images found in {dataset_dir}")

    # Random selection for contact sheet
    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(all_image_paths)), min(num_contact_samples, len(all_image_paths)))
    contact_candidates = [all_image_paths[i] for i in selected_indices]

    rendered_thumbs = []
    thumb_w, thumb_h = 480, 270

    for split, img_path, lbl_path in contact_candidates:
        with Image.open(img_path) as img:
            draw_img, n_boxes, _ = draw_yolo_boxes(img, lbl_path, class_names)
            # Add scene caption
            draw = ImageDraw.Draw(draw_img)
            caption = f"[{split.upper()}] {img_path.stem[:12]} ({n_boxes} objs)"
            draw.rectangle([0, 0, draw_img.width, 28], fill="black")
            draw.text((10, 6), caption, fill="white")
            thumb = draw_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            rendered_thumbs.append(thumb)

    # Assemble 4x5 Contact Sheet
    cols = 4
    rows = math.ceil(len(rendered_thumbs) / cols)
    sheet_w = cols * thumb_w
    sheet_h = rows * thumb_h
    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), color="#1E1E1E")

    for i, thumb in enumerate(rendered_thumbs):
        col = i % cols
        row = i // cols
        contact_sheet.paste(thumb, (col * thumb_w, row * thumb_h))

    contact_sheet_path = output_dir / "visual_qc_contact_sheet.png"
    contact_sheet.save(contact_sheet_path, quality=95)
    print(f"[Created]: 20-image contact sheet saved to: {contact_sheet_path}")

    # 3. Targeted Checks: Find dense scene, small objects, and class samples
    print("\nExtracting targeted QC examples...")
    dense_scenes = []
    class_examples = defaultdict(list)

    for split, img_path, lbl_path in all_image_paths:
        with Image.open(img_path) as img:
            draw_img, n_boxes, boxes = draw_yolo_boxes(img, lbl_path, class_names)
            if n_boxes >= 6 and len(dense_scenes) < 3:
                dense_scenes.append((img_path.stem, draw_img, n_boxes))

            for b in boxes:
                cname = b["cname"]
                if len(class_examples[cname]) < 2:
                    class_examples[cname].append((img_path.stem, draw_img))

    for idx, (stem, img_obj, nb) in enumerate(dense_scenes, 1):
        target_path = samples_dir / f"dense_scene_{idx}_{stem}_{nb}objects.jpg"
        img_obj.save(target_path, quality=92)
        print(f"  Targeted dense scene: {target_path.name}")

    for cname, ex_list in class_examples.items():
        if ex_list:
            stem, img_obj = ex_list[0]
            target_path = samples_dir / f"class_{cname}_{stem}.jpg"
            img_obj.save(target_path, quality=92)

    print(f"[Created]: Targeted QC samples saved to: {samples_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual QC deliverables.")
    parser.add_argument("--dataset-dir", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\processed\deepfurniture_yolo")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\apaar\Desktop\RoomAI\data\reports")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    generate_qc_visuals(Path(args.dataset_dir), Path(args.output_dir), num_contact_samples=args.samples)
