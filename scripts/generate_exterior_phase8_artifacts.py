"""Generates visual audit contacts and comparison charts for Phase 8 Exterior Audit.
"""

import os
import sys

conda_env = r"C:\Users\apaar\miniconda3\envs\tf-gpu"
for dll_dir in [os.path.join(conda_env, "Library", "bin"), os.path.join(conda_env, "Scripts")]:
    if os.path.exists(dll_dir):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "exterior_phase8"
REPORTS_DIR = REPO_ROOT / "reports"

CANONICAL_EXTERIOR_CLASSES = [
    "building", "facade", "window", "door", "roof_cornice",
    "balcony", "porch", "pillar_column", "shutter_blind", "molding_deco"
]


def create_porch_contact_sheet():
    """Assembles a 2x3 contact sheet of all 6 porch annotations across train, val, and test."""
    porch_dir = ARTIFACTS_DIR / "porch_audit"
    files = sorted(list(porch_dir.glob("porch_*.jpg")))
    if not files:
        print("No porch files found to assemble.")
        return

    # Resize each to 400x300
    w_box, h_box = 400, 300
    images = []
    titles = [
        "Train: oi_ext_train_0012 (Area: 2.29%)",
        "Val: oi_ext_val_0084 (Area: 18.36%)",
        "Val: oi_ext_val_0085 - Box 1 (Area: 6.13%)",
        "Val: oi_ext_val_0085 - Box 2 (Area: 2.64%)",
        "Test: oi_ext_test_0106 [ARTIFACT 100%]",
        "Test: oi_ext_test_0113 (Area: 7.80%)"
    ]

    for f in files[:6]:
        im = Image.open(f).convert("RGB")
        im_resized = im.resize((w_box, h_box), Image.Resampling.LANCZOS)
        images.append(im_resized)

    # Canvas: 2 rows x 3 cols + headers
    canvas_w = 3 * w_box + 40
    canvas_h = 2 * (h_box + 35) + 60
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 248))
    draw = ImageDraw.Draw(canvas)

    # Title header
    draw.rectangle([0, 0, canvas_w, 45], fill=(30, 41, 59))
    draw.text((20, 12), "RoomAI Phase 8 — Exhaustive Porch Annotation Audit (All 6 Instances in Repository)", fill=(255, 255, 255))

    for idx, (im, title) in enumerate(zip(images, titles)):
        row = idx // 3
        col = idx % 3
        x = 15 + col * (w_box + 10)
        y = 55 + row * (h_box + 35)

        # Draw card container
        draw.rectangle([x, y, x + w_box, y + h_box + 25], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        canvas.paste(im, (x, y))

        # Title background
        color = (220, 38, 38) if "ARTIFACT" in title else (15, 23, 42)
        draw.rectangle([x, y + h_box, x + w_box, y + h_box + 25], fill=(241, 245, 249))
        draw.text((x + 8, y + h_box + 5), title, fill=color)

    out_path = ARTIFACTS_DIR / "porch_complete_audit_gallery.jpg"
    canvas.save(out_path, quality=95)
    print(f"Porch gallery saved to {out_path}")


def create_b2_vs_b3_comparison_chart():
    """Creates a visual comparison chart of B2 vs B3 per-class mAP50."""
    with open(REPORTS_DIR / "exterior_phase8_data_audit.json", "r") as f:
        d = json.load(f)

    comp = d["b2_vs_b3_fair_comparison"]["per_class_comparison"]

    chart_w, chart_h = 900, 520
    canvas = Image.new("RGB", (chart_w, chart_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, chart_w, 55], fill=(15, 23, 42))
    draw.text((25, 16), "Exterior Detector Fair Evaluation: Baseline B2 vs Candidate B3 (RFS)", fill="white")

    # Legend
    draw.rectangle([chart_w - 260, 20, chart_w - 245, 35], fill=(59, 130, 246))
    draw.text((chart_w - 240, 20), "Model B2 (Standard)", fill="white")
    draw.rectangle([chart_w - 130, 20, chart_w - 115, 35], fill=(16, 185, 129))
    draw.text((chart_w - 110, 20), "Model B3 (RFS)", fill="white")

    # Bar chart area
    origin_x = 140
    origin_y = 440
    bar_area_w = 720
    bar_area_h = 360

    # Draw grid lines
    for pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = int(origin_y - pct * bar_area_h)
        draw.line([(origin_x, y), (origin_x + bar_area_w, y)], fill=(226, 232, 240), width=1)
        draw.text((origin_x - 45, y - 6), f"{int(pct*100)}%", fill=(100, 116, 139))

    num_classes = len(CANONICAL_EXTERIOR_CLASSES)
    group_width = bar_area_w / num_classes
    bar_width = int(group_width * 0.35)

    for i, cname in enumerate(CANONICAL_EXTERIOR_CLASSES):
        stats = comp.get(cname, {})
        b2_map = stats.get("b2_mAP50", 0.0)
        b3_map = stats.get("b3_mAP50", 0.0)

        gx = int(origin_x + i * group_width)
        x_b2 = gx + int(group_width * 0.12)
        x_b3 = x_b2 + bar_width + 4

        h_b2 = int(b2_map * bar_area_h)
        h_b3 = int(b3_map * bar_area_h)

        # Draw B2 bar (blue)
        draw.rectangle([x_b2, origin_y - h_b2, x_b2 + bar_width, origin_y], fill=(59, 130, 246))
        # Draw B3 bar (green)
        draw.rectangle([x_b3, origin_y - h_b3, x_b3 + bar_width, origin_y], fill=(16, 185, 129))

        # Class label
        draw.text((gx + 4, origin_y + 8), cname[:11], fill=(51, 65, 85))
        # Values above bars
        if b2_map > 0.05:
            draw.text((x_b2, origin_y - h_b2 - 14), f"{b2_map:.2f}", fill=(59, 130, 246))
        if b3_map > 0.05:
            draw.text((x_b3, origin_y - h_b3 - 14), f"{b3_map:.2f}", fill=(16, 185, 129))
        elif b3_map == 0.0:
            draw.text((x_b3 - 2, origin_y - 15), "0%", fill=(220, 38, 38))

    # Footer note
    draw.text((25, 485), "* Evaluated under standard Ultralytics COCO 101-point PR interpolation. Porch remains 0.00% across both models due to 1-sample starvation & test artifact.", fill=(100, 116, 139))

    out_chart = ARTIFACTS_DIR / "b2_vs_b3_per_class_comparison.png"
    canvas.save(out_chart)
    print(f"Comparison chart saved to {out_chart}")


if __name__ == "__main__":
    create_porch_contact_sheet()
    create_b2_vs_b3_comparison_chart()
