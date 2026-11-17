"""Visual Prediction Audit on representative scenes from untouched test split."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = REPO_ROOT / "data" / "deepfurniture" / "images" / "test"
TEST_LBL_DIR = REPO_ROOT / "data" / "deepfurniture" / "labels" / "test"
PRED_DIR = REPO_ROOT / "runs" / "predictions" / "deepfurniture_yolo11n_finetune_test"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "deepfurniture_yolo11n_finetune_best.pt"

CLASS_NAMES = {
    0: "cabinet#shelf",
    1: "table",
    2: "chair#stool",
    3: "lamp",
    4: "door",
    5: "bed",
    6: "sofa",
    7: "plant",
    8: "decoration",
    9: "curtain",
    10: "home-appliance",
}

# Color palette for 11 classes
COLORS = [
    (230, 25, 75),   # cabinet#shelf: red
    (60, 180, 75),   # table: green
    (255, 225, 25),  # chair#stool: yellow
    (0, 130, 200),   # lamp: blue
    (245, 130, 48),  # door: orange
    (145, 30, 180),  # bed: purple
    (70, 240, 240),  # sofa: cyan
    (240, 50, 230),  # plant: magenta
    (210, 245, 60),  # decoration: lime
    (250, 190, 212), # curtain: pink
    (0, 128, 128),   # home-appliance: teal
]


def load_ground_truth(stem: str, img_w: int, img_h: int):
    """Loads ground truth bounding boxes and class names."""
    lbl_file = TEST_LBL_DIR / f"{stem}.txt"
    gt_boxes = []
    if lbl_file.exists():
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw / 2.0) * img_w)
                    y1 = int((yc - bh / 2.0) * img_h)
                    x2 = int((xc + bw / 2.0) * img_w)
                    y2 = int((yc + bh / 2.0) * img_h)
                    gt_boxes.append({
                        "class_id": cls_id,
                        "label": CLASS_NAMES.get(cls_id, str(cls_id)),
                        "box": [x1, y1, x2, y2],
                    })
    return gt_boxes


def run_visual_audit():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print(" TASK B: VISUAL PREDICTION AUDIT ON UNTOUCHED TEST SPLIT")
    print("=" * 65)

    model = YOLO(str(CHECKPOINT_PATH))

    # Categorize test scenes by content
    test_stems = [p.stem for p in sorted(TEST_IMG_DIR.glob("*.jpg"))]
    scene_profiles = []

    for stem in test_stems:
        lbl_file = TEST_LBL_DIR / f"{stem}.txt"
        classes_present = []
        if lbl_file.exists():
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        classes_present.append(int(parts[0]))
        scene_profiles.append({
            "stem": stem,
            "classes": classes_present,
            "count": len(classes_present),
            "has_sofa": 6 in classes_present,
            "has_chair": 2 in classes_present,
            "has_appliance": 10 in classes_present,
            "has_bed": 5 in classes_present,
        })

    # Pick 8 diverse representative scenes:
    # 1-2: Crowded multi-furniture scenes
    # 3-4: Living room with sofas
    # 5-6: Dining / study rooms with chairs/stools
    # 7: Kitchen / home-appliance
    # 8: Bedroom scene
    selected_stems = []

    # Crowded (high count)
    crowded = sorted([s for s in scene_profiles if s["count"] >= 10], key=lambda x: -x["count"])
    if crowded:
        selected_stems.extend([s["stem"] for s in crowded[:2]])

    # Living room with sofa
    sofa_scenes = [s for s in scene_profiles if s["has_sofa"] and s["stem"] not in selected_stems]
    if sofa_scenes:
        selected_stems.extend([s["stem"] for s in sofa_scenes[:2]])

    # Rooms with chairs/stools
    chair_scenes = [s for s in scene_profiles if s["has_chair"] and s["stem"] not in selected_stems]
    if chair_scenes:
        selected_stems.extend([s["stem"] for s in chair_scenes[:2]])

    # Kitchen / home appliance
    app_scenes = [s for s in scene_profiles if s["has_appliance"] and s["stem"] not in selected_stems]
    if app_scenes:
        selected_stems.append(app_scenes[0]["stem"])

    # Bedroom
    bed_scenes = [s for s in scene_profiles if s["has_bed"] and s["stem"] not in selected_stems]
    if bed_scenes:
        selected_stems.append(bed_scenes[0]["stem"])

    # Fallback to make at least 8
    for s in scene_profiles:
        if len(selected_stems) >= 8:
            break
        if s["stem"] not in selected_stems:
            selected_stems.append(s["stem"])

    print(f"Selected {len(selected_stems)} representative test scenes:")
    for idx, stem in enumerate(selected_stems, 1):
        print(f"  {idx}. {stem}.jpg")

    audit_summary = []

    for idx, stem in enumerate(selected_stems, 1):
        img_path = TEST_IMG_DIR / f"{stem}.jpg"
        orig_img = Image.open(img_path).convert("RGB")
        w, h = orig_img.size

        gt_items = load_ground_truth(stem, w, h)

        # Run model prediction
        results = model.predict(
            source=orig_img,
            imgsz=640,
            conf=0.30,
            iou=0.45,
            device="0" if torch.cuda.is_available() else "cpu",
            half=False,
            verbose=False,
        )[0]

        pred_items = []
        for box in results.boxes:
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            pred_items.append({
                "class_id": cid,
                "label": CLASS_NAMES.get(cid, str(cid)),
                "confidence": round(conf, 3),
                "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
            })

        # Draw Side-by-Side: Ground Truth (Left) vs Model Prediction (Right)
        canvas = Image.new("RGB", (w * 2, h))
        gt_draw_img = orig_img.copy()
        pred_draw_img = orig_img.copy()

        d_gt = ImageDraw.Draw(gt_draw_img)
        d_pred = ImageDraw.Draw(pred_draw_img)

        # Draw Ground Truth
        for item in gt_items:
            cid = item["class_id"]
            col = COLORS[cid % len(COLORS)]
            x1, y1, x2, y2 = item["box"]
            d_gt.rectangle([x1, y1, x2, y2], outline=col, width=4)
            d_gt.text((x1 + 4, max(0, y1 - 18)), f"GT: {item['label']}", fill=col)

        # Draw Predictions
        for item in pred_items:
            cid = item["class_id"]
            col = COLORS[cid % len(COLORS)]
            x1, y1, x2, y2 = item["box"]
            d_pred.rectangle([x1, y1, x2, y2], outline=col, width=4)
            d_pred.text((x1 + 4, max(0, y1 - 18)), f"{item['label']} {item['confidence']:.2f}", fill=col)

        # Combine onto canvas with headers
        d_gt.text((20, 20), f"[GROUND TRUTH] Objects: {len(gt_items)}", fill=(255, 255, 255))
        d_pred.text((20, 20), f"[YOLO11n PREDICTION] Detected: {len(pred_items)}", fill=(255, 255, 255))

        canvas.paste(gt_draw_img, (0, 0))
        canvas.paste(pred_draw_img, (w, 0))

        out_path = PRED_DIR / f"test_comparison_{idx:02d}_{stem}.jpg"
        canvas.save(out_path, quality=90)

        # Record comparative metrics for this image
        gt_labels = [item["label"] for item in gt_items]
        pred_labels = [item["label"] for item in pred_items]

        audit_summary.append({
            "sample_index": idx,
            "scene_id": stem,
            "image_dimensions": [w, h],
            "gt_count": len(gt_items),
            "pred_count": len(pred_items),
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "output_image": str(out_path),
        })

        print(f"Processed sample {idx}/{len(selected_stems)}: {stem} -> GT: {len(gt_items)} objs, Pred: {len(pred_items)} objs")

    summary_file = PRED_DIR / "visual_audit_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\nVisual audit complete! Saved comparative images to:\n  {PRED_DIR}")
    print(f"Summary JSON: {summary_file}")


if __name__ == "__main__":
    run_visual_audit()
