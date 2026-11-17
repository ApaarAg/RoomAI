"""Generate side-by-side comparative error analysis gallery for Phase 20 (Model B6 vs Model B7)."""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_IMG_DIR = REPO_ROOT / "data" / "pilot_b7_dataset" / "images" / "val"
VAL_LBL_DIR = REPO_ROOT / "data" / "pilot_b7_dataset" / "labels" / "val"
GALLERY_DIR = REPO_ROOT / "reports" / "phase20_pilot_gallery"

B6_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
B7_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b7_pilot_best.pt"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def main():
    print("=== Generating Phase 20 B6 vs B7 Comparative Error Gallery ===")
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))
    all_gts = {}
    for img_p in val_images:
        lbl_p = VAL_LBL_DIR / f"{img_p.stem}.txt"
        boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        all_gts[img_p.name] = boxes

    model_b6 = YOLO(str(B6_CKPT))
    model_b7 = YOLO(str(B7_CKPT))

    # Identify scenes with ground-truth porches or doors/pillars
    porch_scenes = [img_p for img_p in val_images if any(g[0] == 6 for g in all_gts[img_p.name])]
    print(f"Found {len(porch_scenes)} validation scenes containing ground-truth porches.")

    # Select 6 informative scenes
    selected_scenes = porch_scenes[:6]

    for idx, img_p in enumerate(selected_scenes):
        img_name = img_p.name
        gts = all_gts[img_name]

        res_b6 = model_b6.predict(source=str(img_p), conf=0.20, imgsz=640, device=0, verbose=False)[0]
        res_b7 = model_b7.predict(source=str(img_p), conf=0.20, imgsz=640, device=0, verbose=False)[0]

        with Image.open(img_p) as im:
            w, h = im.size

            # Canvas B6
            im_b6 = im.copy()
            draw_b6 = ImageDraw.Draw(im_b6)
            # Canvas B7
            im_b7 = im.copy()
            draw_b7 = ImageDraw.Draw(im_b7)

            # Draw Ground Truths (Green)
            for cid, xc, yc, bw, bh in gts:
                x1 = int((xc - bw / 2.0) * w)
                y1 = int((yc - bh / 2.0) * h)
                x2 = int((xc + bw / 2.0) * w)
                y2 = int((yc + bh / 2.0) * h)
                cname = CANONICAL_CLASSES[cid]
                color = (0, 255, 0)
                if cid == 6:  # Highlight porch GT
                    draw_b6.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw_b6.text((x1, max(0, y1 - 12)), f"GT: {cname}", fill=color)
                    draw_b7.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw_b7.text((x1, max(0, y1 - 12)), f"GT: {cname}", fill=color)

            # Draw B6 Predictions
            for b in res_b6.boxes:
                cid = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                if cid == 6:  # porch
                    xyxy = b.xyxy[0].tolist()
                    draw_b6.rectangle(xyxy, outline=(255, 255, 0), width=3)
                    draw_b6.text((int(xyxy[0]), max(0, int(xyxy[1]) - 12)), f"B6 Porch {conf:.2f}", fill=(255, 255, 0))

            # Draw B7 Predictions
            for b in res_b7.boxes:
                cid = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                if cid == 6:  # porch
                    xyxy = b.xyxy[0].tolist()
                    draw_b7.rectangle(xyxy, outline=(0, 255, 255), width=3)
                    draw_b7.text((int(xyxy[0]), max(0, int(xyxy[1]) - 12)), f"B7 Porch {conf:.2f}", fill=(0, 255, 255))

            # Create side-by-side collage
            banner_h = 30
            collage = Image.new("RGB", (w * 2, h + banner_h), color=(30, 30, 30))
            draw_coll = ImageDraw.Draw(collage)
            draw_coll.text((10, 8), f"Model B6 Baseline (Yellow: Porch pred | Green: GT)", fill=(255, 255, 255))
            draw_coll.text((w + 10, 8), f"Model B7 Pilot (Cyan: Porch pred | Green: GT)", fill=(255, 255, 255))

            collage.paste(im_b6, (0, banner_h))
            collage.paste(im_b7, (w, banner_h))

            out_path = GALLERY_DIR / f"comparative_eval_{idx:02d}_{img_p.stem}.jpg"
            collage.save(out_path, quality=92)
            print(f"  Saved comparison: {out_path.name}")

    print(f"All comparative error analysis gallery images saved to {GALLERY_DIR}")


if __name__ == "__main__":
    main()
