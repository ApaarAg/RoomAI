"""Prepare a controlled evaluation dataset of representative furniture crops from indoor_canonical."""

import os
import json
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def main():
    detector_path = "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt"
    fastsam_path = "FastSAM-s.pt"
    test_img_dir = "data/indoor_canonical/images/test"
    output_dir = "artifacts/phase_5_9_furniture_crops"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb_bbox"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fastsam_masked"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "white_bg"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "padded_masked"), exist_ok=True)

    from ultralytics import YOLO, FastSAM
    detector = YOLO(detector_path)
    fastsam = FastSAM(fastsam_path)

    # Class targets
    # 0: cabinet#shelf, 1: table, 2: chair#stool, 5: bed, 6: sofa
    target_classes = {
        0: "cabinet",
        1: "table",
        2: "chair",
        5: "bed",
        6: "sofa"
    }

    # Desired count per class: 3 each = 15 total
    counts = {cls_name: 0 for cls_name in target_classes.values()}
    max_per_class = 3
    manifest = []

    test_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Total test images: {len(test_files)}")

    for fname in test_files:
        if all(c >= max_per_class for c in counts.values()):
            break

        fpath = os.path.join(test_img_dir, fname)
        img_pil = Image.open(fpath).convert("RGB")
        w, h = img_pil.size

        # Run A3 detector
        det_results = detector.predict(fpath, imgsz=800, conf=0.40, verbose=False)[0]

        for box in det_results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id not in target_classes:
                continue

            cls_name = target_classes[cls_id]
            if counts[cls_name] >= max_per_class:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            bw = x2 - x1
            bh = y2 - y1

            # Require reasonable crop size (> 100x100)
            if bw < 100 or bh < 100 or bw * bh < 15000:
                continue

            # Crop 1: RGB Bbox Crop
            crop_rgb = img_pil.crop((x1, y1, x2, y2))
            crop_id = f"{cls_name}_{counts[cls_name]+1:02d}_{fname.split('.')[0][:12]}"

            # Crop 2: FastSAM mask for this specific box
            # Call FastSAM with single bbox prompt on numpy image
            fs_results = fastsam(
                np.array(img_pil),
                bboxes=[[x1, y1, x2, y2]],
                device="cuda:0",
                verbose=False
            )[0]

            mask_np = None
            if fs_results.masks is not None and len(fs_results.masks.data) > 0:
                # Extract first mask
                m = fs_results.masks.data[0].cpu().numpy().astype(bool)
                # Resize mask to original image if needed
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_np = m[y1:y2, x1:x2]

            if mask_np is None or mask_np.sum() == 0:
                mask_np = np.ones((bh, bw), dtype=bool)

            # Crop 3: White background crop
            img_arr = np.array(crop_rgb)
            white_bg = np.full_like(img_arr, 255)
            white_bg[mask_np] = img_arr[mask_np]
            white_bg_pil = Image.fromarray(white_bg)

            # Crop 4: Masked with alpha (RGBA)
            rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_arr
            rgba[:, :, 3] = (mask_np.astype(np.uint8) * 255)
            rgba_pil = Image.fromarray(rgba, "RGBA")

            # Crop 5: Padded centered square (standard format for 3D generative models like TripoSR/SF3D)
            # 3D models expect a centered object with ~15% border padding on white or transparent background
            side = int(max(bw, bh) * 1.25)
            square_img = Image.new("RGBA", (side, side), (255, 255, 255, 0))
            offset_x = (side - bw) // 2
            offset_y = (side - bh) // 2
            square_img.paste(rgba_pil, (offset_x, offset_y), mask=rgba_pil)

            # Save artifacts
            p_rgb = os.path.join(output_dir, "rgb_bbox", f"{crop_id}_bbox.png")
            p_white = os.path.join(output_dir, "white_bg", f"{crop_id}_whitebg.png")
            p_padded = os.path.join(output_dir, "padded_masked", f"{crop_id}_padded.png")

            crop_rgb.save(p_rgb)
            white_bg_pil.save(p_white)
            square_img.save(p_padded)

            counts[cls_name] += 1
            manifest.append({
                "crop_id": crop_id,
                "category": cls_name,
                "source_image": fname,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "crop_size": [bw, bh],
                "paths": {
                    "rgb_bbox": p_rgb,
                    "white_bg": p_white,
                    "padded_masked": p_padded
                }
            })
            print(f"Extracted {crop_id} (conf={conf:.2f}, size={bw}x{bh})")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCompleted! Extracted {len(manifest)} crops across classes: {counts}")
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
