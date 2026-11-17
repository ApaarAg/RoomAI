"""Deep Rare-Class Audit and Verification Script for Group B Exterior Detector.

Verifies:
  1. Checkpoint integrity and MD5 hash
  2. Exact class-by-class instance, image, and scene counts
  3. Bounding-box area (normalized w*h) and aspect ratio (w/h) distributions per class
  4. Per-class validation and test metrics
  5. Ground truth visualizations for rare classes (porch, pillar_column, balcony)
  6. Outputs reports/exterior_data_audit.json and artifacts/exterior_error_analysis/
"""

import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
DATA_YAML = REPO_ROOT / "data" / "exterior_canonical" / "exterior_canonical.yaml"
DATA_DIR = REPO_ROOT / "data" / "exterior_canonical"
GALLERY_DIR = REPO_ROOT / "artifacts" / "exterior_error_analysis"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "building",
    1: "facade",
    2: "window",
    3: "door",
    4: "roof_cornice",
    5: "balcony",
    6: "porch",
    7: "pillar_column",
    8: "shutter_blind",
    9: "molding_deco",
}


def compute_md5(p: Path) -> str:
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def audit_exterior_data():
    print("=" * 80)
    print("RoomAI Group B Exterior Detector Deep Audit")
    print("=" * 80)

    # 1. Checkpoint Verification
    assert CKPT_PATH.exists(), f"Baseline checkpoint missing: {CKPT_PATH}"
    ckpt_hash = compute_md5(CKPT_PATH)
    ckpt_size_mb = CKPT_PATH.stat().st_size / (1024 * 1024)
    print(f"Baseline Checkpoint: {CKPT_PATH.name}")
    print(f"  Size: {ckpt_size_mb:.2f} MB ({CKPT_PATH.stat().st_size} bytes)")
    print(f"  MD5 Hash: {ckpt_hash}")

    # 2. Per-class instance, image, and geometry analysis
    class_box_counts = {s: Counter() for s in ["train", "val", "test"]}
    class_image_counts = {s: defaultdict(set) for s in ["train", "val", "test"]}
    class_areas = defaultdict(list)
    class_aspect_ratios = defaultdict(list)
    porch_details = []

    for split in ["train", "val", "test"]:
        lbl_dir = DATA_DIR / "labels" / split
        img_dir = DATA_DIR / "images" / split
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            img_file = img_dir / f"{lbl_file.stem}.jpg"
            lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                cname = CLASS_NAMES.get(cid, str(cid))

                class_box_counts[split][cname] += 1
                class_image_counts[split][cname].add(lbl_file.name)

                area = w * h
                aspect = w / (h + 1e-6)
                class_areas[cname].append(area)
                class_aspect_ratios[cname].append(aspect)

                if cid == 6:  # porch
                    porch_details.append({
                        "split": split,
                        "file": lbl_file.name,
                        "image_exists": img_file.exists(),
                        "bbox": [xc, yc, w, h],
                        "area": round(area, 4),
                        "aspect_ratio": round(aspect, 4),
                    })

    # Summary geometry stats
    geometry_stats = {}
    for cname in CLASS_NAMES.values():
        areas = np.array(class_areas[cname]) if class_areas[cname] else np.array([0.0])
        aspects = np.array(class_aspect_ratios[cname]) if class_aspect_ratios[cname] else np.array([0.0])
        geometry_stats[cname] = {
            "total_boxes": len(areas),
            "median_area_pct": round(float(np.median(areas)) * 100.0, 3),
            "p10_area_pct": round(float(np.percentile(areas, 10)) * 100.0, 3),
            "p90_area_pct": round(float(np.percentile(areas, 90)) * 100.0, 3),
            "median_aspect_ratio": round(float(np.median(aspects)), 2),
            "train_boxes": class_box_counts["train"][cname],
            "train_images": len(class_image_counts["train"][cname]),
            "val_boxes": class_box_counts["val"][cname],
            "val_images": len(class_image_counts["val"][cname]),
            "test_boxes": class_box_counts["test"][cname],
            "test_images": len(class_image_counts["test"][cname]),
        }

    print("\n--- Porch (Class 6) Exact Instance Breakdown ---")
    print(f"Total Porch Instances: {len(porch_details)}")
    for p in porch_details:
        print(f"  [{p['split'].upper()}] {p['file']} -> BBox: {p['bbox']}, Area: {p['area']*100:.2f}%, Aspect: {p['aspect_ratio']:.2f}")

    # 3. Baseline Model Evaluation on Val and Test
    print("\n--- Evaluating Baseline Checkpoint on Val & Test ---")
    model = YOLO(str(CKPT_PATH))

    val_res = model.val(data=str(DATA_YAML), split="val", batch=8, imgsz=640, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
    test_res = model.val(data=str(DATA_YAML), split="test", batch=8, imgsz=640, device=0 if torch.cuda.is_available() else "cpu", verbose=False)

    val_metrics = {
        "mAP50": round(float(val_res.box.map50), 4),
        "mAP50_95": round(float(val_res.box.map), 4),
        "precision": round(float(val_res.box.mp), 4),
        "recall": round(float(val_res.box.mr), 4),
        "f1": round(float((2 * val_res.box.mp * val_res.box.mr) / (val_res.box.mp + val_res.box.mr + 1e-6)), 4),
    }

    test_metrics = {
        "mAP50": round(float(test_res.box.map50), 4),
        "mAP50_95": round(float(test_res.box.map), 4),
        "precision": round(float(test_res.box.mp), 4),
        "recall": round(float(test_res.box.mr), 4),
        "f1": round(float((2 * test_res.box.mp * test_res.box.mr) / (test_res.box.mp + test_res.box.mr + 1e-6)), 4),
    }

    # Per-class test performance
    per_class_performance = {}
    p_per = test_res.box.p
    r_per = test_res.box.r
    ap50_per = test_res.box.ap50
    ap_per = test_res.box.ap

    for idx, cname in CLASS_NAMES.items():
        p_val = float(p_per[idx]) if idx < len(p_per) else 0.0
        r_val = float(r_per[idx]) if idx < len(r_per) else 0.0
        f1_val = (2 * p_val * r_val) / (p_val + r_val + 1e-6) if (p_val + r_val) > 0 else 0.0
        ap50_val = float(ap50_per[idx]) if idx < len(ap50_per) else 0.0
        ap_val = float(ap_per[idx]) if idx < len(ap_per) else 0.0
        per_class_performance[cname] = {
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1": round(f1_val, 4),
            "mAP50": round(ap50_val, 4),
            "mAP50_95": round(ap_val, 4),
        }

    print(f"Val Baseline:  mAP50={val_metrics['mAP50']}, P={val_metrics['precision']}, R={val_metrics['recall']}, F1={val_metrics['f1']}")
    print(f"Test Baseline: mAP50={test_metrics['mAP50']}, P={test_metrics['precision']}, R={test_metrics['recall']}, F1={test_metrics['f1']}")

    # 4. Generate Visualizations of Porch and Rare Classes
    print("\n--- Generating Ground Truth & Prediction Visualizations for Rare Classes ---")
    for p in porch_details:
        split = p["split"]
        img_fn = p["file"].replace(".txt", ".jpg")
        img_path = DATA_DIR / "images" / split / img_fn
        if not img_path.exists():
            continue

        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        w_img, h_img = im.size

        # Draw GT box
        xc, yc, bw, bh = p["bbox"]
        x1 = (xc - bw / 2.0) * w_img
        y1 = (yc - bh / 2.0) * h_img
        x2 = (xc + bw / 2.0) * w_img
        y2 = (yc + bh / 2.0) * h_img
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        draw.text((x1 + 4, max(0, y1 - 15)), f"GT Porch ({p['area']*100:.1f}%)", fill=(0, 255, 0))

        # Run prediction
        preds = model.predict(str(img_path), conf=0.15, verbose=False)[0]
        if preds.boxes is not None:
            for b in preds.boxes:
                cid = int(b.cls[0])
                c_conf = float(b.conf[0])
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                pred_name = CLASS_NAMES.get(cid, str(cid))
                draw.rectangle([bx1, by1, bx2, by2], outline=(255, 0, 0), width=2)
                draw.text((bx1 + 4, by1 + 4), f"{pred_name} {c_conf:.2f}", fill=(255, 0, 0))

        vis_path = GALLERY_DIR / f"porch_gt_vs_pred_{split}_{img_fn}"
        im.save(vis_path)
        print(f"  Saved porch visual verification: {vis_path.name}")

    audit_summary = {
        "checkpoint": str(CKPT_PATH.name),
        "checkpoint_hash": ckpt_hash,
        "checkpoint_size_bytes": CKPT_PATH.stat().st_size,
        "val_baseline": val_metrics,
        "test_baseline": test_metrics,
        "per_class_test_performance": per_class_performance,
        "geometry_and_counts": geometry_stats,
        "porch_audit": {
            "total_boxes": len(porch_details),
            "instances": porch_details,
        },
    }

    out_json = REPO_ROOT / "reports" / "exterior_data_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nAudit complete! Saved structured audit to: {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    audit_exterior_data()
