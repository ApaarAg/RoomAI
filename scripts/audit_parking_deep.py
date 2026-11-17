"""Deep Rare-Class and Error Audit Script for Group C Parking & Grounds Detector.

Verifies:
  1. Checkpoint integrity and MD5 hash
  2. Exact class-by-class instance, image, and scene counts across train, val, test splits
  3. Bounding-box area (w*h) and aspect ratio (w/h) distributions per class
  4. Per-class validation and test metrics on the baseline C2 model
  5. Ground truth visualizations for lagging/rare classes (bicycle, tree, outdoor_fixture)
  6. Outputs reports/parking_data_audit.json and artifacts/parking_error_analysis/
"""

import hashlib
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "checkpoints" / "parking_detector_c2_yolo11s_pilot_best.pt"
DATA_YAML = REPO_ROOT / "data" / "parking_canonical" / "parking_canonical.yaml"
DATA_DIR = REPO_ROOT / "data" / "parking_canonical"
GALLERY_DIR = REPO_ROOT / "artifacts" / "parking_error_analysis"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "car_vehicle",
    1: "motorcycle",
    2: "bicycle",
    3: "outdoor_seating",
    4: "outdoor_table",
    5: "plant_bush",
    6: "tree",
    7: "outdoor_fixture",
}


def compute_md5(p: Path) -> str:
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def audit_parking_data():
    print("=" * 80)
    print("RoomAI Group C Parking & Grounds Detector Deep Audit")
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
    
    total_images_per_split = {}

    for split in ["train", "val", "test"]:
        lbl_dir = DATA_DIR / "labels" / split
        img_dir = DATA_DIR / "images" / split
        lbl_files = sorted(list(lbl_dir.glob("*.txt")))
        total_images_per_split[split] = len(lbl_files)
        
        for lbl_file in lbl_files:
            img_stem = lbl_file.stem
            lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                cname = CLASS_NAMES.get(cid, str(cid))

                class_box_counts[split][cname] += 1
                class_image_counts[split][cname].add(img_stem)

                area = w * h
                aspect = w / (h + 1e-6)
                class_areas[cname].append(area)
                class_aspect_ratios[cname].append(aspect)

    print(f"\nImage counts per split: {total_images_per_split}")
    print(f"Total images: {sum(total_images_per_split.values())}")

    # 3. Geometry Statistics per class
    geometry_stats = {}
    print("\n--- Class Geometry & Box Distributions ---")
    for cid in range(8):
        cname = CLASS_NAMES[cid]
        areas = np.array(class_areas.get(cname, [0.0]))
        aspects = np.array(class_aspect_ratios.get(cname, [1.0]))

        train_boxes = class_box_counts["train"][cname]
        val_boxes = class_box_counts["val"][cname]
        test_boxes = class_box_counts["test"][cname]
        total_boxes = train_boxes + val_boxes + test_boxes

        train_imgs = len(class_image_counts["train"][cname])
        val_imgs = len(class_image_counts["val"][cname])
        test_imgs = len(class_image_counts["test"][cname])

        med_area = float(np.median(areas)) * 100.0 if len(areas) else 0.0
        p10_area = float(np.percentile(areas, 10)) * 100.0 if len(areas) else 0.0
        p90_area = float(np.percentile(areas, 90)) * 100.0 if len(areas) else 0.0
        med_aspect = float(np.median(aspects)) if len(aspects) else 1.0

        geometry_stats[cname] = {
            "class_id": cid,
            "total_boxes": int(total_boxes),
            "median_area_pct": round(med_area, 3),
            "p10_area_pct": round(p10_area, 3),
            "p90_area_pct": round(p90_area, 3),
            "median_aspect_ratio": round(med_aspect, 2),
            "train_boxes": int(train_boxes),
            "train_images": int(train_imgs),
            "val_boxes": int(val_boxes),
            "val_images": int(val_imgs),
            "test_boxes": int(test_boxes),
            "test_images": int(test_imgs),
        }
        print(f"[{cid}] {cname:16s} | Total: {total_boxes:4d} boxes ({train_boxes:4d} train, {val_boxes:3d} val, {test_boxes:3d} test) | "
              f"Images: {train_imgs:3d} tr, {val_imgs:2d} val, {test_imgs:2d} te | "
              f"Med Area: {med_area:6.3f}% | Aspect: {med_aspect:.2f}")

    # 4. Baseline Model Validation & Test Split Reproduction
    print("\n--- Reproducing Baseline Model C2 Metrics ---")
    model = YOLO(str(CKPT_PATH))

    # Evaluate Validation Split
    print("Evaluating C2 on 'val' split (120 images)...")
    val_res = model.val(
        data=str(DATA_YAML),
        split="val",
        batch=8,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        verbose=False,
        save_json=False,
    )
    val_map50 = round(float(val_res.box.map50), 4)
    val_map = round(float(val_res.box.map), 4)
    val_p = round(float(val_res.box.mp), 4)
    val_r = round(float(val_res.box.mr), 4)
    val_f1 = round((2 * val_p * val_r) / (val_p + val_r + 1e-6) if (val_p + val_r) > 0 else 0.0, 4)

    # Evaluate Held-Out Test Split
    print("Evaluating C2 on held-out 'test' split (120 images)...")
    test_res = model.val(
        data=str(DATA_YAML),
        split="test",
        batch=8,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        verbose=False,
        save_json=False,
    )
    test_map50 = round(float(test_res.box.map50), 4)
    test_map = round(float(test_res.box.map), 4)
    test_p = round(float(test_res.box.mp), 4)
    test_r = round(float(test_res.box.mr), 4)
    test_f1 = round((2 * test_p * test_r) / (test_p + test_r + 1e-6) if (test_p + test_r) > 0 else 0.0, 4)

    print(f"\n[BASELINE C2 REPRODUCED]")
    print(f"  Val  mAP50: {val_map50:.4f} | mAP50-95: {val_map:.4f} | P: {val_p:.4f} | R: {val_r:.4f} | F1: {val_f1:.4f}")
    print(f"  Test mAP50: {test_map50:.4f} | mAP50-95: {test_map:.4f} | P: {test_p:.4f} | R: {test_r:.4f} | F1: {test_f1:.4f}")

    # Extract per-class test metrics
    per_class_test = {}
    class_names = test_res.names
    ap_class_indices = getattr(test_res.box, "ap_class_index", [])
    for i, c_id in enumerate(ap_class_indices):
        name = class_names.get(int(c_id), f"class_{c_id}")
        p_val = round(float(test_res.box.p[i]), 4) if hasattr(test_res.box, "p") and len(test_res.box.p) > i else 0.0
        r_val = round(float(test_res.box.r[i]), 4) if hasattr(test_res.box, "r") and len(test_res.box.r) > i else 0.0
        ap50 = round(float(test_res.box.ap50[i]), 4) if hasattr(test_res.box, "ap50") and len(test_res.box.ap50) > i else 0.0
        ap_val = round(float(test_res.box.maps[int(c_id)]), 4) if hasattr(test_res.box, "maps") and len(test_res.box.maps) > int(c_id) else 0.0
        f1_c = round((2 * p_val * r_val) / (p_val + r_val + 1e-6) if (p_val + r_val) > 0 else 0.0, 4)
        per_class_test[name] = {
            "class_id": int(c_id),
            "precision": p_val,
            "recall": r_val,
            "f1": f1_c,
            "mAP50": ap50,
            "mAP50_95": ap_val,
        }
    for c_id, name in class_names.items():
        if name not in per_class_test:
            per_class_test[name] = {
                "class_id": int(c_id),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mAP50": 0.0,
                "mAP50_95": 0.0,
            }

    print("\nPer-class test metrics (Baseline C2):")
    for cname, m in per_class_test.items():
        print(f"  {cname:18s}: mAP50={m['mAP50']:.4f} | mAP50-95={m['mAP50_95']:.4f} | P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f}")

    # 5. Generate Ground Truth vs Prediction Visual Galleries for Failure Classes
    print("\n--- Generating Visual Error Analysis Galleries ---")
    test_img_dir = DATA_DIR / "images" / "test"
    test_lbl_dir = DATA_DIR / "labels" / "test"

    # Select representative samples containing bicycle, tree, outdoor_table, outdoor_fixture
    target_samples = []
    for lf in sorted(list(test_lbl_dir.glob("*.txt"))):
        lines = lf.read_text(encoding="utf-8").strip().splitlines()
        classes = {int(l.split()[0]) for l in lines if l.strip()}
        if 2 in classes or 6 in classes or 4 in classes:  # bicycle, tree, outdoor_table
            target_samples.append(lf.stem)
            if len(target_samples) >= 6:
                break

    device = 0 if torch.cuda.is_available() else "cpu"
    for stem in target_samples:
        img_p = test_img_dir / f"{stem}.jpg"
        if not img_p.exists():
            img_p = test_img_dir / f"{stem}.png"
        if not img_p.exists():
            continue

        raw = cv2.imread(str(img_p))
        if raw is None:
            continue
        h, w = raw.shape[:2]

        # Ground Truth
        gt_img = raw.copy()
        lbl_p = test_lbl_dir / f"{stem}.txt"
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    color = (0, 255, 0)
                    cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        gt_img,
                        CLASS_NAMES.get(cid, str(cid)),
                        (x1, max(y1 - 4, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )
        cv2.putText(gt_img, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Baseline C2 Prediction (conf=0.25)
        pred_res = model.predict(raw, conf=0.25, device=device, verbose=False)[0]
        c2_img = raw.copy()
        if pred_res.boxes is not None:
            for box in pred_res.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                color = (0, 165, 255)
                cv2.rectangle(c2_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(
                    c2_img,
                    f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}",
                    (xyxy[0], max(xyxy[1] - 4, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
        cv2.putText(c2_img, "Baseline C2 Pred", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

        side_by_side = np.hstack([gt_img, c2_img])
        out_f = GALLERY_DIR / f"gt_vs_c2_pred_{stem}.jpg"
        cv2.imwrite(str(out_f), side_by_side)
        print(f"  Saved gallery image: {out_f.name}")

    # 6. Save Audit JSON
    audit_report = {
        "checkpoint": CKPT_PATH.name,
        "checkpoint_hash": ckpt_hash,
        "checkpoint_size_bytes": CKPT_PATH.stat().st_size,
        "total_images_per_split": total_images_per_split,
        "val_baseline": {
            "mAP50": val_map50,
            "mAP50_95": val_map,
            "precision": val_p,
            "recall": val_r,
            "f1": val_f1,
        },
        "test_baseline": {
            "mAP50": test_map50,
            "mAP50_95": test_map,
            "precision": test_p,
            "recall": test_r,
            "f1": test_f1,
        },
        "per_class_test_performance": per_class_test,
        "geometry_and_counts": geometry_stats,
    }

    out_json = REPO_ROOT / "reports" / "parking_data_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n[REPORT] Saved deep parking data audit to: {out_json}")
    return audit_report


if __name__ == "__main__":
    audit_parking_data()
