"""Phase 15 Audit Script: Deep Error Analysis and Scene Overlap Verification.

Audits:
1. Verify B2, B5, B6 checkpoint hashes.
2. Confirm validation (154) and test (154) scenes do not overlap with training data.
3. Detailed error categorization for:
   - porch, pillar_column, facade, door, roof_cornice, balcony
   - Breakdown of:
     - Missed objects (FN)
     - False positives (FP)
     - Cross-class confusion (pred class != GT class on same bounding region)
     - Localization errors (0.10 <= IoU < 0.50)
     - Background false positives (IoU < 0.10 with any object)
4. Saves audit results to reports/phase15_audit_details.json.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VAL_IMG_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "val"
VAL_LBL_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "labels" / "val"
TRAIN_IMG_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "train"
TEST_IMG_DIR = REPO_ROOT / "data" / "exterior_b6_expanded" / "images" / "test"

B2_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b2_yolo11s_best.pt"
B5_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b5_transfer_best.pt"
B6_CKPT = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def box_iou(box1: List[float], box2: List[float]) -> float:
    x1_min, x1_max = box1[0] - box1[2] / 2.0, box1[0] + box1[2] / 2.0
    y1_min, y1_max = box1[1] - box1[3] / 2.0, box1[1] + box1[3] / 2.0
    x2_min, x2_max = box2[0] - box2[2] / 2.0, box2[0] + box2[2] / 2.0
    y2_min, y2_max = box2[1] - box2[3] / 2.0, box2[1] + box2[3] / 2.0

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def main():
    print("=== Phase 15 Stage A: Detailed Error and Data Audit ===")

    # 1. Hashes
    expected_hashes = {
        "B2": ("checkpoints/exterior_detector_b2_yolo11s_best.pt", "14a931dd0adabc16cbd7b28ed1d17f66"),
        "B5": ("checkpoints/exterior_detector_b5_transfer_best.pt", "e12b4181fee244a3569dfa782ba3c3c3"),
        "B6": ("checkpoints/exterior_detector_b6_expanded_best.pt", "f5ceea56b0610ea14b647014e48152dc"),
    }
    hash_status = {}
    for name, (rel_p, exp_h) in expected_hashes.items():
        p = REPO_ROOT / rel_p
        act_h = hashlib.md5(p.read_bytes()).hexdigest()
        assert act_h == exp_h, f"Hash mismatch for {name}: {act_h} vs {exp_h}"
        hash_status[name] = {"path": str(p), "md5": act_h, "verified": True}
        print(f"[VERIFIED] {name}: {act_h}")

    # 2. Check overlap between train, val, test
    train_names = set(p.stem for p in TRAIN_IMG_DIR.glob("*.jpg"))
    val_names = set(p.stem for p in VAL_IMG_DIR.glob("*.jpg"))
    test_names = set(p.stem for p in TEST_IMG_DIR.glob("*.jpg"))

    val_overlap = train_names.intersection(val_names)
    test_overlap = train_names.intersection(test_names)
    val_test_overlap = val_names.intersection(test_names)

    print(f"\nSplit Overlap Audit:")
    print(f"  Train: {len(train_names)} images | Val: {len(val_names)} images | Test: {len(test_names)} images")
    print(f"  Train & Val overlap: {len(val_overlap)}")
    print(f"  Train & Test overlap: {len(test_overlap)}")
    print(f"  Val & Test overlap: {len(val_test_overlap)}")
    assert len(val_overlap) == 0, "Train and Val split collision!"
    assert len(test_overlap) == 0, "Train and Test split collision!"

    # 3. Load GT annotations
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

    # 4. Predict with B6 on validation set
    print("\nRunning inference with Model B6 on validation set...")
    model_b6 = YOLO(str(B6_CKPT))
    all_preds_b6 = {}
    for img_p in val_images:
        res = model_b6.predict(source=str(img_p), conf=0.01, imgsz=640, device=0, verbose=False)[0]
        preds = []
        for i in range(len(res.boxes)):
            cid = int(res.boxes.cls[i].item())
            conf = float(res.boxes.conf[i].item())
            xywhn = res.boxes.xywhn[i].tolist()
            preds.append((cid, conf, xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
        all_preds_b6[img_p.name] = preds

    # 5. Error Categorization for porch, pillar_column, facade, door, roof_cornice, balcony
    target_cids = [6, 7, 1, 3, 4, 5]  # porch, pillar_column, facade, door, roof_cornice, balcony
    error_analysis = {}

    conf_thresh = 0.25
    iou_match_thresh = 0.50

    for cid in target_cids:
        cname = CANONICAL_CLASSES[cid]
        missed_gt_count = 0
        tp_count = 0
        fp_count = 0
        loc_error_count = 0  # 0.1 <= IoU < 0.5 with correct class
        cross_class_confusions = {}  # other_class -> count
        background_fp_count = 0  # IoU < 0.1 with any GT
        fp_examples = []

        for img_name, gts in all_gts.items():
            preds = all_preds_b6.get(img_name, [])
            c_gts = [g for g in gts if g[0] == cid]
            c_preds = [p for p in preds if p[0] == cid and p[1] >= conf_thresh]
            c_preds.sort(key=lambda x: x[1], reverse=True)

            matched_gt = set()
            for p in c_preds:
                p_box = [p[2], p[3], p[4], p[5]]
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g in enumerate(c_gts):
                    if g_idx in matched_gt:
                        continue
                    g_box = [g[1], g[2], g[3], g[4]]
                    iou = box_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_match_thresh and best_gt_idx != -1:
                    tp_count += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp_count += 1
                    # Diagnose FP cause:
                    if 0.10 <= best_iou < 0.50:
                        loc_error_count += 1
                        fp_examples.append({"img": img_name, "conf": round(p[1], 3), "reason": f"localization (IoU={best_iou:.2f})"})
                    else:
                        # Check cross-class confusion with any other GT box
                        best_other_iou = 0.0
                        best_other_cls = None
                        for g in gts:
                            if g[0] != cid:
                                g_box = [g[1], g[2], g[3], g[4]]
                                oi = box_iou(p_box, g_box)
                                if oi > best_other_iou:
                                    best_other_iou = oi
                                    best_other_cls = CANONICAL_CLASSES[g[0]]
                        if best_other_iou >= 0.30 and best_other_cls is not None:
                            cross_class_confusions[best_other_cls] = cross_class_confusions.get(best_other_cls, 0) + 1
                            if len(fp_examples) < 8:
                                fp_examples.append({"img": img_name, "conf": round(p[1], 3), "reason": f"confused with {best_other_cls} (IoU={best_other_iou:.2f})"})
                        else:
                            background_fp_count += 1
                            if len(fp_examples) < 8:
                                fp_examples.append({"img": img_name, "conf": round(p[1], 3), "reason": "background/context"})

            missed_gt_count += (len(c_gts) - len(matched_gt))

        total_gt = tp_count + missed_gt_count
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        error_analysis[cname] = {
            "gt_instances": total_gt,
            "tp": tp_count,
            "fp": fp_count,
            "fn_missed": missed_gt_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "localization_errors": loc_error_count,
            "cross_class_confusions": cross_class_confusions,
            "background_fps": background_fp_count,
            "sample_fp_diagnostics": fp_examples[:6],
        }

        print(f"\n--- {cname.upper()} (GT: {total_gt}, TP: {tp_count}, FP: {fp_count}, FN: {missed_gt_count}) ---")
        print(f"  P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
        print(f"  Localization Errors (0.1<=IoU<0.5): {loc_error_count}")
        print(f"  Cross-class confusions: {cross_class_confusions}")
        print(f"  Background/unlabeled FPs: {background_fp_count}")

    # Save detailed JSON
    out_path = REPO_ROOT / "reports" / "phase15_audit_details.json"
    audit_data = {
        "hash_status": hash_status,
        "split_integrity": {
            "train_count": len(train_names),
            "val_count": len(val_names),
            "test_count": len(test_names),
            "val_overlap": len(val_overlap),
            "test_overlap": len(test_overlap),
        },
        "weak_class_error_analysis": error_analysis,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\n[SAVED] Audit details saved to {out_path}")


if __name__ == "__main__":
    main()
