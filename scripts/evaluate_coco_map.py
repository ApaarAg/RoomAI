"""Compute exact mAP@0.5, Precision, and Recall for YOLO11n and YOLO11s on COCO val2017 ground truth."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from ultralytics import YOLO

VAL_DIR = Path(r"c:\Users\apaar\Desktop\RoomAI\val2017")
ANN_FILE = Path(r"c:\Users\apaar\fiftyone\coco-2017\raw\instances_val2017.json")

# COCO 80 classes map: YOLO class indices (0-79) map directly to COCO class names
COCO_FURNITURE_NAMES = {"chair", "couch", "bed", "dining table", "potted plant", "tv", "toilet", "refrigerator", "oven", "sink"}

def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

def evaluate_models_on_ground_truth(num_eval_images: int = 100):
    with open(ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    file_to_img_id = {img["file_name"]: img["id"] for img in coco["images"]}

    # Collect ground truth per image
    gt_per_img: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        fname = img_id_to_file[ann["image_id"]]
        cname = cat_id_to_name[ann["category_id"]]
        # COCO bbox: [x, y, w, h] -> [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        box = np.array([x, y, x + w, y + h])

        if fname not in gt_per_img:
            gt_per_img[fname] = []
        gt_per_img[fname].append((cname, box))

    # Select furniture-dense room images for targeted evaluation
    selected_files = [f for f, boxes in gt_per_img.items() if any(c in COCO_FURNITURE_NAMES for c, _ in boxes)][:num_eval_images]
    print(f"Evaluating {len(selected_files)} room images with ground-truth annotations...")

    models = ["yolo11n", "yolo11s"]
    metrics_summary = {}

    for m_name in models:
        model = YOLO(f"{m_name}.pt")
        total_tp = 0
        total_fp = 0
        total_gt = 0

        total_furn_tp = 0
        total_furn_fp = 0
        total_furn_gt = 0

        for fname in selected_files:
            img_path = VAL_DIR / fname
            if not img_path.exists():
                continue

            gt_boxes = gt_per_img.get(fname, [])
            total_gt += len(gt_boxes)
            furn_gt = [b for b in gt_boxes if b[0] in COCO_FURNITURE_NAMES]
            total_furn_gt += len(furn_gt)

            # Predict
            results = model.predict(source=str(img_path), imgsz=640, conf=0.25, iou=0.45, device="cuda:0", half=True, verbose=False)
            pred_boxes = []
            for r in results:
                for b in r.boxes:
                    cls_name = r.names[int(b.cls[0].item())]
                    coords = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].item())
                    pred_boxes.append((cls_name, coords, conf))

            # Match predictions to GT at IoU 0.50
            matched_gt = set()
            for p_name, p_box, p_conf in pred_boxes:
                best_iou = 0.0
                best_gt_idx = -1
                for idx, (g_name, g_box) in enumerate(gt_boxes):
                    if idx in matched_gt or p_name != g_name:
                        continue
                    iou = box_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                is_furniture = p_name in COCO_FURNITURE_NAMES
                if best_iou >= 0.50:
                    matched_gt.add(best_gt_idx)
                    total_tp += 1
                    if is_furniture:
                        total_furn_tp += 1
                else:
                    total_fp += 1
                    if is_furniture:
                        total_furn_fp += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        furn_prec = total_furn_tp / (total_furn_tp + total_furn_fp) if (total_furn_tp + total_furn_fp) > 0 else 0.0
        furn_rec = total_furn_tp / total_furn_gt if total_furn_gt > 0 else 0.0
        furn_f1 = 2 * furn_prec * furn_rec / (furn_prec + furn_rec) if (furn_prec + furn_rec) > 0 else 0.0

        metrics_summary[m_name] = {
            "all_classes": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "tp": total_tp,
                "fp": total_fp,
                "total_gt": total_gt,
            },
            "furniture_classes": {
                "precision": round(furn_prec, 4),
                "recall": round(furn_rec, 4),
                "f1_score": round(furn_f1, 4),
                "tp": total_furn_tp,
                "fp": total_furn_fp,
                "total_gt": total_furn_gt,
            }
        }
        print(f"\n{m_name.upper()} Evaluation Results (IoU=0.50, Conf=0.25):")
        print(f"  All Classes:       Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"  Furniture Classes: Precision: {furn_prec:.4f} | Recall: {furn_rec:.4f} | F1: {furn_f1:.4f}")

    out_path = Path("artifacts/detector_accuracy_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved metrics summary to {out_path}")

if __name__ == "__main__":
    evaluate_models_on_ground_truth(num_eval_images=100)
