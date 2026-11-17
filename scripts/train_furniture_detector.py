"""Fine-tuning experiment for YOLO on room furniture dataset."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

DATASET_YAML = Path(r"c:\Users\apaar\Desktop\RoomAI\data\furniture\furniture.yaml").as_posix()
RUNS_DIR = Path(r"c:\Users\apaar\Desktop\RoomAI\runs\train")
TEST_IMG_DIR = Path(r"c:\Users\apaar\Desktop\RoomAI\data\furniture\images\test")
TEST_LBL_DIR = Path(r"c:\Users\apaar\Desktop\RoomAI\data\furniture\labels\test")

TARGET_CLASSES = [
    "chair", "couch", "bed", "dining table", "potted plant",
    "tv", "toilet", "refrigerator", "oven", "sink"
]
NAME_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


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


def evaluate_test_set(model: YOLO, imgsz: int = 640, is_pretrained_80_class: bool = False):
    """Evaluates either an 80-class pretrained or 10-class fine-tuned model against the test set ground truth."""
    test_files = list(TEST_IMG_DIR.glob("*.jpg"))
    total_tp = 0
    total_fp = 0
    total_gt = 0

    ious_tp = []

    for img_p in test_files:
        lbl_p = TEST_LBL_DIR / (img_p.stem + ".txt")
        if not lbl_p.exists():
            continue

        # Load GT boxes [cls_id, x1, y1, x2, y2] in pixels
        from PIL import Image
        with Image.open(img_p) as im:
            w, h = im.size

        gt_boxes = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    gt_boxes.append((cid, np.array([x1, y1, x2, y2])))

        total_gt += len(gt_boxes)

        # Run inference
        results = model.predict(source=str(img_p), imgsz=imgsz, device="cuda:0", half=True, conf=0.25, iou=0.45, verbose=False)

        pred_boxes = []
        for r in results:
            for b in r.boxes:
                c_idx = int(b.cls[0].item())
                if is_pretrained_80_class:
                    cname = r.names[c_idx]
                    if cname not in NAME_TO_ID:
                        continue
                    mapped_cid = NAME_TO_ID[cname]
                else:
                    mapped_cid = c_idx

                coords = b.xyxy[0].cpu().numpy()
                pred_boxes.append((mapped_cid, coords, float(b.conf[0].item())))

        # Match at IoU 0.50
        matched_gt = set()
        for p_cid, p_box, p_conf in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for g_idx, (g_cid, g_box) in enumerate(gt_boxes):
                if g_idx in matched_gt or p_cid != g_cid:
                    continue
                iou = box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx

            if best_iou >= 0.50:
                matched_gt.add(best_idx)
                total_tp += 1
                ious_tp.append(best_iou)
            else:
                total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious_tp)) if ious_tp else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou_tp": round(mean_iou, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "ground_truth_total": total_gt,
    }


def run_experiment(
    base_model: str = "yolo11n.pt",
    epochs: int = 5,
    batch_size: int = 8,
    imgsz: int = 640,
    lr0: float = 0.01,
    experiment_name: str = "furniture_finetune",
    seed: int = 42,
):
    print("=" * 60)
    print(f"Starting Fine-Tuning Experiment: {experiment_name}")
    print(f"Base Model:       {base_model}")
    print(f"Dataset:          {DATASET_YAML}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Resolution:       {imgsz}")
    print(f"Learning Rate:    {lr0}")
    print(f"Device:           CUDA ({torch.cuda.get_device_name(0)})")
    print(f"Seed:             {seed}")
    print("=" * 60)

    # 1. Evaluate Pretrained Baseline
    print("\n--- Evaluating Pretrained Baseline on Test Split ---")
    baseline_model = YOLO(base_model)
    base_metrics = evaluate_test_set(baseline_model, imgsz=imgsz, is_pretrained_80_class=True)
    print("Baseline Metrics (Furniture Test Set):")
    print(f"  Precision: {base_metrics['precision']:.4f}")
    print(f"  Recall:    {base_metrics['recall']:.4f}")
    print(f"  F1-Score:  {base_metrics['f1_score']:.4f}")
    print(f"  Mean IoU:  {base_metrics['mean_iou_tp']:.4f}")

    # 2. Train
    print("\n--- Starting Model Training ---")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    train_model = YOLO(base_model)

    train_model.train(
        data=DATASET_YAML,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        seed=seed,
        device="cuda:0",
        amp=False,  # Avoid downloading external models
        workers=0,  # Avoid Windows multiprocessing subprocess issues
        project=str(RUNS_DIR),
        name=experiment_name,
        exist_ok=True,
        patience=5,
        save=True,
        verbose=True,
    )
    train_duration_sec = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    best_ckpt = RUNS_DIR / experiment_name / "weights" / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = RUNS_DIR / experiment_name / "weights" / "last.pt"

    print(f"\nTraining completed in {train_duration_sec:.1f}s.")
    print(f"Peak VRAM: {peak_vram_mb:.1f} MB | Checkpoint: {best_ckpt}")

    # 3. Evaluate Fine-Tuned Model
    print("\n--- Evaluating Fine-Tuned Model on Test Split ---")
    ft_model = YOLO(str(best_ckpt))
    ft_metrics = evaluate_test_set(ft_model, imgsz=imgsz, is_pretrained_80_class=False)
    print("Fine-Tuned Metrics (Furniture Test Set):")
    print(f"  Precision: {ft_metrics['precision']:.4f}")
    print(f"  Recall:    {ft_metrics['recall']:.4f}")
    print(f"  F1-Score:  {ft_metrics['f1_score']:.4f}")
    print(f"  Mean IoU:  {ft_metrics['mean_iou_tp']:.4f}")

    # Latency check
    test_img = r"c:\Users\apaar\Desktop\RoomAI\artifacts\sample_room.jpg"
    lats = []
    for _ in range(15):
        t_s = time.perf_counter()
        _ = ft_model.predict(source=test_img, imgsz=imgsz, device="cuda:0", half=True, verbose=False)
        lats.append((time.perf_counter() - t_s) * 1000)
    mean_lat = float(np.mean(lats[3:]))

    ckpt_size_mb = best_ckpt.stat().st_size / (1024**2) if best_ckpt.exists() else 0.0

    summary = {
        "experiment_name": experiment_name,
        "base_model": base_model,
        "dataset": DATASET_YAML,
        "epochs": epochs,
        "batch_size": batch_size,
        "resolution": imgsz,
        "learning_rate": lr0,
        "seed": seed,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "train_duration_sec": round(train_duration_sec, 1),
        "best_checkpoint": str(best_ckpt),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "inference_latency_ms": round(mean_lat, 2),
        "baseline_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "delta": {
            "precision": round(ft_metrics["precision"] - base_metrics["precision"], 4),
            "recall": round(ft_metrics["recall"] - base_metrics["recall"], 4),
            "f1_score": round(ft_metrics["f1_score"] - base_metrics["f1_score"], 4),
            "mean_iou": round(ft_metrics["mean_iou_tp"] - base_metrics["mean_iou_tp"], 4),
        },
    }

    out_file = Path("artifacts/finetune_experiment_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved full experiment results to:", out_file)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--name", type=str, default="furniture_yolo11n_finetune")
    args = parser.parse_args()

    run_experiment(
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        experiment_name=args.name,
    )
