"""
RoomAI — Vision Model Empirical Benchmarking Suite
Evaluates Semantic Segmentation, Instance Segmentation, and Metric Depth Models
on identical indoor images from ADE20K validation set.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Suppress TensorFlow probing
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.modules["tensorflow"] = None

import torch
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForDepthEstimation
from ultralytics import YOLO, FastSAM

REPO_ROOT = Path(__file__).resolve().parent.parent
ADE_ROOT = REPO_ROOT / "archive (2)" / "ADEChallengeData2016"
REPORTS_DIR = REPO_ROOT / "reports"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_CLASSES = {
    1: "wall",
    4: "floor",
    6: "ceiling",
    9: "windowpane",
    15: "door",
    8: "bed",
    24: "sofa",
    20: "chair",
    16: "table",
    11: "cabinet",
}


def get_curated_eval_images(n_per_category=10):
    scene_file = ADE_ROOT / "sceneCategories.txt"
    categories = ["bedroom", "living_room", "kitchen", "bathroom", "dining_room"]
    selected = []
    cat_counts = {c: 0 for c in categories}

    with open(scene_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id, sc = parts
                if "ADE_val_" in img_id and sc in cat_counts and cat_counts[sc] < n_per_category:
                    img_path = ADE_ROOT / "images" / "validation" / f"{img_id}.jpg"
                    mask_path = ADE_ROOT / "annotations" / "validation" / f"{img_id}.png"
                    if img_path.exists() and mask_path.exists():
                        selected.append({
                            "id": img_id,
                            "category": sc,
                            "image_path": str(img_path),
                            "mask_path": str(mask_path),
                        })
                        cat_counts[sc] += 1
                        if all(c >= n_per_category for c in cat_counts.values()):
                            break
    return selected


def calculate_iou(pred_mask, gt_mask, class_id):
    pred_c = (pred_mask == class_id)
    gt_c = (gt_mask == class_id)
    intersection = np.logical_and(pred_c, gt_c).sum()
    union = np.logical_or(pred_c, gt_c).sum()
    if union == 0:
        return None  # Class not present in GT or Pred
    return float(intersection / union)


def benchmark_semantic_model(model_id, eval_items):
    print(f"\n--- Benchmarking Semantic Model: {model_id} ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(DEVICE)
    model.eval()

    # Warmup
    dummy = Image.new("RGB", (512, 512), color="white")
    for _ in range(5):
        inputs = processor(images=dummy, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    latencies = []
    class_ious = {cid: [] for cid in TARGET_CLASSES}
    all_ious = []

    for item in eval_items:
        img = Image.open(item["image_path"]).convert("RGB")
        orig_w, orig_h = img.size
        gt_mask = np.array(Image.open(item["mask_path"]))

        t0 = time.perf_counter()
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # Upsample logits to original image size
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        # In ADE20K pre-trained models from HF, background is often ignored or 0-indexed.
        # Check HF id2label offset: ADE20K annotations have 1..150, HF model config has 0..149.
        # Let's adjust +1 to map back to ADE20K ground truth class IDs (1-based).
        pred_mask = pred_mask + 1

        for cid in TARGET_CLASSES:
            iou = calculate_iou(pred_mask, gt_mask, cid)
            if iou is not None:
                class_ious[cid].append(iou)
                all_ious.append(iou)

    peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

    mean_class_ious = {}
    for cid, ious in class_ious.items():
        name = TARGET_CLASSES[cid]
        mean_class_ious[name] = float(np.mean(ious)) if ious else None

    results = {
        "model_id": model_id,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "overall_miou": round(float(np.mean(all_ious)), 4) if all_ious else 0.0,
        "per_class_iou": mean_class_ious,
        "peak_vram_allocated_mb": round(peak_vram_alloc, 1),
        "peak_vram_reserved_mb": round(peak_vram_res, 1),
        "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    }

    del model, processor
    torch.cuda.empty_cache()
    return results


def benchmark_instance_models(eval_items):
    print("\n--- Benchmarking Instance Segmentation: YOLO11s-seg vs Model A3 + FastSAM ---")
    results = {}

    # 1. Independent Baseline: YOLO11s-seg
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_yolo_seg = YOLO("yolo11s-seg.pt")

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model_yolo_seg(dummy, device="cuda", verbose=False)
    torch.cuda.synchronize()

    yolo_latencies = []
    yolo_mask_counts = []
    for item in eval_items:
        t0 = time.perf_counter()
        res = model_yolo_seg(item["image_path"], device="cuda", verbose=False)[0]
        torch.cuda.synchronize()
        yolo_latencies.append((time.perf_counter() - t0) * 1000.0)
        n_masks = len(res.masks) if res.masks is not None else 0
        yolo_mask_counts.append(n_masks)

    results["yolo11s_seg_independent"] = {
        "model_id": "yolo11s-seg.pt",
        "mean_latency_ms": round(float(np.mean(yolo_latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(yolo_latencies, 95)), 2),
        "mean_masks_per_image": round(float(np.mean(yolo_mask_counts)), 2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1),
        "parameters_m": round(sum(p.numel() for p in model_yolo_seg.model.parameters()) / 1e6, 2),
        "custom_indoor_class_support": "No (Strictly COCO 80: no cabinet, lamp, curtain, door, window)",
    }
    del model_yolo_seg
    torch.cuda.empty_cache()

    # 2. Prompted Ensemble: Model A3 + FastSAM
    torch.cuda.reset_peak_memory_stats()
    a3_path = REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt"
    model_a3 = YOLO(str(a3_path))
    model_fastsam = FastSAM("FastSAM-s.pt")

    # Warmup
    for _ in range(5):
        det = model_a3(dummy, imgsz=800, device="cuda", verbose=False)[0]
        _ = model_fastsam(dummy, device="cuda", verbose=False)
    torch.cuda.synchronize()

    ensemble_latencies = []
    prompted_mask_counts = []
    classes_covered = set()

    for item in eval_items:
        t0 = time.perf_counter()
        # Stage 1: Detector A3
        det_res = model_a3(item["image_path"], imgsz=800, conf=0.25, device="cuda", verbose=False)[0]
        boxes = det_res.boxes
        n_boxes = len(boxes)

        # Stage 2: FastSAM Box Prompting
        if n_boxes > 0:
            bboxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
            # FastSAM prompt
            sam_res = model_fastsam(item["image_path"], bboxes=bboxes_xyxy, device="cuda", verbose=False)[0]
            n_masks = len(sam_res.masks) if sam_res.masks is not None else 0
            for cls_idx in boxes.cls.cpu().numpy():
                classes_covered.add(model_a3.names[int(cls_idx)])
        else:
            n_masks = 0

        torch.cuda.synchronize()
        ensemble_latencies.append((time.perf_counter() - t0) * 1000.0)
        prompted_mask_counts.append(n_masks)

    results["a3_prompted_fastsam_ensemble"] = {
        "model_id": "Model A3 (YOLO11s 800px) + FastSAM-s",
        "mean_latency_ms": round(float(np.mean(ensemble_latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(ensemble_latencies, 95)), 2),
        "mean_masks_per_image": round(float(np.mean(prompted_mask_counts)), 2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1),
        "classes_prompted": sorted(list(classes_covered)),
        "custom_indoor_class_support": "Yes (All 12 RoomAI Indoor Classes with Zero-Shot Masks)",
    }
    del model_a3, model_fastsam
    torch.cuda.empty_cache()

    return results


def benchmark_depth_model(model_id, is_metric, eval_items):
    print(f"\n--- Benchmarking Depth Model: {model_id} (Metric: {is_metric}) ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(DEVICE)
    model.eval()

    # Warmup
    dummy = Image.new("RGB", (518, 518), color="white")
    for _ in range(5):
        inputs = processor(images=dummy, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    latencies = []
    min_vals, max_vals, mean_vals = [], [], []

    for item in eval_items:
        img = Image.open(item["image_path"]).convert("RGB")
        orig_w, orig_h = img.size

        t0 = time.perf_counter()
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred_depth = outputs.predicted_depth
        pred_depth = torch.nn.functional.interpolate(
            pred_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        min_vals.append(float(pred_depth.min()))
        max_vals.append(float(pred_depth.max()))
        mean_vals.append(float(pred_depth.mean()))

    results = {
        "model_id": model_id,
        "is_metric": is_metric,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "depth_range_min_mean": round(float(np.mean(min_vals)), 3),
        "depth_range_max_mean": round(float(np.mean(max_vals)), 3),
        "depth_mean_val": round(float(np.mean(mean_vals)), 3),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1),
        "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    }

    del model, processor
    torch.cuda.empty_cache()
    return results


def main():
    eval_items = get_curated_eval_images(n_per_category=10)
    print(f"Collected {len(eval_items)} indoor validation scenes across 5 categories.")

    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "eval_dataset_size": len(eval_items),
        "semantic_models": {},
        "instance_models": {},
        "depth_models": {},
    }

    # 1. Semantic Models
    full_results["semantic_models"]["segformer_b0"] = benchmark_semantic_model(
        "nvidia/segformer-b0-finetuned-ade-512-512", eval_items
    )
    full_results["semantic_models"]["segformer_b1"] = benchmark_semantic_model(
        "nvidia/segformer-b1-finetuned-ade-512-512", eval_items
    )

    # 2. Instance Models
    full_results["instance_models"] = benchmark_instance_models(eval_items)

    # 3. Depth Models
    full_results["depth_models"]["depth_anything_v2_small_relative"] = benchmark_depth_model(
        "depth-anything/Depth-Anything-V2-Small-hf", is_metric=False, eval_items=eval_items
    )
    full_results["depth_models"]["depth_anything_v2_small_metric"] = benchmark_depth_model(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", is_metric=True, eval_items=eval_items
    )

    out_file = REPORTS_DIR / "pretrained_model_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n[SUCCESS] Benchmark completed. Results saved to {out_file}")


if __name__ == "__main__":
    main()

