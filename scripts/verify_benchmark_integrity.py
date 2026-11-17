"""
RoomAI — Independent Benchmark Verification & Rigorous Integrity Suite
Audits Phases A-D:
1. Exact 50 benchmark images audit & leakage check.
2. Complete ADE20K to RoomAI 150-class mapping.
3. Independent mIoU recalculation (ad-hoc 10-class vs full dataset standard).
4. Instance segmentation box-to-mask correspondence & failure analysis.
5. Metric depth model card audit & ground-truth availability check.
6. Unified 4-model simultaneous pipeline latency & peak VRAM benchmark.
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
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForDepthEstimation
from ultralytics import YOLO, FastSAM

REPO_ROOT = Path(__file__).resolve().parent.parent
ADE_ROOT = REPO_ROOT / "archive (2)" / "ADEChallengeData2016"
REPORTS_DIR = REPO_ROOT / "reports"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_CLASSES_10 = {
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


def audit_eval_images():
    print("\n=======================================================")
    print("[1] AUDITING EVALUATION IMAGES & LEAKAGE CHECK")
    print("=======================================================")
    scene_file = ADE_ROOT / "sceneCategories.txt"
    categories = ["bedroom", "living_room", "kitchen", "bathroom", "dining_room"]
    selected = []
    cat_counts = {c: 0 for c in categories}

    with open(scene_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id, sc = parts
                if "ADE_val_" in img_id and sc in cat_counts and cat_counts[sc] < 10:
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
                        if all(c >= 10 for c in cat_counts.values()):
                            break

    print(f"Total benchmark images: {len(selected)}")
    print("Category breakdown:", cat_counts)

    # Check for leakage against training sets
    train_ade_images = set(p.stem for p in (ADE_ROOT / "images" / "training").glob("*.jpg"))
    indoor_canonical_images = set()
    canonical_dir = REPO_ROOT / "data" / "indoor_canonical"
    if canonical_dir.exists():
        for p in canonical_dir.glob("**/*.jpg"):
            indoor_canonical_images.add(p.stem)

    val_stems = set(item["id"] for item in selected)
    leakage_train = val_stems.intersection(train_ade_images)
    leakage_canonical = val_stems.intersection(indoor_canonical_images)

    print(f"Leakage with ADE20K training set: {len(leakage_train)} images")
    print(f"Leakage with RoomAI indoor canonical set: {len(leakage_canonical)} images")
    return selected


def independent_semantic_recalculation(eval_items):
    print("\n=======================================================")
    print("[2] INDEPENDENT SEMANTIC EVALUATION & mIoU AUDIT")
    print("=======================================================")
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512").to(DEVICE)
    model.eval()

    # 1. Re-run previous ad-hoc calculation (10 classes, image-instance averaging)
    adhoc_all_ious = []
    adhoc_class_ious = {cid: [] for cid in TARGET_CLASSES_10}

    # 2. Standard Dataset-Level Confusion Matrix (150 classes)
    num_classes = 150
    total_confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for item in eval_items:
        img = Image.open(item["image_path"]).convert("RGB")
        orig_w, orig_h = img.size
        gt_mask = np.array(Image.open(item["mask_path"]))

        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy() + 1  # 1-based ADE class index

        # Ad-hoc evaluation (reproducing phase C)
        for cid in TARGET_CLASSES_10:
            pred_c = (pred_mask == cid)
            gt_c = (gt_mask == cid)
            intersection = np.logical_and(pred_c, gt_c).sum()
            union = np.logical_or(pred_c, gt_c).sum()
            if union > 0:
                iou = float(intersection / union)
                adhoc_class_ious[cid].append(iou)
                adhoc_all_ious.append(iou)

        # Standard Confusion Matrix (ignoring 0 = unlabeled)
        valid_mask = (gt_mask > 0) & (gt_mask <= num_classes)
        gt_valid = gt_mask[valid_mask]
        pred_valid = pred_mask[valid_mask]
        for g, p in zip(gt_valid.ravel(), pred_valid.ravel()):
            total_confusion_matrix[g, p] += 1

    # Ad-hoc result
    reproduced_adhoc_miou = float(np.mean(adhoc_all_ious))
    print(f"Reproduced Phase C Ad-Hoc 10-Class Instance-Averaged mIoU: {reproduced_adhoc_miou:.4f}")

    # Standard Dataset mIoU over 10 target classes
    target_10_dataset_ious = {}
    for cid, name in TARGET_CLASSES_10.items():
        tp = total_confusion_matrix[cid, cid]
        fp = total_confusion_matrix[:, cid].sum() - tp
        fn = total_confusion_matrix[cid, :].sum() - tp
        denom = tp + fp + fn
        target_10_dataset_ious[name] = float(tp / denom) if denom > 0 else 0.0

    std_10_miou = float(np.mean(list(target_10_dataset_ious.values())))
    print(f"Standard Dataset-Aggregated IoU for 10 Target Classes: {std_10_miou:.4f}")
    for k, v in target_10_dataset_ious.items():
        print(f"  {k:12s}: {v:.4f}")

    # Standard Dataset mIoU over all classes present in validation set
    present_classes = []
    present_ious = []
    for c in range(1, num_classes + 1):
        tp = total_confusion_matrix[c, c]
        fp = total_confusion_matrix[:, c].sum() - tp
        fn = total_confusion_matrix[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            present_classes.append(c)
            present_ious.append(tp / denom)

    std_present_miou = float(np.mean(present_ious))
    print(f"Standard Dataset mIoU over ALL {len(present_classes)} classes present in 50 images: {std_present_miou:.4f}")

    del model, processor
    torch.cuda.empty_cache()
    return {
        "reproduced_adhoc_miou": round(reproduced_adhoc_miou, 4),
        "standard_10_class_dataset_miou": round(std_10_miou, 4),
        "standard_all_present_classes_miou": round(std_present_miou, 4),
        "target_10_per_class_iou": {k: round(v, 4) for k, v in target_10_dataset_ious.items()},
    }


def audit_instance_segmentation(eval_items):
    print("\n=======================================================")
    print("[3] INSTANCE SEGMENTATION BOX-TO-MASK AUDIT")
    print("=======================================================")
    a3_path = REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt"
    model_a3 = YOLO(str(a3_path))
    model_fastsam = FastSAM("FastSAM-s.pt")

    total_boxes = 0
    total_masks = 0
    box_to_mask_coverage = []
    merged_or_empty_masks = 0
    per_class_box_counts = {name: 0 for name in model_a3.names.values()}
    per_class_mask_areas = {name: [] for name in model_a3.names.values()}

    for item in eval_items:
        det = model_a3(item["image_path"], imgsz=800, conf=0.25, device="cuda", verbose=False)[0]
        boxes = det.boxes
        n_b = len(boxes)
        total_boxes += n_b

        if n_b > 0:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            classes = [model_a3.names[int(c)] for c in boxes.cls.cpu().numpy()]
            sam_res = model_fastsam(item["image_path"], bboxes=boxes_xyxy.tolist(), device="cuda", verbose=False)[0]

            if sam_res.masks is not None:
                n_m = len(sam_res.masks)
                total_masks += n_m
                # FastSAM returns masks.data of shape (N, H_mask, W_mask)
                # Resized to orig image size for accurate geometry check
                masks_data = sam_res.masks.data.cpu().numpy()
                orig_h, orig_w = sam_res.orig_shape

                for i, (box, cls_name) in enumerate(zip(boxes_xyxy, classes)):
                    per_class_box_counts[cls_name] += 1
                    if i < len(masks_data):
                        mask_raw = masks_data[i]
                        # Resize to orig image size
                        mask_pil = Image.fromarray((mask_raw * 255).astype(np.uint8)).resize(
                            (orig_w, orig_h), Image.NEAREST
                        )
                        mask_arr = np.array(mask_pil) > 127
                        mask_area = mask_arr.sum()
                        per_class_mask_areas[cls_name].append(int(mask_area))

                        # Check bounding box IoU of mask vs input box
                        box_area = (box[2] - box[0]) * (box[3] - box[1])
                        if mask_area == 0:
                            merged_or_empty_masks += 1
                            box_to_mask_coverage.append(0.0)
                        else:
                            # Fraction of mask that falls inside the prompt box
                            y1, y2 = int(box[1]), int(box[3])
                            x1, x2 = int(box[0]), int(box[2])
                            mask_inside_box = mask_arr[y1:y2, x1:x2].sum()
                            in_box_ratio = float(mask_inside_box / mask_area)
                            box_to_mask_coverage.append(in_box_ratio)
            else:
                merged_or_empty_masks += n_b

    mean_coverage = float(np.mean(box_to_mask_coverage)) if box_to_mask_coverage else 0.0
    print(f"Total A3 Bounding Boxes Detected: {total_boxes}")
    print(f"Total FastSAM Masks Generated: {total_masks}")
    print(f"Mask-Inside-Prompt-Box Containment Ratio: {mean_coverage * 100:.2f}%")
    print(f"Empty or Failed Masks: {merged_or_empty_masks}")

    print("\nPer-Class Detection and Mask Generation Summary:")
    for cls_name, count in per_class_box_counts.items():
        areas = per_class_mask_areas[cls_name]
        mean_area = float(np.mean(areas)) if areas else 0.0
        print(f"  {cls_name:16s}: {count:3d} boxes, mean mask pixels: {mean_area:.1f}")

    del model_a3, model_fastsam
    torch.cuda.empty_cache()
    return {
        "total_boxes": total_boxes,
        "total_masks": total_masks,
        "mask_in_box_containment_ratio": round(mean_coverage, 4),
        "empty_masks": merged_or_empty_masks,
        "class_box_counts": per_class_box_counts,
    }


def audit_metric_depth():
    print("\n=======================================================")
    print("[4] METRIC DEPTH AUDIT & GROUND-TRUTH AVAILABILITY")
    print("=======================================================")
    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(DEVICE)
    model.eval()

    print(f"Checkpoint Verified: {model_id}")
    print(f"Architecture: {model.config.architectures}")
    print(f"Max depth setting: {model.config.max_depth} meters")
    print(f"Depth estimation type: {model.config.depth_estimation_type}")

    # Check for ground-truth depth in workspace
    gt_depth_candidates = list(REPO_ROOT.glob("**/*depth*.npy"))
    print(f"Workspace Ground-Truth Depth Maps Found: {len(gt_depth_candidates)}")
    if len(gt_depth_candidates) == 0:
        print("  [AUDIT FINDING] Ground-truth depth maps (NYUv2 / ScanNet / SUN RGB-D) are NOT present in the workspace.")
        print("  [AUDIT FINDING] Quantitative metrics (AbsRel, RMSE, log RMSE, Delta accuracy) CANNOT be computed without ground-truth depth.")
        print("  [AUDIT FINDING] Evaluation is strictly limited to model specification verification, inference latency, VRAM allocation, and physical range plausibility.")

    del model, processor
    torch.cuda.empty_cache()


def benchmark_unified_pipeline(eval_items):
    print("\n=======================================================")
    print("[5] UNIFIED 4-TIER SIMULTANEOUS PIPELINE BENCHMARK")
    print("=======================================================")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load ALL 4 models into VRAM simultaneously
    t_load_start = time.perf_counter()
    a3_path = REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt"
    model_a3 = YOLO(str(a3_path))
    model_fastsam = FastSAM("FastSAM-s.pt")
    proc_sem = AutoImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    model_sem = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512").to(DEVICE)
    proc_depth = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    model_depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf").to(DEVICE)

    model_sem.eval()
    model_depth.eval()
    t_load = (time.perf_counter() - t_load_start) * 1000.0

    vram_static_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_static_res = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"All 4 Models Loaded in VRAM in {t_load:.1f} ms.")
    print(f"Static VRAM Allocated: {vram_static_alloc:.1f} MB, Reserved: {vram_static_res:.1f} MB")

    # Warmup (5 cycles)
    dummy_img = Image.new("RGB", (640, 480), color="white")
    dummy_np = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model_a3(dummy_np, imgsz=800, verbose=False)
        _ = model_fastsam(dummy_np, bboxes=[[10, 10, 100, 100]], verbose=False)
        in_s = proc_sem(images=dummy_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model_sem(**in_s)
        in_d = proc_depth(images=dummy_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model_depth(**in_d)
    torch.cuda.synchronize()

    # Latency tracking across pipeline stages
    timings = {
        "preprocess": [],
        "a3_detect": [],
        "fastsam_mask": [],
        "segformer_sem": [],
        "depth_metric": [],
        "fusion_postproc": [],
        "total_e2e": [],
    }

    for item in eval_items:
        t_e2e_start = time.perf_counter()

        # 1. Preprocessing
        t0 = time.perf_counter()
        img = Image.open(item["image_path"]).convert("RGB")
        orig_w, orig_h = img.size
        img_np = np.array(img)
        torch.cuda.synchronize()
        timings["preprocess"].append((time.perf_counter() - t0) * 1000.0)

        # 2. Stage 1: Detector A3
        t0 = time.perf_counter()
        det = model_a3(img_np, imgsz=800, conf=0.25, device="cuda", verbose=False)[0]
        boxes = det.boxes
        torch.cuda.synchronize()
        timings["a3_detect"].append((time.perf_counter() - t0) * 1000.0)

        # 3. Stage 2: FastSAM Box Prompting
        t0 = time.perf_counter()
        if len(boxes) > 0:
            bboxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
            sam_res = model_fastsam(img_np, bboxes=bboxes_xyxy, device="cuda", verbose=False)[0]
            _ = sam_res.masks
        torch.cuda.synchronize()
        timings["fastsam_mask"].append((time.perf_counter() - t0) * 1000.0)

        # 4. Stage 3: SegFormer Semantic Surfaces
        t0 = time.perf_counter()
        inputs_sem = proc_sem(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            sem_out = model_sem(**inputs_sem)
        torch.cuda.synchronize()
        timings["segformer_sem"].append((time.perf_counter() - t0) * 1000.0)

        # 5. Stage 4: Depth Anything V2 Metric
        t0 = time.perf_counter()
        inputs_depth = proc_depth(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            depth_out = model_depth(**inputs_depth)
        torch.cuda.synchronize()
        timings["depth_metric"].append((time.perf_counter() - t0) * 1000.0)

        # 6. Postprocessing & Fusion
        t0 = time.perf_counter()
        sem_logits = torch.nn.functional.interpolate(
            sem_out.logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        sem_pred = sem_logits.argmax(dim=1)[0].cpu().numpy() + 1

        pred_depth = depth_out.predicted_depth
        pred_depth = torch.nn.functional.interpolate(
            pred_depth.unsqueeze(1), size=(orig_h, orig_w), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
        torch.cuda.synchronize()
        timings["fusion_postproc"].append((time.perf_counter() - t0) * 1000.0)

        timings["total_e2e"].append((time.perf_counter() - t_e2e_start) * 1000.0)

    peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

    stats = {}
    for stage, vals in timings.items():
        stats[stage] = {
            "mean_ms": round(float(np.mean(vals)), 2),
            "std_ms": round(float(np.std(vals)), 2),
            "p50_ms": round(float(np.median(vals)), 2),
            "p95_ms": round(float(np.percentile(vals, 95)), 2),
            "p99_ms": round(float(np.percentile(vals, 99)), 2),
        }

    print("\n--- Unified Pipeline Latency Breakdown ---")
    for stage, s in stats.items():
        print(f"  {stage:16s}: Mean = {s['mean_ms']:6.2f} ms | P95 = {s['p95_ms']:6.2f} ms")
    print(f"\nPeak Pipeline VRAM Allocated: {peak_vram_alloc:.1f} MB ({peak_vram_alloc/1024:.2f} GB)")
    print(f"Peak Pipeline VRAM Reserved:  {peak_vram_res:.1f} MB ({peak_vram_res/1024:.2f} GB)")

    del model_a3, model_fastsam, model_sem, model_depth, proc_sem, proc_depth
    torch.cuda.empty_cache()

    return {
        "latency_stats": stats,
        "peak_vram_allocated_mb": round(peak_vram_alloc, 1),
        "peak_vram_reserved_mb": round(peak_vram_res, 1),
        "static_vram_allocated_mb": round(vram_static_alloc, 1),
        "static_vram_reserved_mb": round(vram_static_res, 1),
    }


def main():
    eval_items = audit_eval_images()
    semantic_audit = independent_semantic_recalculation(eval_items)
    instance_audit = audit_instance_segmentation(eval_items)
    audit_metric_depth()
    pipeline_audit = benchmark_unified_pipeline(eval_items)

    full_audit_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_image_count": len(eval_items),
        "eval_images": [item["id"] for item in eval_items],
        "semantic_audit": semantic_audit,
        "instance_audit": instance_audit,
        "pipeline_audit": pipeline_audit,
    }

    out_file = REPORTS_DIR / "benchmark_reproducibility_results.json"
    with open(out_file, "w") as f:
        json.dump(full_audit_results, f, indent=2)
    print(f"\n[SUCCESS] Independent Audit Completed. Results saved to {out_file}")


if __name__ == "__main__":
    main()

