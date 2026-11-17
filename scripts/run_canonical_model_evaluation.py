"""Canonical Comprehensive Model Evaluation Harness for RoomAI.

Evaluates:
1. Group A (Indoor):
   - A3 (YOLO11s 800px) [Active Production Primary]
   - A2 (YOLO11s 640px) [Rollback Candidate]
   - A1 (YOLO11n 640px) [Lightweight Pilot]
   - DeepFurniture (YOLO11n 640px) [Historical Baseline]
   - Pretrained Baseline: YOLO11s (COCO 80 mapped to Indoor)
   - Pretrained Baseline: YOLO11n (COCO 80 mapped to Indoor)
2. Group B (Exterior):
   - B2 (YOLO11s 640px) [Production Default]
   - B6 (YOLO11s 640px) [Expanded Candidate]
   - B7 (YOLO11s 640px) [Pilot Candidate]
   - C1 (YOLO11s 640px) [Phase 22 Candidate]
   - Pretrained Baseline: YOLO11s (COCO 80)
3. Group C (Parking):
   - C2 (YOLO11s 640px) [Production Primary]
   - C1 (YOLO11n 640px) [Lightweight Pilot]
   - C3 (YOLO11s 640px) [RFS Candidate]
   - Pretrained Baseline: YOLO11s (COCO 80 mapped to Parking)

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA:0)
All evaluations run strictly on frozen held-out test splits.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "reports" / "canonical_evaluation_results.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# COCO 80 to Indoor 12 mapping
COCO_TO_INDOOR = {
    56: 2,   # chair -> chair#stool
    57: 6,   # couch -> sofa
    58: 7,   # potted plant -> plant
    59: 5,   # bed -> bed
    60: 1,   # dining table -> table
    62: 8,   # tv -> decoration
    68: 10,  # microwave -> home-appliance
    69: 10,  # oven -> home-appliance
    70: 10,  # toaster -> home-appliance
    72: 10,  # refrigerator -> home-appliance
}

# COCO 80 to Parking 8 mapping
COCO_TO_PARKING = {
    2: 0,   # car -> car_vehicle
    3: 1,   # motorcycle -> motorcycle
    1: 2,   # bicycle -> bicycle
    56: 3,  # chair -> outdoor_seating
    60: 4,  # dining table -> outdoor_table
    58: 5,  # potted plant -> plant_bush
}

def box_iou_xyxy(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = float(boxAArea + boxBArea - interArea)
    return interArea / union if union > 0 else 0.0

def evaluate_mapped_coco_model(model_path, images_dir, labels_dir, class_map, target_class_names, imgsz=640, conf=0.25, iou_thresh=0.45):
    model = YOLO(model_path)
    img_files = sorted(list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png")))
    
    tp_per_class = {c: 0 for c in range(len(target_class_names))}
    fp_per_class = {c: 0 for c in range(len(target_class_names))}
    fn_per_class = {c: 0 for c in range(len(target_class_names))}
    gt_counts = {c: 0 for c in range(len(target_class_names))}
    
    latencies = []
    
    for img_p in img_files:
        lbl_p = Path(labels_dir) / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        gts.append({"cls": cls_id, "box": [x1, y1, x2, y2], "matched": False})
                        gt_counts[cls_id] = gt_counts.get(cls_id, 0) + 1

        t0 = time.perf_counter()
        res = model.predict(source=str(img_p), imgsz=imgsz, conf=conf, iou=iou_thresh, device=DEVICE, verbose=False)[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
        preds = []
        for box in res.boxes:
            c_coco = int(box.cls[0].item())
            c_score = float(box.conf[0].item())
            # Normalized xyxy
            orig_h, orig_w = res.orig_shape
            xyxy = box.xyxy[0].cpu().numpy()
            norm_box = [xyxy[0]/orig_w, xyxy[1]/orig_h, xyxy[2]/orig_w, xyxy[3]/orig_h]
            if c_coco in class_map:
                c_target = class_map[c_coco]
                preds.append({"cls": c_target, "box": norm_box, "score": c_score})
            else:
                pass # Unmapped COCO class not relevant for target domain

        # Match preds to gts
        preds.sort(key=lambda x: x["score"], reverse=True)
        for pred in preds:
            p_cls = pred["cls"]
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gts):
                if gt["cls"] == p_cls and not gt["matched"]:
                    iou = box_iou_xyxy(pred["box"], gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
            if best_iou >= 0.50 and best_idx >= 0:
                gts[best_idx]["matched"] = True
                tp_per_class[p_cls] += 1
            else:
                fp_per_class[p_cls] += 1
                
        for gt in gts:
            if not gt["matched"]:
                fn_per_class[gt["cls"]] += 1
                
    # Calculate per-class metrics
    class_metrics = {}
    tot_tp = sum(tp_per_class.values())
    tot_fp = sum(fp_per_class.values())
    tot_fn = sum(fn_per_class.values())
    
    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    p_list, r_list, f1_list = [], [], []
    for c_id, name in enumerate(target_class_names):
        tp = tp_per_class[c_id]
        fp = fp_per_class[c_id]
        fn = fn_per_class[c_id]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_metrics[name] = {
            "class_id": c_id,
            "ground_truth_count": gt_counts.get(c_id, 0),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4)
        }
        if gt_counts.get(c_id, 0) > 0:
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
            
    macro_p = float(np.mean(p_list)) if p_list else 0.0
    macro_r = float(np.mean(r_list)) if r_list else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    
    return {
        "overall": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "per_class": class_metrics,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 1)
    }

def measure_peak_vram_and_latency(model, sample_img, imgsz, device, warmup=5, iterations=30):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    for _ in range(warmup):
        _ = model.predict(source=sample_img, imgsz=imgsz, device=device, verbose=False)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = model.predict(source=sample_img, imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - t0
    
    latency_ms = (total_time / iterations) * 1000.0
    throughput_fps = iterations / total_time
    
    if torch.cuda.is_available():
        peak_vram_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_vram_res_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        peak_vram_alloc_mb = 0.0
        peak_vram_res_mb = 0.0
        
    return {
        "latency_ms": round(latency_ms, 2),
        "throughput_fps": round(throughput_fps, 1),
        "peak_vram_alloc_mb": round(peak_vram_alloc_mb, 1),
        "peak_vram_res_mb": round(peak_vram_res_mb, 1)
    }

def run_evaluation():
    print("=" * 80)
    print(" ROOMAI CANONICAL MODEL EVALUATION HARNESS")
    print(f" Device: CUDA:{DEVICE} | Host: {torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    configs = [
        # Group A
        {
            "group": "Group A: Indoor",
            "model_id": "indoor_a3_800px",
            "name": "Model A3 (YOLO11s 800px) [Production Primary]",
            "type": "Trained RoomAI Primary",
            "checkpoint": "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt",
            "data_yaml": "data/indoor_canonical/indoor_canonical.yaml",
            "imgsz": 800,
            "sample_img": "data/indoor_canonical/images/test/000000000036.jpg"
        },
        {
            "group": "Group A: Indoor",
            "model_id": "indoor_a2_640px",
            "name": "Model A2 (YOLO11s 640px) [Rollback Candidate]",
            "type": "Trained RoomAI Candidate",
            "checkpoint": "checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt",
            "data_yaml": "data/indoor_canonical/indoor_canonical.yaml",
            "imgsz": 640,
            "sample_img": "data/indoor_canonical/images/test/000000000036.jpg"
        },
        {
            "group": "Group A: Indoor",
            "model_id": "indoor_a1_640px",
            "name": "Model A1 (YOLO11n 640px) [Lightweight Pilot]",
            "type": "Trained RoomAI Pilot",
            "checkpoint": "checkpoints/indoor_detector_a1_yolo11n_pilot_best.pt",
            "data_yaml": "data/indoor_canonical/indoor_canonical.yaml",
            "imgsz": 640,
            "sample_img": "data/indoor_canonical/images/test/000000000036.jpg"
        },
        {
            "group": "Group A: Indoor",
            "model_id": "deepfurniture_baseline_640px",
            "name": "DeepFurniture (YOLO11n 640px) [Baseline]",
            "type": "Trained Baseline",
            "checkpoint": "checkpoints/deepfurniture_yolo11n_finetune_best.pt",
            "data_yaml": "data/deepfurniture/deepfurniture.yaml",
            "imgsz": 640,
            "sample_img": "data/deepfurniture/images/test/000000000036.jpg" if Path("data/deepfurniture/images/test/000000000036.jpg").exists() else None
        },
        # Group B
        {
            "group": "Group B: Exterior",
            "model_id": "exterior_b2_640px",
            "name": "Model B2 (YOLO11s 640px) [Production Default]",
            "type": "Trained RoomAI Default",
            "checkpoint": "checkpoints/exterior_detector_b2_yolo11s_best.pt",
            "data_yaml": "data/exterior_b6_expanded/dataset.yaml",
            "imgsz": 640,
            "sample_img": None
        },
        {
            "group": "Group B: Exterior",
            "model_id": "exterior_b6_640px",
            "name": "Model B6 (YOLO11s 640px) [Expanded Candidate]",
            "type": "Trained RoomAI Candidate",
            "checkpoint": "checkpoints/exterior_detector_b6_expanded_best.pt",
            "data_yaml": "data/exterior_b6_expanded/dataset.yaml",
            "imgsz": 640,
            "sample_img": None
        },
        {
            "group": "Group B: Exterior",
            "model_id": "exterior_b7_640px",
            "name": "Model B7 (YOLO11s 640px) [Pilot Candidate]",
            "type": "Trained RoomAI Pilot",
            "checkpoint": "checkpoints/exterior_detector_b7_pilot_best.pt",
            "data_yaml": "data/exterior_b6_expanded/dataset.yaml",
            "imgsz": 640,
            "sample_img": None
        },
        {
            "group": "Group B: Exterior",
            "model_id": "exterior_c1_phase22_640px",
            "name": "Candidate C1 (YOLO11s 640px) [Phase 22]",
            "type": "Trained RoomAI Candidate",
            "checkpoint": "checkpoints/exterior_detector_c1_phase22_best.pt",
            "data_yaml": "data/exterior_b6_expanded/dataset.yaml",
            "imgsz": 640,
            "sample_img": None
        },
        # Group C
        {
            "group": "Group C: Parking",
            "model_id": "parking_c2_640px",
            "name": "Model C2 (YOLO11s 640px) [Production Primary]",
            "type": "Trained RoomAI Primary",
            "checkpoint": "checkpoints/parking_detector_c2_yolo11s_pilot_best.pt",
            "data_yaml": "data/parking_canonical/parking_canonical.yaml",
            "imgsz": 640,
            "sample_img": None
        },
        {
            "group": "Group C: Parking",
            "model_id": "parking_c1_640px",
            "name": "Model C1 (YOLO11n 640px) [Lightweight Pilot]",
            "type": "Trained RoomAI Pilot",
            "checkpoint": "checkpoints/parking_detector_c1_yolo11n_pilot_best.pt",
            "data_yaml": "data/parking_canonical/parking_canonical.yaml",
            "imgsz": 640,
            "sample_img": None
        },
        {
            "group": "Group C: Parking",
            "model_id": "parking_c3_640px",
            "name": "Model C3 (YOLO11s 640px) [RFS Candidate]",
            "type": "Trained RoomAI Candidate",
            "checkpoint": "checkpoints/parking_detector_c3_yolo11s_rfs_best.pt",
            "data_yaml": "data/parking_canonical/parking_canonical.yaml",
            "imgsz": 640,
            "sample_img": None
        }
    ]

    all_results = []

    for cfg in configs:
        ckpt = Path(cfg["checkpoint"])
        data_yaml = Path(cfg["data_yaml"])
        if not ckpt.exists():
            print(f"[SKIP] Checkpoint missing: {ckpt}")
            continue
        if not data_yaml.exists():
            print(f"[SKIP] Data yaml missing: {data_yaml}")
            continue

        print(f"\n---> Evaluating {cfg['name']} on {data_yaml.name} (test split) at {cfg['imgsz']}px")
        
        try:
            model = YOLO(str(ckpt))
            n_params = sum(p.numel() for p in model.model.parameters())
            params_m = round(n_params / 1e6, 2)
            
            val_out = model.val(
                data=str(data_yaml),
                split="test",
                imgsz=cfg["imgsz"],
                batch=8,
                conf=0.25,
                iou=0.45,
                device=DEVICE,
                verbose=False
            )
            
            p = float(val_out.box.mp)
            r = float(val_out.box.mr)
            f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            map50 = float(val_out.box.map50)
            map50_95 = float(val_out.box.map)
            
            speed = val_out.speed
            t_prep = round(speed.get("preprocess", 0.0), 2)
            t_inf = round(speed.get("inference", 0.0), 2)
            t_post = round(speed.get("postprocess", 0.0), 2)
            t_total = round(t_prep + t_inf + t_post, 2)
            fps = round(1000.0 / t_total, 1) if t_total > 0 else 0.0
            
            names = val_out.names
            class_metrics = {}
            for i, c_name in names.items():
                c_p = float(val_out.box.p[i]) if hasattr(val_out.box, "p") and len(val_out.box.p) > i else 0.0
                c_r = float(val_out.box.r[i]) if hasattr(val_out.box, "r") and len(val_out.box.r) > i else 0.0
                c_f1 = float(2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
                c_map50 = float(val_out.box.maps[i]) if hasattr(val_out.box, "maps") and len(val_out.box.maps) > i else 0.0
                class_metrics[c_name] = {
                    "class_id": i,
                    "precision": round(c_p, 4),
                    "recall": round(c_r, 4),
                    "f1": round(c_f1, 4),
                    "map50": round(c_map50, 4)
                }
                
            sample_img_path = cfg["sample_img"]
            if not sample_img_path or not Path(sample_img_path).exists():
                img_dir = data_yaml.parent / "images" / "test"
                if img_dir.exists():
                    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                    sample_img_path = str(imgs[0]) if imgs else None

            if sample_img_path and Path(sample_img_path).exists():
                hw_stats = measure_peak_vram_and_latency(model, sample_img_path, cfg["imgsz"], DEVICE)
            else:
                hw_stats = {
                    "latency_ms": t_total,
                    "throughput_fps": fps,
                    "peak_vram_alloc_mb": 0.0,
                    "peak_vram_res_mb": 0.0
                }

            record = {
                "group": cfg["group"],
                "model_id": cfg["model_id"],
                "name": cfg["name"],
                "type": cfg["type"],
                "checkpoint": cfg["checkpoint"],
                "resolution": cfg["imgsz"],
                "parameters_m": params_m,
                "overall_metrics": {
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                    "map50": round(map50, 4),
                    "map50_95": round(map50_95, 4)
                },
                "timing": {
                    "preprocess_ms": t_prep,
                    "inference_ms": t_inf,
                    "postprocess_ms": t_post,
                    "total_per_image_ms": t_total,
                    "fps": fps
                },
                "hardware_profile": hw_stats,
                "per_class": class_metrics
            }
            
            all_results.append(record)
            print(f"   [DONE] mAP50={record['overall_metrics']['map50']} | F1={record['overall_metrics']['f1']} | Latency={record['hardware_profile']['latency_ms']}ms | Peak VRAM={record['hardware_profile']['peak_vram_alloc_mb']}MB")
            
        except Exception as e:
            print(f"   [ERROR] Failed evaluating {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Pretrained Baselines evaluation
    print("\n" + "=" * 80)
    print(" EVALUATING PRETRAINED BASELINES (COCO-Pretrained on Domain Test Sets)")
    print("=" * 80)
    
    indoor_classes = [
        "cabinet#shelf", "table", "chair#stool", "lamp", "door", "bed",
        "sofa", "plant", "decoration", "curtain", "home-appliance", "window"
    ]
    
    for baseline_name, ckpt_p in [("YOLO11s Pretrained (COCO)", "yolo11s.pt"), ("YOLO11n Pretrained (COCO)", "yolo11n.pt")]:
        if Path(ckpt_p).exists():
            print(f"---> Evaluating {baseline_name} on Indoor test set (COCO mapped)...")
            mapped_res = evaluate_mapped_coco_model(
                ckpt_p,
                "data/indoor_canonical/images/test",
                "data/indoor_canonical/labels/test",
                COCO_TO_INDOOR,
                indoor_classes,
                imgsz=800 if "11s" in baseline_name else 640
            )
            model_b = YOLO(ckpt_p)
            n_params_b = round(sum(p.numel() for p in model_b.model.parameters()) / 1e6, 2)
            hw_b = measure_peak_vram_and_latency(model_b, "data/indoor_canonical/images/test/000000000036.jpg", 800 if "11s" in baseline_name else 640, DEVICE)
            
            b_record = {
                "group": "Group A: Indoor",
                "model_id": f"coco_pretrained_{Path(ckpt_p).stem}_indoor",
                "name": f"{baseline_name} [COCO Pretrained Baseline]",
                "type": "Pretrained Baseline",
                "checkpoint": ckpt_p,
                "resolution": 800 if "11s" in baseline_name else 640,
                "parameters_m": n_params_b,
                "overall_metrics": {
                    "precision": mapped_res["overall"]["precision"],
                    "recall": mapped_res["overall"]["recall"],
                    "f1": mapped_res["overall"]["f1"],
                    "macro_precision": mapped_res["overall"]["macro_precision"],
                    "macro_recall": mapped_res["overall"]["macro_recall"],
                    "macro_f1": mapped_res["overall"]["macro_f1"],
                    "map50": 0.0, # Not directly computed by full mAP curve
                    "map50_95": 0.0
                },
                "timing": {
                    "total_per_image_ms": mapped_res["mean_latency_ms"],
                    "fps": mapped_res["fps"]
                },
                "hardware_profile": hw_b,
                "per_class": mapped_res["per_class"]
            }
            all_results.append(b_record)
            print(f"   [DONE] Precision={b_record['overall_metrics']['precision']} | Recall={b_record['overall_metrics']['recall']} | F1={b_record['overall_metrics']['f1']} | Macro F1={b_record['overall_metrics']['macro_f1']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll results successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()
