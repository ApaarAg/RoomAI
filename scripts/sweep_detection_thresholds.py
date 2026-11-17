"""Validation Threshold & NMS-IoU Sweeps for RoomAI Detectors.

Evaluates confidence thresholds and NMS-IoU parameters STRICTLY on the
validation split to prevent test-set data snooping.
Finds optimal operating points (Max F1, Balanced P/R, High-Recall) and
verifies them on the held-out test split.
"""

import json
from pathlib import Path
import time
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEPS = [
    {
        "group": "Group A: Indoor Furniture & Household Objects",
        "model_id": "indoor_a2_finetuned",
        "checkpoint": "checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt",
        "data_yaml": "data/indoor_canonical/indoor_canonical.yaml",
        "conf_range": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
        "iou_range": [0.40, 0.45, 0.50, 0.60],
    },
    {
        "group": "Group B: Exterior & Building Architecture",
        "model_id": "exterior_b2_finetuned",
        "checkpoint": "checkpoints/exterior_detector_b2_yolo11s_best.pt",
        "data_yaml": "data/exterior_canonical/exterior_canonical.yaml",
        "conf_range": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "iou_range": [0.40, 0.45, 0.50, 0.60],
    },
    {
        "group": "Group C: Parking & Outdoor Grounds",
        "model_id": "parking_c2_pilot",
        "checkpoint": "checkpoints/parking_detector_c2_yolo11s_pilot_best.pt",
        "data_yaml": "data/parking_canonical/parking_canonical.yaml",
        "conf_range": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "iou_range": [0.40, 0.45, 0.50, 0.60],
    },
]


def run_sweep(cfg: dict) -> dict:
    ckpt_path = REPO_ROOT / cfg["checkpoint"]
    data_path = REPO_ROOT / cfg["data_yaml"]
    print("\n" + "=" * 80)
    print(f"Sweeping Thresholds on VAL split for: {cfg['group']}")
    print(f"Checkpoint: {ckpt_path.name}")
    print("=" * 80)

    model = YOLO(str(ckpt_path))
    sweep_results = []
    best_f1_entry = None
    best_f1 = -1.0

    for conf in cfg["conf_range"]:
        for iou in cfg["iou_range"]:
            val_res = model.val(
                data=str(data_path),
                split="val",
                batch=8,
                imgsz=640,
                conf=conf,
                iou=iou,
                device=0 if torch.cuda.is_available() else "cpu",
                plots=False,
                verbose=False,
                save_json=False,
            )
            mp = float(val_res.box.mp)
            mr = float(val_res.box.mr)
            f1 = (2 * mp * mr) / (mp + mr + 1e-6) if (mp + mr) > 0 else 0.0
            map50 = float(val_res.box.map50)
            map50_95 = float(val_res.box.map)

            entry = {
                "conf": round(conf, 2),
                "iou": round(iou, 2),
                "precision": round(mp, 4),
                "recall": round(mr, 4),
                "f1": round(f1, 4),
                "map50": round(map50, 4),
                "map50_95": round(map50_95, 4),
            }
            sweep_results.append(entry)

            if f1 > best_f1:
                best_f1 = f1
                best_f1_entry = entry

            print(f"  conf={conf:.2f}, iou={iou:.2f} -> P={mp:.3f}, R={mr:.3f}, F1={f1:.3f}, mAP50={map50:.3f}")

    # Now evaluate optimal threshold chosen from VAL on the held-out TEST split
    opt_conf = best_f1_entry["conf"]
    opt_iou = best_f1_entry["iou"]
    print(f"\nEvaluating Optimal Operating Point (conf={opt_conf}, iou={opt_iou}) on held-out TEST split...")

    test_res = model.val(
        data=str(data_path),
        split="test",
        batch=8,
        imgsz=640,
        conf=opt_conf,
        iou=opt_iou,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        verbose=False,
        save_json=False,
    )
    test_mp = float(test_res.box.mp)
    test_mr = float(test_res.box.mr)
    test_f1 = (2 * test_mp * test_mr) / (test_mp + test_mr + 1e-6) if (test_mp + test_mr) > 0 else 0.0
    test_map50 = float(test_res.box.map50)
    test_map50_95 = float(test_res.box.map)

    test_eval_summary = {
        "applied_conf": opt_conf,
        "applied_iou": opt_iou,
        "test_precision": round(test_mp, 4),
        "test_recall": round(test_mr, 4),
        "test_f1": round(test_f1, 4),
        "test_map50": round(test_map50, 4),
        "test_map50_95": round(test_map50_95, 4),
    }
    print(f"  [TEST GENERALIZATION] P: {test_mp:.4f}, R: {test_mr:.4f}, F1: {test_f1:.4f}, mAP50: {test_map50:.4f}")

    return {
        "group": cfg["group"],
        "model_id": cfg["model_id"],
        "checkpoint": cfg["checkpoint"],
        "validation_sweep": sweep_results,
        "optimal_val_operating_point": best_f1_entry,
        "test_generalization_verification": test_eval_summary,
    }


def main():
    print("=" * 80)
    print("RoomAI Detection Threshold & NMS Optimization Sweep")
    print("=" * 80)

    sweep_data = []
    for cfg in SWEEPS:
        res = run_sweep(cfg)
        sweep_data.append(res)

    out_file = OUTPUT_DIR / "detection_threshold_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sweep_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Sweeps complete! Report written to: {out_file.relative_to(REPO_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
