import json
from pathlib import Path

training_plan = {
    "phase": "Phase 19 — Multi-Dataset Expansion and Training Strategy",
    "status": "APPROVED_PLANNING_PHASE_NO_TRAINING_EXECUTED",
    "timestamp": "2026-10-25T12:35:00+05:30",
    "immutable_constraints": {
        "production_default": "Model B2 (checkpoints/exterior_detector_b2_yolo11s_best.pt)",
        "candidate_baseline": "Model B6 (checkpoints/exterior_detector_b6_expanded_best.pt)",
        "specialist_checkpoints": "checkpoints/exterior_specialist_porch_yolo11s_best.pt",
        "checkpoint_protection": "STRICT_READ_ONLY_NO_OVERWRITE",
        "phase17_test_set_status": "QUARANTINED_DO_NOT_TUNE_OR_EVALUATE_DURING_TRAINING",
        "data_download_policy": "NO_INDISCRIMINATE_DOWNLOADS_TARGETED_SUBSETS_ONLY",
        "training_execution_status": "FROZEN_STAGE_AUDIT_AND_PLANNING_ONLY"
    },
    "hardware_constraints": {
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "vram_total_gb": 8.0,
        "vram_safe_budget_gb": 6.5,
        "ram_total_gb": 23.73,
        "ram_available_gb": 5.39,
        "disk_c_total_gb": 474.87,
        "disk_c_free_gb": 27.18,
        "max_allowable_dataset_storage_gb": 4.5,
        "concurrent_gpu_training": "PROHIBITED"
    },
    "candidate_datasets": {
        "open_images_v7": {
            "source_url": "https://storage.googleapis.com/openimages/web/index.html",
            "access_requirements": "Public HTTP/S3, no registration/tokens required",
            "license": "CC BY 4.0 (annotations) / CC BY 2.0 & Flickr (images)",
            "competition_eligibility": "Permitted with attribution",
            "annotation_types": ["2D Bounding Boxes", "Instance Polygons (2.7M instances)", "Visual Relationships"],
            "relevant_classes": {
                "/m/0cgh4": "building",
                "/m/03jm5": "building (House)",
                "/m/021sj1": "building (Office building)",
                "/m/04m6gz": "porch",
                "/m/02dgv": "door",
                "/m/0d4v4": "window",
                "/m/031b6r": "shutter_blind (Window blind)"
            },
            "unannotated_classes": ["facade", "roof_cornice", "balcony", "pillar_column", "molding_deco"],
            "approx_usable_samples": {
                "porch_scenes": 4200,
                "door_scenes": 45000,
                "window_blind_scenes": 8500,
                "building_scenes": 180000
            },
            "storage_full_dataset": "> 500 GB (Prohibited)",
            "storage_targeted_subset": "1.2 GB per 1,000 images (Approved)",
            "known_limitations": "Partial annotation: unannotated classes must NOT be treated as negative background. Sparse labeling of facade details."
        },
        "mapillary_vistas": {
            "source_url": "https://www.mapillary.com/dataset/vistas",
            "access_requirements": "Account registration on Mapillary, auth token required for download portal",
            "license": "Mapillary Non-Commercial Research License (Research Edition) / Commercial License for production",
            "competition_eligibility": "Allowed for academic/research competitions; commercial license required for closed deployment",
            "annotation_types": ["Pixel-level semantic segmentation (65 classes v1.2, 124 classes v2.0)", "Polygon instance annotations"],
            "relevant_classes": {
                "construction--structure--building": "building",
                "construction--barrier--wall": "facade (ambiguous: mixes freestanding perimeter walls with facades)",
                "object--window": "window (v2.0 only)",
                "object--door": "door (v2.0 only)"
            },
            "approx_usable_samples": {
                "total_images": 25000,
                "prominent_facade_street_scenes": 4500
            },
            "storage_full_dataset": "190 GB (Prohibited)",
            "storage_targeted_subset": "1.8 GB per 500 high-res crops (Approved with license review)",
            "known_limitations": "Ego-vehicle perspective with heavy occlusions (trees, parked cars, pedestrians). Distant oblique angles. Merges fine architectural moldings into building."
        },
        "ade20k": {
            "source_url": "https://groups.csail.mit.edu/vision/datasets/ADE20K/",
            "access_requirements": "Direct public HTTP download; no paywall",
            "license": "CC BY-NC-SA 4.0 (Non-Commercial ShareAlike)",
            "competition_eligibility": "Eligible for non-commercial benchmark competitions and research",
            "annotation_types": ["Dense polygon instance masks", "Semantic segmentation", "Hierarchical object-part annotations"],
            "relevant_classes": {
                "building, edifice": "building",
                "house": "building",
                "wall (exterior)": "facade",
                "facade": "facade",
                "windowpane, window": "window",
                "door": "door",
                "roof": "roof_cornice",
                "cornice": "roof_cornice",
                "balcony": "balcony",
                "porch": "porch",
                "column, pillar": "pillar_column",
                "shutter, blind": "shutter_blind",
                "molding": "molding_deco"
            },
            "approx_usable_samples": {
                "total_images": 25210,
                "outdoor_architectural_scenes": 4100,
                "column_instances": 2100,
                "balcony_instances": 1200,
                "porch_instances": 480,
                "shutter_instances": 650
            },
            "storage_full_dataset": "3.8 GB compressed / 6.5 GB uncompressed (Within 27GB disk constraint)",
            "storage_targeted_subset": "850 MB for filtered architectural subset",
            "known_limitations": "Highest taxonomic alignment with RoomAI. Mix of indoor and outdoor scenes requires scene-category filtering ('outdoor/building_facade', 'outdoor/house')."
        },
        "coco": {
            "source_url": "https://cocodataset.org",
            "access_requirements": "Public HTTP direct download",
            "license": "CC BY 4.0 (annotations) / Flickr licenses (images)",
            "competition_eligibility": "Full commercial and competition eligibility",
            "annotation_types": ["80 Object BBoxes/Polygons (Detection)", "Stuff Pixel Masks (COCO-Stuff)"],
            "relevant_classes": {},
            "unsupported_assessment": "Standard COCO-80 has 0 architectural facade classes (no building, window, door, porch, column, facade). Completely unsupported for RoomAI object detection. Do not use.",
            "storage_full_dataset": "19 GB (Not recommended)",
            "known_limitations": "Taxonomic void for architectural elements. Ingesting standard COCO detection labels would contaminate the training pipeline with non-relevant object distributions."
        },
        "cmp_facade": {
            "source_url": "Czech Technical University in Prague (Radim Tylecek & Radim Sara, 2013)",
            "access_requirements": "Already integrated into RoomAI repository",
            "license": "Academic research / free with citation",
            "competition_eligibility": "Standard architectural benchmark dataset",
            "annotation_types": ["Pixel-level segmentation maps converted to dense bounding boxes"],
            "relevant_classes": {
                "facade": "facade",
                "window": "window",
                "door": "door",
                "cornice": "roof_cornice",
                "sill": "molding_deco",
                "balcony": "balcony",
                "blind": "shutter_blind",
                "pillar": "pillar_column",
                "deco": "molding_deco",
                "molding": "molding_deco",
                "shop": "door"
            },
            "approx_usable_samples": {
                "total_images": 606,
                "train_images": 378,
                "val_images": 114,
                "test_images": 114
            },
            "storage_full_dataset": "145 MB",
            "known_limitations": "Exclusively European classical rectified facades. Lacks suburban American residential architecture and modern homes. Porch count is near zero (1 in train)."
        }
    },
    "label_harmonization_strategy": {
        "canonical_taxonomy": [
            {"id": 0, "name": "building", "type": "envelope_box"},
            {"id": 1, "name": "facade", "type": "semantic_mask_preferred_or_macro_box"},
            {"id": 2, "name": "window", "type": "discrete_instance_box"},
            {"id": 3, "name": "door", "type": "discrete_instance_box"},
            {"id": 4, "name": "roof_cornice", "type": "discrete_instance_box"},
            {"id": 5, "name": "balcony", "type": "discrete_instance_box"},
            {"id": 6, "name": "porch", "type": "discrete_instance_box"},
            {"id": 7, "name": "pillar_column", "type": "discrete_instance_box"},
            {"id": 8, "name": "shutter_blind", "type": "discrete_instance_box"},
            {"id": 9, "name": "molding_deco", "type": "discrete_instance_box"}
        ],
        "partial_annotation_handling": {
            "rule": "Sparse datasets (e.g. Open Images) must NOT treat unannotated objects as true negatives.",
            "loss_masking": "Implement class-conditional ignore masks during loss backpropagation. When training on an Open Images porch scene, compute classification loss solely over verified positive classes and explicit background regions.",
            "pseudo_labeling_alternative": "Run frozen high-precision Model B2 over Open Images frames at confidence >= 0.70 to auto-tag obvious windows and moldings before gradient calculation."
        },
        "facade_geometry_evaluation": {
            "verdict": "BOUNDING_BOX_UNSUITABLE_FOR_FACADE",
            "justification": "Facades are non-convex, stepped, L-shaped surfaces frequently occluded by porches, columns, and trees. Axis-aligned bounding boxes force inclusion of large sky and background areas (mean IoU 0.41 in Phase 17). Facade must transition to semantic segmentation masks (YOLO11-seg or Mask2Former)."
        }
    },
    "staged_training_plan": {
        "experiment_a_frozen_baseline": {
            "name": "Experiment A: Frozen Model B6 Baseline",
            "model_path": "checkpoints/exterior_detector_b6_expanded_best.pt",
            "role": "Immutable control benchmark",
            "epochs": 0,
            "status": "COMPLETED_IN_PHASE17"
        },
        "experiment_b_pilot_multisource_detection": {
            "name": "Pilot B-1: Multi-Source Harmonized Detection Pilot",
            "base_checkpoint": "checkpoints/exterior_detector_b6_expanded_best.pt",
            "image_limit": 300,
            "composition": {
                "ade20k_outdoor_curated": 150,
                "open_images_entrance_curated": 150
            },
            "hyperparameters": {
                "imgsz": 800,
                "batch_size": 16,
                "epochs": 30,
                "patience": 15,
                "optimizer": "AdamW",
                "lr0": 0.0003,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "amp": True,
                "save_period": 5,
                "workers": 4
            },
            "source_aware_sampling": {
                "strategy": "Temperature-scaled inverse frequency",
                "temperature": 0.5,
                "porch_oversampling_factor": 3.0,
                "pillar_column_oversampling_factor": 2.0
            },
            "hardware_budget": {
                "max_vram_allocated_mb": 5200,
                "estimated_training_time_min": 18,
                "disk_footprint_mb": 420
            },
            "checkpointing": {
                "save_dir": "runs/train/pilot_b1_multisource",
                "monitor_metric": "val/macro_f1",
                "recovery_policy": "Auto-resume from last.pt upon crash"
            },
            "go_no_go_gate": "Pilot must achieve Porch val F1 >= 0.20 and Macro-F1 >= 0.50 before proceeding to full Experiment B."
        },
        "experiment_b_full_multisource_detection": {
            "name": "Experiment B: Full Multi-Source Detection Detector (Candidate B7)",
            "base_checkpoint": "checkpoints/exterior_detector_b6_expanded_best.pt",
            "target_train_images": 2200,
            "composition": {
                "cmp_facade_train": 378,
                "open_images_b6_existing": 924,
                "ade20k_architectural_subset": 600,
                "curated_entrance_expansion": 300
            },
            "hyperparameters": {
                "imgsz": 800,
                "batch_size": 16,
                "epochs": 80,
                "patience": 20,
                "optimizer": "AdamW",
                "lr0": 0.0005,
                "lrf": 0.01,
                "amp": True,
                "save_period": 10
            },
            "prerequisites": "Successful completion of Pilot B-1 and validation gate verification."
        },
        "experiment_c_facade_segmentation_baseline": {
            "name": "Experiment C: Dedicated Facade Mask Segmentation Baseline",
            "architecture": "YOLO11s-seg (Pretrained on COCO-seg)",
            "target_classes": ["facade", "roof", "opening"],
            "training_sources": {
                "cmp_facade_polygons": 378,
                "ade20k_exterior_wall_polygons": 400
            },
            "hyperparameters": {
                "imgsz": 800,
                "batch_size": 12,
                "epochs": 50,
                "patience": 15,
                "amp": True
            },
            "expected_outcome": "Resolves non-convex facade geometry failure mode without corrupting discrete box detectors."
        }
    },
    "data_integrity_and_evaluation_design": {
        "automated_checks": {
            "md5_collision_screening": "Zero hash collisions permitted across train, val, and test partitions.",
            "phash_near_duplicate_screening": "64-bit DCT perceptual hash Hamming distance >= 5 against protected validation and test splits.",
            "coordinate_normalization_check": "All bounding box coordinates must lie strictly within [0.0, 1.0] with w, h > 0.003.",
            "corruption_screening": "Header integrity verification via PIL.verify() and cv2.imread decode test.",
            "provenance_logging": "Mandatory metadata record for every image: original_id, source_dataset, license, and ingestion_timestamp."
        },
        "independent_holdout_evaluation_set": {
            "name": "Holdout Test Set 2 (Phase 19 Multi-Dataset Evaluation Set)",
            "size": 200,
            "source_composition": {
                "ade20k_held_out_outdoor": 100,
                "open_images_held_out_verified": 100
            },
            "protection_rule": "Strictly quarantined. Cannot be evaluated until final training candidate is selected. Zero parameter tuning."
        },
        "evaluation_metrics": [
            "mAP50", "mAP50_95", "macro_f1", "micro_f1",
            "per_class_precision", "per_class_recall", "per_class_f1", "per_class_ap50",
            "synchronized_latency_median_p90_p95_ms", "peak_allocated_vram_mb",
            "bootstrap_1000_sample_95pct_confidence_intervals"
        ]
    }
}

out_p = Path(r"C:\Users\apaar\Desktop\RoomAI\reports\phase19_training_plan.json")
out_p.parent.mkdir(parents=True, exist_ok=True)
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(training_plan, f, indent=2)

print(f"Exported training plan JSON to {out_p}")
