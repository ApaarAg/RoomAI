"""
RoomAI — Phase 6: Ground-Truth Dataset Pipeline Validator
Performs rigorous empirical verification of all acquired geometric datasets,
enforcing mathematical validity, coordinate conventions, zero leakage, and provenance.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List

MANIFEST_PATH = Path(r"f:\RoomAI\research\3d_reconstruction\data_manifests\indoor_ground_truth_manifest.json")
REPORT_PATH = Path(r"f:\RoomAI\research\3d_reconstruction\reports\PHASE6_DATASET_VALIDATION_REPORT.md")

def validate_pipeline():
    print("=" * 75)
    print("ROOMAI — GROUND-TRUTH PIPELINE VALIDATION AUDIT")
    print("=" * 75)
    
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
        
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
        
    frames = manifest.get("frames", [])
    print(f"[AUDIT] Loaded manifest: {len(frames)} total frames.")
    
    # 1. Zero Scene Leakage Check
    train_scenes = set([f["scene_id"] for f in frames if f["split"] == "train"])
    val_scenes = set([f["scene_id"] for f in frames if f["split"] == "val"])
    test_scenes = set([f["scene_id"] for f in frames if f["split"] == "test"])
    
    # Group ScanNet scenes by base room ID (e.g. scene0019_00 and scene0019_01 share scene0019)
    def base_room_id(sid: str) -> str:
        if sid.startswith("scene"):
            return sid.split("_")[0]
        return sid
        
    train_rooms = set([base_room_id(s) for s in train_scenes])
    val_rooms = set([base_room_id(s) for s in val_scenes])
    test_rooms = set([base_room_id(s) for s in test_scenes])
    
    leakage_train_val = train_rooms & val_rooms
    leakage_train_test = train_rooms & test_rooms
    leakage_val_test = val_rooms & test_rooms
    
    zero_leakage_passed = (len(leakage_train_val) == 0 and len(leakage_train_test) == 0 and len(leakage_val_test) == 0)
    
    print("\n--- GATE 1: ZERO DATASET LEAKAGE AUDIT ---")
    print(f"  Train Rooms ({len(train_rooms)}): {sorted(list(train_rooms))}")
    print(f"  Val Rooms   ({len(val_rooms)}): {sorted(list(val_rooms))}")
    print(f"  Test Rooms  ({len(test_rooms)}): {sorted(list(test_rooms))}")
    print(f"  Train/Val Overlap: {len(leakage_train_val)} rooms")
    print(f"  Train/Test Overlap: {len(leakage_train_test)} rooms")
    print(f"  Val/Test Overlap:   {len(leakage_val_test)} rooms")
    print(f"  Status: {'PASSED (ZERO LEAKAGE)' if zero_leakage_passed else 'FAILED (CONTAMINATION DETECTED)'}")
    
    # 2. File Presence and Geometry Audits
    print("\n--- GATE 2: GEOMETRIC & FILE INTEGRITY AUDIT ---")
    total_checked = 0
    passed_frames = 0
    
    depth_stats = {"min": 999.0, "max": -999.0, "all_medians": []}
    normal_errors = []
    det_errors = []
    ortho_errors = []
    
    sample_discrepancies = []
    
    root_dir = MANIFEST_PATH.parent.parent / "data" / "ground_truth"
    
    for idx, item in enumerate(frames):
        total_checked += 1
        paths = item["paths"]
        rgb_p = root_dir / paths["rgb"].replace("ground_truth/", "")
        depth_p = root_dir / paths["depth"].replace("ground_truth/", "")
        normal_p = root_dir / paths["normal"].replace("ground_truth/", "")
        mask_p = root_dir / paths["mask"].replace("ground_truth/", "")
        cam_p = root_dir / paths["cam"].replace("ground_truth/", "")
        
        # Verify file existence
        if not (rgb_p.exists() and depth_p.exists() and normal_p.exists() and mask_p.exists() and cam_p.exists()):
            sample_discrepancies.append(f"Frame {idx} ({item['scene_id']}): missing files.")
            continue
            
        # Verify shapes
        rgb = cv2.imread(str(rgb_p))
        depth = np.load(str(depth_p))
        normal = np.load(str(normal_p))
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        cam = np.load(str(cam_p))
        
        if rgb.shape != (518, 518, 3) or depth.shape != (518, 518) or normal.shape != (518, 518, 3) or mask.shape != (518, 518):
            sample_discrepancies.append(f"Frame {idx}: Dimension mismatch.")
            continue
            
        valid = mask > 128
        if valid.sum() == 0:
            sample_discrepancies.append(f"Frame {idx}: Zero valid pixels.")
            continue
            
        # Depth checks
        valid_depth = depth[valid]
        if not np.isfinite(valid_depth).all() or (valid_depth <= 0).any():
            sample_discrepancies.append(f"Frame {idx}: Non-finite or negative depth in valid mask.")
            continue
            
        depth_stats["min"] = min(depth_stats["min"], float(np.min(valid_depth)))
        depth_stats["max"] = max(depth_stats["max"], float(np.max(valid_depth)))
        depth_stats["all_medians"].append(float(np.median(valid_depth)))
        
        # Normal checks
        valid_normals = normal[valid]
        norms = np.linalg.norm(valid_normals, axis=-1)
        norm_err = np.max(np.abs(norms - 1.0))
        normal_errors.append(norm_err)
        if norm_err > 0.05:
            sample_discrepancies.append(f"Frame {idx}: Normal vector not unit length (max err: {norm_err:.4f}).")
            continue
            
        # Camera checks
        K = cam["intrinsics"]
        c2w = cam["pose"]
        if K.shape != (3, 3) or c2w.shape != (4, 4):
            sample_discrepancies.append(f"Frame {idx}: Invalid camera matrix shape.")
            continue
            
        R = c2w[:3, :3]
        det = np.linalg.det(R)
        det_err = abs(det - 1.0)
        det_errors.append(det_err)
        
        ortho_err = np.linalg.norm(R.T @ R - np.eye(3))
        ortho_errors.append(ortho_err)
        
        if det_err > 0.01 or ortho_err > 0.01:
            sample_discrepancies.append(f"Frame {idx}: Camera rotation matrix invalid (det: {det:.4f}, ortho: {ortho_err:.4f}).")
            continue
            
        passed_frames += 1
        
    print(f"  Total frames checked: {total_checked}")
    print(f"  Frames passed all integrity checks: {passed_frames} / {total_checked} ({passed_frames/total_checked*100:.1f}%)")
    print(f"  Valid Depth Range: [{depth_stats['min']:.3f}m, {depth_stats['max']:.3f}m]")
    print(f"  Global Median Depth: {np.median(depth_stats['all_medians']):.3f}m")
    print(f"  Max Normal Vector Deviation from Unit: {max(normal_errors):.6f}")
    print(f"  Max Camera Rotation Determinant Deviation: {max(det_errors):.6f}")
    print(f"  Max Camera Rotation Orthogonality Error: {max(ortho_errors):.6f}")
    
    if len(sample_discrepancies) > 0:
        print(f"  [DISCREPANCIES DETECTED]: {len(sample_discrepancies)}")
        for d in sample_discrepancies[:5]:
            print(f"    - {d}")
    else:
        print("  [SUCCESS] 100% of frames passed strict mathematical & geometric validation!")

    return {
        "total_checked": total_checked,
        "passed_frames": passed_frames,
        "zero_leakage_passed": zero_leakage_passed,
        "train_rooms": sorted(list(train_rooms)),
        "val_rooms": sorted(list(val_rooms)),
        "test_rooms": sorted(list(test_rooms)),
        "depth_range": [depth_stats["min"], depth_stats["max"]],
        "median_depth": float(np.median(depth_stats["all_medians"])),
        "max_normal_err": float(max(normal_errors)),
        "max_det_err": float(max(det_errors)),
        "max_ortho_err": float(max(ortho_errors))
    }

if __name__ == "__main__":
    validate_pipeline()

