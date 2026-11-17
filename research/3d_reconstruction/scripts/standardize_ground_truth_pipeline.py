"""
RoomAI — Phase 6: Ground-Truth Standardization Pipeline
Transforms raw Hypersim (synthetic photorealistic) and ScanNet (real RGB-D)
into RoomAI's standardized geometric supervision schema.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import h5py
from pathlib import Path
from typing import Dict, Any, List, Tuple

TARGET_SIZE = (518, 518)  # Standardized input resolution (W, H)

RAW_ROOT = Path(r"f:\RoomAI\research\3d_reconstruction\data\ground_truth\raw")
STANDARDIZED_ROOT = Path(r"f:\RoomAI\research\3d_reconstruction\data\ground_truth\standardized")
MANIFEST_PATH = Path(r"f:\RoomAI\research\3d_reconstruction\data_manifests\indoor_ground_truth_manifest.json")

def compute_depth_normals_from_pinhole(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Computes smooth unit surface normals from a metric depth map via local 3D gradients.
    Normals are defined in OpenCV camera frame (+X right, +Y down, +Z forward).
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    valid = (depth > 0.05) & (depth < 20.0) & np.isfinite(depth)
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    pts = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    
    # Boundary-aware gradients
    dx = np.gradient(pts, axis=1)
    dy = np.gradient(pts, axis=0)
    
    # Cross product dx x dy (OpenCV camera frame: X right, Y down -> normal points towards camera, Z < 0)
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    
    unit_normals = np.where(valid[..., None], normals / np.maximum(norm, 1e-12), 0.0)
    
    # Ensure normal points towards camera (facing viewer, Nz < 0)
    flip_mask = unit_normals[..., 2] > 0
    unit_normals[flip_mask] = -unit_normals[flip_mask]
    
    return unit_normals.astype(np.float32)

def standardize_hypersim_scene(scene_id: str, split: str, meta_df: pd.DataFrame) -> List[Dict[str, Any]]:
    scene_dir = RAW_ROOT / "hypersim" / scene_id / scene_id
    if not scene_dir.exists():
        scene_dir = RAW_ROOT / "hypersim" / scene_id
    
    img_dir = scene_dir / "images"
    if not img_dir.exists():
        print(f"[HYPERSIM WARN] Scene {scene_id} missing images directory.")
        return []

    # Dynamically find camera
    geo_dirs = [d for d in os.listdir(img_dir) if d.endswith("_geometry_hdf5")]
    if not geo_dirs:
        print(f"[HYPERSIM WARN] Scene {scene_id} missing geometry directory.")
        return []
    
    geo_name = geo_dirs[0]  # e.g. scene_cam_00_geometry_hdf5 or scene_cam_01_geometry_hdf5
    cam_name = geo_name.replace("scene_", "").replace("_geometry_hdf5", "")
    
    geo_dir = img_dir / geo_name
    prev_dir = img_dir / f"scene_{cam_name}_final_preview"
    detail_cam = scene_dir / "_detail" / cam_name
    
    if not geo_dir.exists() or not detail_cam.exists():
        print(f"[HYPERSIM WARN] Scene {scene_id} incomplete: geo_dir or detail_cam missing.")
        return []

    # 1. Camera parameters from CSV
    if scene_id not in meta_df.index:
        print(f"[HYPERSIM WARN] {scene_id} not in metadata_camera_parameters.csv.")
        return []
    
    row = meta_df.loc[scene_id]
    orig_w = int(row["settings_output_img_width"])
    orig_h = int(row["settings_output_img_height"])
    meters_per_unit = float(row["settings_units_info_meters_scale"])
    
    M = np.array([
        [float(row["M_cam_from_uv_00"]), float(row["M_cam_from_uv_01"]), float(row["M_cam_from_uv_02"])],
        [float(row["M_cam_from_uv_10"]), float(row["M_cam_from_uv_11"]), float(row["M_cam_from_uv_12"])],
        [float(row["M_cam_from_uv_20"]), float(row["M_cam_from_uv_21"]), float(row["M_cam_from_uv_22"])]
    ])
    
    # Pinhole focal lengths
    fx_orig = orig_w / (2.0 * M[0, 0])
    fy_orig = orig_h / (2.0 * M[1, 1])
    cx_orig = orig_w / 2.0
    cy_orig = orig_h / 2.0
    
    # Scaled pinhole intrinsics for (518, 518)
    scale_x = TARGET_SIZE[0] / orig_w
    scale_y = TARGET_SIZE[1] / orig_h
    K_std = np.array([
        [fx_orig * scale_x, 0.0, cx_orig * scale_x],
        [0.0, fy_orig * scale_y, cy_orig * scale_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # 2. Camera trajectory
    with h5py.File(detail_cam / "camera_keyframe_positions.hdf5", "r") as f:
        positions = f["dataset"][:]
    with h5py.File(detail_cam / "camera_keyframe_orientations.hdf5", "r") as f:
        orientations = f["dataset"][:]
    with h5py.File(detail_cam / "camera_keyframe_frame_indices.hdf5", "r") as f:
        frame_indices = f["dataset"][:]

    # Precalculate ray direction grid and cos(theta) for full resolution
    u_lin = np.linspace(-1.0 + 1.0/orig_w, 1.0 - 1.0/orig_w, orig_w)
    v_lin = np.linspace(1.0 - 1.0/orig_h, -1.0 + 1.0/orig_h, orig_h)
    u_grid, v_grid = np.meshgrid(u_lin, v_lin)
    uv_ones = np.stack([u_grid, v_grid, np.ones_like(u_grid)], axis=-1)  # (H, W, 3)
    rays = uv_ones @ M.T
    ray_norms = np.linalg.norm(rays, axis=-1)
    cos_theta = -rays[..., 2] / ray_norms  # cos(theta) with optical axis (-Z in OpenGL)

    # Find rendered frames
    depth_files = sorted([f for f in os.listdir(geo_dir) if f.endswith(".depth_meters.hdf5")])
    out_scene_dir = STANDARDIZED_ROOT / "hypersim" / split / scene_id
    out_scene_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_entries = []
    print(f"[HYPERSIM] Standardizing {scene_id} ({len(depth_files)} frames) -> {split}...")

    for idx, d_file in enumerate(depth_files):
        fid = int(d_file.split(".")[1])
        rgb_file = prev_dir / f"frame.{fid:04d}.color.jpg"
        normal_file = geo_dir / f"frame.{fid:04d}.normal_cam.hdf5"
        
        if not rgb_file.exists() or not normal_file.exists():
            continue
            
        # Match camera trajectory index
        traj_matches = np.where(frame_indices == fid)[0]
        if len(traj_matches) == 0:
            continue
        traj_idx = traj_matches[0]
        
        # Load raw data
        with h5py.File(geo_dir / d_file, "r") as f:
            ray_dist = f["dataset"][:].astype(np.float32)
        with h5py.File(normal_file, "r") as f:
            norm_raw = f["dataset"][:].astype(np.float32)
            
        rgb_img = cv2.imread(str(rgb_file))
        if rgb_img is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Convert ray distance to orthogonal planar Z-depth: Z = R * cos(theta)
        z_depth = ray_dist * cos_theta
        
        # Validity mask
        valid_depth = np.isfinite(z_depth) & (z_depth > 0.05) & (z_depth < 50.0)
        valid_norm = np.isfinite(norm_raw).all(axis=-1)
        valid_mask = valid_depth & valid_norm
        
        # Normal vector coordinate transform (OpenGL camera -> OpenCV camera)
        # OpenGL: +X right, +Y up, +Z back. OpenCV: +X right, +Y down, +Z forward.
        norm_opencv = norm_raw.copy()
        norm_opencv[..., 1] *= -1.0
        norm_opencv[..., 2] *= -1.0
        # Re-normalize
        norm_len = np.linalg.norm(norm_opencv, axis=-1, keepdims=True)
        norm_len = np.where(norm_len < 1e-4, 1.0, norm_len)
        norm_opencv = norm_opencv / norm_len
        norm_opencv[~valid_mask] = 0.0
        
        # Camera Pose: OpenGL C2W -> OpenCV C2W
        R_opengl = orientations[traj_idx]
        t_world = positions[traj_idx] * meters_per_unit  # Convert asset units to meters
        
        R_flip = np.diag([1.0, -1.0, -1.0])
        R_opencv = R_opengl @ R_flip
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_opencv
        c2w[:3, 3] = t_world
        
        # Standardize resolution to (518, 518)
        rgb_std = cv2.resize(rgb_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        z_depth_std = cv2.resize(z_depth, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        norm_std = cv2.resize(norm_opencv, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        # Re-normalize after resizing
        n_len_std = np.linalg.norm(norm_std, axis=-1, keepdims=True)
        n_len_std = np.where(n_len_std < 1e-4, 1.0, n_len_std)
        norm_std = norm_std / n_len_std
        
        mask_std = cv2.resize((valid_mask * 255).astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        valid_std = mask_std > 128
        z_depth_std[~valid_std] = 0.0
        norm_std[~valid_std] = 0.0
        
        # Save standardized assets
        prefix = f"frame_{idx:04d}"
        rgb_out = out_scene_dir / f"{prefix}_rgb.png"
        depth_out = out_scene_dir / f"{prefix}_depth.npy"
        norm_out = out_scene_dir / f"{prefix}_normal.npy"
        mask_out = out_scene_dir / f"{prefix}_mask.png"
        cam_out = out_scene_dir / f"{prefix}_cam.npz"
        
        cv2.imwrite(str(rgb_out), cv2.cvtColor(rgb_std, cv2.COLOR_RGB2BGR))
        np.save(str(depth_out), z_depth_std.astype(np.float32))
        np.save(str(norm_out), norm_std.astype(np.float32))
        cv2.imwrite(str(mask_out), mask_std)
        np.savez_compressed(str(cam_out), intrinsics=K_std, pose=c2w, scene_id=scene_id, frame_id=fid, dataset="hypersim", split=split)
        
        valid_vals = z_depth_std[valid_std]
        depth_min = float(np.min(valid_vals)) if len(valid_vals) > 0 else 0.0
        depth_max = float(np.max(valid_vals)) if len(valid_vals) > 0 else 0.0
        depth_median = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
        
        manifest_entries.append({
            "scene_id": scene_id,
            "frame_id": fid,
            "frame_index": idx,
            "dataset": "hypersim",
            "split": split,
            "paths": {
                "rgb": str(rgb_out.relative_to(STANDARDIZED_ROOT.parent)),
                "depth": str(depth_out.relative_to(STANDARDIZED_ROOT.parent)),
                "normal": str(norm_out.relative_to(STANDARDIZED_ROOT.parent)),
                "mask": str(mask_out.relative_to(STANDARDIZED_ROOT.parent)),
                "cam": str(cam_out.relative_to(STANDARDIZED_ROOT.parent))
            },
            "resolution": list(TARGET_SIZE),
            "depth_metrics": {
                "min_m": depth_min,
                "max_m": depth_max,
                "median_m": depth_median,
                "valid_pixel_ratio": float(np.mean(valid_std))
            },
            "camera": {
                "fx": float(K_std[0, 0]),
                "fy": float(K_std[1, 1]),
                "cx": float(K_std[0, 2]),
                "cy": float(K_std[1, 2]),
                "pose_det": float(np.linalg.det(c2w[:3, :3])),
                "camera_center_world": [float(c2w[0, 3]), float(c2w[1, 3]), float(c2w[2, 3])]
            },
            "provenance": {
                "rgb": "Synthetic ground truth (V-Ray path tracer)",
                "depth": "Synthetic ground truth (derived planar Z-depth from exact ray distance)",
                "normal": "Synthetic ground truth (CAD renderer surface normal)",
                "pose": "Synthetic ground truth (CAD trajectory)",
                "intrinsics": "Dataset-provided camera calibration"
            }
        })
        
    return manifest_entries

def standardize_scannet_scene(scene_id: str, split: str) -> List[Dict[str, Any]]:
    scene_dir = RAW_ROOT / "scannet" / scene_id
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intr_dir = scene_dir / "intrinsic"
    
    if not color_dir.exists() or not depth_dir.exists() or not pose_dir.exists() or not intr_dir.exists():
        print(f"[SCANNET WARN] Scene {scene_id} incomplete or missing directories.")
        return []
        
    # Intrinsics
    intr_file = intr_dir / "intrinsic_depth.txt"
    if not intr_file.exists():
        intr_file = intr_dir / "intrinsic_color.txt"
    intr_mat = np.loadtxt(str(intr_file))[:3, :3]
    
    orig_w, orig_h = 640, 480
    scale_x = TARGET_SIZE[0] / orig_w
    scale_y = TARGET_SIZE[1] / orig_h
    K_std = np.array([
        [intr_mat[0, 0] * scale_x, 0.0, intr_mat[0, 2] * scale_x],
        [0.0, intr_mat[1, 1] * scale_y, intr_mat[1, 2] * scale_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")])
    out_scene_dir = STANDARDIZED_ROOT / "scannet" / split / scene_id
    out_scene_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_entries = []
    print(f"[SCANNET] Standardizing {scene_id} ({len(color_files)} frames) -> {split}...")

    for idx, c_file in enumerate(color_files):
        fid_str = c_file.replace(".jpg", "")
        fid = int(fid_str)
        
        depth_file = depth_dir / f"{fid_str}.npy"
        pose_file = pose_dir / f"{fid_str}.txt"
        
        if not depth_file.exists() or not pose_file.exists():
            continue
            
        pose = np.loadtxt(str(pose_file)).astype(np.float32)
        # Check for ScanNet tracking failure (infs or non-invertible)
        if not np.isfinite(pose).all() or np.linalg.det(pose[:3, :3]) <= 0.0:
            continue
            
        rgb_img = cv2.imread(str(color_dir / c_file))
        if rgb_img is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        depth_raw = np.load(str(depth_file))
        # ScanNet depth is stored in millimeters (uint16) -> convert to float32 meters
        z_depth = depth_raw.astype(np.float32) / 1000.0
        
        # Valid depth range for Structure Sensor (0.1m to 8.0m)
        valid_mask = (z_depth > 0.1) & (z_depth < 8.0) & np.isfinite(z_depth)
        
        # Derived normals from local depth gradient in camera frame
        normals_derived = compute_depth_normals_from_pinhole(z_depth, intr_mat)
        valid_normals = np.linalg.norm(normals_derived, axis=-1) > 0.5
        valid_mask = valid_mask & valid_normals
        
        # Resize to (518, 518)
        rgb_std = cv2.resize(rgb_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        z_depth_std = cv2.resize(z_depth, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        norm_std = cv2.resize(normals_derived, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask_std = cv2.resize((valid_mask * 255).astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        
        valid_std = mask_std > 128
        
        # Re-normalize normals
        n_len_std = np.linalg.norm(norm_std, axis=-1, keepdims=True)
        n_len_std = np.where(n_len_std < 1e-4, 1.0, n_len_std)
        norm_std = norm_std / n_len_std
        
        z_depth_std[~valid_std] = 0.0
        norm_std[~valid_std] = 0.0
        
        # Save standardized assets
        prefix = f"frame_{idx:04d}"
        rgb_out = out_scene_dir / f"{prefix}_rgb.png"
        depth_out = out_scene_dir / f"{prefix}_depth.npy"
        norm_out = out_scene_dir / f"{prefix}_normal.npy"
        mask_out = out_scene_dir / f"{prefix}_mask.png"
        cam_out = out_scene_dir / f"{prefix}_cam.npz"
        
        cv2.imwrite(str(rgb_out), cv2.cvtColor(rgb_std, cv2.COLOR_RGB2BGR))
        np.save(str(depth_out), z_depth_std.astype(np.float32))
        np.save(str(norm_out), norm_std.astype(np.float32))
        cv2.imwrite(str(mask_out), mask_std)
        np.savez_compressed(str(cam_out), intrinsics=K_std, pose=pose, scene_id=scene_id, frame_id=fid, dataset="scannet", split=split)
        
        valid_vals = z_depth_std[valid_std]
        depth_min = float(np.min(valid_vals)) if len(valid_vals) > 0 else 0.0
        depth_max = float(np.max(valid_vals)) if len(valid_vals) > 0 else 0.0
        depth_median = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
        
        manifest_entries.append({
            "scene_id": scene_id,
            "frame_id": fid,
            "frame_index": idx,
            "dataset": "scannet",
            "split": split,
            "paths": {
                "rgb": str(rgb_out.relative_to(STANDARDIZED_ROOT.parent)),
                "depth": str(depth_out.relative_to(STANDARDIZED_ROOT.parent)),
                "normal": str(norm_out.relative_to(STANDARDIZED_ROOT.parent)),
                "mask": str(mask_out.relative_to(STANDARDIZED_ROOT.parent)),
                "cam": str(cam_out.relative_to(STANDARDIZED_ROOT.parent))
            },
            "resolution": list(TARGET_SIZE),
            "depth_metrics": {
                "min_m": depth_min,
                "max_m": depth_max,
                "median_m": depth_median,
                "valid_pixel_ratio": float(np.mean(valid_std))
            },
            "camera": {
                "fx": float(K_std[0, 0]),
                "fy": float(K_std[1, 1]),
                "cx": float(K_std[0, 2]),
                "cy": float(K_std[1, 2]),
                "pose_det": float(np.linalg.det(pose[:3, :3])),
                "camera_center_world": [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])]
            },
            "provenance": {
                "rgb": "Direct sensor measurement (iPad color camera)",
                "depth": "Direct sensor measurement (Structure Sensor IR structured light)",
                "normal": "Derived annotation (local surface gradient on sensor depth)",
                "pose": "Reconstructed geometry (BundleFusion real-time SLAM trajectory)",
                "intrinsics": "Dataset-provided camera calibration"
            }
        })
        
    return manifest_entries

def main():
    print("=" * 70)
    print("ROOMAI — GROUND-TRUTH STANDARDIZATION & SUPERVISION PIPELINE")
    print("=" * 70)
    STANDARDIZED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Hypersim processing
    meta_csv = RAW_ROOT / "hypersim" / "metadata_camera_parameters.csv"
    if not meta_csv.exists():
        print("[ERROR] metadata_camera_parameters.csv missing.")
        return
    meta_df = pd.read_csv(meta_csv, index_col="scene_name")
    
    hypersim_scenes = {
        "train": ["ai_015_007", "ai_011_010", "ai_047_008"],
        "val": ["ai_016_001"],
        "test": ["ai_013_010"]
    }
    
    scannet_scenes = {
        "train": ["scene0019_00", "scene0019_01", "scene0015_00"],
        "val": ["scene0030_00", "scene0030_01", "scene0030_02"],
        "test": ["scene0011_00", "scene0011_01"]
    }
    
    all_manifest = []
    
    print("\n--- STANDARDIZING HYPERSIM SCENES ---")
    for split, scenes in hypersim_scenes.items():
        for sc in scenes:
            entries = standardize_hypersim_scene(sc, split, meta_df)
            all_manifest.extend(entries)
            
    print("\n--- STANDARDIZING SCANNET SCENES ---")
    for split, scenes in scannet_scenes.items():
        for sc in scenes:
            entries = standardize_scannet_scene(sc, split)
            all_manifest.extend(entries)

    # Save manifest
    manifest_data = {
        "manifest_version": "2.0_ground_truth",
        "created_at": "2026-11-08T04:00:00Z",
        "resolution": list(TARGET_SIZE),
        "total_frames": len(all_manifest),
        "splits": {
            "train": len([e for e in all_manifest if e["split"] == "train"]),
            "val": len([e for e in all_manifest if e["split"] == "val"]),
            "test": len([e for e in all_manifest if e["split"] == "test"])
        },
        "datasets": {
            "hypersim": len([e for e in all_manifest if e["dataset"] == "hypersim"]),
            "scannet": len([e for e in all_manifest if e["dataset"] == "scannet"])
        },
        "frames": all_manifest
    }
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"[COMPLETE] Standardized {len(all_manifest)} frames across Hypersim and ScanNet.")
    print(f"  Train: {manifest_data['splits']['train']} frames")
    print(f"  Val:   {manifest_data['splits']['val']} frames")
    print(f"  Test:  {manifest_data['splits']['test']} frames")
    print(f"[COMPLETE] Saved manifest to: {MANIFEST_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
