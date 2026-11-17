"""
ROOMAI — Ground-Truth Standardizer for Newly Acquired Hypersim Scenes (Downloads)
Processes all 31 scenes from F:/RoomAI/downloads into the canonical 518x518 square
standardized schema under F:/RoomAI/research/3d_reconstruction/data/ground_truth/standardized/hypersim/
and constructs the expanded canonical manifest for continuous training.
"""

import os
import sys
import json
import math
import cv2
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

TARGET_SIZE = (518, 518)  # (W, H)
REPO_ROOT = Path("F:/RoomAI")
DOWNLOADS_DIR = REPO_ROOT / "downloads"
CAMERA_CSV = REPO_ROOT / "research/3d_reconstruction/data/ground_truth/raw/hypersim/metadata_camera_parameters.csv"
STANDARDIZED_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth/standardized/hypersim"
CANONICAL_MANIFEST_PATH = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest_expanded.json"

# Load Camera Parameter Metadata
meta_df = pd.read_csv(CAMERA_CSV, index_col="scene_name")

# Define deterministic scene-level splits for newly downloaded scenes
LOCKED_TEST_SCENES = {"ai_001_003", "ai_002_010", "ai_003_010"}
VAL_SCENES = {"ai_002_008", "ai_003_002", "ai_004_001", "ai_004_002"}

def standardize_scene(scene_id: str, split: str, stride: int = 4):
    scene_dir = DOWNLOADS_DIR / scene_id
    img_dir = scene_dir / "images"
    detail_dir = scene_dir / "_detail"

    if not img_dir.exists() or not detail_dir.exists():
        print(f"[{scene_id}] Missing images or _detail")
        return []

    geo_dirs = [d for d in os.listdir(img_dir) if d.endswith("_geometry_hdf5")]
    if not geo_dirs:
        print(f"[{scene_id}] No geometry directory")
        return []

    # Camera metadata
    if scene_id not in meta_df.index:
        print(f"[{scene_id}] Not in camera parameters CSV")
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

    fx_orig = orig_w / (2.0 * M[0, 0])
    fy_orig = orig_h / (2.0 * M[1, 1])
    cx_orig = orig_w / 2.0
    cy_orig = orig_h / 2.0

    scale_x = TARGET_SIZE[0] / orig_w
    scale_y = TARGET_SIZE[1] / orig_h
    K_std = np.array([
        [fx_orig * scale_x, 0.0, cx_orig * scale_x],
        [0.0, fy_orig * scale_y, cy_orig * scale_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Ray direction grid
    u_lin = np.linspace(-1.0 + 1.0/orig_w, 1.0 - 1.0/orig_w, orig_w)
    v_lin = np.linspace(1.0 - 1.0/orig_h, -1.0 + 1.0/orig_h, orig_h)
    u_grid, v_grid = np.meshgrid(u_lin, v_lin)
    uv_ones = np.stack([u_grid, v_grid, np.ones_like(u_grid)], axis=-1)
    rays = uv_ones @ M.T
    ray_norms = np.linalg.norm(rays, axis=-1)
    cos_theta = -rays[..., 2] / np.maximum(ray_norms, 1e-12)

    scene_entries = []

    for geo_name in geo_dirs:
        cam_name = geo_name.replace("scene_", "").replace("_geometry_hdf5", "")
        cam_detail = detail_dir / cam_name
        geo_dir = img_dir / geo_name
        prev_dir = img_dir / f"scene_{cam_name}_final_preview"

        if not cam_detail.exists() or not prev_dir.exists():
            continue

        pos_file = cam_detail / "camera_keyframe_positions.hdf5"
        ori_file = cam_detail / "camera_keyframe_orientations.hdf5"
        idx_file = cam_detail / "camera_keyframe_frame_indices.hdf5"
        if not (pos_file.exists() and ori_file.exists() and idx_file.exists()):
            continue

        with h5py.File(pos_file, "r") as f:
            positions = f["dataset"][:]
        with h5py.File(ori_file, "r") as f:
            orientations = f["dataset"][:]
        with h5py.File(idx_file, "r") as f:
            frame_indices = f["dataset"][:].flatten()

        depth_files = sorted([f for f in os.listdir(geo_dir) if f.endswith(".depth_meters.hdf5")])

        out_scene_dir = STANDARDIZED_ROOT / split / scene_id / cam_name
        out_scene_dir.mkdir(parents=True, exist_ok=True)

        for d_file in depth_files[::stride]:
            fid = int(d_file.split(".")[1])
            rgb_file = prev_dir / f"frame.{fid:04d}.color.jpg"
            normal_file = geo_dir / f"frame.{fid:04d}.normal_cam.hdf5"

            if not rgb_file.exists():
                continue

            traj_matches = np.where(frame_indices == fid)[0]
            if len(traj_matches) == 0:
                continue
            traj_idx = traj_matches[0]

            try:
                with h5py.File(geo_dir / d_file, "r") as f:
                    ray_dist = f["dataset"][:].astype(np.float32)

                if normal_file.exists():
                    with h5py.File(normal_file, "r") as f:
                        norm_raw = f["dataset"][:].astype(np.float32)
                else:
                    norm_raw = None

                rgb_img = cv2.imread(str(rgb_file))
                if rgb_img is None:
                    continue
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

                # Planar Z-depth: Z = ray * cos(theta)
                z_depth = ray_dist * cos_theta
                valid_mask = np.isfinite(z_depth) & (z_depth > 0.05) & (z_depth < 30.0)

                # Resize to (518, 518)
                rgb_res = cv2.resize(rgb_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                depth_res = cv2.resize(z_depth, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                mask_res = cv2.resize(valid_mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0

                if mask_res.sum() < 200:
                    continue

                if norm_raw is not None:
                    norm_res = cv2.resize(norm_raw, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    n_len = np.linalg.norm(norm_res, axis=-1, keepdims=True)
                    norm_res = np.where(n_len > 1e-6, norm_res / np.maximum(n_len, 1e-6), 0.0)
                else:
                    norm_res = np.zeros((*TARGET_SIZE, 3), dtype=np.float32)

                # Camera pose c2w
                R = orientations[traj_idx]
                t = positions[traj_idx] * meters_per_unit
                pose_c2w = np.eye(4, dtype=np.float32)
                pose_c2w[:3, :3] = R
                pose_c2w[:3, 3] = t

                # Save standardized frames
                base_name = f"frame_{fid:04d}"
                rgb_out = out_scene_dir / f"{base_name}_rgb.png"
                depth_out = out_scene_dir / f"{base_name}_depth.npy"
                normal_out = out_scene_dir / f"{base_name}_normal.npy"
                mask_out = out_scene_dir / f"{base_name}_mask.png"
                cam_out = out_scene_dir / f"{base_name}_cam.npz"

                cv2.imwrite(str(rgb_out), cv2.cvtColor(rgb_res, cv2.COLOR_RGB2BGR))
                np.save(str(depth_out), depth_res.astype(np.float32))
                np.save(str(normal_out), norm_res.astype(np.float32))
                cv2.imwrite(str(mask_out), (mask_res * 255).astype(np.uint8))
                np.savez_compressed(str(cam_out), intrinsics=K_std, pose=pose_c2w,
                                    scene_id=scene_id, frame_id=fid, dataset="hypersim", split=split)

                # Canonical normalization scale factor
                scale_factor = float(np.median(depth_res[mask_res]))
                if scale_factor < 0.1:
                    scale_factor = 1.0

                rel_paths = {
                    "rgb": str(rgb_out.relative_to(REPO_ROOT / "research/3d_reconstruction/data/ground_truth")),
                    "depth": str(depth_out.relative_to(REPO_ROOT / "research/3d_reconstruction/data/ground_truth")),
                    "normal": str(normal_out.relative_to(REPO_ROOT / "research/3d_reconstruction/data/ground_truth")),
                    "mask": str(mask_out.relative_to(REPO_ROOT / "research/3d_reconstruction/data/ground_truth")),
                    "cam": str(cam_out.relative_to(REPO_ROOT / "research/3d_reconstruction/data/ground_truth")),
                }

                scene_entries.append({
                    "scene_id": scene_id,
                    "frame_id": fid,
                    "frame_index": fid,
                    "dataset": "hypersim",
                    "split": split,
                    "paths": rel_paths,
                    "resolution": [TARGET_SIZE[1], TARGET_SIZE[0]],
                    "canonical_normalization": {
                        "scene_scale_factor": scale_factor
                    }
                })
            except Exception as e:
                continue

    return scene_entries


def main():
    print("=== STANDARDIZING NEW 31 HYPERSIM SCENES ===")
    scenes = sorted([d for d in os.listdir(DOWNLOADS_DIR) if d.startswith("ai_")])
    print(f"Total scenes to process: {len(scenes)}")

    all_new_entries = []

    for idx, sc in enumerate(scenes):
        if sc in LOCKED_TEST_SCENES:
            split = "test"
        elif sc in VAL_SCENES:
            split = "val"
        else:
            split = "train"

        # Use stride=3 for training to maximize diversity while preventing redundant samples (~33 frames/cam)
        # Use stride=4 for val/test
        stride = 3 if split == "train" else 4
        print(f"[{idx+1}/{len(scenes)}] Processing {sc} ({split}, stride={stride})...")
        entries = standardize_scene(sc, split=split, stride=stride)
        print(f"  -> Generated {len(entries)} frames")
        all_new_entries.extend(entries)

    # Load existing canonical manifest to preserve existing frames
    existing_manifest_file = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json"
    with open(existing_manifest_file, "r") as f:
        existing_data = json.load(f)

    existing_frames = existing_data["frames"]
    print(f"Existing canonical frames: {len(existing_frames)}")
    print(f"Newly standardized frames: {len(all_new_entries)}")

    combined_frames = existing_frames + all_new_entries
    print(f"Combined total canonical frames: {len(combined_frames)}")

    # Split distribution
    splits = {}
    for f in combined_frames:
        s = f["split"]
        splits[s] = splits.get(s, 0) + 1
    print("Final Split Breakdown:", splits)

    expanded_manifest = {
        "metadata": {
            "version": "2.0_expanded",
            "resolution": [518, 518],
            "total_frames": len(combined_frames),
            "splits": splits
        },
        "frames": combined_frames
    }

    with open(CANONICAL_MANIFEST_PATH, "w") as f:
        json.dump(expanded_manifest, f, indent=2)

    print(f"[SUCCESS] Expanded canonical manifest saved to: {CANONICAL_MANIFEST_PATH}")

if __name__ == "__main__":
    main()

