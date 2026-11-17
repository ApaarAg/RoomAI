"""
Compute official WorldMirror canonical scene scale factor s for all scenes in
indoor_ground_truth_manifest.json and generate indoor_canonical_manifest.json.

For each scene:
  1. Gathers all frames {depth, intrinsics, camera_poses}.
  2. Computes 3D world coordinates: X_w = R * X_cam + t.
  3. Gathers valid pixels: depth > 0 and isfinite.
  4. Aggregates all valid 3D points across all views in the scene.
  5. Computes scene-wide scale factor: s = nanmedian(||X_w||_2).
  6. Annotates each frame with:
     - 'scene_scale_factor': float
     - 'normalized_depth_median': float
     - 'normalized_depth_min': float
     - 'normalized_depth_max': float
"""

import json
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path("f:/RoomAI")
INPUT_MANIFEST = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_ground_truth_manifest.json"
OUTPUT_MANIFEST = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json"
STANDARDIZED_ROOT = REPO_ROOT / "research/3d_reconstruction/data/ground_truth"


def compute_world_pts(depth: np.ndarray, intr: np.ndarray, c2w: np.ndarray):
    H, W = depth.shape
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    valid = (depth > 0.0) & np.isfinite(depth)
    
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z_cam = np.where(valid, depth, 0.0).astype(np.float32)
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy
    X_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    X_w = np.einsum("ik, vuk -> vui", R, X_cam) + t[None, None, :]
    return X_w, valid


def main():
    print(f"Loading manifest: {INPUT_MANIFEST}")
    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    print(f"Total frames: {len(frames)}")

    # Group frames by scene_id
    scene_frames = {}
    for f_idx, f_info in enumerate(frames):
        s_id = f_info["scene_id"]
        if s_id not in scene_frames:
            scene_frames[s_id] = []
        scene_frames[s_id].append((f_idx, f_info))

    print(f"Grouped into {len(scene_frames)} distinct scenes.")
    scene_scales = {}

    for s_id, f_list in sorted(scene_frames.items()):
        print(f"\nProcessing Scene: {s_id} ({len(f_list)} frames, dataset: {f_list[0][1]['dataset']})...")
        t0 = time.time()
        
        all_scene_pts = []
        
        # Load and compute world points for each frame
        for f_idx, f_info in f_list:
            paths = f_info["paths"]
            depth_path = STANDARDIZED_ROOT / paths["depth"]
            cam_path = STANDARDIZED_ROOT / paths["cam"]
            
            depth = np.load(depth_path).astype(np.float32)
            cam_data = np.load(cam_path)
            intr = cam_data["intrinsics"].astype(np.float32)
            c2w = cam_data["pose"].astype(np.float32)
            
            X_w, valid = compute_world_pts(depth, intr, c2w)
            valid_pts = X_w[valid]
            if len(valid_pts) > 0:
                # Subsample if large to keep memory reasonable
                if len(valid_pts) > 10000:
                    sub_idx = np.random.choice(len(valid_pts), 10000, replace=False)
                    all_scene_pts.append(valid_pts[sub_idx])
                else:
                    all_scene_pts.append(valid_pts)

        if len(all_scene_pts) > 0:
            concat_pts = np.concatenate(all_scene_pts, axis=0)
            dis = np.linalg.norm(concat_pts, ord=2, axis=-1)
            scale = float(np.nanmedian(dis))
            if np.isnan(scale) or np.isinf(scale) or scale <= 0:
                scale = 1.0
                bad_case = True
            else:
                bad_case = False
        else:
            scale = 1.0
            bad_case = True

        scene_scales[s_id] = {
            "scale_factor": scale,
            "bad_case": bad_case,
            "dataset": f_list[0][1]["dataset"],
            "split": f_list[0][1]["split"],
            "frame_count": len(f_list),
        }
        elapsed = time.time() - t0
        print(f"  -> Scene scale factor s: {scale:.4f} (bad_case: {bad_case}, {elapsed:.2f}s)")

    # Annotate frames with canonical normalization data
    for s_id, f_list in scene_frames.items():
        s = scene_scales[s_id]["scale_factor"]
        for f_idx, f_info in f_list:
            raw_median = f_info["depth_metrics"]["median_m"]
            raw_min = f_info["depth_metrics"]["min_m"]
            raw_max = f_info["depth_metrics"]["max_m"]
            
            f_info["canonical_normalization"] = {
                "scene_scale_factor": s,
                "normalized_depth_median": round(raw_median / s, 4),
                "normalized_depth_min": round(raw_min / s, 4),
                "normalized_depth_max": round(raw_max / s, 4),
            }

    # Add summary
    data["canonical_normalization_summary"] = {
        "formula": "s = nanmedian(||X_w||_2)",
        "scenes": scene_scales,
    }

    print(f"\nWriting canonical manifest to: {OUTPUT_MANIFEST}")
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    np.random.seed(42)
    main()
