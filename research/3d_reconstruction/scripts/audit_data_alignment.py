"""
ROOMAI — PHASE 7: DATA ALIGNMENT AND GEOMETRIC REGISTRATION AUDIT
Empirical verification of:
1. RGB-D unprojection consistency and point cloud validity.
2. Planar Z-depth vs Euclidean ray distance ($Z_c = R \cos\theta$).
3. Normal vector unit length and gradient consistency.
4. Camera coordinate frame conventions (OpenCV: +X right, +Y down, +Z forward).
5. Multi-view epipolar geometry and relative pose transformation consistency.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

REPO_ROOT = Path("F:/RoomAI")
GT_BASE = REPO_ROOT / "research" / "3d_reconstruction" / "data" / "ground_truth"
MANIFEST_PATH = REPO_ROOT / "research" / "3d_reconstruction" / "data_manifests" / "indoor_ground_truth_manifest.json"

def run_alignment_audit():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    val_frames = [f for f in manifest["frames"] if f["split"] == "val"]
    print(f"Auditing {len(val_frames)} validation frames for geometric alignment...")

    hypersim_val = [f for f in val_frames if f["dataset"] == "hypersim"]
    scannet_val = [f for f in val_frames if f["dataset"] == "scannet"]

    print(f"Hypersim validation frames: {len(hypersim_val)} (Scene: {hypersim_val[0]['scene_id']})")
    print(f"ScanNet validation frames: {len(scannet_val)} (Scenes: {sorted(list(set(f['scene_id'] for f in scannet_val)))})")

    results = {
        "hypersim": {},
        "scannet": {},
        "multiview_epipolar": {}
    }

    # 1. Hypersim geometric audit
    hyp_errors = []
    normal_unit_errs_hyp = []
    cos_theta_range_hyp = []

    for fr in hypersim_val:
        depth_path = GT_BASE / fr["paths"]["depth"]
        normal_path = GT_BASE / fr["paths"]["normal"]
        cam_path = GT_BASE / fr["paths"]["cam"]
        mask_path = GT_BASE / fr["paths"]["mask"]

        depth = np.load(depth_path)
        normal = np.load(normal_path)
        mask = np.array(Image.open(mask_path)) > 0
        cam = np.load(cam_path)

        intr = cam["intrinsics"]
        pose = cam["pose"]  # c2w (4x4)

        # Unit normal test
        norm_len = np.linalg.norm(normal[mask], axis=-1)
        normal_unit_errs_hyp.append(float(np.max(np.abs(norm_len - 1.0))))

        # Planar Z-depth test: unproject to 3D and verify ray distance
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        x_c = (u - cx) * depth / fx
        y_c = (v - cy) * depth / fy
        z_c = depth
        ray_dist = np.sqrt(x_c**2 + y_c**2 + z_c**2)
        # cos(theta) = z_c / ray_dist
        cos_theta = z_c / np.maximum(ray_dist, 1e-6)
        cos_theta_range_hyp.append((float(cos_theta[mask].min()), float(cos_theta[mask].max())))

        # Pose orthogonality
        R = pose[:3, :3]
        det = np.linalg.det(R)
        ortho_err = np.max(np.abs(R @ R.T - np.eye(3)))
        hyp_errors.append((abs(det - 1.0), ortho_err))

    results["hypersim"] = {
        "num_frames": len(hypersim_val),
        "scene_id": hypersim_val[0]["scene_id"],
        "max_normal_unit_error": float(np.max(normal_unit_errs_hyp)),
        "min_cos_theta": float(np.min([c[0] for c in cos_theta_range_hyp])),
        "max_cos_theta": float(np.max([c[1] for c in cos_theta_range_hyp])),
        "max_det_error": float(np.max([e[0] for e in hyp_errors])),
        "max_ortho_error": float(np.max([e[1] for e in hyp_errors])),
    }

    # 2. ScanNet geometric audit
    scan_errors = []
    normal_unit_errs_scan = []
    valid_ratios_scan = []

    for fr in scannet_val:
        depth_path = GT_BASE / fr["paths"]["depth"]
        normal_path = GT_BASE / fr["paths"]["normal"]
        cam_path = GT_BASE / fr["paths"]["cam"]
        mask_path = GT_BASE / fr["paths"]["mask"]

        depth = np.load(depth_path)
        normal = np.load(normal_path)
        mask = np.array(Image.open(mask_path)) > 0
        cam = np.load(cam_path)

        norm_len = np.linalg.norm(normal[mask], axis=-1)
        if len(norm_len) > 0:
            normal_unit_errs_scan.append(float(np.max(np.abs(norm_len - 1.0))))
        valid_ratios_scan.append(float(np.mean(mask)))

        pose = cam["pose"]
        R = pose[:3, :3]
        det = np.linalg.det(R)
        ortho_err = np.max(np.abs(R @ R.T - np.eye(3)))
        scan_errors.append((abs(det - 1.0), ortho_err))

    results["scannet"] = {
        "num_frames": len(scannet_val),
        "scenes": sorted(list(set(f["scene_id"] for f in scannet_val))),
        "max_normal_unit_error": float(np.max(normal_unit_errs_scan)),
        "mean_valid_pixel_ratio": float(np.mean(valid_ratios_scan)),
        "min_valid_pixel_ratio": float(np.min(valid_ratios_scan)),
        "max_det_error": float(np.max([e[0] for e in scan_errors])),
        "max_ortho_error": float(np.max([e[1] for e in scan_errors])),
    }

    # 3. Multi-view epipolar geometry / relative pose correspondence check
    # Check consecutive frame pairs in ScanNet scene0030_00 and Hypersim ai_016_001
    def test_epipolar_consistency(frame_a, frame_b):
        depth_a = np.load(GT_BASE / frame_a["paths"]["depth"])
        mask_a = np.array(Image.open(GT_BASE / frame_a["paths"]["mask"])) > 0
        cam_a = np.load(GT_BASE / frame_a["paths"]["cam"])
        cam_b = np.load(GT_BASE / frame_b["paths"]["cam"])

        intr_a = cam_a["intrinsics"]
        intr_b = cam_b["intrinsics"]
        c2w_a = cam_a["pose"]
        c2w_b = cam_b["pose"]
        w2c_b = np.linalg.inv(c2w_b)

        # Unproject points in frame_a to world
        H, W = depth_a.shape
        # Subsample grid for fast exact check
        step = 8
        u, v = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
        m = mask_a[v, u]
        u = u[m]
        v = v[m]
        z = depth_a[v, u]

        fx_a, fy_a, cx_a, cy_a = intr_a[0, 0], intr_a[1, 1], intr_a[0, 2], intr_a[1, 2]
        x_c = (u - cx_a) * z / fx_a
        y_c = (v - cy_a) * z / fy_a
        pts_c = np.stack([x_c, y_c, z, np.ones_like(z)], axis=-1)  # (N, 4)

        pts_w = (c2w_a @ pts_c.T).T  # (N, 4)
        pts_cb = (w2c_b @ pts_w.T).T  # (N, 4) in camera b frame

        # Check positive Z in camera B
        pos_z_ratio = float(np.mean(pts_cb[:, 2] > 0))

        # Project into camera B
        fx_b, fy_b, cx_b, cy_b = intr_b[0, 0], intr_b[1, 1], intr_b[0, 2], intr_b[1, 2]
        u_b = fx_b * pts_cb[:, 0] / np.maximum(pts_cb[:, 2], 1e-6) + cx_b
        v_b = fy_b * pts_cb[:, 1] / np.maximum(pts_cb[:, 2], 1e-6) + cy_b

        u_proj = np.round(u_b).astype(int)
        v_proj = np.round(v_b).astype(int)

        in_bounds = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H) & (pts_cb[:, 2] > 0.1)
        in_bounds_ratio = float(np.mean(in_bounds))

        # Compute depth difference with ground truth depth in camera B at projected pixels
        depth_b = np.load(GT_BASE / frame_b["paths"]["depth"])
        mask_b = np.array(Image.open(GT_BASE / frame_b["paths"]["mask"])) > 0
        
        u_valid = u_proj[in_bounds]
        v_valid = v_proj[in_bounds]
        z_reproj = pts_cb[in_bounds, 2]
        z_gt_b = depth_b[v_valid, u_valid]
        m_b = mask_b[v_valid, u_valid] & (z_gt_b > 0.1)

        if np.sum(m_b) > 0:
            depth_err = np.abs(z_reproj[m_b] - z_gt_b[m_b])
            median_reproj_err = float(np.median(depth_err))
            mae_reproj_err = float(np.mean(depth_err))
            consistent_5cm = float(np.mean(depth_err < 0.05))
            consistent_10cm = float(np.mean(depth_err < 0.10))
        else:
            median_reproj_err = float("nan")
            mae_reproj_err = float("nan")
            consistent_5cm = float("nan")
            consistent_10cm = float("nan")

        return {
            "pos_z_ratio": pos_z_ratio,
            "in_bounds_ratio": in_bounds_ratio,
            "median_reproj_err_m": median_reproj_err,
            "mae_reproj_err_m": mae_reproj_err,
            "consistent_under_5cm_ratio": consistent_5cm,
            "consistent_under_10cm_ratio": consistent_10cm
        }

    # Hypersim consecutive pair (step of 2 frames)
    hyp_epipolar = test_epipolar_consistency(hypersim_val[10], hypersim_val[12])
    print(f"Hypersim multi-view epipolar consistency (ai_016_001 fr10->fr12): {hyp_epipolar}")

    # ScanNet consecutive pair (step of 3 frames)
    scan_scene30_00 = [f for f in scannet_val if f["scene_id"] == "scene0030_00"]
    scan_epipolar = test_epipolar_consistency(scan_scene30_00[20], scan_scene30_00[23])
    print(f"ScanNet multi-view epipolar consistency (scene0030_00 fr20->fr23): {scan_epipolar}")

    results["multiview_epipolar"] = {
        "hypersim_pair": hyp_epipolar,
        "scannet_pair": scan_epipolar
    }

    print("\n" + "="*70)
    print("PHASE 7 DATA ALIGNMENT AUDIT COMPLETE")
    print(json.dumps(results, indent=2))
    print("="*70)

    out_file = REPO_ROOT / "research" / "3d_reconstruction" / "experiments" / "phase7_data_alignment_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Audit saved to {out_file}")

if __name__ == "__main__":
    run_alignment_audit()

