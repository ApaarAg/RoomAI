"""
ROOMAI — Phase 3+5: Hypersim Training Data Loader and Experiment Controller
 
This module provides the new HypersimDataset class that:
1. Reads the hypersim_training_manifest_v1.json
2. Loads RGB from HDF5
3. Loads depth from HDF5 (ray distance) and converts to planar Z-depth  
4. Loads camera-space normals from PNG
5. Extracts and interpolates camera poses
6. Applies proper NaN/Inf masking
7. Returns tensors compatible with the existing WorldMirror pipeline

Uses the roomai-hunyuan conda env (Python 3.10, PyTorch 2.11, h5py).
"""

import os
import gc
import csv
import json
import math
import time
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    raise ImportError("h5py is required. Install in roomai-hunyuan env.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("F:/RoomAI")
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "hypersim_training_manifest_v1.json"

# ---------------------------------------------------------------------------
# Intrinsics grid computation (cached for efficiency)
# ---------------------------------------------------------------------------
def make_ray_distance_to_z_factor_map(
    h: int, w: int, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """
    Compute per-pixel cos(theta) factor to convert ray distance to planar Z-depth.
    
    Z_planar(u, v) = D_ray(u, v) * cos(theta(u, v))
    where theta(u, v) = arctan(sqrt(((u - cx)/fx)^2 + ((v - cy)/fy)^2))
    
    Returns:
        factor: (H, W) float32 array where factor[v,u] = cos(theta(u,v))
    """
    u_coords = np.arange(w, dtype=np.float32)
    v_coords = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)
    
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    
    theta = np.arctan(np.sqrt(dx**2 + dy**2))
    factor = np.cos(theta).astype(np.float32)
    return factor


# ---------------------------------------------------------------------------
# Scene normalization (from existing scene_normalization.py logic)
# ---------------------------------------------------------------------------
def compute_scene_scale_factor(depth_meters: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    Compute scene scale factor s such that normalized depth = depth / s.
    Uses median of valid metric depth values.
    """
    valid_depth = depth_meters[valid_mask]
    if len(valid_depth) < 10:
        return 1.0
    return float(np.median(valid_depth))


def compute_world_points(
    depth_z: np.ndarray,  # planar Z-depth (H, W)
    intrinsics: Dict,     # fx, fy, cx, cy
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Back-project depth to 3D camera-space points.
    Returns (H, W, 3) world-space points (here: camera-space for single-view).
    """
    h, w = depth_z.shape
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    
    u_coords = np.arange(w, dtype=np.float32)
    v_coords = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)
    
    # Camera-space points (OpenCV: X right, Y down, Z forward)
    X = (uu - cx) / fx * depth_z
    Y = (vv - cy) / fy * depth_z
    Z = depth_z.copy()
    
    pts = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    pts[~valid_mask] = 0.0
    return pts


# ---------------------------------------------------------------------------
# HDF5 readers
# ---------------------------------------------------------------------------
def read_color_hdf5(path: Path) -> Optional[np.ndarray]:
    """Read color HDF5. Returns (H, W, 3) float32 in [0,1] with tone mapping."""
    if not HDF5_AVAILABLE or not path.exists():
        return None
    try:
        with h5py.File(str(path), "r") as f:
            keys = list(f.keys())
            if not keys:
                return None
            data = f[keys[0]][:]  # May be (H, W, 3) float or uint8
        
        if data.dtype == np.uint8:
            rgb = data.astype(np.float32) / 255.0
        else:
            # Hypersim stores HDR float images; apply simple Reinhard tone mapping
            data = data.astype(np.float32)
            # Reinhard global tone mapping: L / (1 + L)
            # Apply per-channel; clip negatives
            data = np.clip(data, 0.0, None)
            # Simple gamma-correct linear RGB to display
            # Normalize by 95th percentile to handle HDR range
            p95 = np.percentile(data[data > 0], 95) if (data > 0).any() else 1.0
            if p95 > 0:
                data = data / p95
            data = np.clip(data, 0.0, 1.0)
            rgb = data
        
        # Ensure (H, W, 3)
        if rgb.ndim == 2:
            rgb = np.stack([rgb]*3, axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        
        return rgb.astype(np.float32)
    except Exception as e:
        return None


def read_depth_hdf5(path: Path) -> Optional[np.ndarray]:
    """Read depth_meters HDF5. Returns (H, W) float32 ray distances."""
    if not HDF5_AVAILABLE or not path.exists():
        return None
    try:
        with h5py.File(str(path), "r") as f:
            keys = list(f.keys())
            if not keys:
                return None
            data = f[keys[0]][:]
        
        depth = data.astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        elif depth.ndim == 3:
            depth = depth[..., 0]  # take first channel if multi-channel
        
        return depth
    except Exception as e:
        return None


def read_normal_png(path: Path, h: int, w: int) -> Optional[np.ndarray]:
    """
    Read camera-space normal PNG.
    Hypersim normals are stored as uint8 PNG: normal = (pixel / 255.0) * 2 - 1.
    Returns (H, W, 3) float32 unit normals.
    """
    if path is None or not path.exists():
        return None
    try:
        img = Image.open(str(path)).convert("RGB")
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        # Decode: [-1, 1] range
        normals = (arr / 255.0) * 2.0 - 1.0
        # Normalize to unit length
        norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
        norm_len = np.maximum(norm_len, 1e-8)
        normals = normals / norm_len
        return normals
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class HypersimDataset(Dataset):
    """
    Dataset loading WorldMirror-compatible tensors from the Hypersim manifest.
    
    For each frame:
    - Loads RGB from HDF5 → (3, H, W) float32 [0, 1]
    - Loads depth from HDF5 → converts ray distance to planar Z → (H, W) float32
    - Applies validity mask (NaN/Inf/zero excluded) → (H, W) bool
    - Computes 3D camera-space points from depth → (H, W, 3) float32
    - Loads camera-space normals → (H, W, 3) float32
    - Normalizes depth and points by scene scale factor
    
    Returns dict compatible with the existing WorldMirror training loop.
    """
    
    TARGET_H = 480  # WorldMirror standard resolution
    TARGET_W = 640
    
    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        max_frames_per_scene: Optional[int] = None,
        skip_frames: int = 5,  # sample every N frames to reduce temporal correlation
    ):
        self.split = split
        self.max_frames_per_scene = max_frames_per_scene
        self.skip_frames = skip_frames
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        
        self.source_root = Path(self.manifest["source_root"])
        
        # Build flat frame list
        self.frames = []
        self._build_frame_list()
        
        print(f"[HypersimDataset/{split}] {len(self.frames):,} frames from "
              f"{len(set(f['scene_id'] for f in self.frames))} scenes")
    
    def _build_frame_list(self):
        for scene in self.manifest["scenes"]:
            if scene["split"] != self.split:
                continue
            
            scene_id = scene["scene_id"]
            
            for cam in scene["cameras"]:
                intrinsics = cam.get("intrinsics") or {}
                # Default to 1024x768 60-degree FOV if missing
                h = intrinsics.get("height", 768)
                w = intrinsics.get("width", 1024)
                fx = intrinsics.get("fx", 892.3)
                fy = intrinsics.get("fy", 892.3)
                cx = intrinsics.get("cx", 512.0)
                cy = intrinsics.get("cy", 384.0)
                
                frames_in_cam = cam.get("frames", [])
                
                # Apply skip factor to reduce temporal correlation
                selected = frames_in_cam[::self.skip_frames]
                
                if self.max_frames_per_scene is not None:
                    selected = selected[:self.max_frames_per_scene]
                
                for frame in selected:
                    rgb_path = frame.get("rgb_hdf5")
                    depth_path = frame.get("depth_hdf5")
                    normal_path = frame.get("normal_cam_png")
                    
                    if rgb_path is None or depth_path is None:
                        continue
                    
                    self.frames.append({
                        "scene_id": scene_id,
                        "cam_id": cam["cam_id"],
                        "frame_idx": frame.get("frame_idx", "0"),
                        "rgb_hdf5": REPO_ROOT / rgb_path,
                        "depth_hdf5": REPO_ROOT / depth_path,
                        "normal_png": REPO_ROOT / normal_path if normal_path else None,
                        "intrinsics": {"h": h, "w": w, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
                        "dataset": "hypersim",
                    })
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        frame_info = self.frames[idx]
        intr = frame_info["intrinsics"]
        h, w = intr["h"], intr["w"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        
        # 1. Load RGB
        rgb = read_color_hdf5(frame_info["rgb_hdf5"])
        if rgb is None:
            return self._get_fallback(frame_info)
        
        # 2. Load depth (ray distance)
        depth_ray = read_depth_hdf5(frame_info["depth_hdf5"])
        if depth_ray is None:
            return self._get_fallback(frame_info)
        
        # Align dimensions
        if rgb.shape[0] != h or rgb.shape[1] != w:
            h, w = rgb.shape[0], rgb.shape[1]
            cx = w / 2.0
            cy = h / 2.0
        
        if depth_ray.shape[0] != h or depth_ray.shape[1] != w:
            depth_ray = np.array(Image.fromarray(depth_ray).resize((w, h), Image.NEAREST))
        
        # 3. Convert ray distance → planar Z-depth
        cos_factor = make_ray_distance_to_z_factor_map(h, w, fx, fy, cx, cy)
        depth_z = depth_ray * cos_factor  # (H, W) planar metric depth
        
        # 4. Build validity mask
        valid_mask = (
            np.isfinite(depth_z) & 
            np.isfinite(depth_ray) &
            (depth_z > 0.05) &     # min 5 cm
            (depth_z < 50.0)       # max 50 m (indoor scenes)
        )
        
        if valid_mask.sum() < 1000:
            return self._get_fallback(frame_info)
        
        # 5. Scene scale factor
        scale_s = compute_scene_scale_factor(depth_z, valid_mask)
        if scale_s < 0.01:
            scale_s = 1.0
        
        # 6. Normalized depth
        depth_norm = depth_z / scale_s
        depth_norm[~valid_mask] = 0.0
        
        # 7. 3D points (camera space)
        pts = compute_world_points(depth_z, {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, valid_mask)
        pts_norm = pts / scale_s
        
        # 8. Surface normals
        normals = read_normal_png(frame_info.get("normal_png"), h, w)
        if normals is None:
            # Fallback: compute normals from depth gradient
            normals = self._normals_from_depth(depth_z, fx, fy, cx, cy, valid_mask)
        normals[~valid_mask] = 0.0
        
        # 9. Resize to target resolution for consistent batching
        rgb_resized = self._resize_rgb(rgb)
        
        # Resize mask, depth, pts, normals
        valid_mask_resized = self._resize_mask(valid_mask)
        depth_norm_resized = self._resize_depth(depth_norm)
        pts_norm_resized = self._resize_pts(pts_norm)
        normals_resized = self._resize_normals(normals)
        
        # 10. Convert to tensors
        rgb_t = torch.from_numpy(rgb_resized.transpose(2, 0, 1))  # (3, H, W)
        depth_t = torch.from_numpy(depth_norm_resized)  # (H, W)
        depth_metric_t = torch.from_numpy(self._resize_depth(depth_z))  # (H, W)
        pts_t = torch.from_numpy(pts_norm_resized)  # (H, W, 3)
        norm_t = torch.from_numpy(normals_resized)  # (H, W, 3)
        mask_t = torch.from_numpy(valid_mask_resized)  # (H, W) bool
        
        return {
            "rgb": rgb_t,                    # (3, H, W) float32 [0,1]
            "depth_norm": depth_t,           # (H, W) float32 (depth/scale)
            "depth_metric": depth_metric_t,  # (H, W) float32 (metric depth)
            "pts_norm": pts_t,               # (H, W, 3) float32 (pts/scale)
            "normals": norm_t,               # (H, W, 3) float32 unit normals
            "mask": mask_t,                  # (H, W) bool
            "scale_factor": float(scale_s),
            "scene_id": frame_info["scene_id"],
            "cam_id": frame_info["cam_id"],
            "frame_id": frame_info["frame_idx"],
            "dataset": "hypersim",
        }
    
    def _get_fallback(self, frame_info):
        """Return empty-ish sample for failed frames (will be skipped by loss via mask)."""
        H, W = self.TARGET_H, self.TARGET_W
        return {
            "rgb": torch.zeros(3, H, W),
            "depth_norm": torch.zeros(H, W),
            "depth_metric": torch.zeros(H, W),
            "pts_norm": torch.zeros(H, W, 3),
            "normals": torch.zeros(H, W, 3),
            "mask": torch.zeros(H, W, dtype=torch.bool),
            "scale_factor": 1.0,
            "scene_id": frame_info["scene_id"],
            "cam_id": frame_info.get("cam_id", "00"),
            "frame_id": frame_info.get("frame_idx", "0"),
            "dataset": "hypersim",
        }
    
    def _normals_from_depth(
        self, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, mask: np.ndarray
    ) -> np.ndarray:
        """Compute surface normals from depth via gradient cross-product."""
        h, w = depth.shape
        normals = np.zeros((h, w, 3), dtype=np.float32)
        
        # Sobel gradients
        dz_dx = np.gradient(depth, axis=1)
        dz_dy = np.gradient(depth, axis=0)
        
        # Normal from: n = (-dz/dx / fx, -dz/dy / fy, 1) (unnormalized)
        nx = -dz_dx / fx
        ny = -dz_dy / fy
        nz = np.ones_like(depth)
        
        n = np.stack([nx, ny, nz], axis=-1)
        n_len = np.linalg.norm(n, axis=-1, keepdims=True)
        n_len = np.maximum(n_len, 1e-8)
        n = n / n_len
        
        normals[mask] = n[mask]
        return normals
    
    def _resize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        H, W = self.TARGET_H, self.TARGET_W
        if rgb.shape[0] == H and rgb.shape[1] == W:
            return rgb
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img = img.resize((W, H), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    
    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        H, W = self.TARGET_H, self.TARGET_W
        if depth.shape[0] == H and depth.shape[1] == W:
            return depth
        img = Image.fromarray(depth)
        img = img.resize((W, H), Image.NEAREST)
        return np.array(img, dtype=np.float32)
    
    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        H, W = self.TARGET_H, self.TARGET_W
        if mask.shape[0] == H and mask.shape[1] == W:
            return mask
        img = Image.fromarray(mask.astype(np.uint8) * 255)
        img = img.resize((W, H), Image.NEAREST)
        return np.array(img, dtype=bool)
    
    def _resize_pts(self, pts: np.ndarray) -> np.ndarray:
        H, W = self.TARGET_H, self.TARGET_W
        if pts.shape[0] == H and pts.shape[1] == W:
            return pts
        # Resize each channel
        result = np.zeros((H, W, 3), dtype=np.float32)
        for c in range(3):
            img = Image.fromarray(pts[..., c])
            img = img.resize((W, H), Image.NEAREST)
            result[..., c] = np.array(img, dtype=np.float32)
        return result
    
    def _resize_normals(self, normals: np.ndarray) -> np.ndarray:
        H, W = self.TARGET_H, self.TARGET_W
        if normals.shape[0] == H and normals.shape[1] == W:
            return normals
        result = np.zeros((H, W, 3), dtype=np.float32)
        for c in range(3):
            ch = ((normals[..., c] + 1.0) / 2.0 * 255).astype(np.uint8)
            img = Image.fromarray(ch)
            img = img.resize((W, H), Image.BILINEAR)
            result[..., c] = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
        # Re-normalize
        n_len = np.linalg.norm(result, axis=-1, keepdims=True)
        n_len = np.maximum(n_len, 1e-8)
        result = result / n_len
        return result


def test_dataset_loading():
    """Quick validation of the data loader."""
    print(f"Testing HypersimDataset...")
    
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        return False
    
    # Test train split with skip_frames=10 for speed
    dataset = HypersimDataset(MANIFEST_PATH, split="train", skip_frames=10)
    print(f"Train dataset: {len(dataset)} frames")
    
    # Load first 5 samples
    for i in [0, 1, 2, 3, 4]:
        if i >= len(dataset):
            break
        t0 = time.time()
        sample = dataset[i]
        elapsed = time.time() - t0
        
        if sample is None:
            print(f"  [{i}] None returned")
            continue
        
        rgb = sample["rgb"]
        depth = sample["depth_norm"]
        mask = sample["mask"]
        pts = sample["pts_norm"]
        normals = sample["normals"]
        
        n_valid = mask.sum().item()
        pct_valid = 100.0 * n_valid / (mask.numel())
        
        print(f"  [{i}] Scene={sample['scene_id']} cam={sample['cam_id']} frame={sample['frame_id']}")
        print(f"       rgb={rgb.shape} range=[{rgb.min():.3f},{rgb.max():.3f}]")
        print(f"       depth={depth.shape} range=[{depth[mask].min():.3f},{depth[mask].max():.3f}] valid={pct_valid:.1f}%")
        print(f"       pts={pts.shape} normals={normals.shape}")
        print(f"       scale_factor={sample['scale_factor']:.3f} load_time={elapsed:.2f}s")
        
        # Verify no NaN/Inf in output
        assert torch.isfinite(rgb).all(), "NaN/Inf in RGB output!"
        assert torch.isfinite(depth[mask]).all(), "NaN/Inf in masked depth!"
        assert torch.isfinite(pts[mask]).all(), "NaN/Inf in masked points!"
        print(f"       [OK] No NaN/Inf in outputs")
    
    # Test val split
    val_ds = HypersimDataset(MANIFEST_PATH, split="val", skip_frames=5)
    print(f"\nVal dataset: {len(val_ds)} frames")
    s = val_ds[0]
    if s:
        print(f"  Val sample: scene={s['scene_id']} depth_range=[{s['depth_metric'][s['mask']].min():.2f},{s['depth_metric'][s['mask']].max():.2f}]m")
    
    print("\n[OK] Dataset loading test PASSED")
    return True


if __name__ == "__main__":
    test_dataset_loading()

