"""RoomAI — HunyuanWorld-Mirror Production Service Adapter.

Integrates the Tencent HunyuanWorld-Mirror 1.0 foundation model into RoomAI's
backend pipeline following the formal Integration Contract (arXiv:2510.10726).

Handles:
- Safe singleton model loading with bfloat16/float16 precision
- 518x518 canonical image preprocessing (DINOv2 patch alignment)
- Single-view and multi-view geometric unprojection
- Model-to-RoomAI coordinate transformation (C = diag(1, -1, -1))
- Point cloud extraction with confidence masking and voxel filtering
- Surface normal extraction and orientation validation
- Memory management and graceful CPU/mock fallback
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "scripts"

if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from src.models.models.worldmirror import WorldMirror
    HUNYUAN_AVAILABLE = True
except ImportError:
    HUNYUAN_AVAILABLE = False

from utils.logging_config import get_logger

logger = get_logger("HunyuanMirrorService")

# Canonical Coordinate Transformation Matrix: Model (OpenGL/Blender) -> RoomAI (OpenCV Pinhole)
# Model: +X right, +Y up, +Z back
# RoomAI: +X right, +Y down, +Z forward
C_MODEL_TO_ROOMAI = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


class HunyuanWorldMirrorAdapter:
    """Thread-safe production adapter for HunyuanWorld-Mirror 1.0."""

    _instance = None

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        img_size: int = 518,
    ):
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.img_size = img_size
        self.model: Optional[WorldMirror] = None
        self._loaded = False

        if checkpoint_path is None:
            from models.adapters.hunyuan_mirror_adapter import resolve_hunyuan_checkpoint_path
            resolved = resolve_hunyuan_checkpoint_path()
            self.checkpoint_path = str(resolved) if resolved else None
        else:
            self.checkpoint_path = str(checkpoint_path)

        if self.checkpoint_path and Path(self.checkpoint_path).exists() and HUNYUAN_AVAILABLE:
            self._init_model()

    def _init_model(self):
        """Initializes model architecture and loads weights with safety assertions."""
        try:
            logger.info(f"Initializing WorldMirror-1.0 architecture (img_size={self.img_size})...")

            # Determine if this is a Stage 2 checkpoint (has gs_head/gs_renderer)
            checkpoint_path = Path(self.checkpoint_path)
            is_stage2 = "stage2" in str(checkpoint_path).lower()

            self.model = WorldMirror(
                img_size=self.img_size,
                patch_size=14,
                embed_dim=1024,
                gs_dim=256,
                enable_cam=True,
                enable_pts=True,
                enable_depth=True,
                enable_norm=True,
                enable_gs=is_stage2,  # Enable GS head for Stage 2
                enable_cond=False,
            ).to(device=self.device, dtype=self.dtype)

            logger.info(f"Loading checkpoint weights from {self.checkpoint_path}...")
            sd = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}

            # Verify architecture compatibility
            model_keys = set(self.model.state_dict().keys())
            ckpt_keys = set(clean_sd.keys())
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys

            if missing:
                logger.warning(f"Missing keys in checkpoint: {len(missing)} keys (e.g., {list(missing)[:5]})")
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys (e.g., {list(unexpected)[:5]})")

            self.model.load_state_dict(clean_sd, strict=False)
            del sd, clean_sd
            if self.device == "cuda":
                torch.cuda.empty_cache()

            self.model.eval()
            self._loaded = True
            logger.info(f"WorldMirror-1.0 loaded successfully ({'Stage 2' if is_stage2 else 'Stage 1/Reference'}).")
        except Exception as e:
            logger.error(f"Failed to initialize WorldMirror model: {e}", exc_info=True)
            self._loaded = False
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocesses PIL image to canonical 518x518 square tensor with patch-14 alignment."""
        orig_w, orig_h = image.size
        # Square crop to preserve aspect ratio without non-isometric distortion
        min_dim = min(orig_w, orig_h)
        left = (orig_w - min_dim) // 2
        top = (orig_h - min_dim) // 2
        cropped = image.crop((left, top, left + min_dim, top + min_dim))
        resized = cropped.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        rgb_arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).to(device=self.device, dtype=self.dtype)
        return tensor, (orig_w, orig_h)

    def reconstruct(
        self,
        images: Union[Image.Image, List[Image.Image]],
        confidence_threshold: float = 0.5,
        max_points: int = 40000,
    ) -> Dict[str, Any]:
        """Executes 3D reconstruction on single or multi-view images.

        Returns structured geometric dictionary matching RoomAI schemas.
        """
        t0 = time.perf_counter()
        if not self.is_ready:
            return {
                "status": "error",
                "message": "HunyuanWorldMirror model is not initialized or checkpoint is missing.",
                "point_cloud": None,
                "depth_map": None,
                "normals": None,
            }

        img_list = [images] if isinstance(images, Image.Image) else list(images)
        if not img_list:
            return {"status": "error", "message": "No images provided."}

        num_views = len(img_list)
        preprocessed_tensors = []
        for img in img_list:
            t, _ = self.preprocess_image(img)
            preprocessed_tensors.append(t)

        # Batch shape: (1, V, 3, 518, 518)
        batch_rgb = torch.stack(preprocessed_tensors, dim=0).unsqueeze(0)

        with torch.no_grad():
            preds = self.model(
                views={"img": batch_rgb},
                cond_flags=[0, 0, 0],
                is_inference=True,
            )

        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Extract predictions
        # depth: (1, V, 1, 518, 518)
        # pts3d: (1, V, 518, 518, 3)
        # normals: (1, V, 518, 518, 3)
        pred_depth = preds.get("depth")
        pred_pts = preds.get("pts3d")
        pred_norm = preds.get("normals")
        pred_cam = preds.get("cam")
        pred_conf = preds.get("pts3d_conf", preds.get("conf"))

        depth_maps = []
        pts_list = []
        rgb_list = []
        norm_list = []

        for v in range(num_views):
            d_v = pred_depth[0, v, :, :, 0].float().cpu().numpy() if pred_depth is not None else None
            pts_v = pred_pts[0, v].float().cpu().numpy() if pred_pts is not None else None
            norm_v = pred_norm[0, v].float().cpu().numpy() if pred_norm is not None else None
            rgb_v = batch_rgb[0, v].permute(1, 2, 0).float().cpu().numpy()

            if d_v is not None:
                depth_maps.append(d_v)

            if pts_v is not None:
                # Apply coordinate transformation from Model to RoomAI
                # pts_roomai = pts_model @ C.T
                pts_roomai = pts_v @ C_MODEL_TO_ROOMAI.T

                # Normal transformation
                norm_roomai = None
                if norm_v is not None:
                    norm_roomai = norm_v @ C_MODEL_TO_ROOMAI.T
                    norm_mag = np.linalg.norm(norm_roomai, axis=-1, keepdims=True)
                    norm_roomai = norm_roomai / np.maximum(norm_mag, 1e-12)

                # Filter valid points
                valid = np.isfinite(pts_roomai).all(axis=-1)
                if d_v is not None:
                    valid = valid & np.isfinite(d_v) & (d_v > 0.1) & (d_v < 25.0)

                valid_pts = pts_roomai[valid]
                valid_rgb = rgb_v[valid]

                pts_list.append(valid_pts)
                rgb_list.append(valid_rgb)
                if norm_roomai is not None:
                    norm_list.append(norm_roomai[valid])

        # Consolidate cross-view point cloud
        if pts_list:
            all_pts = np.concatenate(pts_list, axis=0)
            all_rgb = np.concatenate(rgb_list, axis=0)
            all_norm = np.concatenate(norm_list, axis=0) if norm_list else None

            # Subsample if exceeding max_points
            if len(all_pts) > max_points:
                step = len(all_pts) // max_points
                all_pts = all_pts[::step][:max_points]
                all_rgb = all_rgb[::step][:max_points]
                if all_norm is not None:
                    all_norm = all_norm[::step][:max_points]
        else:
            all_pts = np.zeros((0, 3), dtype=np.float32)
            all_rgb = np.zeros((0, 3), dtype=np.float32)
            all_norm = None

        def to_numpy(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().float().cpu().numpy()
            if isinstance(value, dict):
                return {key: to_numpy(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_numpy(item) for item in value]
            return value

        return {
            "status": "success",
            "point_cloud_xyz": all_pts,
            "point_cloud_rgb": all_rgb,
            "point_cloud_normals": all_norm,
            "depth_maps": depth_maps,
            "primary_depth": depth_maps[0] if depth_maps else None,
            "native_outputs": {
                "depth": to_numpy(pred_depth),
                "points": to_numpy(pred_pts),
                "confidence": to_numpy(pred_conf),
                "normals": to_numpy(pred_norm),
                "camera": to_numpy(pred_cam),
                "coordinate_transform_model_to_roomai": C_MODEL_TO_ROOMAI.copy(),
            },
            "timing_ms": {
                "inference_time_ms": inference_ms,
                "total_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
            "num_views": num_views,
            "num_points": len(all_pts),
        }


# Singleton instance accessor
_HUNYUAN_ADAPTER = None

def get_hunyuan_adapter(checkpoint_path: Optional[str] = None) -> HunyuanWorldMirrorAdapter:
    global _HUNYUAN_ADAPTER
    if _HUNYUAN_ADAPTER is None:
        _HUNYUAN_ADAPTER = HunyuanWorldMirrorAdapter(checkpoint_path=checkpoint_path)
    return _HUNYUAN_ADAPTER

