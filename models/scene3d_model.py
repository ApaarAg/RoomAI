"""3D Vision and Monocular Depth Estimation module for RoomAI.

Integrates Depth Anything V2 Small (DINOv2 ViT-Small + DPT Head) for real-time monocular
depth estimation. Supports relative depth mode from local checkpoint.

NOTE: The available checkpoint provides RELATIVE depth, not metric depth.
Metric depth requires a separate checkpoint calibrated for meters.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from models.interfaces import BaseScene3D, ModelNotImplementedError
from schemas.responses import Scene3DResponse
from utils.logging_config import get_logger

logger = get_logger("Scene3DModel")

# Camera intrinsics presets for common indoor capture scenarios
class CustomScene3DModel(BaseScene3D):
    """Real Monocular Depth Estimation model using Depth Anything V2.

    This implementation uses the RELATIVE depth checkpoint available at
    checkpoints/depth_anything_v2_small/

    Output:
    - Relative inverse depth: higher values = closer to camera
    - Range: [0, 1] (normalized)
    - NOT metric depth (not in meters)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Store whether config was explicitly provided (to distinguish from None)
        self._config_supplied = config is not None
        super().__init__(config)
        self.model = None
        self.processor = None
        self.resolved_device = "cpu"
        self.depth_mode = "relative"  # Always relative for this checkpoint
        self.model_name = "Depth Anything V2 Small (Relative)"
        self._init_error = None  # Store initialization errors for debugging
        self._model_loaded = False

        self.REPO_ROOT = Path(__file__).resolve().parent.parent
        default_checkpoint = self.REPO_ROOT / "checkpoints" / "depth_anything_v2_small"

        # Check if enabled - if explicitly false, don't load model
        enabled = self.config.get("enabled", True)
        if enabled is False:
            logger.info("Scene3D model disabled by config (enabled: false). Skipping initialization.")
            self._model_loaded = False
            self.checkpoint_path = default_checkpoint
            self.local_checkpoint = self.checkpoint_path
            self.camera_intrinsics = self.config.get("camera_intrinsics", {})
            return

        # Determine checkpoint path
        cp = self.config.get("checkpoint_path", "")
        if cp:
            self.checkpoint_path = Path(cp)
            if not self.checkpoint_path.is_absolute():
                self.checkpoint_path = self.REPO_ROOT / cp
        else:
            # No explicit checkpoint_path, use default
            self.checkpoint_path = default_checkpoint
        self.local_checkpoint = self.checkpoint_path

        # Auto-load if config was supplied and checkpoint exists
        if self._config_supplied and self.local_checkpoint.exists():
            try:
                self.load_model()
                self._model_loaded = True
            except ModelNotImplementedError as e:
                self._init_error = str(e)
                logger.warning(f"Model load skipped due to missing implementation: {e}")
            except Exception as e:
                self._init_error = f"Initialization error: {e}"
                logger.error(f"Failed to load model: {e}")
                raise
        elif not self._config_supplied:
            # No config supplied, use nonexistent path to indicate unconfigured
            self.local_checkpoint = Path("nonexistent_unconfigured_path")

        # Camera intrinsics for point cloud generation
        self.camera_intrinsics = self.config.get("camera_intrinsics", {})

    def load_model(self) -> None:
        """Loads Depth Anything V2 processor and weights from local checkpoint."""
        if not self.local_checkpoint.exists():
            raise ModelNotImplementedError(
                f"Checkpoint not found at {self.local_checkpoint}. "
                "Expected: transformers-compatible checkpoint with model.safetensors",
                __file__
            )

        try:
            import torch
        except ImportError:
            raise ModelNotImplementedError(
                f"PyTorch not available. Cannot load depth model.",
                __file__
            )

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise ModelNotImplementedError(
                f"Transformers not available: {e}",
                __file__
            )

        target_device = self.config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        if "cuda" in str(target_device) and not torch.cuda.is_available():
            logger.warning(f"CUDA '{target_device}' requested but unavailable. Falling back to CPU.")
            self.resolved_device = "cpu"
        else:
            self.resolved_device = target_device

        logger.info(f"Loading Depth Anything V2 from local checkpoint on {self.resolved_device}...")

        try:
            # Load from local checkpoint directory
            checkpoint_path = str(self.local_checkpoint)
            self.processor = AutoImageProcessor.from_pretrained(checkpoint_path)
            self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint_path)
            self.model.to(self.resolved_device)
            self.model.eval()

            # Determine depth mode (metric vs relative)
            is_metric_config = getattr(self.model.config, "depth_estimation_type", "") == "metric"
            is_metric_path = "metric" in checkpoint_path.lower()
            is_metric_req = self.config.get("depth_mode") == "metric" or self.config.get("metric_depth", False)

            if is_metric_config or is_metric_path or is_metric_req:
                self.depth_mode = "metric"
                self.model_name = "Depth Anything V2 Metric Indoor Small"
            else:
                self.depth_mode = "relative"
                self.model_name = "Depth Anything V2 Small (Relative Depth)"

            logger.info(f"{self.model_name} loaded successfully ({sum(p.numel() for p in self.model.parameters()):,} params) on {self.resolved_device}")
        except Exception as e:
            raise ModelNotImplementedError(
                f"Failed to load depth model from {self.local_checkpoint}: {e}",
                __file__
            )

    def is_ready(self) -> bool:
        return self._model_loaded and self.model is not None and self.processor is not None

    def get_model_name(self) -> str:
        if self.depth_mode == "metric":
            return "Depth Anything V2 Metric Indoor Small"
        return "Depth Anything V2 Small (Relative Depth)"

    def _unproject_depth_to_pointcloud(
        self,
        depth_map: np.ndarray,
        image: Image.Image,
        max_points: int = 200000,
    ) -> Dict[str, List]:
        """Unproject depth map to 3D point cloud using camera intrinsics.

        Args:
            depth_map: Relative depth map (0-1, higher = closer)
            image: Original RGB PIL Image
            max_points: Maximum points to return

        Returns:
            Dict with 'xyz', 'rgb', 'depth' as lists
        """
        h, w = depth_map.shape
        fx = float(self.camera_intrinsics.get("fx", max(w, h)))
        fy = float(self.camera_intrinsics.get("fy", max(w, h)))
        cx = float(self.camera_intrinsics.get("cx", (w - 1) / 2.0))
        cy = float(self.camera_intrinsics.get("cy", (h - 1) / 2.0))

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # For relative depth: higher value = closer
        # Convert to proxy Z values (we'll use the relative depth as-is)
        valid = depth_map > 0
        z = depth_map  # Using relative depth as proxy for Z

        valid_indices = np.where(valid)
        if len(valid_indices[0]) == 0:
            return {"xyz": [], "rgb": [], "depth": []}

        # Downsample if too many points
        n_valid = len(valid_indices[0])
        if n_valid > max_points:
            idx = np.random.choice(n_valid, max_points, replace=False)
            valid_indices = (valid_indices[0][idx], valid_indices[1][idx])

        v_idx, u_idx = valid_indices
        z_vals = z[v_idx, u_idx]

        # Unproject: x = (u - cx) * z / fx, y = (v - cy) * z / fy
        xx = (u_idx - cx) * z_vals / fx
        yy = (v_idx - cy) * z_vals / fy

        xyz = np.stack([xx, yy, z_vals], axis=1).astype(np.float32)

        # Get RGB colors
        img_arr = np.array(image.convert("RGB"))
        rgb = img_arr[v_idx, u_idx].astype(np.float32) / 255.0

        return {
            "xyz": xyz.tolist(),
            "rgb": rgb.tolist(),
            "depth": z_vals.tolist(),
        }

    def reconstruct(
        self, image: Image.Image
    ) -> Tuple[Optional[np.ndarray], Scene3DResponse]:
        """Performs monocular depth estimation using Depth Anything V2.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Tuple of:
              - Normalized 2D float32 depth map for visualization
              - Scene3DResponse with point cloud and metadata
        """
        if not self.is_ready():
            raise ModelNotImplementedError("CustomScene3DModel", __file__)

        if image is None:
            raise ValueError("Input image cannot be None")

        import torch

        # Preprocess
        image_rgb = image.convert("RGB")
        orig_w, orig_h = image_rgb.size

        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.resolved_device) for k, v in inputs.items()}

        # Sync GPU before timing
        is_cuda = "cuda" in str(self.resolved_device) and torch.cuda.is_available()
        if is_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Sync after inference
        if is_cuda:
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated(self.resolved_device) / (1024 * 1024)
        else:
            peak_vram_mb = 0.0

        inference_time_ms = (time.perf_counter() - t_start) * 1000

        # Resize to original dimensions
        pred_aligned = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        raw_depth = pred_aligned.cpu().numpy().astype(np.float32)

        # Handle non-finite values
        raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_raw_depth = raw_depth

        # Separate normalization logic for metric vs relative:
        if self.depth_mode == "metric":
            # Physical meters: valid indoor range typically 0.1m to 25m
            valid_mask = (raw_depth > 0.05) & (raw_depth < 30.0)
            valid_depths = raw_depth[valid_mask]
            d_min = float(valid_depths.min()) if len(valid_depths) > 0 else 0.0
            d_max = float(valid_depths.max()) if len(valid_depths) > 0 else 0.0

            # Invert visualization so closer points are brighter
            p2 = float(np.percentile(valid_depths, 2)) if len(valid_depths) > 0 else 0.5
            p98 = float(np.percentile(valid_depths, 98)) if len(valid_depths) > 0 else 10.0
            if p98 <= p2:
                p2, p98 = d_min, max(d_max, d_min + 0.1)
            clipped = np.clip(raw_depth, p2, p98)
            norm_depth = (p98 - clipped) / (p98 - p2) if (p98 - p2) > 0 else np.zeros_like(clipped)
            norm_depth = np.clip(norm_depth, 0.0, 1.0).astype(np.float32)

            msg = f"Depth Anything V2 Metric Indoor depth estimation completed (range: {d_min:.2f}m - {d_max:.2f}m)."
            is_rel = False
            unit = "meters"
            disc = "Metric indoor depth map: values represent absolute physical distance in meters (0 to 20m). Point cloud Z represents physical metric distance."
        else:
            # Relative inverse depth: higher value = closer
            valid_mask = raw_depth > 0
            valid_depths = raw_depth[valid_mask]
            d_min = float(valid_depths.min()) if len(valid_depths) > 0 else 0.0
            d_max = float(valid_depths.max()) if len(valid_depths) > 0 else 0.0

            p2, p98 = np.percentile(raw_depth[valid_mask], [2, 98]) if np.any(valid_mask) else (0.0, 1.0)
            if p98 <= p2:
                p2, p98 = float(raw_depth.min()), float(raw_depth.max())
            clipped = np.clip(raw_depth, p2, p98)
            norm_depth = (p98 - clipped) / (p98 - p2) if (p98 - p2) > 0 else np.zeros_like(clipped)
            norm_depth = np.clip(norm_depth, 0.0, 1.0).astype(np.float32)

            msg = "Depth Anything V2 (Relative) depth estimation completed."
            is_rel = True
            unit = "relative (inverse depth, NOT metric)"
            disc = (
                "Relative depth map output: higher values indicate proximity to camera. "
                "Value range is normalized; NOT metric distance in meters. "
                "Point cloud Z-coordinates are proxy values for visualization only."
            )

        # Generate point cloud
        point_cloud = self._unproject_depth_to_pointcloud(raw_depth, image_rgb)

        response = Scene3DResponse(
            status="Complete",
            depth_available=True,
            point_cloud_available=True,
            is_simulated=False,
            message=msg,
            model_name=self.get_model_name(),
            inference_time_ms=round(inference_time_ms, 2),
            device=str(self.resolved_device),
            peak_memory_mb=round(peak_vram_mb, 1),
            depth_min=round(d_min, 6),
            depth_max=round(d_max, 6),
            relative_depth_only=is_rel,
            depth_unit=unit,
            disclaimer=disc,
            raw_depth_map=raw_depth.tolist() if raw_depth.size < 500000 else None,
            camera_intrinsics=self.camera_intrinsics,
            point_cloud=point_cloud,
        )

        return norm_depth, response


class MockScene3DModel(BaseScene3D):
    """Deterministic mock 3D model for validating the 3D analysis UI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._ready = True
        self.camera_intrinsics = config.get("camera_intrinsics") if config else None
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {"fx": 1000, "fy": 1000, "cx": 0.5, "cy": 0.5}

    def load_model(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "Deterministic Mock 3D Model (Simulated)"

    def _generate_mock_point_cloud(self, image: Image.Image) -> Dict[str, List]:
        """Generate a synthetic point cloud for mock mode testing."""
        width, height = image.size
        fx = float(self.camera_intrinsics.get("fx", max(width, height)))
        fy = float(self.camera_intrinsics.get("fy", max(width, height)))
        cx = float(self.camera_intrinsics.get("cx", (width - 1) / 2.0))
        cy = float(self.camera_intrinsics.get("cy", (height - 1) / 2.0))

        # Create a simple synthetic depth map (gradient)
        y_coords = np.linspace(0.5, 3.0, height, dtype=np.float32)
        depth_map = np.tile(y_coords[:, np.newaxis], (1, width))

        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Unproject to 3D
        zz = depth_map
        xx = (u - cx) * zz / fx
        yy = (v - cy) * zz / fy

        # Get RGB (use image colors or generate synthetic)
        img_arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        rgb = img_arr.reshape(-1, 3)
        xyz = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)

        # Downsample for performance
        max_points = 100000
        n_points = width * height
        if n_points > max_points:
            idx = np.random.choice(n_points, max_points, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]

        return {
            "xyz": xyz.tolist(),
            "rgb": rgb.tolist(),
            "depth": depth_map.reshape(-1).tolist(),
        }

    def reconstruct(
        self, image: Image.Image
    ) -> Tuple[Optional[np.ndarray], Scene3DResponse]:
        width, height = image.size
        # Generate synthetic monocular depth gradient (0=near, 255=far)
        y = np.linspace(255, 0, height, dtype=np.float32)
        depth_map = np.tile(y[:, np.newaxis], (1, width))

        # Generate mock point cloud
        point_cloud = self._generate_mock_point_cloud(image)

        response = Scene3DResponse(
            status="Complete",
            depth_available=True,
            point_cloud_available=True,
            is_simulated=True,
            message="Deterministic synthetic depth map and point cloud for testing the 3D extension interface.",
            model_name=self.get_model_name(),
            device="cpu",
            inference_time_ms=1.2,
            point_cloud=point_cloud,
        )
        return depth_map, response
