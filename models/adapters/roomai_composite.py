"""RoomAI Composite Model Adapter — Wraps Production YOLO11s + Depth Anything V2 + SIFT/Kabsch."""

import gc
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import yaml

from models.adapters.base import ModelCapabilities, ReconstructionModelAdapter
from schemas.scene import SceneGraph
from services.multiview_fusion_service import MultiViewFusionService
from services.scene_fusion_service import SceneFusionService
from utils.device import get_device_string
from utils.gpu_concurrency import GPU_GATE
from utils.logging_config import get_logger
from utils.reconstruction_contract import normalize_intrinsics

logger = get_logger("RoomAICompositeAdapter")
REPO_ROOT = Path(__file__).resolve().parents[2]


class RoomAICompositeAdapter(ReconstructionModelAdapter):
    """Production adapter wrapping RoomAI's verified single-view and multi-view pipelines."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.config_path = Path(config_path) if config_path else (REPO_ROOT / "config" / "model_config.yaml")
        self.mock_mode = mock_mode
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as exc:
                logger.error(f"Failed to load model config: {exc}")
                self.config = {}

        self._single_view_service: Optional[SceneFusionService] = None
        self._multi_view_service: Optional[MultiViewFusionService] = None
        self._is_initialized = False

    def get_capabilities(self) -> ModelCapabilities:
        """Declares composite pipeline capabilities."""
        return ModelCapabilities(
            model_id="roomai_composite_v1",
            name="RoomAI Composite Geometry Pipeline",
            version="1.4.0",
            status="PRODUCTION_READY",
            supports_single_view=True,
            supports_multi_view=True,
            min_views=1,
            max_views=10,
            supports_metric_depth=True,
            supports_surface_normals=False,
            supports_3dgs=False,
            supports_mesh_glb=False,
            supports_point_cloud_ply=True,
            device=get_device_string(),
            memory_required_mb=1500,
            description=(
                "Single-view indoor perception with YOLO11s A3 (800px), YOLO11n-Seg, and a metric-trained "
                "Depth Anything V2 Indoor Small estimator. Produces a visible-surface colored point-cloud candidate; "
                "absolute scale and camera calibration require independent validation. It does not produce a complete "
                "surface mesh, and multi-view alignment remains experimental."
            ),
        )

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Verifies checkpoint presence on disk."""
        if self.mock_mode:
            return True, None

        required_checkpoints = [
            REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt",
            REPO_ROOT / "checkpoints" / "yolo11n-seg.pt",
            REPO_ROOT / "checkpoints" / "depth_anything_v2_metric_indoor_small",
        ]
        missing = [str(p.name) for p in required_checkpoints if not p.exists()]
        if missing:
            return False, f"Missing required checkpoints: {', '.join(missing)}"

        if torch.cuda.is_available():
            free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            if free_vram_mb < 500:
                return False, f"Insufficient VRAM: {free_vram_mb:.1f} MB available, requires at least 500 MB."

        return True, None

    def initialize(self) -> None:
        """Initializes underlying fusion services."""
        if self._is_initialized:
            return

        available, reason = self.is_available()
        if not available and not self.mock_mode:
            raise RuntimeError(f"Cannot initialize RoomAICompositeAdapter: {reason}")

        self._single_view_service = SceneFusionService(mock_mode=self.mock_mode, config=self.config)
        self._multi_view_service = MultiViewFusionService(mock_mode=self.mock_mode, config=self.config)
        self._is_initialized = True
        logger.info("RoomAICompositeAdapter initialized successfully.")

    def validate_inputs(self, images: List[Image.Image]) -> Tuple[bool, Optional[str]]:
        """Validates view count and image validity."""
        if not images:
            return False, "At least one input image is required."
        if len(images) > 10:
            return False, f"Maximum supported view count is 10, received {len(images)} images."
        for idx, img in enumerate(images):
            if not isinstance(img, Image.Image):
                return False, f"Image at index {idx} is not a valid PIL Image."
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Image at index {idx} has invalid dimensions: {w}x{h}."
            if w > 8192 or h > 8192:
                return False, f"Image at index {idx} exceeds maximum dimension bound (8192px): {w}x{h}."
        return True, None

    def reconstruct(
        self,
        images: List[Image.Image],
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> SceneGraph:
        """Executes 3D reconstruction."""
        if not self._is_initialized:
            self.initialize()

        valid, err = self.validate_inputs(images)
        if not valid:
            raise ValueError(f"Input validation failed: {err}")

        opts = options or {}
        refine_depth = opts.get("refine_depth", True)
        if len(images) == 1 and opts.get("camera_intrinsics") is not None:
            normalize_intrinsics(images[0].width, images[0].height, opts["camera_intrinsics"])

        def report(stage: str, frac: float):
            if progress_callback:
                try:
                    progress_callback(stage, frac)
                except Exception as exc:
                    logger.debug(f"Progress callback ignored error: {exc}")

        report("VALIDATING", 0.10)

        # Acquire GPU gate for thread safety and memory isolation
        with GPU_GATE.acquire():
            if len(images) == 1:
                report("RECONSTRUCTING", 0.35)
                # Single-view reconstruction
                assert self._single_view_service is not None
                report("FUSING", 0.70)
                scene = self._single_view_service.fuse(
                    images[0], camera_intrinsics=opts.get("camera_intrinsics")
                )
                report("EXPORTING", 0.95)
                return scene
            else:
                report("RECONSTRUCTING", 0.30)
                # Multi-view reconstruction
                assert self._multi_view_service is not None
                valid_imgs, warnings, _ = self._multi_view_service.validate_multiview_sequence(images)
                active_images = valid_imgs if valid_imgs else images
                report("FUSING", 0.65)
                scene = self._multi_view_service.fuse_multiview(active_images, refine_depth=refine_depth)
                report("EXPORTING", 0.95)
                return scene

    def cleanup(self) -> None:
        """Releases memory and flushes CUDA cache."""
        self._single_view_service = None
        self._multi_view_service = None
        self._is_initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("RoomAICompositeAdapter cleaned up.")
