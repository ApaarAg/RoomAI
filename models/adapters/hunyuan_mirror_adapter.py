"""HunyuanWorld-Mirror Model Adapter — Production Foundation Model Reconstructor."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import yaml

from models.adapters.base import ModelCapabilities, ReconstructionModelAdapter
from schemas.scene import SceneGraph
from utils.device import get_device_string
from utils.logging_config import get_logger

logger = get_logger("HunyuanMirrorAdapter")
REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical Checkpoint Locations
DEFAULT_EXP05_CHECKPOINT = REPO_ROOT / "results" / "final_execution" / "checkpoints" / "exp_05_expanded_lora_rank8_final.ckpt"
DEFAULT_BASELINE_CHECKPOINT = REPO_ROOT / "research" / "3d_reconstruction" / "checkpoints" / "production_candidate" / "production_model.ckpt"


def resolve_hunyuan_checkpoint_path() -> Path:
    """Dynamically resolves the Hunyuan checkpoint path from env, config, or default paths."""
    # 1. Environment variable override
    env_ckpt = os.getenv("ROOMAI_HUNYUAN_CHECKPOINT")
    if env_ckpt:
        p = Path(env_ckpt)
        if p.exists():
            return p

    # 2. Model configuration file
    cfg_path = REPO_ROOT / "config" / "model_config.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            h_cfg = cfg.get("hunyuan_world_mirror_1b", {})
            ckpt_rel = h_cfg.get("checkpoint_path")
            if ckpt_rel:
                p = (REPO_ROOT / ckpt_rel).resolve()
                if p.exists():
                    return p
        except Exception as exc:
            logger.warning(f"Failed to read checkpoint from {cfg_path}: {exc}")

    # 3. Winning production candidate Exp 05 LoRA
    if DEFAULT_EXP05_CHECKPOINT.exists():
        return DEFAULT_EXP05_CHECKPOINT

    # 4. Pretrained baseline reference
    if DEFAULT_BASELINE_CHECKPOINT.exists():
        return DEFAULT_BASELINE_CHECKPOINT

    return DEFAULT_EXP05_CHECKPOINT


class HunyuanMirrorAdapter(ReconstructionModelAdapter):
    """Adapter for Tencent HunyuanWorld-Mirror 3D foundation model.
    
    Conforms to research/3d_reconstruction/INTEGRATION_CONTRACT.md.
    Provides direct feed-forward 3D room reconstruction using WorldMirror-1.0.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        mock_mode: bool = False,
    ):
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = resolve_hunyuan_checkpoint_path()
        self.mock_mode = mock_mode
        self._is_initialized = False

    def get_capabilities(self) -> ModelCapabilities:
        """Declares foundation model capabilities."""
        available, _ = self.is_available()
        status = "STAGED" if available else "UNAVAILABLE"

        return ModelCapabilities(
            model_id="hunyuan_world_mirror_1b",
            name="HunyuanWorld-Mirror (1B Foundation Model)",
            version="1.0.0-lora-exp05" if "exp_05" in str(self.checkpoint_path) else "1.0.0-production",
            status=status,
            supports_single_view=True,
            supports_multi_view=True,
            min_views=1,
            max_views=6,
            supports_metric_depth=False,
            supports_surface_normals=True,
            supports_3dgs=False,
            supports_mesh_glb=False,
            supports_point_cloud_ply=True,
            device=get_device_string(),
            memory_required_mb=4800,
            description=(
                "Research HunyuanWorld-Mirror 1.0 output with scene-normalized geometry semantics. "
                "Metric depth and surface meshes are not currently supported or product-approved."
            ),
        )

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Verifies checkpoint availability."""
        if self.mock_mode:
            return True, None
        if not self.checkpoint_path.exists():
            return (
                False,
                f"HunyuanWorld-Mirror 1B checkpoint not found at {self.checkpoint_path}. "
                "Training / evaluation is currently being conducted by the independent ML Research Agent.",
            )
        if torch.cuda.is_available():
            from services.hunyuan_mirror_service import get_hunyuan_adapter
            adapter = get_hunyuan_adapter(str(self.checkpoint_path))
            if not adapter.is_ready:
                free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                if free_vram_mb < 4500:
                    return False, f"Insufficient VRAM for 1B model: {free_vram_mb:.1f} MB available, requires 4800 MB."
        return True, None

    def initialize(self) -> None:
        """Initializes model pipeline."""
        available, reason = self.is_available()
        if not available and not self.mock_mode:
            raise RuntimeError(f"HunyuanMirrorAdapter unavailable: {reason}")
        self._is_initialized = True
        logger.info("HunyuanMirrorAdapter initialized.")

    def validate_inputs(self, images: List[Image.Image]) -> Tuple[bool, Optional[str]]:
        """Validates input sequence."""
        if not images:
            return False, "At least one input image is required."
        if len(images) > 6:
            return False, f"HunyuanWorld-Mirror supports max 6 views on 8GB GPU, received {len(images)}."
        return True, None

    def reconstruct(
        self,
        images: List[Image.Image],
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> SceneGraph:
        """Executes reconstruction or raises if not ready."""
        available, reason = self.is_available()
        if not available:
            raise NotImplementedError(
                f"Cannot execute HunyuanWorld-Mirror reconstruction (Model Unavailable): {reason}. "
                "Use 'roomai_composite_v1' for verified production reconstruction."
            )

        if self.mock_mode:
            w, h = images[0].size
            import numpy as np
            xs = np.linspace(-2.0, 2.0, 50, dtype=np.float32)
            zs = np.linspace(1.0, 5.0, 50, dtype=np.float32)
            gx, gz = np.meshgrid(xs, zs)
            gy = np.full_like(gx, 1.4)
            pts = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1).tolist()
            rgb = np.full_like(np.array(pts), 0.7).tolist()
            return SceneGraph(
                image_id="mock_hunyuan",
                width=w,
                height=h,
                model_name="HunyuanWorld-Mirror 1.0 (Mock)",
                depth_mode="metric",
                camera_intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 0.5, "cy": 0.5},
                point_cloud={"xyz": pts, "rgb": rgb},
            )

        from services.hunyuan_mirror_service import get_hunyuan_adapter
        adapter = get_hunyuan_adapter(str(self.checkpoint_path))
        if not adapter.is_ready:
            raise RuntimeError("HunyuanWorldMirrorAdapter could not be loaded on this system.")

        if progress_callback:
            progress_callback("Running feed-forward transformer geometry prediction...", 0.5)

        res = adapter.reconstruct(images)
        if res["status"] != "success":
            raise RuntimeError(f"Hunyuan reconstruction error: {res.get('message')}")

        w, h = images[0].size
        pts_xyz = res["point_cloud_xyz"].tolist() if hasattr(res["point_cloud_xyz"], "tolist") else res["point_cloud_xyz"]
        pts_rgb = res["point_cloud_rgb"].tolist() if hasattr(res["point_cloud_rgb"], "tolist") else res["point_cloud_rgb"]
        n_pts = len(pts_xyz) if pts_xyz else 0

        d_map = res.get("primary_depth")
        depth_min = 0.0
        depth_max = 0.0
        if d_map is not None:
            depth_min = float(d_map.min())
            depth_max = float(d_map.max())
        elif n_pts > 0 and isinstance(pts_xyz, list) and len(pts_xyz[0]) >= 3:
            zs = [p[2] for p in pts_xyz if len(p) >= 3]
            if zs:
                depth_min = float(min(zs))
                depth_max = float(max(zs))

        heatmap_img = None
        if d_map is not None:
            from services.image_service import create_depth_heatmap
            try:
                heatmap_img = create_depth_heatmap(d_map)
            except Exception as e:
                logger.warning(f"Could not create depth heatmap: {e}")

        if progress_callback:
            progress_callback("Consolidating 3D point cloud...", 0.9)

        return SceneGraph(
            image_id="hunyuan_recon",
            width=w,
            height=h,
            model_name="Tencent HunyuanWorld-Mirror 1.0",
            depth_mode="metric",
            camera_intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 0.5, "cy": 0.5},
            point_cloud={"xyz": pts_xyz, "rgb": pts_rgb},
            n_total_points=n_pts,
            n_valid_points=n_pts,
            depth_min_m=depth_min,
            depth_max_m=depth_max,
            depth_map=d_map,
            heatmap_image=heatmap_img,
            model_native_outputs=res.get("native_outputs"),
            model_coordinate_transform="diag(1,-1,-1)",
            depth_semantics="scene_normalized",
            checkpoint_path=str(self.checkpoint_path),
        )

    def cleanup(self) -> None:
        """Releases resources."""
        self._is_initialized = False
