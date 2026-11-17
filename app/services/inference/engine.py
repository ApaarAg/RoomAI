"""RoomAI Inference Engine.

Unified interface for loading models, running inference, and managing GPU resources.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from app.core.config import settings
from app.core.exceptions import GPUError, ModelError, InferenceError
from app.core.logging import get_logger, log_inference_event
from app.services.model_registry.registry import ModelRegistryService, get_model_registry_service


logger = get_logger("InferenceEngine")


class ModelInstance:
    """Wrapper for a loaded model with metadata."""

    def __init__(
        self,
        model_id: str,
        model: Any,
        device: torch.device,
        dtype: torch.dtype,
        config: Dict[str, Any],
    ):
        self.model_id = model_id
        self.model = model
        self.device = device
        self.dtype = dtype
        self.config = config
        self.loaded_at = time.time()
        self.last_used = time.time()
        self.inference_count = 0

    def to(self, device: torch.device):
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device


class InferenceEngine:
    """Manages model loading, inference execution, and GPU memory."""

    def __init__(self, registry: Optional[ModelRegistryService] = None):
        self.registry = registry or get_model_registry_service()
        self._loaded_models: Dict[str, ModelInstance] = {}
        self._device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self._default_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._max_models_in_memory = settings.max_concurrent_jobs  # Typically 1 for RTX 4060

    @property
    def device(self) -> torch.device:
        return self._device

    def load_model(self, model_id: str, force_reload: bool = False) -> ModelInstance:
        """Load a model from checkpoint."""
        # Check if already loaded
        if model_id in self._loaded_models and not force_reload:
            instance = self._loaded_models[model_id]
            instance.last_used = time.time()
            logger.info(f"Model {model_id} already loaded, reusing")
            return instance

        # Enforce single-model limit (evict oldest if needed)
        if len(self._loaded_models) >= self._max_models_in_memory:
            self._evict_oldest()

        # Get model spec
        model_spec = self.registry.get_model(model_id)
        checkpoint_path = self.registry.get_checkpoint_path(model_id)

        if not checkpoint_path.exists():
            raise ModelError(f"Checkpoint not found: {checkpoint_path}", model_id=model_id)

        # Load model architecture and weights
        instance = self._load_model_from_spec(model_id, model_spec, checkpoint_path)

        self._loaded_models[model_id] = instance
        logger.info(
            f"Loaded model {model_id} on {self._device}",
            extra={
                "model_id": model_id,
                "device": str(self._device),
                "checkpoint": str(checkpoint_path),
            },
        )
        return instance

    def _load_model_from_spec(
        self,
        model_id: str,
        spec: Dict[str, Any],
        checkpoint_path: Path,
    ) -> ModelInstance:
        """Load model based on its specification."""
        caps = spec.get("capabilities", {})

        # For HunyuanWorld-Mirror models
        if "worldmirror" in model_id.lower() or "hunyuan" in model_id.lower():
            return self._load_hunyuan_model(model_id, spec, checkpoint_path)

        # For YOLO/composite models
        if "composite" in model_id.lower() or "yolo" in str(spec.get("checkpoint_path", "")).lower():
            return self._load_composite_model(model_id, spec, checkpoint_path)

        raise ModelError(f"Unknown model type for {model_id}", model_id=model_id)

    def _load_hunyuan_model(
        self,
        model_id: str,
        spec: Dict[str, Any],
        checkpoint_path: Path,
    ) -> ModelInstance:
        """Load HunyuanWorld-Mirror model."""
        import sys

        # Add Hunyuan path
        repo_root = Path(__file__).resolve().parents[3]
        hunyuan_path = repo_root / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
        gsplat_path = hunyuan_path / "submodules" / "gsplat"
        scripts_path = repo_root / "research" / "3d_reconstruction" / "scripts"

        for p in [str(hunyuan_path), str(gsplat_path), str(scripts_path)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from src.models.models.worldmirror import WorldMirror
        except ImportError as exc:
            raise ModelError(f"Failed to import WorldMirror: {exc}", model_id=model_id)

        # Model config from spec or defaults
        img_size = 518
        patch_size = 14
        embed_dim = 1024
        gs_dim = 256

        # Determine which heads to enable based on model version
        if "stage2" in model_id.lower():
            enable_gs = True
        else:
            enable_gs = False

        model = WorldMirror(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            gs_dim=gs_dim,
            enable_cam=True,
            enable_pts=True,
            enable_depth=True,
            enable_norm=True,
            enable_gs=enable_gs,
            enable_cond=False,
        ).to(device=self._device, dtype=self._default_dtype)

        # Load checkpoint
        logger.info(f"Loading weights from {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd, strict=False)
        del sd, clean_sd

        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        model.eval()

        return ModelInstance(
            model_id=model_id,
            model=model,
            device=self._device,
            dtype=self._default_dtype,
            config={
                "img_size": img_size,
                "patch_size": patch_size,
                "enable_gs": enable_gs,
            },
        )

    def _load_composite_model(
        self,
        model_id: str,
        spec: Dict[str, Any],
        checkpoint_path: Path,
    ) -> ModelInstance:
        """Load RoomAI composite model (YOLO + Depth + Fusion)."""
        # This delegates to the existing RoomAICompositeAdapter
        # For now return a placeholder - the actual implementation
        # uses the existing services.roomai_backend pipeline
        from models.adapters.roomai_composite import RoomAICompositeAdapter

        adapter = RoomAICompositeAdapter(mock_mode=False)
        if not adapter.is_available()[0]:
            raise ModelError("Composite model not available", model_id=model_id)

        # Wrap adapter in ModelInstance-like object
        class CompositeWrapper:
            def __init__(self, adapter):
                self.adapter = adapter

            def reconstruct(self, images, options=None, progress_callback=None):
                return self.adapter.reconstruct(images, options, progress_callback)

            def validate_inputs(self, images):
                return self.adapter.validate_inputs(images)

            def get_capabilities(self):
                return self.adapter.get_capabilities()

        wrapper = CompositeWrapper(adapter)

        return ModelInstance(
            model_id=model_id,
            model=wrapper,
            device=self._device,
            dtype=self._default_dtype,
            config={},
        )

    def _evict_oldest(self) -> None:
        """Evict least recently used model from memory."""
        if not self._loaded_models:
            return

        oldest_id = min(self._loaded_models.keys(), key=lambda k: self._loaded_models[k].last_used)
        instance = self._loaded_models.pop(oldest_id)

        # Clear GPU memory
        del instance.model
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Evicted model {oldest_id} from memory")

    def unload_model(self, model_id: str) -> bool:
        """Explicitly unload a model."""
        if model_id in self._loaded_models:
            instance = self._loaded_models.pop(model_id)
            del instance.model
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model {model_id}")
            return True
        return False

    def unload_all(self) -> None:
        """Unload all models."""
        for model_id in list(self._loaded_models.keys()):
            self.unload_model(model_id)

    def run_inference(
        self,
        model_id: str,
        images: List[Image.Image],
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Any:
        """Run inference with specified model."""
        start_time = time.time()

        # Load model
        instance = self.load_model(model_id)
        instance.inference_count += 1
        instance.last_used = time.time()

        try:
            # Run inference based on model type
            if hasattr(instance.model, 'reconstruct'):
                # Hunyuan or composite adapter
                result = instance.model.reconstruct(
                    images=images,
                    options=options or {},
                    progress_callback=progress_callback,
                )
            elif hasattr(instance.model, 'forward'):
                # Raw PyTorch model (Hunyuan)
                result = self._run_hunyuan_forward(instance, images, options)
            else:
                raise InferenceError(f"Model {model_id} has no inference method")

            duration_ms = (time.time() - start_time) * 1000

            # Log inference event
            log_inference_event(
                logger,
                "inference_complete",
                model_id,
                input_shape=(len(images), 3, 518, 518),
                duration_ms=duration_ms,
                vram_mb=torch.cuda.memory_allocated() / (1024*1024) if self._device.type == "cuda" else 0,
            )

            return result

        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise GPUError(
                "GPU out of memory during inference",
                operation=f"inference:{model_id}",
            ) from exc
        except Exception as exc:
            logger.error(f"Inference failed for {model_id}: {exc}", exc_info=True)
            raise InferenceError(f"Inference failed: {exc}", model_id=model_id) from exc

    def _run_hunyuan_forward(
        self,
        instance: ModelInstance,
        images: List[Image.Image],
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run HunyuanWorld-Mirror forward pass."""
        model = instance.model
        config = instance.config
        img_size = config.get("img_size", 518)

        # Preprocess images to 518x518
        tensors = []
        orig_sizes = []
        for img in images:
            # Square crop
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            cropped = img.crop((left, top, left + min_dim, top + min_dim))
            resized = cropped.resize((img_size, img_size), Image.Resampling.BILINEAR)

            arr = np.array(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)
            orig_sizes.append((w, h))

        batch = torch.stack(tensors, dim=0).unsqueeze(0).to(instance.device, instance.dtype)

        with torch.no_grad():
            preds = model(
                views={"img": batch},
                cond_flags=[0, 0, 0],
                is_inference=True,
            )

        # Extract predictions
        pred_depth = preds.get("depth")
        pred_depth_conf = preds.get("depth_conf")
        pred_pts = preds.get("pts3d")
        pred_norm = preds.get("normals")
        pred_camera_poses = preds.get("camera_poses")
        pred_camera_intrinsics = preds.get("camera_intrs")

        # Process outputs (similar to hunyuan_mirror_service)
        C_MODEL_TO_ROOMAI = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

        depth_maps = []
        depth_confidence_maps = []
        finite_depth_masks = []
        point_export_masks = []
        camera_poses = []
        camera_intrinsics = []
        all_pts = []
        all_rgb = []
        all_norm = []

        num_views = len(images)

        def scalar_map_for_view(tensor: Optional[torch.Tensor], view_index: int, name: str) -> Optional[np.ndarray]:
            if tensor is None:
                return None
            view = tensor[0, view_index]
            if view.ndim == 3 and view.shape[-1] == 1:
                view = view[..., 0]
            elif view.ndim == 3 and view.shape[0] == 1:
                view = view[0]
            elif view.ndim != 2:
                raise InferenceError(f"Unexpected {name} output shape: {tuple(tensor.shape)}")
            expected = (batch.shape[-2], batch.shape[-1])
            if tuple(view.shape) != expected:
                raise InferenceError(
                    f"{name} output shape {tuple(view.shape)} does not match model RGB {expected}"
                )
            return view.float().cpu().numpy()

        for v in range(num_views):
            # WorldMirror dense heads return channel-last predictions [B,V,H,W,C].
            # Keep the 2D depth map aligned with the model input; indexing [0,v,0]
            # would select only the first image row and broadcast it over points.
            d_v = scalar_map_for_view(pred_depth, v, "depth")
            conf_v = scalar_map_for_view(pred_depth_conf, v, "depth confidence")
            pts_v = pred_pts[0, v].float().cpu().numpy() if pred_pts is not None else None
            norm_v = pred_norm[0, v].float().cpu().numpy() if pred_norm is not None else None
            rgb_v = batch[0, v].permute(1, 2, 0).float().cpu().numpy()

            if d_v is not None:
                depth_maps.append(d_v)
                finite_depth_masks.append(np.isfinite(d_v))
            if conf_v is not None:
                depth_confidence_maps.append(conf_v)

            if pred_camera_poses is not None:
                camera_poses.append(pred_camera_poses[0, v].float().cpu().numpy())
            if pred_camera_intrinsics is not None:
                camera_intrinsics.append(pred_camera_intrinsics[0, v].float().cpu().numpy())

            if pts_v is not None:
                pts_roomai = pts_v @ C_MODEL_TO_ROOMAI.T
                norm_roomai = norm_v @ C_MODEL_TO_ROOMAI.T if norm_v is not None else None
                if norm_roomai is not None:
                    norm_mag = np.linalg.norm(norm_roomai, axis=-1, keepdims=True)
                    norm_roomai = norm_roomai / np.maximum(norm_mag, 1e-12)

                valid = np.isfinite(pts_roomai).all(axis=-1)
                if d_v is not None:
                    valid = valid & np.isfinite(d_v) & (d_v > 0.1) & (d_v < 25.0)

                point_export_masks.append(valid)

                all_pts.append(pts_roomai[valid])
                all_rgb.append(rgb_v[valid])
                if norm_roomai is not None:
                    all_norm.append(norm_roomai[valid])

        # Consolidate
        if all_pts:
            all_pts = np.concatenate(all_pts, axis=0)
            all_rgb = np.concatenate(all_rgb, axis=0)
            all_norm = np.concatenate(all_norm, axis=0) if all_norm else None
        else:
            all_pts = np.zeros((0, 3), dtype=np.float32)
            all_rgb = np.zeros((0, 3), dtype=np.float32)
            all_norm = None

        return {
            "status": "success",
            "point_cloud_xyz": all_pts,
            "point_cloud_rgb": all_rgb,
            "point_cloud_normals": all_norm,
            "depth_maps": depth_maps,
            "depth_confidence_maps": depth_confidence_maps,
            "finite_depth_masks": finite_depth_masks,
            "point_export_masks": point_export_masks,
            "camera_poses": camera_poses,
            "camera_intrinsics": camera_intrinsics,
            "num_views": num_views,
            "num_points": len(all_pts),
        }

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs."""
        return list(self._loaded_models.keys())

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics."""
        if self._device.type != "cuda":
            return {"device": "cpu"}

        return {
            "device": str(self._device),
            "total_vram_mb": torch.cuda.get_device_properties(0).total_memory / (1024*1024),
            "allocated_mb": torch.cuda.memory_allocated() / (1024*1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024*1024),
            "loaded_models": len(self._loaded_models),
        }


# Singleton
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
