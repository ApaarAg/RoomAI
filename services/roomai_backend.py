"""RoomAI Unified Backend Facade — Production Service API for Frontend Integration.

This module provides a unified, exception-safe, high-level interface for frontend
applications (Streamlit, FastAPI, REST, or CLI). Frontend developers can execute
detection, segmentation, depth estimation, 3D room reconstruction, and asset export
without touching model-loading boilerplate, neural network inference loops, or
coordinate-transformation math.

Usage:
    from services.roomai_backend import RoomAIBackend, BACKEND

    # 1. System capabilities and health check
    capabilities = BACKEND.get_capabilities()
    health = BACKEND.get_system_health()

    # 2. Object detection
    det_result = BACKEND.detect_objects(image, task_group="Group A: Indoor Furniture & Household Objects")

    # 3. 3D room reconstruction (single-view or multi-view)
    recon_result = BACKEND.reconstruct_room_3d(image)

    # 4. Export 3D assets
    exports = BACKEND.export_3d_assets(recon_result["scene_graph"])
    # exports["glb"] -> glTF 2.0 binary bytes
    # exports["ply"] -> ASCII PLY bytes
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import yaml

# Decompression bomb DOS protection
Image.MAX_IMAGE_PIXELS = 100_000_000

from models.registry import GLOBAL_MODEL_REGISTRY, ModelRegistry
from schemas.requests import DesignRequest
from schemas.responses import (
    DetectionResponse,
    SegmentationResponse,
    Scene3DResponse,
    VLMAnswerResponse,
    DesignGenerationResponse,
)
from schemas.scene import SceneGraph
from services.classification_service import ClassificationService
from services.detection_service import DetectionService
from services.generation_service import GenerationService
from services.multiview_fusion_service import MultiViewFusionService
from services.scene3d_service import Scene3DService
from services.scene_fusion_service import SceneFusionService
from services.segmentation_service import SegmentationService
from services.vision_language_service import VisionLanguageService
from utils.export_3d import export_point_cloud_ply, export_scene_glb, validate_glb_bytes, validate_ply_bytes
from utils.logging_config import get_logger

logger = get_logger("RoomAIBackend")
REPO_ROOT = Path(__file__).resolve().parent.parent


def validate_input_image(image: Any, max_dim: int = 8192) -> Tuple[bool, Optional[str]]:
    """Validates that input is a valid PIL Image within allowable dimension bounds."""
    if image is None:
        return False, "Input image cannot be None."
    if not isinstance(image, Image.Image):
        return False, f"Expected PIL Image, got {type(image).__name__}."
    w, h = image.size
    if w <= 0 or h <= 0:
        return False, f"Invalid image dimensions: {w}x{h}."
    if w > max_dim or h > max_dim:
        return False, f"Image dimensions ({w}x{h}) exceed maximum allowable size ({max_dim}x{max_dim})."
    return True, None


class RoomAIBackend:
    """Unified, thread-safe production backend facade for RoomAI."""

    def __init__(self, config_path: Optional[str] = None):
        """Initializes backend configuration and service registries."""
        self.config_path = Path(config_path) if config_path else (REPO_ROOT / "config" / "model_config.yaml")
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as exc:
                logger.error(f"Failed to load backend configuration from {self.config_path}: {exc}")
                self.config = {}

        self.registry: ModelRegistry = GLOBAL_MODEL_REGISTRY

    def get_capabilities(self) -> Dict[str, Any]:
        """Returns the capability matrix and operational readiness status of all backend subsystems."""
        cuda_avail = torch.cuda.is_available()
        device_str = "cuda" if cuda_avail else "cpu"

        return {
            "indoor_detection": {
                "status": "PRODUCTION_READY",
                "model": "YOLO11s (A3 Fine-Tuned 800px)",
                "checkpoint": "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt",
                "device": device_str,
                "notes": "Verified mAP50=0.512 on indoor validation set. Primary indoor detector.",
            },
            "exterior_detection": {
                "status": "PRODUCTION_READY",
                "model": "YOLO11s (B2 Fine-Tuned)",
                "checkpoint": "checkpoints/exterior_detector_b2_yolo11s_best.pt",
                "device": device_str,
                "notes": "Architectural exterior elements detector.",
            },
            "parking_detection": {
                "status": "PRODUCTION_READY",
                "model": "YOLO11s (C2 Pilot)",
                "checkpoint": "checkpoints/parking_detector_c2_yolo11s_pilot_best.pt",
                "device": device_str,
                "notes": "Vehicles and parking asset detector.",
            },
            "segmentation": {
                "status": "PRODUCTION_READY",
                "model": "YOLO11n-Seg",
                "checkpoint": "checkpoints/yolo11n-seg.pt",
                "device": device_str,
                "notes": "Instance segmentation for room geometry and boundary extraction.",
            },
            "depth_estimation_metric": {
                "status": "PRODUCTION_READY",
                "model": "Depth Anything V2 Metric Indoor Small",
                "checkpoint": "checkpoints/depth_anything_v2_metric_indoor_small",
                "device": device_str,
                "notes": "Metric physical depth in meters with guided bilateral edge refinement.",
            },
            "depth_estimation_relative": {
                "status": "PRODUCTION_READY",
                "model": "Depth Anything V2 Small",
                "checkpoint": "checkpoints/depth_anything_v2_small",
                "device": device_str,
                "notes": "Normalized relative inverse depth [0, 1].",
            },
            "single_view_3d_reconstruction": {
                "status": "PRODUCTION_READY",
                "model": "RoomAI Unified Scene Fusion (A3 + YOLO11-Seg + Metric Depth + RANSAC)",
                "device": device_str,
                "notes": "Complete end-to-end 3D room reconstruction, physical grounding, and layout estimation.",
            },
            "multi_view_3d_reconstruction": {
                "status": "EXPERIMENTAL",
                "model": "SIFT + FLANN + Umeyama RANSAC + Voxel Fusion",
                "notes": "Algorithmic pipeline implemented; unvalidated against authentic multi-view ground truth scans. Falls back cleanly to single-view reference when consensus is unachieved.",
            },
            "vlm_qa": {
                "status": "UNAVAILABLE",
                "model": "Qwen2-VL / MiniCPM-V (Offline)",
                "notes": "Model checkpoint offline to preserve 8 GB VRAM budget for computer vision models. Returns structured degradation card without crashing.",
            },
            "room_classification": {
                "status": "UNAVAILABLE",
                "model": "None (Offline)",
                "notes": "Not deployed in active production pipeline.",
            },
            "generative_redesign": {
                "status": "UNAVAILABLE",
                "model": "SDXL / ControlNet (Offline)",
                "notes": "Model checkpoint offline to preserve 8 GB VRAM budget. Returns structured degradation response.",
            },
            "asset_export_ply": {
                "status": "PRODUCTION_READY",
                "format": "ASCII PLY with RGB point colors",
                "notes": "Validated against PLY format specification.",
            },
            "asset_export_glb": {
                "status": "PRODUCTION_READY",
                "format": "Binary glTF 2.0 (GLB) with point cloud and 3D bounding boxes",
                "notes": "Validated against glTF 2.0 specification.",
            },
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Returns comprehensive hardware telemetry, VRAM utilization, and model availability.

        Returns:
            Dict containing:
                - cuda_available: bool
                - device_name: str
                - total_vram_mb: float
                - allocated_vram_mb: float
                - reserved_vram_mb: float
                - vram_headroom_mb: float
                - checkpoints_status: Dict[str, bool] (checkpoint availability on disk)
                - registered_models: List[str]
        """
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        total_vram = 0.0
        allocated_vram = 0.0
        reserved_vram = 0.0

        if cuda_avail:
            props = torch.cuda.get_device_properties(0)
            total_vram = round(props.total_memory / (1024 * 1024), 1)
            allocated_vram = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            reserved_vram = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)

        headroom_vram = max(0.0, round(total_vram - allocated_vram, 1))

        # Checkpoint presence audit
        checkpoints_to_check = {
            "indoor_a3": "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt",
            "indoor_a2": "checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt",
            "exterior_b2": "checkpoints/exterior_detector_b2_yolo11s_best.pt",
            "parking_c2": "checkpoints/parking_detector_c2_yolo11s_pilot_best.pt",
            "yolo11_seg": "checkpoints/yolo11n-seg.pt",
            "depth_anything_v2_metric": "checkpoints/depth_anything_v2_metric_indoor_small",
            "depth_anything_v2_relative": "checkpoints/depth_anything_v2_small",
        }
        checkpoints_status = {}
        for key, rel_path in checkpoints_to_check.items():
            full_path = REPO_ROOT / rel_path
            checkpoints_status[key] = full_path.exists()

        registered_ids = list(self.registry._registry.keys())

        return {
            "cuda_available": cuda_avail,
            "device_name": device_name,
            "total_vram_mb": total_vram,
            "allocated_vram_mb": allocated_vram,
            "reserved_vram_mb": reserved_vram,
            "vram_headroom_mb": headroom_vram,
            "checkpoints_status": checkpoints_status,
            "registered_models": registered_ids,
        }

    def detect_objects(
        self,
        image: Image.Image,
        task_group: str = "Group A: Indoor Furniture & Household Objects",
        checkpoint_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        use_clahe: bool = False,
        mock_mode: bool = False,
    ) -> Dict[str, Any]:
        """Executes object detection using task-specific YOLO11 models."""
        t0 = time.perf_counter()
        valid, err_msg = validate_input_image(image)
        if not valid:
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "INVALID_INPUT_IMAGE",
                    "message": err_msg or "Invalid input image.",
                    "remediation": "Provide a valid RGB PIL Image instance.",
                },
                "warnings": [],
                "provenance": {"task_group": task_group},
                "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                "success": False,
                "response": None,
                "annotated_image": None,
                "timing_metrics": {},
                "error_message": err_msg,
            }

        try:
            cfg = dict(self.config.get("detection", {}))
            cfg["task_group"] = task_group
            if checkpoint_path:
                cfg["checkpoint_path"] = checkpoint_path
            if use_clahe:
                cfg["use_clahe"] = True

            # Match task group to correct default config section if not overridden
            if not checkpoint_path:
                if "exterior" in task_group.lower():
                    cfg = dict(self.config.get("exterior_detection", cfg))
                elif "parking" in task_group.lower():
                    cfg = dict(self.config.get("parking_detection", cfg))

            service = DetectionService(mock_mode=mock_mode, config=cfg)
            res = service.run(image, confidence_threshold=confidence_threshold)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

            if res["success"]:
                resp = res.get("response")
                return {
                    "status": "success",
                    "capability_status": "PRODUCTION_READY",
                    "data": {
                        "response": resp,
                        "annotated_image": res.get("annotated_image"),
                        "detections": resp.detections if resp else [],
                    },
                    "error": None,
                    "warnings": [],
                    "provenance": {
                        "model_name": resp.model_name if resp else "YOLO11s",
                        "task_group": task_group,
                        "is_simulated": getattr(resp, "is_simulated", mock_mode),
                    },
                    "timing": {
                        "inference_time_ms": res.get("timing_metrics", {}).get("inference_time_ms", 0.0),
                        "total_time_ms": total_ms,
                    },
                    "success": True,
                    "response": resp,
                    "annotated_image": res.get("annotated_image"),
                    "timing_metrics": res.get("timing_metrics", {}),
                    "error_message": None,
                }
            else:
                return {
                    "status": "error",
                    "capability_status": "PRODUCTION_READY",
                    "data": None,
                    "error": {
                        "code": "DETECTION_FAILED",
                        "message": res.get("error_message", "Unknown detection error."),
                        "remediation": "Verify confidence threshold and image contents.",
                    },
                    "warnings": [],
                    "provenance": {"task_group": task_group},
                    "timing": {"total_time_ms": total_ms},
                    "success": False,
                    "response": None,
                    "annotated_image": None,
                    "timing_metrics": res.get("timing_metrics", {}),
                    "error_message": res.get("error_message"),
                }
        except Exception as exc:
            logger.error(f"Detection failed in backend facade: {exc}", exc_info=True)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "DETECTION_EXCEPTION",
                    "message": str(exc),
                    "remediation": "Check logs and GPU availability.",
                },
                "warnings": [],
                "provenance": {"task_group": task_group},
                "timing": {"total_time_ms": total_ms},
                "success": False,
                "response": None,
                "annotated_image": None,
                "timing_metrics": {},
                "error_message": str(exc),
            }

    def segment_scene(
        self,
        image: Image.Image,
        mock_mode: bool = False,
    ) -> Dict[str, Any]:
        """Segments room semantic categories and furniture instances."""
        t0 = time.perf_counter()
        valid, err_msg = validate_input_image(image)
        if not valid:
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "INVALID_INPUT_IMAGE",
                    "message": err_msg or "Invalid input image.",
                    "remediation": "Provide a valid RGB PIL Image instance.",
                },
                "warnings": [],
                "provenance": {},
                "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                "success": False,
                "response": None,
                "overlay_image": None,
                "mask_array": None,
                "error_message": err_msg,
            }

        try:
            cfg = self.config.get("segmentation", {})
            service = SegmentationService(mock_mode=mock_mode, config=cfg)
            res = service.run(image)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

            if res["success"]:
                resp = res.get("response")
                return {
                    "status": "success",
                    "capability_status": "PRODUCTION_READY",
                    "data": {
                        "response": resp,
                        "overlay_image": res.get("overlay_image"),
                        "mask_array": res.get("mask_array"),
                    },
                    "error": None,
                    "warnings": [],
                    "provenance": {
                        "model_name": resp.model_name if resp else "YOLO11n-Seg",
                        "is_simulated": getattr(resp, "is_simulated", mock_mode),
                    },
                    "timing": {
                        "inference_time_ms": getattr(resp, "inference_time_ms", 0.0) if resp else 0.0,
                        "total_time_ms": total_ms,
                    },
                    "success": True,
                    "response": resp,
                    "overlay_image": res.get("overlay_image"),
                    "mask_array": res.get("mask_array"),
                    "error_message": None,
                }
            else:
                return {
                    "status": "error",
                    "capability_status": "PRODUCTION_READY",
                    "data": None,
                    "error": {
                        "code": "SEGMENTATION_FAILED",
                        "message": res.get("error_message", "Segmentation error."),
                        "remediation": "Verify segmentation checkpoint and input image.",
                    },
                    "warnings": [],
                    "provenance": {},
                    "timing": {"total_time_ms": total_ms},
                    "success": False,
                    "response": None,
                    "overlay_image": None,
                    "mask_array": None,
                    "error_message": res.get("error_message"),
                }
        except Exception as exc:
            logger.error(f"Segmentation failed in backend facade: {exc}", exc_info=True)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "SEGMENTATION_EXCEPTION",
                    "message": str(exc),
                    "remediation": "Check logs and GPU availability.",
                },
                "warnings": [],
                "provenance": {},
                "timing": {"total_time_ms": total_ms},
                "success": False,
                "response": None,
                "overlay_image": None,
                "mask_array": None,
                "error_message": str(exc),
            }

    def estimate_depth(
        self,
        image: Image.Image,
        mode: str = "metric",
        mock_mode: bool = False,
    ) -> Dict[str, Any]:
        """Estimates monocular depth map and generates 3D point cloud proxy."""
        t0 = time.perf_counter()
        valid, err_msg = validate_input_image(image)
        if not valid:
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "INVALID_INPUT_IMAGE",
                    "message": err_msg or "Invalid input image.",
                    "remediation": "Provide a valid RGB PIL Image instance.",
                },
                "warnings": [],
                "provenance": {"mode": mode},
                "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                "success": False,
                "response": None,
                "depth_map": None,
                "heatmap_image": None,
                "depth_unit": "unknown",
                "relative_depth_only": mode != "metric",
                "error_message": err_msg,
            }

        try:
            cfg_key = "scene3d_metric" if mode == "metric" else "scene3d"
            cfg = self.config.get(cfg_key, {})
            service = Scene3DService(mock_mode=mock_mode, config=cfg)
            res = service.run(image)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

            if res["success"]:
                resp = res.get("response")
                depth_unit = getattr(resp, "depth_unit", "meters" if mode == "metric" else "relative")
                rel_only = getattr(resp, "relative_depth_only", mode != "metric")

                return {
                    "status": "success",
                    "capability_status": "PRODUCTION_READY",
                    "data": {
                        "response": resp,
                        "depth_map": res.get("depth_map"),
                        "heatmap_image": res.get("heatmap_image"),
                        "point_cloud": res.get("point_cloud"),
                        "depth_unit": depth_unit,
                    },
                    "error": None,
                    "warnings": [],
                    "provenance": {
                        "model_name": getattr(resp, "model_name", "Depth Anything V2"),
                        "depth_mode": mode,
                        "is_simulated": getattr(resp, "is_simulated", mock_mode),
                    },
                    "timing": {
                        "inference_time_ms": res.get("timing", {}).get("inference_time_ms", 0.0),
                        "total_time_ms": total_ms,
                    },
                    "success": True,
                    "response": resp,
                    "depth_map": res.get("depth_map"),
                    "heatmap_image": res.get("heatmap_image"),
                    "point_cloud": res.get("point_cloud"),
                    "depth_unit": depth_unit,
                    "relative_depth_only": rel_only,
                    "error_message": None,
                }
            else:
                return {
                    "status": "error",
                    "capability_status": "PRODUCTION_READY",
                    "data": None,
                    "error": {
                        "code": "DEPTH_ESTIMATION_FAILED",
                        "message": res.get("error_message", "Depth estimation failed."),
                        "remediation": "Verify depth model checkpoint.",
                    },
                    "warnings": [],
                    "provenance": {"mode": mode},
                    "timing": {"total_time_ms": total_ms},
                    "success": False,
                    "response": None,
                    "depth_map": None,
                    "heatmap_image": None,
                    "depth_unit": "unknown",
                    "relative_depth_only": mode != "metric",
                    "error_message": res.get("error_message"),
                }
        except Exception as exc:
            logger.error(f"Depth estimation failed in backend facade: {exc}", exc_info=True)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "DEPTH_ESTIMATION_EXCEPTION",
                    "message": str(exc),
                    "remediation": "Check logs and GPU availability.",
                },
                "warnings": [],
                "provenance": {"mode": mode},
                "timing": {"total_time_ms": total_ms},
                "success": False,
                "response": None,
                "depth_map": None,
                "heatmap_image": None,
                "depth_unit": "unknown",
                "relative_depth_only": mode != "metric",
                "error_message": str(exc),
            }

    def reconstruct_room_3d(
        self,
        images: Union[Image.Image, List[Image.Image]],
        refine_depth: bool = True,
        mock_mode: bool = False,
    ) -> Dict[str, Any]:
        """Performs full 3D room scene reconstruction from single or multi-view photographs.

        If a single image is provided, executes single-view reconstruction using
        Model A3, YOLO11-Seg, Depth Anything V2 Metric, and RANSAC layout estimation (PRODUCTION_READY).
        If multiple images are provided (2-10), performs SIFT+FLANN feature registration,
        rigid 3D alignment, voxel downsampling, and cross-view geometry consolidation (EXPERIMENTAL).
        """
        t0 = time.perf_counter()
        img_list = [images] if isinstance(images, Image.Image) else list(images) if images else []

        if not img_list:
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "NO_IMAGES_PROVIDED",
                    "message": "No images provided for 3D reconstruction.",
                    "remediation": "Pass at least one PIL RGB Image.",
                },
                "warnings": ["No images provided"],
                "provenance": {"view_count": 0},
                "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                "success": False,
                "scene_graph": None,
                "point_cloud": None,
                "objects": [],
                "surfaces": [],
                "room_layout": None,
                "relationships": [],
                "telemetry": [],
                "per_view_provenance": [],
                "validation_warnings": ["No images provided"],
                "error_message": "No images provided for 3D reconstruction.",
            }

        # Validate each image
        for idx, img in enumerate(img_list):
            valid, err_msg = validate_input_image(img)
            if not valid:
                return {
                    "status": "error",
                    "capability_status": "PRODUCTION_READY" if len(img_list) == 1 else "EXPERIMENTAL",
                    "data": None,
                    "error": {
                        "code": "INVALID_INPUT_IMAGE",
                        "message": f"Image at index {idx} is invalid: {err_msg}",
                        "remediation": "Ensure all inputs are valid PIL Images within 8192px.",
                    },
                    "warnings": [],
                    "provenance": {"view_count": len(img_list), "invalid_index": idx},
                    "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                    "success": False,
                    "scene_graph": None,
                    "point_cloud": None,
                    "objects": [],
                    "surfaces": [],
                    "room_layout": None,
                    "relationships": [],
                    "telemetry": [],
                    "per_view_provenance": [],
                    "validation_warnings": [f"Image {idx} invalid: {err_msg}"],
                    "error_message": f"Image {idx} invalid: {err_msg}",
                }

        is_multiview = len(img_list) > 1
        cap_status = "EXPERIMENTAL" if is_multiview else "PRODUCTION_READY"

        try:
            mv_service = MultiViewFusionService(mock_mode=mock_mode, config=self.config)
            valid_imgs, val_warnings, _ = mv_service.validate_multiview_sequence(img_list)
            active_images = valid_imgs if valid_imgs else img_list

            scene = mv_service.fuse_multiview(active_images, refine_depth=refine_depth)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

            telemetry = getattr(scene, "multiview_telemetry", [])
            per_view_prov = getattr(scene, "multiview_per_view_provenance", [])

            warnings = list(val_warnings)
            if is_multiview:
                warnings.append(
                    "Multi-view 3D reconstruction is an EXPERIMENTAL capability and has not been validated against authentic ground truth scans."
                )

            return {
                "status": "success",
                "capability_status": cap_status,
                "data": {
                    "scene_graph": scene,
                    "point_cloud": scene.point_cloud,
                    "objects": scene.objects,
                    "surfaces": scene.surfaces,
                    "room_layout": scene.room_layout,
                    "relationships": scene.relationships,
                },
                "error": None,
                "warnings": warnings,
                "provenance": {
                    "model_name": scene.model_name,
                    "depth_mode": scene.depth_mode,
                    "views_processed": len(active_images),
                    "total_views_input": len(img_list),
                    "is_multiview": is_multiview,
                },
                "timing": {
                    "total_time_ms": total_ms,
                },
                "success": True,
                "scene_graph": scene,
                "point_cloud": scene.point_cloud,
                "objects": scene.objects,
                "surfaces": scene.surfaces,
                "room_layout": scene.room_layout,
                "relationships": scene.relationships,
                "telemetry": telemetry,
                "per_view_provenance": per_view_prov,
                "validation_warnings": warnings,
                "error_message": None,
            }
        except Exception as exc:
            logger.error(f"3D reconstruction failed in backend facade: {exc}", exc_info=True)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "capability_status": cap_status,
                "data": None,
                "error": {
                    "code": "RECONSTRUCTION_EXCEPTION",
                    "message": str(exc),
                    "remediation": "Verify GPU VRAM and input image validity.",
                },
                "warnings": [],
                "provenance": {"view_count": len(img_list)},
                "timing": {"total_time_ms": total_ms},
                "success": False,
                "scene_graph": None,
                "point_cloud": None,
                "objects": [],
                "surfaces": [],
                "room_layout": None,
                "relationships": [],
                "telemetry": [],
                "per_view_provenance": [],
                "validation_warnings": [],
                "error_message": f"3D reconstruction error: {str(exc)}",
            }

    def reconstruct_room_multiview(
        self,
        images: List[Image.Image],
        refine_depth: bool = True,
        mock_mode: bool = False,
    ) -> Dict[str, Any]:
        """Convenience method for multi-view 3D room reconstruction (EXPERIMENTAL).

        Notice: Multi-view reconstruction requires 2 to 10 photographs of the same physical room.
        This feature is classified as EXPERIMENTAL. If feature alignment fails, it falls back
        gracefully to single-view reference geometry.
        """
        return self.reconstruct_room_3d(images, refine_depth=refine_depth, mock_mode=mock_mode)

    def reconstruct_room_hunyuan(
        self,
        images: Union[Image.Image, List[Image.Image]],
        confidence_threshold: float = 0.5,
        max_points: int = 40000,
        mock_mode: bool = False,
        allow_composite_fallback: bool = False,
        strict_no_fallback: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Executes 3D room reconstruction using HunyuanWorld-Mirror 1.0 foundation model.

        Adheres strictly to the RoomAI 3D Reconstruction Integration Contract:
        - 518x518 canonical image preprocessing (patch-14 alignment)
        - Joint multi-view and single-view geometric inference
        - Rigid coordinate mapping: T_model_to_roomai via C = diag(1, -1, -1)
        - Metric scale calibration and surface normal extraction
        - Strict failure semantics by default (NO silent fallback unless explicitly requested via allow_composite_fallback=True)
        """
        if strict_no_fallback is not None:
            allow_composite_fallback = not strict_no_fallback

        t0 = time.perf_counter()
        img_list = [images] if isinstance(images, Image.Image) else list(images) if images else []

        if not img_list:
            return {
                "status": "error",
                "capability_status": "PRODUCTION_CANDIDATE",
                "data": None,
                "error": {
                    "code": "NO_IMAGES_PROVIDED",
                    "message": "No images provided for Hunyuan 3D reconstruction.",
                    "remediation": "Provide 1 to 10 PIL Images.",
                },
                "success": False,
                "scene_graph": None,
                "point_cloud": None,
                "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
            }

        # In mock mode or if GPU is constrained, provide high-fidelity synthetic proxy
        if mock_mode:
            w, h = img_list[0].size
            # Generate representative 3D room geometry proxy
            xs = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
            zs = np.linspace(1.0, 5.0, 100, dtype=np.float32)
            grid_x, grid_z = np.meshgrid(xs, zs)
            grid_y = np.full_like(grid_x, 1.4)  # Floor at y = 1.4m
            pts_xyz = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1)
            pts_rgb = np.full_like(pts_xyz, 0.7)

            scene = SceneGraph(
                image_id="mock_hunyuan_001",
                width=w,
                height=h,
                model_name="HunyuanWorld-Mirror 1.0 (Mock Mode)",
                depth_mode="metric",
                camera_intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 0.5, "cy": 0.5},
                point_cloud={"xyz": pts_xyz.tolist(), "rgb": pts_rgb.tolist()},
            )
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "success",
                "capability_status": "PRODUCTION_CANDIDATE",
                "data": {
                    "scene_graph": scene,
                    "point_cloud": scene.point_cloud,
                    "num_points": len(pts_xyz),
                },
                "error": None,
                "success": True,
                "scene_graph": scene,
                "point_cloud": scene.point_cloud,
                "timing": {"total_time_ms": total_ms},
            }

        try:
            from services.hunyuan_mirror_service import get_hunyuan_adapter
            adapter = get_hunyuan_adapter()
            if not adapter.is_ready:
                if not allow_composite_fallback:
                    return {
                        "status": "error",
                        "capability_status": "PRODUCTION_CANDIDATE",
                        "data": None,
                        "error": {
                            "code": "HUNYUAN_ADAPTER_NOT_READY",
                            "message": "HunyuanWorld-Mirror adapter is not ready or failed to load.",
                            "remediation": "Check GPU VRAM and verify checkpoint existence.",
                        },
                        "success": False,
                        "scene_graph": None,
                        "point_cloud": None,
                        "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                    }
                logger.warning("Hunyuan adapter not ready. Explicitly allowed fallback to standard 3D reconstruction.")
                return self.reconstruct_room_3d(images, refine_depth=True, mock_mode=False)

            result = adapter.reconstruct(
                img_list,
                confidence_threshold=confidence_threshold,
                max_points=max_points,
            )

            if result["status"] != "success":
                if not allow_composite_fallback:
                    return {
                        "status": "error",
                        "capability_status": "PRODUCTION_CANDIDATE",
                        "data": None,
                        "error": {
                            "code": "HUNYUAN_INFERENCE_ERROR",
                            "message": f"Hunyuan inference returned error: {result.get('message')}",
                            "remediation": "Check image dimensions, VRAM headroom, and input validity.",
                        },
                        "success": False,
                        "scene_graph": None,
                        "point_cloud": None,
                        "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                    }
                logger.warning(f"Hunyuan inference returned error: {result.get('message')}. Explicitly allowed fallback.")
                return self.reconstruct_room_3d(images, refine_depth=True, mock_mode=False)

            w, h = img_list[0].size
            pts_xyz = result["point_cloud_xyz"]
            pts_rgb = result["point_cloud_rgb"]
            pts_norm = result["point_cloud_normals"]

            pts_xyz_arr = np.asarray(pts_xyz, dtype=np.float32) if len(pts_xyz) > 0 else np.zeros((0, 3), dtype=np.float32)
            n_pts = len(pts_xyz_arr)
            z_vals = pts_xyz_arr[:, 2] if n_pts > 0 else np.array([0.0])
            d_min = float(np.min(z_vals)) if n_pts > 0 else 0.0
            d_max = float(np.max(z_vals)) if n_pts > 0 else 0.0

            scene = SceneGraph(
                image_id="hunyuan_recon_001",
                width=w,
                height=h,
                model_name="Tencent HunyuanWorld-Mirror 1.0",
                depth_mode="metric",
                camera_intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 0.5, "cy": 0.5},
                point_cloud={"xyz": pts_xyz_arr.tolist(), "rgb": (pts_rgb.tolist() if hasattr(pts_rgb, "tolist") else list(pts_rgb)) if pts_rgb is not None else []},
                n_valid_points=n_pts,
                depth_min_m=d_min,
                depth_max_m=d_max,
            )

            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "success",
                "capability_status": "PRODUCTION_CANDIDATE",
                "data": {
                    "scene_graph": scene,
                    "point_cloud": scene.point_cloud,
                    "num_points": len(pts_xyz),
                    "depth_maps": result.get("depth_maps"),
                },
                "error": None,
                "success": True,
                "scene_graph": scene,
                "point_cloud": scene.point_cloud,
                "timing": {
                    "inference_time_ms": result.get("timing_ms", {}).get("inference_time_ms", total_ms),
                    "total_time_ms": total_ms,
                },
            }
        except Exception as exc:
            logger.error(f"Hunyuan 3D reconstruction failed: {exc}", exc_info=True)
            if not allow_composite_fallback:
                return {
                    "status": "error",
                    "capability_status": "PRODUCTION_CANDIDATE",
                    "data": None,
                    "error": {
                        "code": "HUNYUAN_EXCEPTION",
                        "message": str(exc),
                        "remediation": "Check logs and GPU availability.",
                    },
                    "success": False,
                    "scene_graph": None,
                    "point_cloud": None,
                    "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                }
            logger.info("Falling back to standard 3D reconstruction pipeline via explicit allow_composite_fallback=True...")
            return self.reconstruct_room_3d(images, refine_depth=True, mock_mode=False)

    def export_3d_assets(
        self,
        scene_graph: SceneGraph,
        max_points: int = 40000,
    ) -> Dict[str, Any]:
        """Generates standard binary GLB (glTF 2.0) and ASCII PLY export assets."""
        t0 = time.perf_counter()
        if scene_graph is None:
            return {
                "status": "error",
                "capability_status": "PRODUCTION_READY",
                "data": None,
                "error": {
                    "code": "INVALID_SCENE_GRAPH",
                    "message": "SceneGraph is None.",
                    "remediation": "Pass a valid SceneGraph produced by reconstruct_room_3d.",
                },
                "warnings": [],
                "glb": b"",
                "ply": b"",
                "glb_valid": False,
                "ply_valid": False,
                "timing": {"total_time_ms": 0.0},
            }

        pts = []
        rgbs = None
        if scene_graph.point_cloud and scene_graph.point_cloud.get("xyz"):
            pts = scene_graph.point_cloud["xyz"][:max_points]
            rgbs = scene_graph.point_cloud.get("rgb")
            if rgbs:
                rgbs = rgbs[:max_points]

        ply_bytes = export_point_cloud_ply(pts, rgbs)
        glb_bytes = export_scene_glb(point_cloud={"xyz": pts}, objects=scene_graph.objects)

        ply_res = validate_ply_bytes(ply_bytes)
        glb_res = validate_glb_bytes(glb_bytes)
        ply_ok = ply_res["is_valid"]
        ply_err = ply_res["error"]
        glb_ok = glb_res["is_valid"]
        glb_err = glb_res["error"]

        warnings = []
        if not ply_ok:
            warnings.append(f"PLY validation warning: {ply_err}")
        if not glb_ok:
            warnings.append(f"GLB validation warning: {glb_err}")

        total_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "status": "success" if (ply_ok and glb_ok) else "warning",
            "capability_status": "PRODUCTION_READY",
            "data": {
                "glb_bytes_length": len(glb_bytes),
                "ply_bytes_length": len(ply_bytes),
                "points_exported": len(pts),
                "objects_exported": len(scene_graph.objects),
            },
            "error": None,
            "warnings": warnings,
            "glb": glb_bytes,
            "ply": ply_bytes,
            "glb_valid": glb_ok,
            "ply_valid": ply_ok,
            "timing": {"total_time_ms": total_ms},
        }

    def ask_vlm(
        self,
        image: Image.Image,
        question: str,
        mock_mode: bool = False,
    ) -> Dict[str, Any]:
        """Queries the Vision-Language Model interface regarding room features."""
        t0 = time.perf_counter()
        valid, err_msg = validate_input_image(image)
        if not valid:
            return {
                "status": "error",
                "capability_status": "UNAVAILABLE",
                "data": None,
                "error": {
                    "code": "INVALID_INPUT_IMAGE",
                    "message": err_msg or "Invalid input image.",
                    "remediation": "Provide a valid RGB PIL Image instance.",
                },
                "warnings": [],
                "provenance": {},
                "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                "success": False,
                "response": None,
                "is_implemented": True,
                "error_message": err_msg,
            }

        try:
            cfg = self.config.get("vision_language", {})
            service = VisionLanguageService(mock_mode=mock_mode, config=cfg)
            res = service.run(image, question)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

            if res["is_implemented"] and res["success"]:
                return {
                    "status": "success",
                    "capability_status": "UNAVAILABLE" if not mock_mode else "EXPERIMENTAL",
                    "data": {
                        "response": res.get("response"),
                        "answer": res.get("response").answer if res.get("response") else "",
                    },
                    "error": None,
                    "warnings": ["VLM weights are offline; output simulated in mock mode."] if mock_mode else [],
                    "provenance": {"is_simulated": mock_mode},
                    "timing": {"total_time_ms": total_ms},
                    "success": True,
                    "response": res.get("response"),
                    "is_implemented": True,
                    "error_message": None,
                }
            else:
                return {
                    "status": "unsupported",
                    "capability_status": "UNAVAILABLE",
                    "data": None,
                    "error": {
                        "code": "VLM_OFFLINE",
                        "message": res.get("error_message", "VLM model offline to preserve 8 GB VRAM budget."),
                        "remediation": "Model weights are not hosted locally to protect VRAM. Enable via external service.",
                    },
                    "warnings": ["Vision-Language subsystem is currently UNAVAILABLE."],
                    "provenance": {"is_implemented": res.get("is_implemented", False)},
                    "timing": {"total_time_ms": total_ms},
                    "success": False,
                    "response": None,
                    "is_implemented": res.get("is_implemented", False),
                    "error_message": res.get("error_message"),
                }
        except Exception as exc:
            logger.error(f"VLM query failed in backend facade: {exc}", exc_info=True)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "capability_status": "UNAVAILABLE",
                "data": None,
                "error": {
                    "code": "VLM_EXCEPTION",
                    "message": str(exc),
                    "remediation": "Check logs.",
                },
                "warnings": [],
                "provenance": {},
                "timing": {"total_time_ms": total_ms},
                "success": False,
                "response": None,
                "is_implemented": False,
                "error_message": str(exc),
            }

    def generate_redesign(
        self,
        image: Image.Image,
        target_style: str = "Scandinavian",
        custom_prompt: str = "",
        mock_mode: bool = False,
    ) -> Dict[str, Any]:
        """Generates an aesthetic interior redesign concept."""
        t0 = time.perf_counter()
        valid, err_msg = validate_input_image(image)
        if not valid:
            return {
                "status": "error",
                "capability_status": "UNAVAILABLE",
                "data": None,
                "error": {
                    "code": "INVALID_INPUT_IMAGE",
                    "message": err_msg or "Invalid input image.",
                    "remediation": "Provide a valid RGB PIL Image instance.",
                },
                "warnings": [],
                "provenance": {},
                "timing": {"total_time_ms": round((time.perf_counter() - t0) * 1000, 2)},
                "success": False,
                "response": None,
                "generated_image": None,
                "comparison_image": None,
                "is_implemented": True,
                "error_message": err_msg,
            }

        try:
            cfg = self.config.get("generation", {})
            service = GenerationService(mock_mode=mock_mode, config=cfg)
            req = DesignRequest(
                target_style=target_style,
                custom_prompt=custom_prompt,
            )
            res = service.run(image, req)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)

            if res["is_implemented"] and res["success"]:
                return {
                    "status": "success",
                    "capability_status": "UNAVAILABLE" if not mock_mode else "EXPERIMENTAL",
                    "data": {
                        "response": res.get("response"),
                        "generated_image": res.get("generated_image"),
                        "comparison_image": res.get("comparison_image"),
                    },
                    "error": None,
                    "warnings": ["Generative weights are offline; output simulated in mock mode."] if mock_mode else [],
                    "provenance": {"is_simulated": mock_mode},
                    "timing": {"total_time_ms": total_ms},
                    "success": True,
                    "response": res.get("response"),
                    "generated_image": res.get("generated_image"),
                    "comparison_image": res.get("comparison_image"),
                    "is_implemented": True,
                    "error_message": None,
                }
            else:
                return {
                    "status": "unsupported",
                    "capability_status": "UNAVAILABLE",
                    "data": None,
                    "error": {
                        "code": "GENERATION_OFFLINE",
                        "message": res.get("error_message", "Generative redesign offline to preserve 8 GB VRAM budget."),
                        "remediation": "Model weights are not hosted locally to protect VRAM.",
                    },
                    "warnings": ["Generative Redesign subsystem is currently UNAVAILABLE."],
                    "provenance": {"is_implemented": res.get("is_implemented", False)},
                    "timing": {"total_time_ms": total_ms},
                    "success": False,
                    "response": None,
                    "generated_image": None,
                    "comparison_image": None,
                    "is_implemented": res.get("is_implemented", False),
                    "error_message": res.get("error_message"),
                }
        except Exception as exc:
            logger.error(f"Generation failed in backend facade: {exc}", exc_info=True)
            total_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "status": "error",
                "capability_status": "UNAVAILABLE",
                "data": None,
                "error": {
                    "code": "GENERATION_EXCEPTION",
                    "message": str(exc),
                    "remediation": "Check logs.",
                },
                "warnings": [],
                "provenance": {},
                "timing": {"total_time_ms": total_ms},
                "success": False,
                "response": None,
                "generated_image": None,
                "comparison_image": None,
                "is_implemented": False,
                "error_message": str(exc),
            }


# Global singleton instance for convenient frontend imports
BACKEND = RoomAIBackend()
