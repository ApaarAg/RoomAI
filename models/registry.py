"""Model Registry and Dynamic VRAM Routing for RoomAI.

Coordinates task-specific pretrained computer vision architectures,
enforcing GPU memory headroom (8 GB VRAM budget) and controlled lazy loading.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from utils.logging_config import get_logger

logger = get_logger("ModelRegistry")
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelSpec:
    """Specification and resource metadata for a registered model architecture."""

    task_id: str
    model_id: str
    display_name: str
    architecture: str
    checkpoint_rel_path: str
    classes: List[str]
    input_resolution: Tuple[int, int]
    expected_vram_mb: float
    license_type: str
    output_schema: str
    is_primary: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_path(self) -> Path:
        return REPO_ROOT / self.checkpoint_rel_path

    def exists(self) -> bool:
        return self.checkpoint_path.exists()


class ModelRegistry:
    """Central registry and lifecycle manager for all RoomAI computer vision models."""

    def __init__(self, max_vram_mb: float = 4500.0):
        self.max_vram_mb = max_vram_mb
        self._registry: Dict[str, ModelSpec] = {}
        self._active_models: Dict[str, Any] = {}
        self._last_accessed: Dict[str, float] = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Populates the canonical model registry."""
        # 1. Group A: Indoor Furniture & Household Objects
        self.register(
            ModelSpec(
                task_id="indoor_detection",
                model_id="indoor_a3_800px",
                display_name="Model A3 (Higher-Resolution YOLO11s 800px)",
                architecture="YOLO11s",
                checkpoint_rel_path="checkpoints/indoor_detector_a3_yolo11s_800px_best.pt",
                classes=[
                    "cabinet#shelf", "table", "chair#stool", "lamp", "door", "bed",
                    "sofa", "plant", "decoration", "curtain", "home-appliance", "window"
                ],
                input_resolution=(800, 800),
                expected_vram_mb=850.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=True,
                metadata={"test_map50": 0.7760, "test_map50_95": 0.6509, "precision": 0.8301, "latency_ms": 13.57},
            )
        )
        self.register(
            ModelSpec(
                task_id="indoor_detection",
                model_id="indoor_a2_finetuned",
                display_name="Fine-Tuned Model A2 (YOLO11s 12-Class Rollback)",
                architecture="YOLO11s",
                checkpoint_rel_path="checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt",
                classes=[
                    "cabinet#shelf", "table", "chair#stool", "lamp", "door", "bed",
                    "sofa", "plant", "decoration", "curtain", "home-appliance", "window"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=850.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=False,
                metadata={"test_map50": 0.7464, "precision": 0.7549, "latency_ms": 13.05},
            )
        )
        self.register(
            ModelSpec(
                task_id="indoor_detection",
                model_id="indoor_a1_pilot",
                display_name="Model A1 (YOLO11n 12-Class Pilot)",
                architecture="YOLO11n",
                checkpoint_rel_path="checkpoints/indoor_detector_a1_yolo11n_pilot_best.pt",
                classes=[
                    "cabinet#shelf", "table", "chair#stool", "lamp", "door", "bed",
                    "sofa", "plant", "decoration", "curtain", "home-appliance", "window"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=550.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=False,
                metadata={"test_map50": 0.6642, "precision": 0.7102, "latency_ms": 23.46},
            )
        )
        self.register(
            ModelSpec(
                task_id="indoor_detection",
                model_id="indoor_baseline_rollback",
                display_name="Rollback Baseline (DeepFurniture YOLO11n 11-Class)",
                architecture="YOLO11n",
                checkpoint_rel_path="checkpoints/deepfurniture_yolo11n_finetune_best.pt",
                classes=[
                    "cabinet#shelf", "table", "chair#stool", "lamp", "door", "bed",
                    "sofa", "plant", "decoration", "curtain", "home-appliance"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=550.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=False,
                metadata={"test_map50": 0.7311, "precision": 0.7592, "classes_count": 11},
            )
        )

        # 2. Group B: Exterior & Building Architecture
        self.register(
            ModelSpec(
                task_id="exterior_detection",
                model_id="exterior_b2_finetuned",
                display_name="Fine-Tuned Model B2 (YOLO11s Exterior)",
                architecture="YOLO11s",
                checkpoint_rel_path="checkpoints/exterior_detector_b2_yolo11s_best.pt",
                classes=[
                    "building", "facade", "window", "door", "roof_cornice",
                    "balcony", "porch", "pillar_column", "shutter_blind", "molding_deco"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=850.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=True,
                metadata={"test_f1": 0.6018, "test_map50": 0.4487, "latency_ms": 29.61},
            )
        )
        self.register(
            ModelSpec(
                task_id="exterior_detection",
                model_id="exterior_b1_pilot",
                display_name="Model B1 (YOLO11n Exterior Pilot)",
                architecture="YOLO11n",
                checkpoint_rel_path="checkpoints/exterior_detector_b1_yolo11n_pilot_best.pt",
                classes=[
                    "building", "facade", "window", "door", "roof_cornice",
                    "balcony", "porch", "pillar_column", "shutter_blind", "molding_deco"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=550.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=False,
                metadata={"test_f1": 0.008, "latency_ms": 51.17},
            )
        )
        self.register(
            ModelSpec(
                task_id="exterior_detection",
                model_id="exterior_b5_transfer",
                display_name="Model B5 (Transfer Learning YOLO11s)",
                architecture="YOLO11s",
                checkpoint_rel_path="checkpoints/exterior_detector_b5_transfer_best.pt",
                classes=[
                    "building", "facade", "window", "door", "roof_cornice",
                    "balcony", "porch", "pillar_column", "shutter_blind", "molding_deco"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=850.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=False,
                metadata={"val_map50": 0.4869, "val_macro_f1": 0.4960, "status": "experimental_candidate"},
            )
        )
        self.register(
            ModelSpec(
                task_id="exterior_detection",
                model_id="exterior_b6_expanded",
                display_name="Model B6 (Targeted Data Expansion YOLO11s)",
                architecture="YOLO11s",
                checkpoint_rel_path="checkpoints/exterior_detector_b6_expanded_best.pt",
                classes=[
                    "building", "facade", "window", "door", "roof_cornice",
                    "balcony", "porch", "pillar_column", "shutter_blind", "molding_deco"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=850.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=False,
                metadata={"val_map50": 0.4920, "val_macro_f1": 0.5069, "porch_ap50": 0.1437, "status": "experimental_candidate"},
            )
        )

        # 3. Group C: Parking & Outdoor Property
        self.register(
            ModelSpec(
                task_id="parking_detection",
                model_id="parking_c2_pilot",
                display_name="Model C2 (YOLO11s Parking/Outdoor)",
                architecture="YOLO11s",
                checkpoint_rel_path="checkpoints/parking_detector_c2_yolo11s_pilot_best.pt",
                classes=[
                    "car_vehicle", "motorcycle", "bicycle", "outdoor_seating",
                    "outdoor_table", "plant_bush", "tree", "outdoor_fixture"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=850.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=True,
                metadata={"test_f1": 0.4029, "latency_ms": 18.33, "fps": 54.6},
            )
        )
        self.register(
            ModelSpec(
                task_id="parking_detection",
                model_id="parking_c1_pilot",
                display_name="Model C1 (YOLO11n Parking/Outdoor)",
                architecture="YOLO11n",
                checkpoint_rel_path="checkpoints/parking_detector_c1_yolo11n_pilot_best.pt",
                classes=[
                    "car_vehicle", "motorcycle", "bicycle", "outdoor_seating",
                    "outdoor_table", "plant_bush", "tree", "outdoor_fixture"
                ],
                input_resolution=(640, 640),
                expected_vram_mb=550.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="DetectionResponse",
                is_primary=False,
                metadata={"test_f1": 0.3083, "latency_ms": 21.94, "fps": 45.6},
            )
        )

        # 4. Semantic / Instance Segmentation
        self.register(
            ModelSpec(
                task_id="segmentation",
                model_id="yolo11_seg_instance",
                display_name="Ultralytics YOLO11-Seg (Pretrained Segmenter)",
                architecture="YOLO11n-Seg",
                checkpoint_rel_path="checkpoints/yolo11n-seg.pt",
                classes=["wall", "floor", "furniture", "window", "door", "table", "chair"],
                input_resolution=(640, 640),
                expected_vram_mb=650.0,
                license_type="AGPL-3.0 / Enterprise",
                output_schema="SegmentationResponse",
                is_primary=True,
                metadata={"real_time_fps": 45.0, "latency_ms": 22.0},
            )
        )

        # 5. Monocular Depth & 3D Understanding (Relative & Metric)
        self.register(
            ModelSpec(
                task_id="scene3d_depth_metric",
                model_id="depth_anything_v2_metric_indoor_small",
                display_name="Depth Anything V2 Metric Indoor Small (Metric Geometry)",
                architecture="DINOv2 ViT-S + DPT Head (Metric Fine-Tuned)",
                checkpoint_rel_path="checkpoints/depth_anything_v2_metric_indoor_small",
                classes=["metric_depth_meters"],
                input_resolution=(518, 518),
                expected_vram_mb=920.0,
                license_type="Apache-2.0",
                output_schema="Scene3DResponse",
                is_primary=True,
                metadata={"latency_ms": 22.0, "params": "24.8M", "depth_unit": "meters", "depth_range": "0.1m - 20m"},
            )
        )
        self.register(
            ModelSpec(
                task_id="scene3d_depth",
                model_id="depth_anything_v2_small",
                display_name="Depth Anything V2 Small (Monocular Geometry)",
                architecture="DINOv2 ViT-S + DPT Head",
                checkpoint_rel_path="checkpoints/depth_anything_v2_small",
                classes=["depth_map_continuous"],
                input_resolution=(518, 518),
                expected_vram_mb=920.0,
                license_type="Apache-2.0",
                output_schema="Scene3DResponse",
                is_primary=True,
                metadata={"latency_ms": 21.0, "params": "24.8M", "depth_unit": "relative"},
            )
        )

        # 6. Vision-Language Q&A
        self.register(
            ModelSpec(
                task_id="vision_language",
                model_id="moondream2_vlm",
                display_name="Moondream2 Vision-Language Model (1.86B)",
                architecture="SigLIP + Phi-1.5 / Moondream Decoder",
                checkpoint_rel_path="checkpoints/moondream2_weights",
                classes=["natural_language_qa"],
                input_resolution=(378, 378),
                expected_vram_mb=2200.0,
                license_type="Apache-2.0",
                output_schema="VLMAnswerResponse",
                is_primary=True,
                metadata={"quantization": "4-bit/FP16", "latency_ms": 650.0},
            )
        )

    def register(self, spec: ModelSpec) -> None:
        """Registers a model specification."""
        self._registry[spec.model_id] = spec

    def get_spec(self, model_id: str) -> Optional[ModelSpec]:
        return self._registry.get(model_id)

    def get_model_spec(self, model_id: str) -> Optional[ModelSpec]:
        """Alias for get_spec for backward compatibility."""
        return self.get_spec(model_id)

    def get_primary_for_task(self, task_id: str) -> Optional[ModelSpec]:
        """Returns the primary recommended model for a given task."""
        for spec in self._registry.values():
            if spec.task_id == task_id and spec.is_primary:
                return spec
        return None

    def list_by_task(self, task_id: str) -> List[ModelSpec]:
        """Lists all registered models supporting a specific task."""
        return [spec for spec in self._registry.values() if spec.task_id == task_id]

    def check_vram_headroom(self) -> Dict[str, float]:
        """Returns current GPU VRAM utilization and available capacity in MB."""
        info = {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_budget_mb": self.max_vram_mb}
        if torch.cuda.is_available():
            info["allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            info["reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
        return info

    def enforce_vram_budget(self) -> None:
        """Evicts least recently used models if allocated memory exceeds the target budget."""
        if not torch.cuda.is_available():
            return

        allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        if allocated_mb > self.max_vram_mb:
            logger.warning(
                f"VRAM allocation ({allocated_mb:.1f} MB) exceeds budget ({self.max_vram_mb:.1f} MB). Evicting idle models..."
            )
            # Sort active models by last accessed timestamp
            sorted_keys = sorted(self._last_accessed.keys(), key=lambda k: self._last_accessed[k])
            for k in sorted_keys[:-1]:  # Keep the most recently used model
                if k in self._active_models:
                    model = self._active_models.pop(k)
                    try:
                        if hasattr(model, "to"):
                            model.to("cpu")
                    except Exception:
                        pass
                    del model
            torch.cuda.empty_cache()
            new_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
            logger.info(f"VRAM budget enforced. New allocation: {new_alloc:.1f} MB")


# Global singleton registry instance
GLOBAL_MODEL_REGISTRY = ModelRegistry()
