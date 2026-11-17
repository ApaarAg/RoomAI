"""Object Detection model implementation using Ultralytics YOLO and deterministic mock."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image

from models.interfaces import BaseDetector, ModelNotImplementedError
from schemas.responses import BoundingBox, DetectionItem, DetectionResponse
from utils.device import get_optimal_device
from utils.logging_config import get_logger

logger = get_logger("DetectionModel")

# Module-level cache for YOLO models to avoid reloading
_YOLO_MODEL_CACHE: Dict[str, Any] = {}


def get_cached_yolo(checkpoint_path: str) -> Any:
    """Get a cached YOLO model instance, loading it if not cached.

    Args:
        checkpoint_path: Path to the YOLO checkpoint file.

    Returns:
        The YOLO model instance.
    """
    if checkpoint_path not in _YOLO_MODEL_CACHE:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model from {checkpoint_path} (new cache entry)")
        _YOLO_MODEL_CACHE[checkpoint_path] = YOLO(checkpoint_path)
    else:
        logger.debug(f"Using cached YOLO model for {checkpoint_path}")
    return _YOLO_MODEL_CACHE[checkpoint_path]


def clear_detector_cache() -> None:
    """Clear the YOLO model cache. Useful for testing."""
    global _YOLO_MODEL_CACHE
    _YOLO_MODEL_CACHE.clear()
    logger.debug("YOLO model cache cleared")


def validate_per_class_thresholds(config: Dict[str, Any]) -> None:
    """Validates per-class confidence thresholds configuration."""
    if config.get("use_per_class_thresholds", False):
        pct = config.get("per_class_thresholds")
        if not pct or not isinstance(pct, dict):
            raise ValueError("per_class_thresholds is missing or empty when use_per_class_thresholds is True")
        for cls_name, val in pct.items():
            if not isinstance(val, (int, float)):
                raise ValueError(f"Invalid threshold value for '{cls_name}': must be numeric")
            if not (0.0 < float(val) <= 1.0):
                raise ValueError(f"Threshold for '{cls_name}' must be in (0.0, 1.0]")


class CustomObjectDetector(BaseDetector):
    """Production Object Detector powered by Ultralytics YOLO (YOLO11s/YOLO11n)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        validate_per_class_thresholds(self.config)
        self.use_per_class_thresholds = bool(self.config.get("use_per_class_thresholds", False))
        self.per_class_thresholds = self.config.get("per_class_thresholds", {})
        self.checkpoint_path = self.config.get(
            "checkpoint_path", "checkpoints/deepfurniture_yolo11n_finetune_best.pt"
        )
        self.imgsz = int(self.config.get("imgsz", 640))
        self.iou_threshold = float(self.config.get("iou_threshold", 0.45))
        self.half = bool(self.config.get("half", True))

        # Device selection: use cuda if available, else cpu
        req_device = self.config.get("device", "cuda:0")
        optimal = get_optimal_device()
        self.device = req_device if optimal == "cuda" else "cpu"
        self.model = None

    def _resolve_checkpoint_path(self) -> Path:
        """Resolves relative checkpoint path against working directory and repository root."""
        p = Path(self.checkpoint_path)
        if p.exists():
            return p
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent
            candidate = repo_root / p
            if candidate.exists():
                return candidate
        return p

    def load_model(self) -> None:
        """Loads the Ultralytics YOLO model from checkpoint with caching and existence checks."""
        resolved = self._resolve_checkpoint_path()

        if not resolved.exists() and ("checkpoints" in str(self.checkpoint_path) or "/" in str(self.checkpoint_path) or "\\" in str(self.checkpoint_path)):
            raise FileNotFoundError(f"Detector checkpoint '{self.checkpoint_path}' does not exist on disk.")

        try:
            logger.info(f"Loading YOLO detector checkpoint '{resolved}' on {self.device}...")
            self.model = get_cached_yolo(str(resolved))
            logger.info(f"YOLO detector successfully loaded: {resolved}")
        except FileNotFoundError:
            self.model = None
            raise
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}", exc_info=True)
            self.model = None
            raise RuntimeError(f"Could not load detector checkpoint '{self.checkpoint_path}': {str(exc)}")

    def is_ready(self) -> bool:
        return self.model is not None

    def get_model_name(self) -> str:
        ckpt_name = Path(self.checkpoint_path).name
        return f"Ultralytics YOLO ({ckpt_name})"

    def predict(
        self, image: Image.Image, confidence_threshold: float = 0.35
    ) -> DetectionResponse:
        """Performs actual object detection on the provided PIL image.

        Args:
            image: Validated PIL Image in RGB mode.
            confidence_threshold: Minimum detection confidence score.

        Returns:
            DetectionResponse containing detected objects, bounding boxes, latency, and device.
        """
        if not self.is_ready():
            self.load_model()

        img_w, img_h = image.size
        t0 = time.perf_counter()

        # Run inference
        # If running on CPU, half precision is not supported by PyTorch on Windows CPU, so fallback to FP32 on CPU
        use_half = self.half if "cuda" in self.device.lower() else False

        try:
            results = self.model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                half=use_half,
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"Inference error in YOLO detector: {exc}", exc_info=True)
            raise RuntimeError(f"Detection inference failed on {self.device}: {str(exc)}")

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = r.names[cls_id]
                conf = float(box.conf[0].item())

                # Per-class threshold filtering
                thresh = (
                    self.per_class_thresholds.get(label, confidence_threshold)
                    if self.use_per_class_thresholds
                    else confidence_threshold
                )
                if conf < thresh:
                    continue

                # Normalized coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x_min = max(0.0, min(1.0, float(xyxy[0]) / img_w))
                y_min = max(0.0, min(1.0, float(xyxy[1]) / img_h))
                x_max = max(0.0, min(1.0, float(xyxy[2]) / img_w))
                y_max = max(0.0, min(1.0, float(xyxy[3]) / img_h))

                # Ensure x_max >= x_min and y_max >= y_min for strict Pydantic bounds
                if x_max < x_min:
                    x_max = x_min
                if y_max < y_min:
                    y_max = y_min

                bbox = BoundingBox(
                    x_min=round(x_min, 4),
                    y_min=round(y_min, 4),
                    x_max=round(x_max, 4),
                    y_max=round(y_max, 4),
                )

                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        label=label,
                        confidence=round(conf, 4),
                        box=bbox,
                    )
                )

        return DetectionResponse(
            detections=detections,
            total_objects=len(detections),
            is_simulated=False,
            model_name=self.get_model_name(),
            checkpoint=Path(self.checkpoint_path).name,
            inference_time_ms=round(inference_time_ms, 2),
            device=self.device,
        )


class MockObjectDetector(BaseDetector):
    """Deterministic mock object detector for testing the UI and application flow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        validate_per_class_thresholds(self.config)
        self.use_per_class_thresholds = bool(self.config.get("use_per_class_thresholds", False))
        self.per_class_thresholds = self.config.get("per_class_thresholds", {})
        self._ready = True

    def load_model(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "Deterministic Mock Detector (Simulated)"

    def predict(
        self, image: Image.Image, confidence_threshold: float = 0.40
    ) -> DetectionResponse:
        # Deterministic realistic furniture detections for UI testing
        simulated_items = [
            DetectionItem(
                class_id=1,
                label="couch",
                confidence=0.92,
                box=BoundingBox(x_min=0.15, y_min=0.45, x_max=0.75, y_max=0.85),
            ),
            DetectionItem(
                class_id=2,
                label="dining table",
                confidence=0.86,
                box=BoundingBox(x_min=0.30, y_min=0.68, x_max=0.60, y_max=0.88),
            ),
            DetectionItem(
                class_id=3,
                label="chair",
                confidence=0.81,
                box=BoundingBox(x_min=0.80, y_min=0.25, x_max=0.95, y_max=0.80),
            ),
            DetectionItem(
                class_id=4,
                label="potted plant",
                confidence=0.77,
                box=BoundingBox(x_min=0.05, y_min=0.40, x_max=0.20, y_max=0.75),
            ),
            DetectionItem(
                class_id=5,
                label="tv",
                confidence=0.89,
                box=BoundingBox(x_min=0.35, y_min=0.12, x_max=0.58, y_max=0.35),
            ),
        ]

        filtered = []
        for item in simulated_items:
            thresh = (
                self.per_class_thresholds.get(item.label, confidence_threshold)
                if self.use_per_class_thresholds
                else confidence_threshold
            )
            if item.confidence >= thresh:
                filtered.append(item)

        return DetectionResponse(
            detections=filtered,
            total_objects=len(filtered),
            is_simulated=True,
            model_name=self.get_model_name(),
            checkpoint="mock_simulated.pt",
            inference_time_ms=1.5,
            device="cpu",
        )
