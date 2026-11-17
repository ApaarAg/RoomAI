"""Image Segmentation model implementations and mock representations."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from models.interfaces import BaseSegmenter, ModelNotImplementedError
from schemas.responses import SegmentationMask, SegmentationResponse
from utils.logging_config import get_logger

logger = get_logger("SegmentationModel")

# High-contrast hex palette strictly aligned with image_service PALETTE
HEX_PALETTE = [
    "#3B82F6",  # Royal Blue
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber
    "#EF4444",  # Crimson Red
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#F97316",  # Orange
]


class CustomRoomSegmenter(BaseSegmenter):
    """Production Semantic & Instance Segmentation powered by Ultralytics YOLO-Seg."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        self._is_configured = bool(config and ("checkpoint_path" in config or config.get("enabled", False)))
        self.checkpoint_path = self.config.get("checkpoint_path", "checkpoints/yolo11n-seg.pt")
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.25))
        self.segmentation_mode = str(self.config.get("segmentation_mode", self.config.get("mode", "instance")))
        self.model = None

    def load_model(self) -> None:
        """Loads the Ultralytics segmentation model from checkpoint."""
        if not self._is_configured:
            raise ModelNotImplementedError("CustomRoomSegmenter", __file__)

        ckpt_p = Path(self.checkpoint_path)
        if not ckpt_p.exists():
            # Check relative to root
            repo_root = Path(__file__).resolve().parent.parent
            ckpt_p = repo_root / self.checkpoint_path
            if not ckpt_p.exists():
                raise FileNotFoundError(f"Segmentation checkpoint not found: {self.checkpoint_path}")

        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO-Seg checkpoint '{ckpt_p}'...")
            self.model = YOLO(str(ckpt_p))
            logger.info(f"YOLO-Seg successfully loaded: {ckpt_p}")
        except Exception as exc:
            logger.error(f"Failed to load segmentation model: {exc}", exc_info=True)
            self.model = None
            raise RuntimeError(f"Could not load segmenter checkpoint '{self.checkpoint_path}': {str(exc)}")

    def is_ready(self) -> bool:
        return self._is_configured and self.model is not None

    def get_model_name(self) -> str:
        ckpt_name = Path(self.checkpoint_path).name
        return f"Ultralytics YOLO-Seg ({ckpt_name})"

    def predict(
        self, image: Image.Image
    ) -> Tuple[SegmentationResponse, Optional[np.ndarray]]:
        """Performs segmentation inference on the input image."""
        if not self._is_configured:
            raise ModelNotImplementedError("CustomRoomSegmenter", __file__)

        if not self.is_ready():
            self.load_model()

        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        mask_array = np.zeros((height, width), dtype=np.uint8)
        masks_meta = []
        categories_found = []

        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                imgsz=640,
                verbose=False
            )[0]
        except Exception as exc:
            logger.error(f"Inference error in YOLO-Seg: {exc}", exc_info=True)
            raise RuntimeError(f"Segmentation inference failed: {str(exc)}")

        if results.masks is not None and len(results.masks.data) > 0:
            for idx in range(len(results.masks.data)):
                m_data = results.masks.data[idx].cpu().numpy().astype(bool)
                if m_data.shape != (height, width):
                    m_data = cv2.resize(
                        m_data.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                mask_id = idx + 1
                mask_array[m_data] = mask_id

                cov_pct = (float(m_data.sum()) / float(width * height)) * 100.0
                cls_id = int(results.boxes.cls[idx].item()) if results.boxes is not None else 0
                cat_name = results.names.get(cls_id, f"object_{cls_id}")

                color_hex = HEX_PALETTE[idx % len(HEX_PALETTE)]
                conf_val = float(results.boxes.conf[idx].item()) if results.boxes is not None and results.boxes.conf is not None else None
                masks_meta.append(
                    SegmentationMask(
                        category=cat_name,
                        coverage_percentage=round(cov_pct, 2),
                        mask_shape=(height, width),
                        color_hex=color_hex,
                        instance_id=mask_id,
                        confidence=conf_val,
                    )
                )
                if cat_name not in categories_found:
                    categories_found.append(cat_name)

        response = SegmentationResponse(
            masks=masks_meta,
            categories_found=categories_found,
            is_simulated=False,
            model_name=self.get_model_name(),
        )
        return response, mask_array


class MockRoomSegmenter(BaseSegmenter):
    """Deterministic mock segmentation model for testing UI and visualization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.segmentation_mode = str(self.config.get("segmentation_mode", self.config.get("mode", "semantic")))
        self._ready = True

    def load_model(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "Deterministic Mock Segmenter (Simulated)"

    def predict(
        self, image: Image.Image
    ) -> Tuple[SegmentationResponse, Optional[np.ndarray]]:
        width, height = image.size
        # Generate synthetic regions: 0=wall, 1=floor, 2=furniture
        mask = np.zeros((height, width), dtype=np.uint8)

        # Floor (bottom 35% of room)
        floor_start = int(height * 0.65)
        mask[floor_start:, :] = 1

        # Central furniture area
        furn_y1, furn_y2 = int(height * 0.50), int(height * 0.85)
        furn_x1, furn_x2 = int(width * 0.20), int(width * 0.80)
        mask[furn_y1:furn_y2, furn_x1:furn_x2] = 2

        masks_meta = [
            SegmentationMask(
                category="wall",
                coverage_percentage=48.5,
                mask_shape=(height, width),
                color_hex="#3B82F6",
            ),
            SegmentationMask(
                category="floor",
                coverage_percentage=27.5,
                mask_shape=(height, width),
                color_hex="#10B981",
            ),
            SegmentationMask(
                category="furniture",
                coverage_percentage=24.0,
                mask_shape=(height, width),
                color_hex="#F59E0B",
            ),
        ]

        response = SegmentationResponse(
            masks=masks_meta,
            categories_found=["wall", "floor", "furniture"],
            is_simulated=True,
            model_name=self.get_model_name(),
        )
        return response, mask
