"""RoomAI Phase 15: Specialist Ensemble Service for Exterior Architecture Detection.

Coordinates multi-model detection between general detector (B6/B2) and
domain-specific specialist detectors (e.g., Porch Specialist).

Architectures Implemented:
1. 'general_only': Baseline execution of general detector (B6 or B2).
2. 'parallel_ensemble': Both general and specialist run on full image;
   calibrated WBF/IoU fusion merges porch predictions with consensus bonus.
3. 'cascaded': General detector runs first. If entrance cues or low-confidence
   porch candidates (0.10 <= conf < 0.25) are found, the specialist is routed.
4. 'high_res_crop': General detector localizes potential entrance regions;
   specialist evaluates high-resolution crops and maps coordinates back to full image.

Safeguards:
- Canonical schema: schemas.responses.DetectionResponse, DetectionItem, BoundingBox.
- Provenance tracking on every box (metadata: 'model_provenance').
- Cross-class conflict resolution (porch vs facade/building containment).
- Deterministic Mock Mode.
- Explicit failure on missing checkpoints.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch

from models.detection_model import CustomObjectDetector, MockObjectDetector
from models.interfaces import BaseDetector
from schemas.responses import BoundingBox, DetectionItem, DetectionResponse
from utils.logging_config import get_logger

logger = get_logger("SpecialistEnsembleService")

CANONICAL_CLASSES = {
    0: "building", 1: "facade", 2: "window", 3: "door", 4: "roof_cornice",
    5: "balcony", 6: "porch", 7: "pillar_column", 8: "shutter_blind", 9: "molding_deco"
}


def box_iou_xyxy(box1: List[float], box2: List[float]) -> float:
    """Computes IoU for boxes formatted as [x_min, y_min, x_max, y_max]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


class SpecialistEnsembleDetector(BaseDetector):
    """Orchestrates general and specialist exterior object detectors."""

    def __init__(
        self,
        general_model: BaseDetector,
        specialist_model: Optional[BaseDetector] = None,
        mode: str = "parallel_ensemble",
        general_conf_thresh: float = 0.25,
        specialist_conf_thresh: float = 0.20,
        fusion_iou_thresh: float = 0.40,
        weights: Optional[List[float]] = None,
        consensus_bonus: float = 0.05,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.general_model = general_model
        self.specialist_model = specialist_model
        self.mode = mode.lower()
        self.general_conf_thresh = general_conf_thresh
        self.specialist_conf_thresh = specialist_conf_thresh
        self.fusion_iou_thresh = fusion_iou_thresh
        self.weights = weights or [0.8, 1.2]  # [general_weight, specialist_weight]
        self.consensus_bonus = consensus_bonus
        self.last_timing_breakdown: Dict[str, float] = {}

    def load_model(self) -> None:
        """Loads underlying models if needed."""
        if self.general_model and hasattr(self.general_model, "load_model"):
            self.general_model.load_model()
        if self.specialist_model and hasattr(self.specialist_model, "load_model"):
            self.specialist_model.load_model()

    def is_ready(self) -> bool:
        """Returns True if the underlying models are loaded."""
        gen_ready = self.general_model.is_ready() if hasattr(self.general_model, "is_ready") else True
        spec_ready = self.specialist_model.is_ready() if (self.specialist_model and hasattr(self.specialist_model, "is_ready")) else True
        return gen_ready and spec_ready

    def get_model_name(self) -> str:

        spec_name = self.specialist_model.get_model_name() if self.specialist_model else "None"
        return f"SpecialistEnsemble[{self.mode.upper()}]({self.general_model.get_model_name()} + {spec_name})"

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None,
    ) -> DetectionResponse:
        """Executes detection pipeline according to configured specialist mode."""
        if confidence_threshold is None:
            confidence_threshold = self.general_conf_thresh

        t0 = time.perf_counter()

        # Handle Mock Mode
        if getattr(self.general_model, "is_mock", False) or isinstance(self.general_model, MockObjectDetector):
            mock_resp = self.general_model.predict(image, confidence_threshold=confidence_threshold)
            return mock_resp

        if self.mode == "general_only" or self.specialist_model is None:
            resp = self.general_model.predict(image, confidence_threshold=confidence_threshold)
            for item in resp.detections:
                item.model_provenance = "general_model"
            self.last_timing_breakdown = {
                "general_ms": resp.inference_time_ms,
                "specialist_ms": 0.0,
                "fusion_ms": 0.0,
                "total_ms": resp.inference_time_ms,
            }
            return resp

        if self.mode == "parallel_ensemble":
            return self._predict_parallel(image, confidence_threshold, t0)
        elif self.mode == "cascaded":
            return self._predict_cascaded(image, confidence_threshold, t0)
        elif self.mode == "high_res_crop":
            return self._predict_high_res_crop(image, confidence_threshold, t0)
        else:
            raise ValueError(f"Unknown specialist ensemble mode: {self.mode}")

    def _predict_parallel(
        self,
        image: Image.Image,
        conf_thresh: float,
        t_start: float,
    ) -> DetectionResponse:
        # 1. Run General Detector at low conf to catch all potential classes
        t_gen_0 = time.perf_counter()
        gen_resp = self.general_model.predict(image, confidence_threshold=min(conf_thresh, 0.15))
        gen_ms = (time.perf_counter() - t_gen_0) * 1000.0

        # 2. Run Specialist Detector for porch
        t_spec_0 = time.perf_counter()
        spec_resp = self.specialist_model.predict(image, confidence_threshold=self.specialist_conf_thresh)
        spec_ms = (time.perf_counter() - t_spec_0) * 1000.0

        # 3. Fuse Predictions
        t_fuse_0 = time.perf_counter()
        fused_detections = self._fuse_predictions(gen_resp.detections, spec_resp.detections, conf_thresh)
        fuse_ms = (time.perf_counter() - t_fuse_0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_timing_breakdown = {
            "general_ms": round(gen_ms, 2),
            "specialist_ms": round(spec_ms, 2),
            "fusion_ms": round(fuse_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        return DetectionResponse(
            detections=fused_detections,
            total_objects=len(fused_detections),
            is_simulated=False,
            model_name=self.get_model_name(),
            checkpoint=f"{getattr(self.general_model, 'checkpoint_path', 'gen')} + {getattr(self.specialist_model, 'checkpoint_path', 'spec')}",
            inference_time_ms=round(total_ms, 2),
            device=getattr(self.general_model, "device", "cuda:0"),
        )

    def _predict_cascaded(
        self,
        image: Image.Image,
        conf_thresh: float,
        t_start: float,
    ) -> DetectionResponse:
        # 1. General model inference first
        t_gen_0 = time.perf_counter()
        gen_resp = self.general_model.predict(image, confidence_threshold=min(conf_thresh, 0.10))
        gen_ms = (time.perf_counter() - t_gen_0) * 1000.0

        # 2. Check cascade trigger rule:
        # Trigger specialist IF:
        # a) low-confidence porch (0.10 <= conf < 0.25), OR
        # b) presence of entrance door (class_id=3) or ground-floor facade (class_id=1, y_max > 0.5) without confident porch
        has_confident_porch = any(d.class_id == 6 and d.confidence >= conf_thresh for d in gen_resp.detections)
        has_weak_porch = any(d.class_id == 6 and (0.10 <= d.confidence < conf_thresh) for d in gen_resp.detections)
        has_entrance_cue = any(d.class_id in [1, 3] and d.box.y_max >= 0.50 for d in gen_resp.detections)

        invoke_specialist = (not has_confident_porch) and (has_weak_porch or has_entrance_cue)

        spec_detections = []
        spec_ms = 0.0
        if invoke_specialist and self.specialist_model is not None:
            t_spec_0 = time.perf_counter()
            spec_resp = self.specialist_model.predict(image, confidence_threshold=self.specialist_conf_thresh)
            spec_detections = spec_resp.detections
            spec_ms = (time.perf_counter() - t_spec_0) * 1000.0

        # 3. Fuse
        t_fuse_0 = time.perf_counter()
        fused_detections = self._fuse_predictions(gen_resp.detections, spec_detections, conf_thresh)
        fuse_ms = (time.perf_counter() - t_fuse_0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_timing_breakdown = {
            "general_ms": round(gen_ms, 2),
            "specialist_ms": round(spec_ms, 2),
            "fusion_ms": round(fuse_ms, 2),
            "total_ms": round(total_ms, 2),
            "cascaded_invoked": invoke_specialist,
        }

        return DetectionResponse(
            detections=fused_detections,
            total_objects=len(fused_detections),
            is_simulated=False,
            model_name=self.get_model_name(),
            checkpoint=f"{getattr(self.general_model, 'checkpoint_path', 'gen')} + {getattr(self.specialist_model, 'checkpoint_path', 'spec')}",
            inference_time_ms=round(total_ms, 2),
            device=getattr(self.general_model, "device", "cuda:0"),
        )

    def _predict_high_res_crop(
        self,
        image: Image.Image,
        conf_thresh: float,
        t_start: float,
    ) -> DetectionResponse:
        t_gen_0 = time.perf_counter()
        gen_resp = self.general_model.predict(image, confidence_threshold=min(conf_thresh, 0.15))
        gen_ms = (time.perf_counter() - t_gen_0) * 1000.0

        spec_detections = []
        spec_ms = 0.0

        # Identify entrance / ground-floor region for high-resolution crop
        # Find union box of lower doors (class 3) and lower facades (class 1)
        w_img, h_img = image.size
        entrance_boxes = [d.box for d in gen_resp.detections if d.class_id in [1, 3, 6] and d.box.y_max >= 0.40]

        if entrance_boxes and self.specialist_model is not None:
            t_spec_0 = time.perf_counter()
            x1 = max(0.0, min(b.x_min for b in entrance_boxes) - 0.10)
            y1 = max(0.0, min(b.y_min for b in entrance_boxes) - 0.10)
            x2 = min(1.0, max(b.x_max for b in entrance_boxes) + 0.10)
            y2 = min(1.0, max(b.y_max for b in entrance_boxes) + 0.05)

            crop_box = (int(x1 * w_img), int(y1 * h_img), int(x2 * w_img), int(y2 * h_img))
            crop_w = crop_box[2] - crop_box[0]
            crop_h = crop_box[3] - crop_box[1]

            if crop_w >= 64 and crop_h >= 64:
                cropped_img = image.crop(crop_box)
                crop_resp = self.specialist_model.predict(cropped_img, confidence_threshold=self.specialist_conf_thresh)

                # Map coordinates back to full image coordinate space
                for d in crop_resp.detections:
                    orig_xmin = x1 + d.box.x_min * (crop_w / w_img)
                    orig_ymin = y1 + d.box.y_min * (crop_h / h_img)
                    orig_xmax = x1 + d.box.x_max * (crop_w / w_img)
                    orig_ymax = y1 + d.box.y_max * (crop_h / h_img)

                    remapped_box = BoundingBox(
                        x_min=max(0.0, min(1.0, round(orig_xmin, 4))),
                        y_min=max(0.0, min(1.0, round(orig_ymin, 4))),
                        x_max=max(0.0, min(1.0, round(orig_xmax, 4))),
                        y_max=max(0.0, min(1.0, round(orig_ymax, 4))),
                    )
                    spec_detections.append(DetectionItem(
                        class_id=6,
                        label="porch",
                        confidence=d.confidence,
                        box=remapped_box,
                        model_provenance="porch_specialist_highres_crop",
                    ))
            spec_ms = (time.perf_counter() - t_spec_0) * 1000.0

        # Fuse
        t_fuse_0 = time.perf_counter()
        fused_detections = self._fuse_predictions(gen_resp.detections, spec_detections, conf_thresh)
        fuse_ms = (time.perf_counter() - t_fuse_0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.last_timing_breakdown = {
            "general_ms": round(gen_ms, 2),
            "specialist_ms": round(spec_ms, 2),
            "fusion_ms": round(fuse_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        return DetectionResponse(
            detections=fused_detections,
            total_objects=len(fused_detections),
            is_simulated=False,
            model_name=self.get_model_name(),
            checkpoint=f"{getattr(self.general_model, 'checkpoint_path', 'gen')} + {getattr(self.specialist_model, 'checkpoint_path', 'spec')}",
            inference_time_ms=round(total_ms, 2),
            device=getattr(self.general_model, "device", "cuda:0"),
        )

    def _fuse_predictions(
        self,
        gen_items: List[DetectionItem],
        spec_items: List[DetectionItem],
        conf_thresh: float,
    ) -> List[DetectionItem]:
        """Merges general detections with specialist porch detections."""
        non_porch_items = [d for d in gen_items if d.class_id != 6 and d.confidence >= conf_thresh]
        for item in non_porch_items:
            item.model_provenance = "general_b6"

        gen_porches = [d for d in gen_items if d.class_id == 6 and d.confidence >= 0.15]
        spec_porches = [d for d in spec_items if d.confidence >= self.specialist_conf_thresh]

        fused_porches: List[DetectionItem] = []
        matched_spec_indices = set()

        w_gen, w_spec = self.weights

        for g in gen_porches:
            g_box = [g.box.x_min, g.box.y_min, g.box.x_max, g.box.y_max]
            best_iou = 0.0
            best_s_idx = -1

            for s_idx, s in enumerate(spec_porches):
                if s_idx in matched_spec_indices:
                    continue
                s_box = [s.box.x_min, s.box.y_min, s.box.x_max, s.box.y_max]
                iou = box_iou_xyxy(g_box, s_box)
                if iou > best_iou:
                    best_iou = iou
                    best_s_idx = s_idx

            if best_iou >= self.fusion_iou_thresh and best_s_idx != -1:
                # Weighted Box Fusion (WBF)
                s = spec_porches[best_s_idx]
                s_box = [s.box.x_min, s.box.y_min, s.box.x_max, s.box.y_max]

                denom = (w_gen * g.confidence) + (w_spec * s.confidence)
                if denom > 0:
                    f_xmin = (w_gen * g.confidence * g_box[0] + w_spec * s.confidence * s_box[0]) / denom
                    f_ymin = (w_gen * g.confidence * g_box[1] + w_spec * s.confidence * s_box[1]) / denom
                    f_xmax = (w_gen * g.confidence * g_box[2] + w_spec * s.confidence * s_box[2]) / denom
                    f_ymax = (w_gen * g.confidence * g_box[3] + w_spec * s.confidence * s_box[3]) / denom
                else:
                    f_xmin, f_ymin, f_xmax, f_ymax = g_box

                f_conf = min(1.0, max(g.confidence, s.confidence) + self.consensus_bonus)
                if f_conf >= conf_thresh:
                    fused_porches.append(DetectionItem(
                        class_id=6,
                        label="porch",
                        confidence=round(f_conf, 4),
                        box=BoundingBox(
                            x_min=round(f_xmin, 4),
                            y_min=round(f_ymin, 4),
                            x_max=round(f_xmax, 4),
                            y_max=round(f_ymax, 4),
                        ),
                        model_provenance="fused_consensus_wbf",
                    ))
                matched_spec_indices.add(best_s_idx)
            else:
                # Solo general porch
                if g.confidence >= conf_thresh:
                    g.model_provenance = "general_b6_solo"
                    fused_porches.append(g)

        # Unmatched solo specialist porches
        for s_idx, s in enumerate(spec_porches):
            if s_idx not in matched_spec_indices:
                if s.confidence >= conf_thresh:
                    s_item = DetectionItem(
                        class_id=6,
                        label="porch",
                        confidence=round(s.confidence, 4),
                        box=s.box,
                        model_provenance=s.model_provenance or "porch_specialist_solo",
                    )
                    fused_porches.append(s_item)

        # Cross-Class Conflict Resolution:
        # Porch vs Building/Facade: porches sit physically on facades, so both are preserved.
        # Deduplicate overlapping solo specialist porches
        cleaned_porches = self._deduplicate_same_class(fused_porches, iou_thresh=0.50)

        return non_porch_items + cleaned_porches

    def _deduplicate_same_class(self, items: List[DetectionItem], iou_thresh: float = 0.50) -> List[DetectionItem]:
        """Suppresses internal overlapping duplicates for the same class."""
        if len(items) <= 1:
            return items
        items.sort(key=lambda x: x.confidence, reverse=True)
        keep = []
        for i, it in enumerate(items):
            box_i = [it.box.x_min, it.box.y_min, it.box.x_max, it.box.y_max]
            suppress = False
            for prev in keep:
                box_p = [prev.box.x_min, prev.box.y_min, prev.box.x_max, prev.box.y_max]
                if box_iou_xyxy(box_i, box_p) >= iou_thresh:
                    suppress = True
                    break
            if not suppress:
                keep.append(it)
        return keep
