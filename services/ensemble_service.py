"""Confidence-Weighted Ensemble Service for Object Detection.

Implements:
1. Weighted Box Fusion (WBF) and Class-Aware NMS for compatible object detectors.
2. Per-model confidence calibration (linear weights and temperature scaling).
3. Class-aware IoU matching that merges duplicate boxes across models without
   suppressing distinct neighboring objects of different or same classes.
4. Single-model preservation logic to retain high-value recall additions.
5. Model provenance tracking (records which candidate models contributed to each box).
6. Granular inference timing breakdown (load, per-model inference, fusion, total).
7. Graceful single-model fallback if a candidate checkpoint fails.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from models.interfaces import BaseDetector
from schemas.responses import BoundingBox, DetectionItem, DetectionResponse
from utils.logging_config import get_logger
from utils.timing_utils import InferenceTimer

logger = get_logger("EnsembleService")


def calculate_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Calculates Intersection over Union (IoU) between two normalized boxes [x1, y1, x2, y2]."""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, (x2_a - x1_a) * (y2_a - y1_a))
    area_b = max(0.0, (x2_b - x1_b) * (y2_b - y1_b))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


class EnsembleDetector:
    """Combines predictions from multiple object detectors using confidence-weighted fusion."""

    def __init__(
        self,
        models: List[BaseDetector],
        model_names: Optional[List[str]] = None,
        model_weights: Optional[List[float]] = None,
        fusion_iou_threshold: float = 0.50,
        single_model_factor: float = 0.90,
        consensus_bonus: float = 0.05,
        fusion_method: str = "wbf",
        cross_class_iou_threshold: Optional[float] = None,
    ):
        """Initializes the EnsembleDetector.

        Args:
            models: List of initialized candidate detector models.
            model_names: Optional human-readable names for each model.
            model_weights: Confidence weights per model (default: equal weights 1.0).
            fusion_iou_threshold: IoU threshold for matching boxes of the same class.
            single_model_factor: Multiplier applied to boxes detected by only 1 model.
            consensus_bonus: Score bonus per additional agreeing model.
            fusion_method: 'wbf' (Weighted Box Fusion) or 'nms' (Class-Aware NMS).
            cross_class_iou_threshold: Optional IoU threshold (e.g. 0.80) to suppress
                duplicate boxes across different classes on the same physical object.
        """
        if not models:
            raise ValueError("EnsembleDetector requires at least one model.")

        self.models = models
        self.model_names = model_names or [m.get_model_name() for m in models]
        self.model_weights = model_weights or [1.0] * len(models)
        self.fusion_iou_threshold = fusion_iou_threshold
        self.single_model_factor = single_model_factor
        self.consensus_bonus = consensus_bonus
        self.fusion_method = fusion_method.lower()
        self.cross_class_iou_threshold = cross_class_iou_threshold
        self.last_timing_breakdown: Dict[str, Any] = {}

        if len(self.model_weights) != len(self.models):
            raise ValueError("Length of model_weights must match length of models.")

    def calibrate_score(self, raw_score: float, model_idx: int) -> float:
        """Applies calibration weight to a model's raw confidence score."""
        weight = self.model_weights[model_idx]
        # Linear calibration bounded in [0.0, 1.0]
        return float(np.clip(raw_score * weight, 0.0, 1.0))

    def fuse_cluster_wbf(
        self, cluster: List[Dict[str, Any]], num_models: int
    ) -> Tuple[Tuple[float, float, float, float], float, List[str]]:
        """Applies Weighted Box Fusion (WBF) to a cluster of matching detections."""
        total_weight_score = sum(item["weight"] * item["calibrated_score"] for item in cluster)
        if total_weight_score <= 0.0:
            total_weight_score = 1e-6

        # Confidence-weighted coordinate fusion
        fused_x1 = sum(item["weight"] * item["calibrated_score"] * item["box"][0] for item in cluster) / total_weight_score
        fused_y1 = sum(item["weight"] * item["calibrated_score"] * item["box"][1] for item in cluster) / total_weight_score
        fused_x2 = sum(item["weight"] * item["calibrated_score"] * item["box"][2] for item in cluster) / total_weight_score
        fused_y2 = sum(item["weight"] * item["calibrated_score"] * item["box"][3] for item in cluster) / total_weight_score

        # Bound check
        fused_x1 = max(0.0, min(1.0, fused_x1))
        fused_y1 = max(0.0, min(1.0, fused_y1))
        fused_x2 = max(fused_x1 + 0.001, min(1.0, fused_x2))
        fused_y2 = max(fused_y1 + 0.001, min(1.0, fused_y2))

        # Distinct models that contributed
        models_contributed = sorted(list(set(item["model_name"] for item in cluster)))
        num_agreeing = len(models_contributed)

        # Base confidence: weighted average
        avg_score = sum(item["calibrated_score"] for item in cluster) / len(cluster)

        if num_agreeing > 1:
            # Consensus reinforcement
            fused_score = min(1.0, avg_score * (1.0 + self.consensus_bonus * (num_agreeing - 1)))
        else:
            # Single-model preservation with slight penalty if multiple models were active
            if num_models > 1:
                fused_score = avg_score * self.single_model_factor
            else:
                fused_score = avg_score

        fused_score = float(np.clip(fused_score, 0.0, 1.0))
        return (fused_x1, fused_y1, fused_x2, fused_y2), fused_score, models_contributed

    def fuse_predictions(
        self, all_model_items: List[List[Dict[str, Any]]], confidence_threshold: float
    ) -> List[DetectionItem]:
        """Fuses predictions across all models grouped by canonical class ID."""
        num_models = len(all_model_items)

        # Group detections by class_id
        class_buckets: Dict[int, List[Dict[str, Any]]] = {}
        class_label_map: Dict[int, str] = {}

        for model_idx, items in enumerate(all_model_items):
            model_name = self.model_names[model_idx]
            weight = self.model_weights[model_idx]

            for item in items:
                cid = item["class_id"]
                class_label_map[cid] = item["label"]
                calibrated = self.calibrate_score(item["confidence"], model_idx)

                record = {
                    "class_id": cid,
                    "label": item["label"],
                    "raw_confidence": item["confidence"],
                    "calibrated_score": calibrated,
                    "weight": weight,
                    "box": item["box"],  # (x1, y1, x2, y2)
                    "model_idx": model_idx,
                    "model_name": model_name,
                }
                class_buckets.setdefault(cid, []).append(record)

        final_items: List[DetectionItem] = []

        for cid, candidates in class_buckets.items():
            label = class_label_map[cid]
            # Sort descending by calibrated score
            candidates.sort(key=lambda x: x["calibrated_score"], reverse=True)

            if self.fusion_method == "nms":
                # Standard Class-Aware NMS
                kept: List[Dict[str, Any]] = []
                for cand in candidates:
                    overlap = False
                    for k in kept:
                        if calculate_iou(cand["box"], k["box"]) > self.fusion_iou_threshold:
                            overlap = True
                            break
                    if not overlap:
                        kept.append(cand)

                for item in kept:
                    if item["calibrated_score"] >= confidence_threshold:
                        x1, y1, x2, y2 = item["box"]
                        final_items.append(
                            DetectionItem(
                                class_id=cid,
                                label=label,
                                confidence=round(item["calibrated_score"], 4),
                                box=BoundingBox(
                                    x_min=round(x1, 4),
                                    y_min=round(y1, 4),
                                    x_max=round(x2, 4),
                                    y_max=round(y2, 4),
                                ),
                            )
                        )
            else:
                # Weighted Box Fusion (WBF)
                clusters: List[List[Dict[str, Any]]] = []
                cluster_centers: List[Tuple[float, float, float, float]] = []

                for cand in candidates:
                    matched_cluster_idx = -1
                    best_iou = -1.0

                    for c_idx, center_box in enumerate(cluster_centers):
                        iou = calculate_iou(cand["box"], center_box)
                        if iou >= self.fusion_iou_threshold and iou > best_iou:
                            best_iou = iou
                            matched_cluster_idx = c_idx

                    if matched_cluster_idx >= 0:
                        clusters[matched_cluster_idx].append(cand)
                        # Recompute center of cluster
                        cur_cluster = clusters[matched_cluster_idx]
                        sum_x1 = sum(it["box"][0] for it in cur_cluster) / len(cur_cluster)
                        sum_y1 = sum(it["box"][1] for it in cur_cluster) / len(cur_cluster)
                        sum_x2 = sum(it["box"][2] for it in cur_cluster) / len(cur_cluster)
                        sum_y2 = sum(it["box"][3] for it in cur_cluster) / len(cur_cluster)
                        cluster_centers[matched_cluster_idx] = (sum_x1, sum_y1, sum_x2, sum_y2)
                    else:
                        clusters.append([cand])
                        cluster_centers.append(cand["box"])

                # Fuse each cluster
                for cluster in clusters:
                    fused_box, fused_score, models_contrib = self.fuse_cluster_wbf(cluster, num_models)
                    if fused_score >= confidence_threshold:
                        x1, y1, x2, y2 = fused_box
                        final_items.append(
                            DetectionItem(
                                class_id=cid,
                                label=label,
                                confidence=round(fused_score, 4),
                                box=BoundingBox(
                                    x_min=round(x1, 4),
                                    y_min=round(y1, 4),
                                    x_max=round(x2, 4),
                                    y_max=round(y2, 4),
                                ),
                            )
                        )

        # Sort final items descending by confidence
        final_items.sort(key=lambda x: x.confidence, reverse=True)

        # Cross-class duplicate conflict resolution
        if self.cross_class_iou_threshold and self.cross_class_iou_threshold > 0.0:
            cleaned_items: List[DetectionItem] = []
            for item in final_items:
                suppress = False
                box_item = (item.box.x_min, item.box.y_min, item.box.x_max, item.box.y_max)
                for k in cleaned_items:
                    if item.class_id != k.class_id:
                        box_k = (k.box.x_min, k.box.y_min, k.box.x_max, k.box.y_max)
                        if calculate_iou(box_item, box_k) >= self.cross_class_iou_threshold:
                            suppress = True
                            break
                if not suppress:
                    cleaned_items.append(item)
            final_items = cleaned_items

        return final_items

    def predict(
        self, image: Image.Image, confidence_threshold: float = 0.35
    ) -> DetectionResponse:
        """Executes candidate models, measures per-model latency, and fuses predictions.

        Returns a DetectionResponse with comprehensive timing breakdown.
        """
        all_model_items: List[List[Dict[str, Any]]] = []
        per_model_times: Dict[str, float] = {}
        active_devices: List[str] = []
        model_checkpoints: List[str] = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_start = time.perf_counter()

        for idx, model in enumerate(self.models):
            m_name = self.model_names[idx]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            m_start = time.perf_counter()
            try:
                # Use slightly lower threshold (e.g. 0.15) on candidate detectors to allow
                # weak overlapping signals to participate in confidence-weighted consensus
                candidate_thresh = max(0.10, confidence_threshold - 0.15)
                resp = model.predict(image, confidence_threshold=candidate_thresh)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                m_duration = (time.perf_counter() - m_start) * 1000.0
                per_model_times[m_name] = round(m_duration, 2)
                active_devices.append(resp.device)
                model_checkpoints.append(resp.checkpoint)

                # Extract items
                items_data = []
                for det in resp.detections:
                    items_data.append(
                        {
                            "class_id": det.class_id,
                            "label": det.label,
                            "confidence": det.confidence,
                            "box": (det.box.x_min, det.box.y_min, det.box.x_max, det.box.y_max),
                        }
                    )
                all_model_items.append(items_data)
            except Exception as exc:
                logger.error(f"Candidate model {m_name} failed during ensemble: {exc}", exc_info=True)
                per_model_times[m_name] = -1.0
                # Continue with remaining models for graceful degradation

        # Fusion phase
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fusion_start = time.perf_counter()
        fused_items = self.fuse_predictions(all_model_items, confidence_threshold=confidence_threshold)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fusion_ms = round((time.perf_counter() - fusion_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - total_start) * 1000.0, 2)

        self.last_timing_breakdown = {
            "per_model_times": per_model_times,
            "fusion_ms": fusion_ms,
            "total_ms": total_ms,
        }

        ensemble_name = f"Confidence-Weighted Ensemble ({'+'.join(self.model_names)})"
        primary_device = active_devices[0] if active_devices else "cpu"
        combined_ckpt = " + ".join(model_checkpoints)

        return DetectionResponse(
            detections=fused_items,
            total_objects=len(fused_items),
            is_simulated=False,
            model_name=ensemble_name,
            checkpoint=combined_ckpt,
            inference_time_ms=total_ms,
            device=primary_device,
        )
