"""Object Detection service coordinating model execution, ensembling, and visualization."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from models.detection_model import CustomObjectDetector, MockObjectDetector
from models.interfaces import BaseDetector, ModelNotImplementedError
from schemas.responses import DetectionResponse
from services.ensemble_service import EnsembleDetector
from services.image_service import draw_bounding_boxes
from utils.logging_config import get_logger

logger = get_logger("DetectionService")


def resolve_detector(config: Dict[str, Any], mock_mode: bool = False) -> Tuple[Any, float]:
    """Resolves and instantiates the chosen detector or ensemble, recording load duration."""
    t_load_start = time.perf_counter()

    if mock_mode:
        model = MockObjectDetector(config)
        load_ms = round((time.perf_counter() - t_load_start) * 1000.0, 2)
        return model, load_ms

    repo_root = Path(__file__).resolve().parent.parent
    checkpoints_dir = repo_root / "checkpoints"

    detector_mode = config.get("detector_mode", "single").lower()
    ckpt_path_str = str(config.get("checkpoint_path", "")).lower()
    task_group = config.get("task_group", "").lower()

    if "exterior_b6_clahe" in ckpt_path_str:
        b6_pt = checkpoints_dir / "exterior_detector_b6_expanded_best.pt"
        cfg_copy = dict(config)
        cfg_copy["checkpoint_path"] = str(b6_pt)
        cfg_copy["use_clahe"] = True
        model = CustomObjectDetector(cfg_copy)
        load_ms = round((time.perf_counter() - t_load_start) * 1000.0, 2)
        return model, load_ms

    if "exterior" in task_group or "exterior" in ckpt_path_str:
        if detector_mode == "ensemble" or "ensemble" in ckpt_path_str:
            b1_pt = checkpoints_dir / "exterior_detector_b1_yolo11n_best.pt"
            if not b1_pt.exists():
                b1_pt = checkpoints_dir / "exterior_detector_b1_yolo11n_pilot_best.pt"
            if not b1_pt.exists():
                b1_pt = repo_root / "yolo11n.pt"

            b2_pt = checkpoints_dir / "exterior_detector_b2_yolo11s_best.pt"
            if not b2_pt.exists():
                b2_pt = checkpoints_dir / "exterior_detector_b2_yolo11s_pilot_best.pt"
            if not b2_pt.exists():
                b2_pt = repo_root / "yolo11s.pt"

            device = config.get("device", "cuda:0")
            model_b1 = CustomObjectDetector({"checkpoint_path": str(b1_pt), "device": device})
            model_b2 = CustomObjectDetector({"checkpoint_path": str(b2_pt), "device": device})

            try:
                model_b1.load_model()
                model_b2.load_model()
            except Exception as exc:
                logger.warning(f"Exterior ensemble model preload deferred: {exc}")

            model = EnsembleDetector(
                models=[model_b1, model_b2],
                model_names=["Model B1 (YOLO11n)", "Model B2 (YOLO11s)"],
                model_weights=[0.8, 1.2],
                fusion_iou_threshold=0.50,
                single_model_factor=0.88,
                consensus_bonus=0.06,
                fusion_method="wbf",
            )
        else:
            model = CustomObjectDetector(config)
    elif "parking" in task_group or "parking" in ckpt_path_str:
        if detector_mode == "ensemble" or "ensemble" in ckpt_path_str:
            c1_pt = checkpoints_dir / "parking_detector_c1_yolo11n_best.pt"
            if not c1_pt.exists():
                c1_pt = checkpoints_dir / "parking_detector_c1_yolo11n_pilot_best.pt"
            if not c1_pt.exists():
                c1_pt = repo_root / "yolo11n.pt"

            c2_pt = checkpoints_dir / "parking_detector_c2_yolo11s_best.pt"
            if not c2_pt.exists():
                c2_pt = checkpoints_dir / "parking_detector_c2_yolo11s_pilot_best.pt"
            if not c2_pt.exists():
                c2_pt = repo_root / "yolo11s.pt"

            device = config.get("device", "cuda:0")
            model_c1 = CustomObjectDetector({"checkpoint_path": str(c1_pt), "device": device})
            model_c2 = CustomObjectDetector({"checkpoint_path": str(c2_pt), "device": device})

            try:
                model_c1.load_model()
                model_c2.load_model()
            except Exception as exc:
                logger.warning(f"Parking ensemble model preload deferred: {exc}")

            model = EnsembleDetector(
                models=[model_c1, model_c2],
                model_names=["Model C1 (YOLO11n)", "Model C2 (YOLO11s)"],
                model_weights=[0.8, 1.2],
                fusion_iou_threshold=0.50,
                single_model_factor=0.88,
                consensus_bonus=0.06,
                fusion_method="wbf",
            )
        else:
            model = CustomObjectDetector(config)
    elif detector_mode == "ensemble" or "ensemble" in ckpt_path_str:
        # Resolve A1 checkpoint
        a1_pt = checkpoints_dir / "indoor_detector_a1_yolo11n_best.pt"
        if not a1_pt.exists():
            a1_pt = checkpoints_dir / "indoor_detector_a1_yolo11n_pilot_best.pt"
        if not a1_pt.exists():
            a1_pt = checkpoints_dir / "deepfurniture_yolo11n_finetune_best.pt"

        # Resolve A2 checkpoint
        a2_pt = checkpoints_dir / "indoor_detector_a2_yolo11s_finetune_best.pt"
        if not a2_pt.exists():
            a2_pt = checkpoints_dir / "indoor_detector_a2_yolo11s_best.pt"
        if not a2_pt.exists():
            a2_pt = checkpoints_dir / "indoor_detector_a2_yolo11s_pilot_best.pt"
        if not a2_pt.exists():
            a2_pt = repo_root / "yolo11s.pt"

        device = config.get("device", "cuda:0")
        model_a1 = CustomObjectDetector({"checkpoint_path": str(a1_pt), "device": device})
        model_a2 = CustomObjectDetector({"checkpoint_path": str(a2_pt), "device": device})

        try:
            model_a1.load_model()
            model_a2.load_model()
        except Exception as exc:
            logger.warning(f"Ensemble model preload deferred: {exc}")

        model = EnsembleDetector(
            models=[model_a1, model_a2],
            model_names=["Model A1 (YOLO11n)", "Fine-Tuned Model A2 (YOLO11s)"],
            model_weights=[0.8, 1.2],
            fusion_iou_threshold=0.50,
            single_model_factor=0.88,
            consensus_bonus=0.06,
            fusion_method="wbf",
            cross_class_iou_threshold=0.80,
        )
    else:
        # Individual detector mode (lazy loading on predict)
        model = CustomObjectDetector(config)

    load_ms = round((time.perf_counter() - t_load_start) * 1000.0, 2)
    return model, load_ms


class DetectionService:
    """Coordinates object detection model inference, ensembling, and visualization rendering."""

    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        self.mock_mode = mock_mode
        self.config = config or {}
        self.model, self.load_time_ms = resolve_detector(self.config, mock_mode=mock_mode)

    def run(
        self, image: Image.Image, confidence_threshold: float = 0.35, save_annotated: bool = True
    ) -> Dict[str, Any]:
        """Executes object detection, renders annotations, and tracks granular timing breakdown.

        Returns a dictionary with:
            success: bool
            response: Optional[DetectionResponse]
            annotated_image: Optional[PIL.Image.Image]
            saved_image_path: Optional[str]
            timing_metrics: Dict[str, Any]
            error_message: Optional[str]
            is_implemented: bool
        """
        t_service_start = time.perf_counter()
        if image is None:
            return {
                "success": False,
                "response": None,
                "annotated_image": None,
                "saved_image_path": None,
                "timing_metrics": {"queue_wait_ms": 0.0},
                "error_message": "Input image cannot be None.",
                "is_implemented": True,
            }
        try:
            # Preprocessing timing
            t_pre_start = time.perf_counter()
            if image.mode != "RGB":
                image = image.convert("RGB")
            if self.config.get("use_clahe") or "clahe" in str(self.config.get("checkpoint_path", "")).lower():
                import cv2
                np_img = np.array(image)
                lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                merged = cv2.merge((cl, a, b))
                rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
                image = Image.fromarray(rgb)
            preprocess_ms = round((time.perf_counter() - t_pre_start) * 1000.0, 2)

            # Inference & optional fusion timing
            t_infer_start = time.perf_counter()
            response = self.model.predict(image, confidence_threshold=confidence_threshold)
            infer_duration_ms = round((time.perf_counter() - t_infer_start) * 1000.0, 2)

            # Check if ensemble fusion was performed and extract exact breakdown
            is_ensemble = isinstance(self.model, EnsembleDetector)
            if is_ensemble and hasattr(self.model, "last_timing_breakdown"):
                tb = self.model.last_timing_breakdown
                fusion_ms = float(tb.get("fusion_ms", 0.0))
                per_model_times = tb.get("per_model_times", {})
                model_infer_ms = round(sum(v for v in per_model_times.values() if isinstance(v, (int, float)) and v > 0), 2)
            else:
                fusion_ms = 0.0
                model_infer_ms = response.inference_time_ms

            # Annotation rendering timing
            t_render_start = time.perf_counter()
            annotated_img = draw_bounding_boxes(image, response.detections)
            render_ms = round((time.perf_counter() - t_render_start) * 1000.0, 2)

            # Saving timing
            saved_path = None
            save_ms = 0.0
            if save_annotated:
                t_save_start = time.perf_counter()
                out_dir = Path("artifacts/detections")
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time())
                saved_path = str(out_dir / f"detection_{timestamp}.png")
                annotated_img.save(saved_path, format="PNG")
                save_ms = round((time.perf_counter() - t_save_start) * 1000.0, 2)
                logger.info(f"Annotated detection saved to: {saved_path}")

            total_ms = round((time.perf_counter() - t_service_start) * 1000.0, 2)
            fps = round(1000.0 / total_ms, 1) if total_ms > 0 else 0.0

            task_name = self.config.get("task_group", "Group A: Indoor Furniture & Household Objects")
            use_pct = bool(self.config.get("use_per_class_thresholds", False))
            timing_metrics = {
                "task_name": task_name,
                "model_selected": response.model_name,
                "is_ensemble": is_ensemble,
                "load_time_ms": self.load_time_ms,
                "preprocess_time_ms": preprocess_ms,
                "per_model_inference_ms": model_infer_ms,
                "ensemble_fusion_ms": fusion_ms,
                "rendering_time_ms": render_ms,
                "saving_time_ms": save_ms,
                "total_end_to_end_ms": total_ms,
                "steady_state_fps": fps,
                "total_detections": response.total_objects,
                "confidence_threshold": confidence_threshold,
                "device": response.device,
                "queue_wait_ms": 0.0,
                "use_per_class_thresholds": use_pct,
            }

            return {
                "success": True,
                "response": response,
                "annotated_image": annotated_img,
                "saved_image_path": saved_path,
                "timing_metrics": timing_metrics,
                "error_message": None,
                "is_implemented": True,
            }
        except ModelNotImplementedError as exc:
            logger.warning(f"Detection model not implemented: {exc}")
            return {
                "success": False,
                "response": None,
                "annotated_image": None,
                "saved_image_path": None,
                "timing_metrics": {"queue_wait_ms": 0.0},
                "error_message": str(exc),
                "is_implemented": False,
            }
        except Exception as exc:
            logger.error(f"Error executing detection service: {exc}", exc_info=True)
            return {
                "success": False,
                "response": None,
                "annotated_image": None,
                "saved_image_path": None,
                "timing_metrics": {"queue_wait_ms": 0.0},
                "error_message": f"Object detection failed: {str(exc)}",
                "is_implemented": True,
            }
