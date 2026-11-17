"""High-Performance Object Detection Endpoint for Model A3."""

import asyncio
import hashlib
import io
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
import torch

from api.schemas import (
    BoundingBox2D,
    DetectionItem,
    DetectionResponse,
    ImageDimensions,
    InferenceMetadata,
    TimingBreakdown,
)
from utils.device import get_optimal_device
from utils.logging_config import get_logger

logger = get_logger("DetectRouter")

router = APIRouter(prefix="/api/v1/detect", tags=["Detection"])

# Supported formats and constraints
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB ceiling
MIN_DIMENSION_PX = 32
MAX_DIMENSION_PX = 8192
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
MAX_CACHE_ENTRIES = 64


SUPPORTED_DETECTOR_MODELS: Dict[str, Dict[str, Any]] = {
    "indoor_detector_a3_yolo11s_800px": {
        "canonical_name": "indoor_detector_a3_yolo11s_800px",
        "checkpoint": Path(os.getenv("ROOMAI_MODEL_PATH", "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")),
        "imgsz": 800,
        "aliases": ["indoor_detector_a3_yolo11s_800px", "indoor_a3", "model_a3", "indoor", "a3"],
    },
    "exterior_detector_b2_yolo11s": {
        "canonical_name": "exterior_detector_b2_yolo11s",
        "checkpoint": Path("checkpoints/exterior_detector_b2_yolo11s_best.pt"),
        "imgsz": 640,
        "aliases": ["exterior_detector_b2_yolo11s", "exterior_b2", "model_b2", "exterior", "b2"],
    },
    "parking_detector_c2_yolo11s": {
        "canonical_name": "parking_detector_c2_yolo11s",
        "checkpoint": Path("checkpoints/parking_detector_c2_yolo11s_pilot_best.pt"),
        "imgsz": 640,
        "aliases": ["parking_detector_c2_yolo11s", "parking_c2", "model_c2", "parking", "c2"],
    },
}


class ModelA3InferenceEngine:
    """Singleton inference manager for Model A3 and specialist YOLO11 models.
    
    Guarantees:
    - Primary Model A3 loaded once at application startup.
    - 1 warmup pass executed prior to user requests to prime CUDA context.
    - Thread-safe serialization on GPU via asyncio.Lock.
    - In-memory SHA-256 LRU cache for identical images.
    - On-demand caching for specialist models (B2 Exterior, C2 Parking).
    """

    def __init__(self):
        self.model = None
        self.model_name = "indoor_detector_a3_yolo11s_800px"
        self.checkpoint_path = Path(
            os.getenv("ROOMAI_MODEL_PATH", "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt")
        )
        self.device = "cpu"
        self.imgsz = 800
        self.lock = asyncio.Lock()
        self._cache: OrderedDict[Tuple[str, float, str, int], Dict[str, Any]] = OrderedDict()
        self._models: Dict[str, Any] = {}
        self.is_warmed = False
        self.startup_error: Optional[str] = None

    def load_and_warmup(self) -> None:
        """Loads Model A3 onto target device and runs 1 warmup inference pass."""
        if self.model is not None and self.is_warmed:
            return

        t0 = time.perf_counter()
        optimal = get_optimal_device()
        self.device = "cuda:0" if optimal == "cuda" else "cpu"

        # Resolve and ensure checkpoint presence with integrity check
        from utils.model_downloader import (
            ChecksumMismatchError,
            ModelDownloadError,
            ensure_model_checkpoint,
        )

        try:
            resolved_ckpt = ensure_model_checkpoint(self.checkpoint_path)
        except (ChecksumMismatchError, ModelDownloadError, FileNotFoundError) as exc:
            err_msg = f"Model A3 checkpoint unavailable: {exc}"
            logger.error(err_msg)
            self.startup_error = err_msg
            raise FileNotFoundError(err_msg) from exc
        except Exception as exc:
            err_msg = f"Unexpected error while ensuring Model A3 checkpoint: {exc}"
            logger.error(err_msg, exc_info=True)
            self.startup_error = err_msg
            raise RuntimeError(err_msg) from exc

        logger.info(f"Initializing Model A3 on {self.device} from {resolved_ckpt}...")
        from ultralytics import YOLO

        self.model = YOLO(str(resolved_ckpt))
        self._models[self.model_name] = self.model
        load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.info(f"Model A3 weights loaded in {load_ms}ms")

        # Run startup warmup pass
        try:
            logger.info(f"Executing warmup pass for Model A3 on {self.device}...")
            dummy_img = Image.new("RGB", (self.imgsz, self.imgsz), (128, 128, 128))
            use_half = self.device.startswith("cuda")
            _ = self.model.predict(
                source=dummy_img,
                imgsz=self.imgsz,
                conf=0.25,
                device=self.device,
                half=use_half,
                verbose=False,
            )
            self.is_warmed = True
            self.startup_error = None
            warmup_ms = round((time.perf_counter() - t0) * 1000.0 - load_ms, 2)
            logger.info(f"Model A3 warmup complete in {warmup_ms}ms. Service is READY.")
        except Exception as exc:
            logger.error(f"Warmup pass failed: {exc}", exc_info=True)
            self.startup_error = str(exc)
            raise RuntimeError(f"Failed during Model A3 warmup: {exc}")

    def resolve_model(self, requested_id: Optional[str] = None):
        """Resolves target model, loading on-demand if necessary."""
        if not requested_id:
            return self.model_name, self.model, self.imgsz

        req_lower = requested_id.lower().strip()
        target_key = self.model_name
        for key, spec in SUPPORTED_DETECTOR_MODELS.items():
            if req_lower == key.lower() or req_lower in spec["aliases"]:
                target_key = key
                break

        spec = SUPPORTED_DETECTOR_MODELS.get(target_key, SUPPORTED_DETECTOR_MODELS[self.model_name])
        canonical_name = spec["canonical_name"]
        imgsz = spec["imgsz"]

        if canonical_name in self._models and self._models[canonical_name] is not None:
            return canonical_name, self._models[canonical_name], imgsz

        from ultralytics import YOLO
        resolved_ckpt = spec["checkpoint"]
        if not resolved_ckpt.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            resolved_ckpt = repo_root / spec["checkpoint"]

        if not resolved_ckpt.exists():
            logger.warning(f"Requested checkpoint {resolved_ckpt} missing. Falling back to Model A3.")
            return self.model_name, self.model, self.imgsz

        logger.info(f"Loading {canonical_name} on {self.device} from {resolved_ckpt}...")
        loaded = YOLO(str(resolved_ckpt))
        self._models[canonical_name] = loaded
        return canonical_name, loaded, imgsz

    def _make_cache_key(self, sha256_hash: str, conf_thresh: float, model_name: str, imgsz: int) -> Tuple[str, float, str, int]:
        """Generates deterministic cache key incorporating image hash, threshold, model, and input size."""
        return (sha256_hash, round(conf_thresh, 4), model_name, imgsz)

    def get_cached(self, sha256_hash: str, conf_thresh: float, model_name: str, imgsz: int) -> Optional[Dict[str, Any]]:
        """Retrieves cached detection items if present, moving to end of LRU."""
        key = self._make_cache_key(sha256_hash, conf_thresh, model_name, imgsz)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def store_cache(self, sha256_hash: str, conf_thresh: float, model_name: str, imgsz: int, detections: List[Dict[str, Any]]) -> None:
        """Stores detections in LRU cache, evicting oldest if capacity reached."""
        key = self._make_cache_key(sha256_hash, conf_thresh, model_name, imgsz)
        self._cache[key] = detections
        self._cache.move_to_end(key)
        if len(self._cache) > MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Flushes the in-memory detection cache. Used for testing and cache invalidation."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    async def predict_async(
        self, image: Image.Image, conf_thresh: float, sha256_hash: str, model_id: Optional[str] = None
    ) -> Tuple[List[DetectionItem], float, bool, float, str]:
        """Performs serial GPU inference protected by asyncio.Lock.
        
        Returns:
            Tuple of (detections, infer_ms, is_cached, queue_wait_ms, canonical_model_name)
        """
        if self.model is None:
            if self.startup_error is not None:
                raise FileNotFoundError(self.startup_error)
            self.load_and_warmup()

        canonical_name, active_model, active_imgsz = self.resolve_model(model_id)

        # 1. Check content cache
        cached_raw = self.get_cached(sha256_hash, conf_thresh, canonical_name, active_imgsz)
        if cached_raw is not None:
            items = [DetectionItem(**d) for d in cached_raw]
            return items, 0.0, True, 0.0, canonical_name

        img_w, img_h = image.size
        use_half = self.device.startswith("cuda")

        # 2. Acquire GPU lock and measure queue delay
        t_wait_start = time.perf_counter()
        async with self.lock:
            queue_wait_ms = round((time.perf_counter() - t_wait_start) * 1000.0, 2)

            # Execute forward pass with strict CUDA stream synchronization
            def _sync_predict():
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_infer_start = time.perf_counter()
                with torch.inference_mode():
                    res = active_model.predict(
                        source=image,
                        imgsz=active_imgsz,
                        conf=conf_thresh,
                        device=self.device,
                        half=use_half,
                        verbose=False,
                    )
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_duration_ms = round((time.perf_counter() - t_infer_start) * 1000.0, 2)
                return res, infer_duration_ms

            loop = asyncio.get_running_loop()
            results, infer_ms = await loop.run_in_executor(None, _sync_predict)

        # 3. Postprocess detections into canonical schema
        detections: List[DetectionItem] = []
        raw_to_cache: List[Dict[str, Any]] = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = r.names[cls_id]
                conf = float(box.conf[0].item())

                xyxy = box.xyxy[0].cpu().numpy()
                norm_x1 = max(0.0, min(1.0, float(xyxy[0]) / img_w))
                norm_y1 = max(0.0, min(1.0, float(xyxy[1]) / img_h))
                norm_x2 = max(0.0, min(1.0, float(xyxy[2]) / img_w))
                norm_y2 = max(0.0, min(1.0, float(xyxy[3]) / img_h))

                item = DetectionItem(
                    class_id=cls_id,
                    class_name=label,
                    confidence=round(conf, 4),
                    bbox=BoundingBox2D(
                        x1=round(norm_x1 * img_w, 1),
                        y1=round(norm_y1 * img_h, 1),
                        x2=round(norm_x2 * img_w, 1),
                        y2=round(norm_y2 * img_h, 1),
                        norm_x1=round(norm_x1, 5),
                        norm_y1=round(norm_y1, 5),
                        norm_x2=round(norm_x2, 5),
                        norm_y2=round(norm_y2, 5),
                    ),
                )
                detections.append(item)
                raw_to_cache.append(item.model_dump())

        # 4. Save to content cache
        self.store_cache(sha256_hash, conf_thresh, canonical_name, active_imgsz, raw_to_cache)

        return detections, infer_ms, False, queue_wait_ms, canonical_name


# Global singleton instance
GLOBAL_MODEL_A3_ENGINE = ModelA3InferenceEngine()


@router.post(
    "",
    response_model=DetectionResponse,
    summary="Synchronous 2D Object Detection via Model A3 / Specialists",
    description="Accepts an image (JPEG, PNG, WEBP), performs YOLO inference (Model A3 Indoor, B2 Exterior, C2 Parking), and returns 2D bounding boxes and timing metrics.",
)
async def detect_objects(
    file: UploadFile = File(..., description="Scene image file (JPEG, PNG, WEBP <= 20MB)"),
    confidence_threshold: float = Form(
        DEFAULT_CONFIDENCE_THRESHOLD, ge=0.01, le=1.0, description="Minimum detection confidence"
    ),
    model_id: Optional[str] = Form(
        "indoor_detector_a3_yolo11s_800px",
        description="Target model identifier (indoor_detector_a3_yolo11s_800px, exterior_detector_b2_yolo11s, or parking_detector_c2_yolo11s)",
    ),
):
    """Executes high-performance object detection on the uploaded image."""
    t_start = time.perf_counter()
    request_id = f"det_{uuid.uuid4().hex[:12]}"

    # 1. Content-Type Header Validation
    content_type = file.content_type.lower() if file.content_type else ""
    if content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": f"Media type '{file.content_type}' is unsupported.",
                "remediation": f"Submit an image formatted as {', '.join(sorted(SUPPORTED_MIME_TYPES))}.",
            },
        )

    # 2. File Size Validation (stream reading with 20MB limit)
    t_read_start = time.perf_counter()
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "EMPTY_PAYLOAD",
                "message": "The uploaded file is empty (0 bytes).",
                "remediation": "Provide a non-empty image file.",
            },
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File size ({len(content)} bytes) exceeds the 20MB ceiling.",
                "remediation": "Resize or compress the image before uploading.",
            },
        )

    # 3. SHA-256 Digest for Content Caching
    sha256_hash = hashlib.sha256(content).hexdigest()

    # 4. Image Decoding and Dimension Validation
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "CORRUPT_IMAGE",
                "message": f"Image stream could not be decoded: {str(exc)}",
                "remediation": "Ensure the uploaded file is a valid, uncorrupted JPEG, PNG, or WebP image.",
            },
        )

    orig_w, orig_h = image.size
    if orig_w < MIN_DIMENSION_PX or orig_h < MIN_DIMENSION_PX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "DIMENSIONS_TOO_SMALL",
                "message": f"Dimensions ({orig_w}x{orig_h}) below minimum threshold of {MIN_DIMENSION_PX}px.",
                "remediation": "Submit an image with width and height of at least 32 pixels.",
            },
        )
    if orig_w > MAX_DIMENSION_PX or orig_h > MAX_DIMENSION_PX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "DIMENSIONS_TOO_LARGE",
                "message": f"Dimensions ({orig_w}x{orig_h}) exceed maximum allowable size of {MAX_DIMENSION_PX}px.",
                "remediation": "Downsample the image resolution before submitting.",
            },
        )

    # Convert to RGB mode if RGBA/Grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")

    preprocess_ms = round((time.perf_counter() - t_read_start) * 1000.0, 2)

    # 5. Model Inference
    try:
        detections, infer_ms, is_cached, queue_wait_ms, used_model = await GLOBAL_MODEL_A3_ENGINE.predict_async(
            image=image, conf_thresh=confidence_threshold, sha256_hash=sha256_hash, model_id=model_id
        )
    except FileNotFoundError as exc:
        logger.error(f"Detector checkpoint missing for request {request_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_UNAVAILABLE",
                "message": f"Detector checkpoint unavailable: {str(exc)}",
                "remediation": "Verify that target detector checkpoint exists on the server.",
            },
        )
    except Exception as exc:
        logger.error(f"Inference error for request {request_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INFERENCE_FAILED",
                "message": "Model inference failed while processing the scene image.",
                "remediation": "Verify server GPU/CPU resources or retry with a different image.",
            },
        )

    t_end = time.perf_counter()
    total_server_ms = round((t_end - t_start) * 1000.0, 2)
    postprocess_ms = max(0.0, round(total_server_ms - preprocess_ms - infer_ms - queue_wait_ms, 2))

    return DetectionResponse(
        status="success",
        request_id=request_id,
        image=ImageDimensions(width=orig_w, height=orig_h, channels=3),
        detections=detections,
        inference=InferenceMetadata(
            model=used_model,
            latency_ms=infer_ms,
            device=GLOBAL_MODEL_A3_ENGINE.device,
            cached=is_cached,
        ),
        timing=TimingBreakdown(
            preprocess_ms=preprocess_ms,
            inference_ms=infer_ms,
            postprocess_ms=postprocess_ms,
            total_server_ms=total_server_ms,
            queue_wait_ms=queue_wait_ms,
        ),
        timestamp=time.time(),
    )
