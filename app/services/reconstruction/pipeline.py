"""RoomAI Reconstruction Service.

Orchestrates the full 3D reconstruction pipeline: inference -> geometry -> export.
"""

import time
import uuid
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np

from app.core.config import settings
from app.core.exceptions import ReconstructionError, GPUError
from app.core.logging import get_logger, log_job_event
from app.services.inference.engine import InferenceEngine, get_inference_engine
from app.services.preprocessing.pipeline import PreprocessingPipeline, get_preprocessing_pipeline
from app.services.geometry.postprocessing import GeometryService, get_geometry_service, PointCloud
from app.services.export.exporter import ExportService, get_export_service
from app.storage.artifacts import ArtifactStore, get_artifact_store


logger = get_logger("ReconstructionService")


class ReconstructionService:
    """End-to-end 3D reconstruction pipeline."""

    def __init__(
        self,
        inference_engine: Optional[InferenceEngine] = None,
        preprocessing: Optional[PreprocessingPipeline] = None,
        geometry: Optional[GeometryService] = None,
        exporter: Optional[ExportService] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ):
        self.inference = inference_engine or get_inference_engine()
        self.preprocessing = preprocessing or get_preprocessing_pipeline()
        self.geometry = geometry or get_geometry_service()
        self.exporter = exporter or get_export_service()
        self.artifact_store = artifact_store or get_artifact_store()

    def reconstruct(
        self,
        job_id: str,
        images: List[Image.Image],
        model_id: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Execute full reconstruction pipeline."""
        options = options or {}
        start_time = time.time()

        # Stage: Preprocessing
        log_job_event(logger, "preprocessing_start", job_id, "PREPROCESSING", "Preprocessing input images", 0.1)
        if progress_callback:
            progress_callback("Preprocessing input images", 0.1)

        model_spec = self.inference.registry.get_model(model_id)
        model_config = {
            "input_resolution": (518, 518),
            "patch_size": 14,
        }

        preprocessed = []
        for i, img in enumerate(images):
            # Convert PIL to bytes for hashing
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()

            result = self.preprocessing.process_for_model(
                img_bytes, f"view_{i:02d}.jpg", model_config
            )
            preprocessed.append(result)

        log_job_event(logger, "preprocessing_complete", job_id, "PREPROCESSING", "Preprocessing complete", 0.2)
        if progress_callback:
            progress_callback("Preprocessing complete", 0.2)

        # Stage: Inference
        log_job_event(logger, "inference_start", job_id, "INFERENCE", "Running model inference", 0.2)
        if progress_callback:
            progress_callback("Running model inference", 0.2)

        pil_images = [p["resized_image"] for p in preprocessed]

        try:
            inference_result = self.inference.run_inference(
                model_id=model_id,
                images=pil_images,
                options=options,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            log_job_event(logger, "inference_failed", job_id, "FAILED", f"Inference failed: {exc}", 0.0)
            raise ReconstructionError(f"Inference failed: {exc}", job_id=job_id) from exc

        log_job_event(logger, "inference_complete", job_id, "INFERENCE", "Inference complete", 0.5)
        if progress_callback:
            progress_callback("Inference complete", 0.5)

        depth_artifacts = self._persist_depth_artifacts(
            job_id=job_id,
            inference_result=inference_result,
            preprocessed=preprocessed,
            model_id=model_id,
            options=options,
        )

        # Stage: Geometry Postprocessing
        log_job_event(logger, "geometry_start", job_id, "GEOMETRY_POSTPROCESSING", "Processing geometry", 0.5)
        if progress_callback:
            progress_callback("Processing geometry", 0.5)

        scene_data = self._process_geometry(inference_result, preprocessed, model_id, options)

        log_job_event(logger, "geometry_complete", job_id, "GEOMETRY_POSTPROCESSING", "Geometry processing complete", 0.7)
        if progress_callback:
            progress_callback("Geometry processing complete", 0.7)

        # Stage: Export
        log_job_event(logger, "export_start", job_id, "EXPORTING", "Exporting 3D assets", 0.7)
        if progress_callback:
            progress_callback("Exporting 3D assets", 0.7)

        job_dir = self.artifact_store.get_job_dir(job_id)
        export_result = self.exporter.export_scene(
            scene_data=scene_data,
            output_dir=job_dir,
            formats=["glb", "ply"],
            max_points=options.get("downsample_points", 40000),
        )

        log_job_event(logger, "export_complete", job_id, "EXPORTING", "Export complete", 0.9)
        if progress_callback:
            progress_callback("Export complete", 0.9)

        # Finalize
        total_time = time.time() - start_time

        result = {
            "job_id": job_id,
            "model_id": model_id,
            "status": "COMPLETED",
            "scene_data": scene_data,
            "exports": export_result,
            "timing": {
                "total_seconds": total_time,
                "preprocessing_seconds": 0,  # Would track per-stage
                "inference_seconds": 0,
                "geometry_seconds": 0,
                "export_seconds": 0,
            },
            "preprocessing_info": {
                "version": self.preprocessing.version,
                "model_config": model_config,
                "num_views": len(images),
            },
            "depth_artifacts": depth_artifacts,
        }

        log_job_event(logger, "reconstruction_complete", job_id, "COMPLETED",
                      "Reconstruction completed successfully", 1.0,
                      duration_ms=total_time * 1000)

        if progress_callback:
            progress_callback("Reconstruction completed", 1.0)

        return result

    def _persist_depth_artifacts(
        self,
        job_id: str,
        inference_result: Dict[str, Any],
        preprocessed: List[Dict[str, Any]],
        model_id: str,
        options: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Persist unmodified dense model depth and its provenance before geometry drops it."""
        depth_maps = inference_result.get("depth_maps") or []
        if not depth_maps:
            return {}

        model_spec = self.inference.registry.get_model(model_id)
        checkpoint_path = self.inference.registry.get_checkpoint_path(model_id)
        depth_dir = self.artifact_store.get_job_dir(job_id) / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        upload_ids = options.get("upload_ids") or []
        confidence_maps = inference_result.get("depth_confidence_maps") or []
        finite_masks = inference_result.get("finite_depth_masks") or []
        point_masks = inference_result.get("point_export_masks") or []
        poses = inference_result.get("camera_poses") or []
        intrinsics = inference_result.get("camera_intrinsics") or []
        artifact_records: Dict[str, Dict[str, Any]] = {}
        views_metadata = []

        def finite_json_array(value: Any) -> Optional[List[Any]]:
            array = np.asarray(value)
            if not np.isfinite(array).all():
                return None
            return array.tolist()

        def save_array(key: str, path: Path, value: np.ndarray) -> None:
            np.save(path, value, allow_pickle=False)
            artifact_records[key] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        for view_index, value in enumerate(depth_maps):
            depth = np.asarray(value)
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32, copy=False)
            if depth.ndim != 2:
                raise ReconstructionError(
                    f"Expected a 2D depth map for view {view_index}; received shape {depth.shape}",
                    job_id=job_id,
                )

            processed_size = preprocessed[view_index]["resized_size"]
            expected_shape = (int(processed_size[1]), int(processed_size[0]))
            if depth.shape != expected_shape:
                raise ReconstructionError(
                    f"Depth map shape {depth.shape} does not match processed RGB {expected_shape}",
                    job_id=job_id,
                )

            stem = f"view_{view_index:02d}"
            depth_path = depth_dir / f"{stem}_raw.npy"
            save_array(f"depth_raw_{view_index:02d}", depth_path, depth)
            finite_mask = (
                np.asarray(finite_masks[view_index], dtype=bool)
                if view_index < len(finite_masks)
                else np.isfinite(depth)
            )
            if finite_mask.shape != depth.shape:
                raise ReconstructionError("Finite-depth mask does not match depth dimensions", job_id=job_id)
            finite_mask_path = depth_dir / f"{stem}_finite_mask.npy"
            save_array(f"depth_finite_mask_{view_index:02d}", finite_mask_path, finite_mask)

            confidence_path = None
            if view_index < len(confidence_maps):
                confidence = np.asarray(confidence_maps[view_index], dtype=np.float32)
                if confidence.shape != depth.shape:
                    raise ReconstructionError("Depth-confidence map does not match depth dimensions", job_id=job_id)
                confidence_path = depth_dir / f"{stem}_confidence.npy"
                save_array(f"depth_confidence_{view_index:02d}", confidence_path, confidence)

            export_mask_path = None
            if view_index < len(point_masks):
                export_mask = np.asarray(point_masks[view_index], dtype=bool)
                if export_mask.shape != depth.shape:
                    raise ReconstructionError("Point-export mask does not match depth dimensions", job_id=job_id)
                export_mask_path = depth_dir / f"{stem}_point_export_mask.npy"
                save_array(f"depth_point_export_mask_{view_index:02d}", export_mask_path, export_mask)

            source_image = {
                "upload_id": upload_ids[view_index] if view_index < len(upload_ids) else None,
                "filename": None,
                "sha256": None,
                "dimensions_px": list(preprocessed[view_index].get("original_size", [])),
            }
            if source_image["upload_id"]:
                source_path = self.artifact_store.get_upload_path(source_image["upload_id"])
                if source_path and source_path.is_file():
                    source_image["filename"] = source_path.name
                    source_image["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            finite_values = depth[finite_mask]
            if finite_values.size:
                stats = {
                    "min": float(finite_values.min()),
                    "max": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "percentiles_1_50_95_99": [float(v) for v in np.percentile(finite_values, [1, 50, 95, 99])],
                }
            else:
                stats = {"min": None, "max": None, "mean": None, "percentiles_1_50_95_99": []}

            rgb_pixels = np.asarray(preprocessed[view_index]["resized_image"].convert("RGB"), dtype=np.uint8)
            views_metadata.append({
                "view_index": view_index,
                "source_image": source_image,
                "processed_rgb": {
                    "dimensions_px": [int(processed_size[0]), int(processed_size[1])],
                    "pixel_sha256": hashlib.sha256(rgb_pixels.tobytes(order="C")).hexdigest(),
                    "preprocessing": "EXIF-corrected RGB; center square crop; bilinear resize to model input resolution; inference crop on the square input is a no-op",
                    "preprocessed_source_file_sha256": preprocessed[view_index].get("sha256"),
                },
                "depth": {
                    "path": str(depth_path),
                    "format": "NumPy .npy, unmodified prediction-head values",
                    "dimensions": list(depth.shape),
                    "dtype": str(depth.dtype),
                    "units": "UNKNOWN",
                    "semantics": "raw HunyuanWorld-Mirror depth-head output; no normalization or color mapping applied",
                    "value_statistics": stats,
                },
                "confidence": {
                    "path": str(confidence_path) if confidence_path else None,
                    "semantics": "raw model depth-confidence head output; not thresholded or calibrated",
                },
                "finite_depth_mask": {
                    "path": str(finite_mask_path),
                    "semantics": "true where the raw depth value is finite; not a ground-truth validity label",
                },
                "point_export_mask": {
                    "path": str(export_mask_path) if export_mask_path else None,
                    "semantics": "existing point-export filter: finite depth, finite XYZ, and 0.1 < depth < 25.0; not ground-truth validity",
                },
                "camera": {
                    "pose_c2w": finite_json_array(poses[view_index]) if view_index < len(poses) else None,
                    "intrinsics": finite_json_array(intrinsics[view_index]) if view_index < len(intrinsics) else None,
                    "intrinsics_source": "WorldMirror prediction head" if view_index < len(intrinsics) else "UNKNOWN",
                },
            })

        metadata = {
            "schema_version": "1.0",
            "job_id": job_id,
            "model": {
                "id": model_id,
                "version": model_spec.get("version"),
                "name": model_spec.get("name"),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": model_spec.get("sha256"),
                "source_commit": model_spec.get("source_commit"),
            },
            "depth_unit_evidence": {
                "status": "conflicting repository metadata; units intentionally reported UNKNOWN",
                "registry_capability_claim": "metric_depth=true",
                "registry_manifest_claim": "meters",
                "adapter_semantics": "scene_normalized; metric-depth capability is false",
                "note": "Raw model output is preserved without asserting independent metric calibration.",
            },
            "output_contract": "depth/view_XX_raw.npy contains the original dense model-head values; no PNG normalization or point-cloud inversion was used.",
            "views": views_metadata,
        }
        metadata_path = depth_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        artifact_records["depth_metadata"] = {
            "path": str(metadata_path),
            "size_bytes": metadata_path.stat().st_size,
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
        return artifact_records

    def _process_geometry(
        self,
        inference_result: Dict[str, Any],
        preprocessed: List[Dict[str, Any]],
        model_id: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process raw inference output into structured scene data."""
        # Extract point cloud
        pts_xyz = inference_result.get("point_cloud_xyz", np.zeros((0, 3)))
        pts_rgb = inference_result.get("point_cloud_rgb", None)
        pts_normals = inference_result.get("point_cloud_normals", None)
        depth_maps = inference_result.get("depth_maps", [])

        # Create PointCloud
        pc = PointCloud(
            xyz=pts_xyz,
            rgb=pts_rgb,
            normals=pts_normals,
        )

        # Filter valid points
        if len(pc) > 0:
            valid_mask = self.geometry.filter_valid_points(
                pc.xyz,
                confidence=pc.confidence,
            )
            if valid_mask is not None and not valid_mask.all():
                pc = PointCloud(
                    xyz=pc.xyz[valid_mask],
                    rgb=pc.rgb[valid_mask] if pc.rgb is not None else None,
                    normals=pc.normals[valid_mask] if pc.normals is not None else None,
                    confidence=pc.confidence[valid_mask] if pc.confidence is not None else None,
                )

        # Voxel downsample if too many points
        max_points = options.get("downsample_points", 40000)
        if len(pc) > max_points:
            pc.xyz, attrs = self.geometry.voxel_downsample(
                pc.xyz,
                {"rgb": pc.rgb, "normals": pc.normals} if pc.rgb is not None or pc.normals is not None else None,
            )
            pc.rgb = attrs.get("rgb")
            pc.normals = attrs.get("normals")

        # Validate normals
        if pc.normals is not None:
            pc.normals = self.geometry.validate_normals(pc.normals, pc.xyz)

        # Build scene data structure
        scene_data = {
            "model_name": model_id,
            "model_version": self.inference.registry.get_model(model_id).get("version"),
            "depth_mode": "metric",
            "camera_intrinsics": {
                "fx": 1000.0, "fy": 1000.0, "cx": 0.5, "cy": 0.5
            },
            "point_cloud": {
                "xyz": pc.xyz.tolist() if len(pc) > 0 else [],
                "rgb": pc.rgb.tolist() if pc.rgb is not None else [],
                "normals": pc.normals.tolist() if pc.normals is not None else [],
            },
            "num_points": len(pc),
            "depth_range": [float(np.min(pts_xyz[:, 2])) if len(pts_xyz) > 0 else 0.0,
                           float(np.max(pts_xyz[:, 2])) if len(pts_xyz) > 0 else 0.0],
            "views_processed": len(preprocessed),
            "preprocessing_version": self.preprocessing.version,
            "reconstruction_timestamp": time.time(),
        }

        return scene_data

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job (cooperative)."""
        # This would need integration with the job queue
        logger.info(f"Cancellation requested for job {job_id}")
        return True


# Singleton
_reconstruction_service: Optional[ReconstructionService] = None


def get_reconstruction_service() -> ReconstructionService:
    global _reconstruction_service
    if _reconstruction_service is None:
        _reconstruction_service = ReconstructionService()
    return _reconstruction_service
