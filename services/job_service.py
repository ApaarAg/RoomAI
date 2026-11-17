"""Asynchronous Job Processing and GPU Lifecycle Service for RoomAI."""

import json
import math
from importlib.metadata import PackageNotFoundError, version as package_version
import platform
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
import numpy as np
from PIL import Image
import torch

from models.adapters import RECONSTRUCTION_REGISTRY
from schemas.scene import SceneGraph
from services.job_store import GLOBAL_JOB_STORE, JobStore
from utils.export_3d import (
    export_point_cloud_ply,
    export_colored_point_cloud_glb,
    extract_point_cloud_ply,
    extract_glb_point_cloud,
    validate_glb_bytes,
    validate_ply_bytes,
)
from utils.reconstruction_contract import (
    CAMERA_FRAME,
    CONTRACT_VERSION,
    depth_to_xyz,
    normalize_intrinsics,
    save_rgb_png,
    sha256_file,
    sha256_path,
    transform_intrinsics,
    validate_reconstruction,
    write_manifest,
)
from utils.logging_config import get_logger

logger = get_logger("JobService")
REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_ROOT / "artifacts" / "jobs"


class JobService:
    """Manages asynchronous reconstruction jobs, persistent queue, and GPU worker lifecycle."""

    def __init__(self, store: Optional[JobStore] = None):
        self.store = store or GLOBAL_JOB_STORE
        self.jobs_dir = JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue = queue.Queue()
        self._cancelled_jobs: set = set()
        self._active_job_id: Optional[str] = None
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="JobServiceWorker")
        self._worker_thread.start()

        # Clean up any leftover active jobs from past server process
        self.store.recover_interrupted_jobs()

    def submit_reconstruction_job(
        self,
        images: List[Image.Image],
        model_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Enqueues a new 3D reconstruction job and returns a unique job ID immediately.

        Args:
            images: List of 1 or more PIL Images.
            model_id: Target model adapter (defaults to 'roomai_composite_v1').
            options: Optional configuration dictionary.

        Returns:
            Unique string identifier for the job.
        """
        if not images:
            raise ValueError("Cannot submit reconstruction job without input images.")

        selected_model = model_id or RECONSTRUCTION_REGISTRY.get_default_model_id()
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing reconstruction job artifacts: {job_dir}")
        input_dir = job_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Persist input images safely to disk
        image_paths: List[str] = []
        source_files = list((options or {}).get("source_upload_paths", []))
        source_records = []
        source_dir = job_dir / "source"
        decoded_dir = job_dir / "rgb"
        for idx, img in enumerate(images):
            img_filename = f"view_{idx:02d}.png"
            img_path = input_dir / img_filename
            rgb_img = img.convert("RGB")
            save_rgb_png(rgb_img, img_path)
            decoded_path = decoded_dir / img_filename
            save_rgb_png(rgb_img, decoded_path)
            image_paths.append(str(img_path))
            source_record = {
                "view_index": idx,
                "decoded_rgb_path": str(decoded_path.relative_to(job_dir)),
                "decoded_rgb_sha256": sha256_file(decoded_path),
                "decoded_rgb_format": "PNG",
                "width": rgb_img.width,
                "height": rgb_img.height,
                "source_original_path": None,
                "source_original_sha256": None,
            }
            if idx < len(source_files) and Path(source_files[idx]).is_file():
                original_path = Path(source_files[idx])
                source_dir.mkdir(parents=True, exist_ok=True)
                extension = original_path.suffix.lower() or ".bin"
                preserved_path = source_dir / f"view_{idx:02d}{extension}"
                shutil.copy2(original_path, preserved_path)
                source_record["source_original_path"] = str(preserved_path.relative_to(job_dir))
                source_record["source_original_sha256"] = sha256_file(preserved_path)
                source_record["source_original_bytes"] = preserved_path.stat().st_size
            source_records.append(source_record)

        job_options = dict(options or {})
        job_options.pop("source_upload_paths", None)
        job_options["source_artifacts"] = source_records

        # Record initial QUEUED state in persistent DB
        self.store.create_job(
            job_id=job_id,
            model_id=selected_model,
            image_paths=image_paths,
            options=job_options,
        )

        # Push to in-memory processing queue
        self._queue.put((job_id, image_paths, selected_model, job_options))
        logger.info(f"Reconstruction job {job_id} enqueued successfully (model: {selected_model}, views: {len(images)}).")
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Requests cancellation of a queued or running job."""
        job = self.store.get_job(job_id)
        if not job:
            return False

        if job["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return False

        self._cancelled_jobs.add(job_id)
        if job["status"] == "QUEUED":
            self.store.update_job(
                job_id=job_id,
                status="CANCELLED",
                stage="Cancelled by user while queued",
                progress=0.0,
            )
            logger.info(f"Queued job {job_id} cancelled.")
            return True

        self.store.update_job(
            job_id=job_id,
            status="CANCELLED",
            stage="Cancellation requested by user",
            progress=job.get("progress", 0.0),
        )
        logger.info(f"Cancellation requested for active job {job_id}.")
        return True

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the current state, progress, and stage description for a job."""
        job = self.store.get_job(job_id)
        if not job:
            return None

        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "progress": job["progress"],
            "model_id": job["model_id"],
            "view_count": job["view_count"],
            "timing": job["timing"],
            "error": job["error"],
            "warnings": job["warnings"],
            "asset_acceptance": (job.get("results_summary") or {}).get("asset_acceptance"),
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns structured 3D scene data and object information for a completed job."""
        job = self.store.get_job(job_id)
        if not job:
            return None

        if job["status"] != "COMPLETED":
            return {
                "job_id": job_id,
                "status": job["status"],
                "is_completed": False,
                "message": f"Job is in state '{job['status']}', results not yet available.",
            }

        job_dir = self.jobs_dir / job_id
        scene_json_path = job_dir / "scene_graph.json"
        scene_data = None
        if scene_json_path.exists():
            try:
                scene_data = json.loads(scene_json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error(f"Failed to read scene_graph.json for job {job_id}: {exc}")

        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "is_completed": True,
            "model_id": job["model_id"],
            "timing": job["timing"],
            "results_summary": job["results_summary"],
            "artifacts": job["artifacts"],
            "scene_graph": scene_data,
        }

    def get_export_file(self, job_id: str, fmt: str) -> Tuple[Optional[Path], Optional[str]]:
        """Resolves file path and mime-type for exported 3D assets.
        
        Args:
            job_id: Job identifier.
            fmt: 'glb' or 'ply'.

        Returns:
            Tuple of (file_path: Optional[Path], media_type: Optional[str]).
        """
        job = self.store.get_job(job_id)
        if not job or job["status"] != "COMPLETED":
            return None, None
        acceptance = ((job.get("results_summary") or {}).get("asset_acceptance") or {}).get("status")
        if acceptance == "REJECTED":
            return None, None

        job_dir = self.jobs_dir / job_id
        fmt_clean = fmt.lower().strip()
        if fmt_clean == "glb":
            path = job_dir / "scene_reconstruction.glb"
            if path.exists():
                return path, "model/gltf-binary"
        elif fmt_clean == "ply":
            path = job_dir / "scene_reconstruction.ply"
            if path.exists():
                return path, "application/octet-stream"

        return None, None

    def _persist_and_validate_bundle(
        self,
        scene: SceneGraph,
        images: List[Image.Image],
        job_dir: Path,
        model_id: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist exact model evidence and validated deterministic point exports."""
        rgb = np.asarray(images[0].convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        depth_value = scene.depth_map
        depth: Optional[np.ndarray] = None
        native_depth_path = None
        if depth_value is not None:
            raw_depth_array = np.asarray(depth_value, dtype=np.float32)
            squeezed_depth = raw_depth_array.squeeze()
            if squeezed_depth.ndim == 2:
                depth = squeezed_depth
            else:
                native_depth_path = job_dir / "model_native_depth.npy"
                np.save(native_depth_path, raw_depth_array, allow_pickle=False)

        depth_semantics = (
            "scene_normalized" if "hunyuan" in model_id.lower()
            else "metric_model_estimate" if scene.depth_mode == "metric"
            else "relative_inverse"
        )
        intrinsics, intrinsics_source = normalize_intrinsics(
            width, height, options.get("camera_intrinsics")
        )
        native_outputs_path = None
        raw_native_outputs = getattr(scene, "model_native_outputs", None)
        if isinstance(raw_native_outputs, dict):
            native_arrays: Dict[str, np.ndarray] = {}

            def collect_native(prefix: str, value: Any) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        collect_native(f"{prefix}_{key}" if prefix else str(key), item)
                elif isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                    native_arrays[prefix or "value"] = value
                elif isinstance(value, (list, tuple)):
                    try:
                        array = np.asarray(value)
                        if array.dtype.kind in "biuf":
                            native_arrays[prefix or "value"] = array
                    except (TypeError, ValueError):
                        return

            collect_native("", raw_native_outputs)
            if native_arrays:
                native_outputs_path = job_dir / "model_native_outputs.npz"
                np.savez_compressed(native_outputs_path, **native_arrays)
        point_indices = np.empty((0,), dtype=np.int64)
        canonical = False
        validation: Dict[str, Any] = {"is_valid": False, "errors": []}
        xyz = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.float32)
        depth_range = None

        rgb_path = job_dir / "rgb" / "view_00.png"
        save_rgb_png(images[0], rgb_path)
        source_records = options.get("source_artifacts", [])
        source_record = source_records[0] if source_records else {}
        calibration_evidence = None
        evidence_relative = source_record.get("camera_calibration_evidence_path")
        if options.get("camera_calibration_verified") is True and evidence_relative:
            evidence_path = (job_dir / evidence_relative).resolve()
            try:
                evidence_path.relative_to(job_dir.resolve())
                evidence_hash = sha256_file(evidence_path)
                if evidence_hash == source_record.get("camera_calibration_evidence_sha256"):
                    calibration_evidence = {
                        "path": str(evidence_path.relative_to(job_dir)),
                        "sha256": evidence_hash,
                        "status": "dataset_or_calibration_evidence_hash_verified",
                    }
            except (OSError, ValueError):
                calibration_evidence = None
        if depth is not None:
            np.save(job_dir / "depth_raw.npy", depth.astype(np.float32, copy=False), allow_pickle=False)
            if depth_semantics == "metric_model_estimate":
                mask = np.isfinite(depth) & (depth > 0.05) & (depth < 30.0)
            else:
                mask = np.isfinite(depth) & (depth > 0)
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            mask_image.save(job_dir / "validity_mask.png", format="PNG")
            finite_values = depth[mask]
            if finite_values.size:
                depth_range = {"min": float(finite_values.min()), "max": float(finite_values.max())}

            if "hunyuan" not in model_id.lower() and depth_semantics == "metric_model_estimate" and depth.shape == (height, width) and len(images) == 1:
                max_points = int(options.get("canonical_max_points", 200000))
                xyz, colors, point_indices = depth_to_xyz(
                    depth, rgb, intrinsics, mask, max_points=max_points,
                    seed=int(options.get("sampling_seed", 0)),
                )
                canonical = True
                validation = validate_reconstruction(
                    rgb, depth, mask, xyz, colors, point_indices, intrinsics, depth_semantics
                )
            elif depth.shape != (height, width):
                validation["errors"].append("depth_not_aligned_to_source_rgb")
            else:
                validation["errors"].append("depth_semantics_not_eligible_for_metric_unprojection")
        else:
            validation["errors"].append("raw_depth_unavailable")

        # Preserve untrusted/native model points for forensics when canonical unprojection is unavailable.
        if not canonical and scene.point_cloud:
            native_xyz = np.asarray(scene.point_cloud.get("xyz", []), dtype=np.float32)
            native_rgb = np.asarray(scene.point_cloud.get("rgb", []), dtype=np.float32)
            if native_xyz.ndim == 2 and native_xyz.shape[1] == 3 and native_rgb.shape == native_xyz.shape:
                keep = np.isfinite(native_xyz).all(axis=1) & np.isfinite(native_rgb).all(axis=1)
                xyz = native_xyz[keep]
                colors = np.clip(native_rgb[keep], 0.0, 1.0)
                np.savez_compressed(job_dir / "model_native_geometry.npz", xyz=xyz, rgb=colors)
                validation["native_model_geometry_preserved"] = True
                if len(xyz):
                    span = np.ptp(xyz, axis=0)
                    validation["native_bounds_span"] = span.tolist()
                    covariance = np.linalg.eigvalsh(np.cov(xyz, rowvar=False)) if len(xyz) > 3 else np.zeros(3)
                    validation["native_covariance_eigenvalues"] = np.maximum(covariance, 0).tolist()
                    if len(np.flatnonzero(span > 1e-4)) < 2 or covariance[-2] <= max(covariance[-1] * 1e-6, 1e-10):
                        validation["errors"].append("collapsed_or_rank_deficient_native_geometry")

        point_cloud_path = job_dir / "point_cloud_canonical.npz"
        np.savez_compressed(
            point_cloud_path,
            xyz=xyz.astype(np.float32),
            rgb=colors.astype(np.float32),
            source_pixel_indices=point_indices,
        )
        ply_path = job_dir / "scene_reconstruction.ply"
        glb_path = job_dir / "scene_reconstruction.glb"
        mobile_glb_path = job_dir / "scene_reconstruction_mobile.glb"
        point_units = "m" if depth_semantics == "metric_model_estimate" else "model_native"
        desktop_limit = int(options.get("desktop_max_points", options.get("downsample_points", 40000)))
        mobile_limit = int(options.get("mobile_max_points", 15000))
        sampling_seed = int(options.get("sampling_seed", 0))
        ply_bytes = export_point_cloud_ply(
            xyz=xyz.tolist(), rgb=colors.tolist(), output_path=ply_path,
            units=point_units,
            coordinate_frame=CAMERA_FRAME if canonical else "model_native_unverified",
        )
        glb_bytes = None
        mobile_glb_bytes = None
        if len(xyz):
            glb_bytes = export_colored_point_cloud_glb(
                xyz, colors, glb_path, max_points=desktop_limit,
                sampling_seed=sampling_seed,
                units=point_units,
            )
            mobile_glb_bytes = export_colored_point_cloud_glb(
                xyz, colors, mobile_glb_path, max_points=mobile_limit,
                sampling_seed=sampling_seed,
                units=point_units,
            )

        ply_validation = validate_ply_bytes(ply_bytes)
        glb_validation = validate_glb_bytes(glb_bytes) if glb_bytes else {"is_valid": False, "error": "no points to export"}
        mobile_validation = validate_glb_bytes(mobile_glb_bytes) if mobile_glb_bytes else {"is_valid": False, "error": "no points to export"}
        khronos_validation: Dict[str, Any] = {
            "status": "not_run",
            "validator": "Khronos glTF Validator",
            "pinned_version": "2.0.0-dev.3.10",
            "assets": {},
        }
        node_executable = shutil.which("node")
        khronos_script = REPO_ROOT / "scripts" / "validate_glb_khronos.mjs"
        if node_executable and khronos_script.is_file():
            all_khronos_pass = True
            for label, path, content in (
                ("desktop", glb_path, glb_bytes),
                ("mobile", mobile_glb_path, mobile_glb_bytes),
            ):
                if not content:
                    khronos_validation["assets"][label] = {"status": "missing_asset"}
                    all_khronos_pass = False
                    continue
                try:
                    result = subprocess.run(
                        [node_executable, str(khronos_script), str(path)],
                        capture_output=True, text=True, timeout=60, check=False,
                    )
                    parsed = json.loads(result.stdout) if result.stdout.strip() else None
                    if parsed is not None:
                        report_path = job_dir / f"khronos_{label}_report.json"
                        report_path.write_text(json.dumps(parsed, indent=2, allow_nan=False), encoding="utf-8")
                        asset_result = {
                            "status": "passed" if result.returncode == 0 else "failed",
                            "exit_code": result.returncode,
                            "report_path": report_path.name,
                            "report_sha256": sha256_file(report_path),
                            "report": parsed,
                        }
                    else:
                        asset_result = {
                            "status": "unavailable" if result.returncode == 2 else "failed_to_run",
                            "exit_code": result.returncode,
                            "stderr": result.stderr[-2000:],
                        }
                    khronos_validation["assets"][label] = asset_result
                    if asset_result["status"] != "passed":
                        all_khronos_pass = False
                except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                    khronos_validation["assets"][label] = {"status": "failed_to_run", "error": str(exc)}
                    all_khronos_pass = False
            khronos_validation["status"] = "completed" if all_khronos_pass else "completed_with_failures"
        else:
            khronos_validation["status"] = "unavailable"
            khronos_validation["error"] = "node_or_validator_script_unavailable"
        if khronos_validation["status"] != "completed":
            validation["errors"].append("khronos_gltf_external_validation_failed_or_unavailable")
        parity = {"ply": False, "desktop": False, "mobile": False}
        if ply_validation.get("is_valid"):
            ply_xyz, ply_rgb = extract_point_cloud_ply(ply_bytes)
            parity["ply"] = (
                ply_xyz.shape == xyz.shape
                and np.allclose(ply_xyz, xyz, rtol=0, atol=1e-7)
                and np.allclose(ply_rgb, np.rint(colors * 255.0) / 255.0, rtol=0, atol=1e-7)
            )
        for label, content, limit in (("desktop", glb_bytes, desktop_limit),
                                      ("mobile", mobile_glb_bytes, mobile_limit)):
            if content and validate_glb_bytes(content).get("is_valid"):
                exported_xyz, exported_rgb = extract_glb_point_cloud(content)
                if len(xyz) > limit:
                    indices = np.sort(np.random.default_rng(sampling_seed).choice(len(xyz), size=limit, replace=False))
                    expected_xyz, expected_rgb = xyz[indices], np.rint(colors[indices] * 255.0) / 255.0
                else:
                    expected_xyz, expected_rgb = xyz, np.rint(colors * 255.0) / 255.0
                parity[label] = (
                    exported_xyz.shape == expected_xyz.shape
                    and np.allclose(exported_xyz, expected_xyz, rtol=0, atol=1e-7)
                    and np.allclose(exported_rgb, expected_rgb, rtol=0, atol=1 / 255 + 1e-6)
                )

        if len(xyz) == 0:
            validation["errors"].append("empty_point_cloud")
        if not ply_validation.get("is_valid"):
            validation["errors"].append("ply_validation_failed")
        if not parity["ply"]:
            validation["errors"].append("ply_canonical_point_parity_failed")
        if glb_bytes and (not glb_validation.get("is_valid") or not parity["desktop"]):
            validation["errors"].append("glb_validation_or_ply_parity_failed")
        if mobile_glb_bytes and (not mobile_validation.get("is_valid") or not parity["mobile"]):
            validation["errors"].append("mobile_glb_validation_or_parity_failed")

        if not canonical:
            validation["is_valid"] = False
            validation["errors"] = list(dict.fromkeys(validation.get("errors", [])))
        elif validation.get("errors"):
            validation["is_valid"] = False

        if validation.get("is_valid"):
            acceptance = {
                "status": "REVIEW_REQUIRED",
                "reasons": [
                    "Monocular metric depth is a model estimate, not independent metric calibration.",
                    "Human visual review and real-camera/reference-distance validation are pending.",
                ],
            }
        else:
            acceptance = {"status": "REJECTED", "reasons": validation.get("errors", ["validation_failed"])}

        scene_json_path = job_dir / "scene_graph.json"
        scene_dict = scene.model_dump(exclude={"point_cloud", "depth_map", "heatmap_image", "model_native_outputs"})
        def json_default(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if hasattr(value, "model_dump"):
                return value.model_dump()
            raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

        def make_json_safe(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return make_json_safe(value.tolist())
            if isinstance(value, np.generic):
                return make_json_safe(value.item())
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {str(key): make_json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [make_json_safe(item) for item in value]
            if hasattr(value, "model_dump"):
                return make_json_safe(value.model_dump())
            return value

        scene_json_path.write_text(
            json.dumps(make_json_safe(scene_dict), indent=2, allow_nan=False, default=json_default), encoding="utf-8"
        )
        intrinsics_record = {
            **intrinsics,
            "units": "pixels",
            "source": "verified_calibration_evidence" if calibration_evidence else intrinsics_source,
            "calibrated": calibration_evidence is not None,
            "calibration_status": "evidence_hash_verified" if calibration_evidence else (
                "estimated" if intrinsics_source.startswith("estimated_") else "provided_unverified"
            ),
            "coordinate_frame": CAMERA_FRAME,
        }
        if "hunyuan" in model_id.lower():
            crop_size = min(width, height)
            crop_left = (width - crop_size) // 2
            crop_top = (height - crop_size) // 2
            model_scale = 518.0 / crop_size
            preprocessing_transforms = [
                {"operation": "center_square_crop", "left_px": crop_left, "top_px": crop_top,
                 "width_px": crop_size, "height_px": crop_size},
                {"operation": "resize_bilinear", "input_size_px": [crop_size, crop_size],
                 "output_size_px": [518, 518]},
            ]
            model_input_intrinsics = transform_intrinsics(
                intrinsics, scale_x=model_scale, scale_y=model_scale,
                crop_left=crop_left, crop_top=crop_top,
            )
        else:
            preprocessing_transforms = [
                {"operation": "RGB decode", "input_size_px": [width, height], "output_size_px": [width, height]}
            ]
            model_input_intrinsics = intrinsics
        depth_record = {
            "semantics": depth_semantics,
            "units": "m" if depth_semantics == "metric_model_estimate" else None,
            "scale_to_meters": 1.0 if depth_semantics == "metric_model_estimate" else None,
            "scale_status": "model_estimate_not_independently_calibrated" if depth_semantics == "metric_model_estimate" else "not_metric",
            "range": depth_range,
            "raw_path": "depth_raw.npy" if depth is not None else ("model_native_depth.npy" if native_depth_path is not None else None),
            "validity_mask_path": "validity_mask.png" if depth is not None else None,
            "alignment_status": "aligned_to_decoded_rgb" if depth is not None and depth.shape == (height, width) and len(images) == 1 else "unaligned_or_unknown",
        }
        view_records = []
        for index, record in enumerate(source_records):
            decoded_relative = record.get("decoded_rgb_path", f"rgb/view_{index:02d}.png")
            decoded_path = job_dir / decoded_relative
            view_records.append({
                "view_index": index,
                "decoded_rgb_path": decoded_relative,
                "decoded_rgb_sha256": sha256_file(decoded_path) if decoded_path.is_file() else None,
                "source_original_path": record.get("source_original_path"),
                "source_original_sha256": record.get("source_original_sha256"),
                "width": record.get("width", images[index].width if index < len(images) else None),
                "height": record.get("height", images[index].height if index < len(images) else None),
            })
        if not view_records:
            view_records.append({
                "view_index": 0,
                "decoded_rgb_path": str(rgb_path.relative_to(job_dir)),
                "decoded_rgb_sha256": sha256_file(rgb_path),
                "source_original_path": None,
                "source_original_sha256": None,
                "width": width,
                "height": height,
            })
        manifest = {
            "schema": "roomai.exhibit_manifest",
            "schema_version": CONTRACT_VERSION,
            "job_id": job_dir.name,
            "model_id": model_id,
            "model_name": scene.model_name,
            "model_version": "roomai_composite_v1.4.0" if "composite" in model_id.lower() else "hunyuan_world_mirror_1.0_research",
            "source": source_record,
            "views": view_records,
            "decoded_rgb": {
                "path": str(rgb_path.relative_to(job_dir)),
                "sha256": sha256_file(rgb_path),
                "width": width,
                "height": height,
                "color_space": "sRGB",
                "format": "PNG",
            },
            "preprocessing": {
                "transforms": preprocessing_transforms,
                "camera_intrinsics_after_preprocessing": model_input_intrinsics,
                "intrinsic_pixel_convention": "pixel_center_coordinates",
            },
            "camera": {"intrinsics": intrinsics_record, "calibration_evidence": calibration_evidence,
                       "extrinsics": None, "extrinsics_status": "single_view_camera_origin" if len(images) == 1 else "multiview_pose_not_validated"},
            "coordinate_frame": CAMERA_FRAME if canonical else ("model_native_unverified" if len(images) == 1 else "multiview_unverified"),
            "depth": depth_record,
            "geometry": {
                "representation": "colored_single_view_point_cloud" if canonical else "unverified_model_native_point_cloud",
                "complete_mesh": False,
                "units": "m" if canonical else None,
                "point_count": int(len(xyz)),
                "source_pixel_mapping": "row_major_flat_index" if canonical else None,
                "sampling": {"algorithm": "numpy.default_rng.choice sorted indices", "seed": sampling_seed, "canonical_limit": int(options.get("canonical_max_points", 200000)), "desktop_limit": desktop_limit, "mobile_limit": mobile_limit} if canonical else None,
                "bounds": validation.get("bounds", {"span": validation.get("native_bounds_span")}),
            },
            "model_provenance": {
                "depth_checkpoint_config_sha256": sha256_file(REPO_ROOT / "config" / "model_config.yaml") if (REPO_ROOT / "config" / "model_config.yaml").is_file() else None,
                "checkpoint_sha256": None,
                "checkpoint_status": "not_resolved_by_adapter" if "hunyuan" in model_id.lower() else "local_checkpoint_digest_not_recorded",
                "checkpoint_digests": [],
                "license_status": "review_required_non_commercial_research_weights" if "hunyuan" in model_id.lower() else "not_assessed_in_job_worker",
            },
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": {},
            },
            "code_provenance": [],
            "multiview_provenance": {
                "telemetry": scene.multiview_telemetry,
                "per_view": scene.multiview_per_view_provenance,
            } if len(images) > 1 else None,
            "validation": validation,
            "acceptance": acceptance,
            "assets": {},
        }
        for package in ("numpy", "Pillow", "torch", "transformers", "ultralytics"):
            try:
                manifest["runtime"]["packages"][package] = package_version(package)
            except PackageNotFoundError:
                manifest["runtime"]["packages"][package] = None
        for relative in (
            "models/adapters/roomai_composite.py", "services/scene_fusion_service.py",
            "models/scene3d_model.py", "utils/reconstruction_contract.py",
            "utils/export_3d.py", "services/job_service.py",
            "services/hunyuan_mirror_service.py", "models/adapters/hunyuan_mirror_adapter.py",
        ):
            code_path = REPO_ROOT / relative
            if code_path.is_file():
                manifest["code_provenance"].append({"path": relative, "sha256": sha256_file(code_path)})
        if "hunyuan" in model_id.lower():
            try:
                adapter = RECONSTRUCTION_REGISTRY.get_adapter(model_id)
                checkpoint = Path(adapter.checkpoint_path)
                if checkpoint.is_file():
                    manifest["model_provenance"]["checkpoint_sha256"] = sha256_file(checkpoint)
                    manifest["model_provenance"]["checkpoint_status"] = "sha256_recorded_license_review_required"
                    config_text = (REPO_ROOT / "config" / "model_config.yaml").read_text(encoding="utf-8")
                    import re
                    match = re.search(r'expected_sha256:\s*["\']?([0-9a-fA-F]{64})', config_text)
                    expected_sha = match.group(1).lower() if match else None
                    actual_sha = manifest["model_provenance"]["checkpoint_sha256"].lower()
                    manifest["model_provenance"]["expected_checkpoint_sha256"] = expected_sha
                    manifest["model_provenance"]["checkpoint_hash_matches_expected"] = (
                        actual_sha == expected_sha if expected_sha else None
                    )
                    if expected_sha and actual_sha != expected_sha:
                        validation["errors"].append("hunyuan_checkpoint_sha_mismatch")
            except (AttributeError, OSError, KeyError):
                pass
        elif "composite" in model_id.lower():
            for label, relative in (
                ("detector", "checkpoints/indoor_detector_a3_yolo11s_800px_best.pt"),
                ("segmenter", "checkpoints/yolo11n-seg.pt"),
                ("depth_estimator", "checkpoints/depth_anything_v2_metric_indoor_small"),
            ):
                checkpoint = REPO_ROOT / relative
                if checkpoint.exists():
                    manifest["model_provenance"]["checkpoint_digests"].append({
                        "component": label,
                        "path": relative,
                        "sha256": sha256_path(checkpoint),
                    })
            manifest["model_provenance"]["checkpoint_status"] = (
                "all_runtime_checkpoints_hashed" if len(manifest["model_provenance"]["checkpoint_digests"]) == 3
                else "one_or_more_runtime_checkpoints_unavailable"
            )
        if validation.get("errors"):
            acceptance = {"status": "REJECTED", "reasons": list(dict.fromkeys(validation["errors"]))}
        manifest["acceptance"] = acceptance
        manifest["validation"] = validation

        asset_paths = {
            "decoded_rgb": rgb_path,
            "point_cloud_npz": point_cloud_path,
            "ply": ply_path,
            "scene_graph": scene_json_path,
        }
        for label in ("desktop", "mobile"):
            report_record = khronos_validation.get("assets", {}).get(label, {})
            report_relative = report_record.get("report_path")
            if report_relative:
                asset_paths[f"khronos_{label}_validation_report"] = job_dir / report_relative
        if depth is not None:
            asset_paths["raw_depth"] = job_dir / "depth_raw.npy"
            asset_paths["validity_mask"] = job_dir / "validity_mask.png"
        if native_outputs_path is not None:
            asset_paths["model_native_outputs"] = native_outputs_path
        if native_depth_path is not None:
            asset_paths["model_native_depth"] = native_depth_path
        if calibration_evidence is not None:
            asset_paths["camera_calibration_evidence"] = job_dir / calibration_evidence["path"]
        if glb_bytes:
            asset_paths["desktop_glb"] = glb_path
            asset_paths["mobile_glb"] = mobile_glb_path
        for key, path in asset_paths.items():
            manifest["assets"][key] = {"path": str(path.relative_to(job_dir)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        manifest_path = job_dir / "exhibit_manifest.json"
        write_manifest(manifest_path, manifest)
        validation_path = job_dir / "validation_report.json"
        validation_report = {
            "validation": validation,
            "acceptance": acceptance,
            "ply": ply_validation,
            "desktop_glb": {k: v for k, v in glb_validation.items() if k not in ("binary", "json_data")},
            "mobile_glb": {k: v for k, v in mobile_validation.items() if k not in ("binary", "json_data")},
            "ply_glb_parity": parity,
            "external_khronos_validator": khronos_validation,
        }
        validation_path.write_text(json.dumps(validation_report, indent=2, allow_nan=False), encoding="utf-8")
        manifest["assets"]["validation_report"] = {"path": "validation_report.json", "sha256": sha256_file(validation_path)}
        write_manifest(manifest_path, manifest)

        artifacts = {
            "rgb": {"path": str(rgb_path), "sha256": sha256_file(rgb_path)},
            "source_original": source_record,
            "depth_raw": {"path": str(job_dir / "depth_raw.npy") if depth is not None else None},
            "validity_mask": {"path": str(job_dir / "validity_mask.png") if depth is not None else None},
            "point_cloud_npz": {"path": str(point_cloud_path), "point_count": int(len(xyz))},
            "ply": {"path": str(ply_path), "size_bytes": len(ply_bytes), "is_valid": ply_validation.get("is_valid"), "vertex_count": ply_validation.get("vertex_count")},
            "glb": {"path": str(glb_path) if glb_bytes else None, "size_bytes": len(glb_bytes) if glb_bytes else 0, "is_valid": glb_validation.get("is_valid"), "point_count": len(extract_glb_point_cloud(glb_bytes)[0]) if glb_bytes and glb_validation.get("is_valid") else 0},
            "glb_mobile": {"path": str(mobile_glb_path) if mobile_glb_bytes else None, "size_bytes": len(mobile_glb_bytes) if mobile_glb_bytes else 0, "is_valid": mobile_validation.get("is_valid")},
            "exhibit_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "validation_report": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
            "scene_graph_json": str(scene_json_path),
        }
        if native_outputs_path is not None:
            artifacts["model_native_outputs"] = {"path": str(native_outputs_path), "sha256": sha256_file(native_outputs_path)}
        if native_depth_path is not None:
            artifacts["model_native_depth"] = {"path": str(native_depth_path), "sha256": sha256_file(native_depth_path)}
        return {
            "artifacts": artifacts,
            "acceptance": acceptance,
            "depth_semantics": depth_semantics,
            "depth_range": depth_range,
        }

    def _worker_loop(self) -> None:
        """Continuous background worker executing queued jobs with GPU safety."""
        while True:
            try:
                task = self._queue.get()
                if task is None:
                    break

                job_id, image_paths, model_id, options = task
                if job_id in self._cancelled_jobs:
                    logger.info(f"Skipping cancelled job {job_id}.")
                    self._queue.task_done()
                    continue

                self._active_job_id = job_id
                self._execute_job(job_id, image_paths, model_id, options)
                self._active_job_id = None
                self._queue.task_done()
            except Exception as exc:
                logger.critical(f"Unexpected error in JobService worker loop: {exc}", exc_info=True)
                time.sleep(1.0)

    def _execute_job(
        self,
        job_id: str,
        image_paths: List[str],
        model_id: str,
        options: Dict[str, Any],
    ) -> None:
        """Executes full pipeline for a single reconstruction job."""
        t_start = time.perf_counter()
        started_time = time.time()
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Executing job {job_id} using model '{model_id}'...")
        self.store.update_job(
            job_id=job_id,
            status="VALIDATING",
            stage="Validating input images and checking hardware readiness",
            progress=0.10,
            timing_updates={"started_at": started_time},
        )

        try:
            # 1. Load input images
            images: List[Image.Image] = []
            for path_str in image_paths:
                p = Path(path_str)
                if not p.exists():
                    raise FileNotFoundError(f"Input image not found: {path_str}")
                img = Image.open(p).convert("RGB")
                images.append(img)

            # Check cancellation
            if job_id in self._cancelled_jobs:
                self.store.update_job(job_id=job_id, status="CANCELLED", stage="Cancelled by user")
                return

            # 2. Retrieve model adapter
            self.store.update_job(
                job_id=job_id,
                status="PROCESSING",
                stage="Acquiring GPU execution lock and loading models",
                progress=0.20,
            )

            adapter = RECONSTRUCTION_REGISTRY.get_adapter(model_id)
            is_avail, avail_reason = adapter.is_available()
            if not is_avail:
                raise RuntimeError(f"Selected model '{model_id}' is unavailable: {avail_reason}")

            valid_in, err_in = adapter.validate_inputs(images)
            if not valid_in:
                raise ValueError(f"Input validation rejected: {err_in}")

            # 3. Progress callback closure
            def on_progress(stage_name: str, frac: float):
                if job_id in self._cancelled_jobs:
                    raise InterruptedError("Job cancelled by user during inference.")
                self.store.update_job(
                    job_id=job_id,
                    status=stage_name,
                    stage=f"Inference in progress ({stage_name.lower()})",
                    progress=round(frac, 2),
                )

            # 4. Execute 3D reconstruction
            scene: SceneGraph = adapter.reconstruct(
                images=images,
                options=options,
                progress_callback=on_progress,
            )

            # Check cancellation
            if job_id in self._cancelled_jobs:
                self.store.update_job(job_id=job_id, status="CANCELLED", stage="Cancelled by user")
                return

            # 5. Persist canonical source evidence, geometry, exports, and validation report.
            self.store.update_job(
                job_id=job_id,
                status="EXPORTING",
                stage="Persisting reconstruction evidence and validating exports",
                progress=0.90,
            )

            bundle = self._persist_and_validate_bundle(scene, images, job_dir, model_id, options)

            # 6. Build summary
            t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
            results_summary = {
                "object_count": len(scene.objects),
                "surface_count": len(scene.surfaces),
                "valid_points": bundle["artifacts"]["point_cloud_npz"]["point_count"],
                "depth_range": bundle["depth_range"],
                "depth_semantics": bundle["depth_semantics"],
                "asset_acceptance": bundle["acceptance"],
                "room_layout": scene.room_layout,
                "objects": [
                    {
                        "object_id": obj.object_id,
                        "semantic_class": obj.semantic_class,
                        "confidence": obj.detection_confidence,
                        "extent_m": obj.geometry.extent if obj.geometry else None,
                        "centroid_m": obj.geometry.centroid if obj.geometry else None,
                        "is_reliable": obj.geometry.is_dimension_reliable if obj.geometry else True,
                    }
                    for obj in scene.objects
                ],
            }

            artifacts_summary = bundle["artifacts"]

            # 7. Final completion update
            self.store.update_job(
                job_id=job_id,
                status="COMPLETED",
                stage=f"Inference/export completed; asset acceptance: {bundle['acceptance']['status']}",
                progress=1.00,
                timing_updates={"completed_at": time.time(), "elapsed_ms": t_elapsed_ms},
                results_summary=results_summary,
                artifacts=artifacts_summary,
            )
            logger.info(
                f"Job {job_id} COMPLETED in {t_elapsed_ms:.2f} ms "
                f"({len(scene.objects)} objects, {scene.n_valid_points} points; "
                f"asset={bundle['acceptance']['status']})."
            )

        except InterruptedError:
            self.store.update_job(job_id=job_id, status="CANCELLED", stage="Cancelled by user")
            logger.info(f"Job {job_id} cancelled during execution.")
        except torch.cuda.OutOfMemoryError as oom:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            err_dict = {
                "code": "CUDA_OUT_OF_MEMORY",
                "message": f"GPU memory exhausted: {oom}",
                "remediation": "Reduce input resolution, decrease view count, or select a smaller model.",
            }
            self.store.update_job(job_id=job_id, status="FAILED", stage="Failed due to GPU out-of-memory", error=err_dict)
            logger.error(f"Job {job_id} failed with CUDA OOM: {oom}")
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            err_dict = {
                "code": type(exc).__name__,
                "message": str(exc),
                "remediation": "Inspect server logs for complete stack trace.",
            }
            self.store.update_job(job_id=job_id, status="FAILED", stage=f"Failed: {exc}", error=err_dict)
            logger.error(f"Job {job_id} failed with exception: {exc}", exc_info=True)


# Global singleton instance
GLOBAL_JOB_SERVICE = JobService()

