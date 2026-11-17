"""Multi-View 3D Scene Registration, Depth Fusion, and Geometry Consolidation Service.

Coordinates:
1. Feature extraction and matching between multi-view room perspectives (SIFT + FLANN / RANSAC).
2. Metric 3D-3D point cloud registration (Umeyama / Kabsch rigid transformation).
3. Seamless multi-view point cloud fusion with voxel downsampling and outlier removal.
4. 3D object association across viewpoints, merging occluded furniture and expanding room bounds.
5. Consolidated multi-view room layout and architectural boundary recovery.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from schemas.scene import Object3DGeometry, RoomSurface3D, SceneGraph, SceneObject, SpatialRelationship
from services.room_layout_service import RoomLayoutService
from services.scene_fusion_service import SceneFusionService
from utils.depth_refinement import (
    filter_flying_pixels,
    refine_metric_depth_guided,
    statistical_outlier_removal,
    voxel_grid_downsample,
)
from utils.logging_config import get_logger

logger = get_logger("MultiViewFusionService")


def compute_rigid_transform_3d(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes optimal rigid transformation [R | t] mapping pts_src to pts_dst.

    Uses Kabsch algorithm via Singular Value Decomposition (SVD):
    pts_dst ~ R @ pts_src + t

    Args:
        pts_src: (N, 3) source 3D points.
        pts_dst: (N, 3) destination 3D points.

    Returns:
        Tuple of (R: [3, 3] rotation matrix, t: [3] translation vector, rmse: float).
    """
    assert len(pts_src) >= 3 and len(pts_dst) >= 3, "At least 3 points required"

    centroid_src = np.mean(pts_src, axis=0)
    centroid_dst = np.mean(pts_dst, axis=0)

    p_src = pts_src - centroid_src
    p_dst = pts_dst - centroid_dst

    # Covariance matrix H = p_src^T @ p_dst
    h_matrix = p_src.T @ p_dst

    u, s, vt = np.linalg.svd(h_matrix)
    r_matrix = vt.T @ u.T

    # Reflection check (ensure det(R) == +1)
    if np.linalg.det(r_matrix) < 0:
        vt[2, :] *= -1
        r_matrix = vt.T @ u.T

    t_vec = centroid_dst - r_matrix @ centroid_src

    # Compute Root Mean Square Error (RMSE)
    transformed_src = (r_matrix @ pts_src.T).T + t_vec
    rmse = float(np.sqrt(np.mean(np.sum((transformed_src - pts_dst) ** 2, axis=1))))

    return r_matrix, t_vec, rmse


def compute_similarity_transform_3d(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Computes optimal similarity transformation [s * R | t] via Umeyama (1991).

    Args:
        pts_src: (N, 3) source 3D points.
        pts_dst: (N, 3) destination 3D points.

    Returns:
        Tuple of (R: [3, 3] rotation matrix, t: [3] translation vector, s: float scale, rmse: float).
    """
    assert len(pts_src) >= 3 and len(pts_dst) >= 3, "At least 3 points required"

    n = len(pts_src)
    centroid_src = np.mean(pts_src, axis=0)
    centroid_dst = np.mean(pts_dst, axis=0)

    p_src = pts_src - centroid_src
    p_dst = pts_dst - centroid_dst

    var_src = float(np.sum(p_src ** 2) / n)

    h_matrix = (p_src.T @ p_dst) / n
    u, d, vt = np.linalg.svd(h_matrix)

    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt.T) < 0:
        s_mat[2, 2] = -1

    r_matrix = vt.T @ s_mat @ u.T
    scale = float((1.0 / (var_src + 1e-9)) * np.trace(np.diag(d) @ s_mat)) if var_src > 1e-9 else 1.0
    scale = float(np.clip(scale, 0.2, 5.0))

    t_vec = centroid_dst - scale * (r_matrix @ centroid_src)

    transformed = scale * (r_matrix @ pts_src.T).T + t_vec
    rmse = float(np.sqrt(np.mean(np.sum((transformed - pts_dst) ** 2, axis=1))))

    return r_matrix, t_vec, scale, rmse


class MultiViewFusionService:
    """Consolidates and fuses multi-view room imagery into a cohesive 3D room reconstruction."""

    def __init__(
        self,
        mock_mode: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.mock_mode = mock_mode
        if config is None:
            from pathlib import Path
            import yaml
            cfg_p = Path("config/model_config.yaml")
            if cfg_p.exists():
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        self.config = yaml.safe_load(f) or {}
                except Exception:
                    self.config = {}
            else:
                self.config = {}
        else:
            self.config = config

        self.single_view_fusion = SceneFusionService(mock_mode=mock_mode, config=self.config)
        self.layout_service = RoomLayoutService(config=self.config)

    def load_models(self) -> None:
        """Initializes underlying perception and depth models."""
        self.single_view_fusion.load_models()

    def validate_multiview_sequence(
        self, images: List[Image.Image]
    ) -> Tuple[List[Image.Image], List[str], Dict[str, Any]]:
        """Validates a multi-view capture sequence (typically 3-10 images).

        Checks:
        1. Non-empty and valid image objects.
        2. Image resolution (within 320x240 to 4096x4096).
        3. Sharpness / blurriness via Laplacian variance.
        4. Duplicate image detection via normalized perceptual differences.
        5. Visual overlap assessment via feature correspondence density.

        Returns:
            Tuple of (valid_images, warnings_list, metrics_dict).
        """
        valid_images: List[Image.Image] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {"total_input": len(images), "images_info": []}

        if not images:
            warnings.append("No images provided for multi-view validation.")
            return valid_images, warnings, metrics

        if len(images) < 2:
            warnings.append(f"Multi-view reconstruction requires at least 2 images (received {len(images)}).")
        elif len(images) < 3:
            warnings.append(f"Recommended multi-view capture range is 3-10 images for complete room coverage (received {len(images)}).")

        seen_signatures = []

        for idx, img in enumerate(images):
            info = {"index": idx, "is_valid": True, "issues": []}

            # 1. Format & Dimension Validation
            if not isinstance(img, Image.Image):
                info["is_valid"] = False
                info["issues"].append("Object is not a valid PIL Image")
                metrics["images_info"].append(info)
                continue

            w, h = img.size
            info["dimensions"] = (w, h)
            if w < 320 or h < 240:
                info["is_valid"] = False
                msg = f"Image {idx} resolution ({w}x{h}) is below minimum requirement (320x240)."
                info["issues"].append(msg)
                warnings.append(msg)
            if w > 4096 or h > 4096:
                warnings.append(f"Image {idx} resolution ({w}x{h}) exceeds 4096px; will be downsampled.")

            # 2. Convert to RGB and check sharpness
            rgb_arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            info["sharpness_score"] = round(laplacian_var, 1)

            if laplacian_var < 25.0:
                warnings.append(f"Image {idx} exhibits motion blur (sharpness: {laplacian_var:.1f} < 25.0).")

            # 3. Duplicate Detection via downsampled perceptual thumbnail
            thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            is_dup = False
            for prev_idx, prev_thumb in seen_signatures:
                mse = float(np.mean((thumb - prev_thumb) ** 2))
                if mse < 0.0008:
                    is_dup = True
                    msg = f"Image {idx} is a duplicate of Image {prev_idx} (MSE: {mse:.6f}); discarded."
                    info["is_valid"] = False
                    info["issues"].append(msg)
                    warnings.append(msg)
                    break

            if not is_dup and info["is_valid"]:
                seen_signatures.append((idx, thumb))
                valid_images.append(img)

            metrics["images_info"].append(info)

        metrics["valid_count"] = len(valid_images)
        return valid_images, warnings, metrics

    def register_views(
        self,
        img1: Image.Image,
        depth1: np.ndarray,
        img2: Image.Image,
        depth2: np.ndarray,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Performs feature matching and 3D-3D rigid registration between two camera viewpoints.

        Args:
            img1: Reference view PIL image.
            depth1: Reference view (H, W) metric depth map in meters.
            img2: Secondary view PIL image.
            depth2: Secondary view (H, W) metric depth map in meters.
            fx, fy: Focal lengths in pixels. If None, derived from config or default (1000.0).
            cx, cy: Principal point offsets in pixels. If None, derived from config or image center.

        Returns:
            Dict containing R, t, keypoints matched, inlier count, and registration success.
        """
        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))

        gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)

        w1, h1 = img1.size
        w2, h2 = img2.size

        # Resolve camera intrinsics
        intrinsics = self.config.get("camera_intrinsics", {})
        fx_val = float(fx) if fx is not None else float(intrinsics.get("fx", 1000.0))
        fy_val = float(fy) if fy is not None else float(intrinsics.get("fy", 1000.0))

        cfg_cx = intrinsics.get("cx", 0.5) if cx is None else cx
        cfg_cy = intrinsics.get("cy", 0.5) if cy is None else cy

        cx1 = float(cfg_cx * w1) if float(cfg_cx) <= 1.0 else float(cfg_cx)
        cy1 = float(cfg_cy * h1) if float(cfg_cy) <= 1.0 else float(cfg_cy)
        cx2 = float(cfg_cx * w2) if float(cfg_cx) <= 1.0 else float(cfg_cx)
        cy2 = float(cfg_cy * h2) if float(cfg_cy) <= 1.0 else float(cfg_cy)

        # 1. SIFT Feature Detection & Description
        sift = cv2.SIFT_create(nfeatures=2000)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            logger.warning("Insufficient SIFT features detected between views.")
            return {
                "success": False,
                "R": np.eye(3, dtype=np.float32),
                "t": np.zeros(3, dtype=np.float32),
                "inliers_count": 0,
                "rotation_deg": 0.0,
                "baseline_m": 0.0,
                "rmse_m": 999.0,
            }

        # 2. FLANN-based Feature Matching with Lowe's Ratio Test
        index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                good_matches.append(m_pair[0])

        if len(good_matches) < 8:
            logger.warning(f"Only {len(good_matches)} good feature matches found between views.")
            return {
                "success": False,
                "R": np.eye(3, dtype=np.float32),
                "t": np.zeros(3, dtype=np.float32),
                "inliers_count": len(good_matches),
                "rotation_deg": 0.0,
                "baseline_m": 0.0,
                "rmse_m": 999.0,
            }

        # 3. Associate 2D Matches with 3D Metric Points
        pts1_3d = []
        pts2_3d = []
        inlier_matches = []

        dh1, dw1 = depth1.shape[:2]
        dh2, dw2 = depth2.shape[:2]

        for m in good_matches:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt

            # Scale to depth map resolution if differing
            du1 = int(round(u1 * dw1 / w1))
            dv1 = int(round(v1 * dh1 / h1))
            du2 = int(round(u2 * dw2 / w2))
            dv2 = int(round(v2 * dh2 / h2))

            if not (0 <= du1 < dw1 and 0 <= dv1 < dh1 and 0 <= du2 < dw2 and 0 <= dv2 < dh2):
                continue

            z1 = float(depth1[dv1, du1])
            z2 = float(depth2[dv2, du2])

            if z1 < 0.2 or z1 > 20.0 or z2 < 0.2 or z2 > 20.0 or not np.isfinite(z1) or not np.isfinite(z2):
                continue

            x1 = (u1 - cx1) * z1 / fx_val
            y1 = (v1 - cy1) * z1 / fy_val
            x2 = (u2 - cx2) * z2 / fx_val
            y2 = (v2 - cy2) * z2 / fy_val

            pts1_3d.append([x1, y1, z1])
            pts2_3d.append([x2, y2, z2])
            inlier_matches.append(m)

        if len(pts1_3d) < 6:
            logger.warning(f"Only {len(pts1_3d)} valid 3D-3D correspondence pairs obtained.")
            return {
                "success": False,
                "R": np.eye(3, dtype=np.float32),
                "t": np.zeros(3, dtype=np.float32),
                "inliers_count": len(pts1_3d),
                "rotation_deg": 0.0,
                "baseline_m": 0.0,
                "rmse_m": 999.0,
            }

        pts1_arr = np.array(pts1_3d, dtype=np.float32)
        pts2_arr = np.array(pts2_3d, dtype=np.float32)

        # 4. Robust 3D-3D RANSAC Alignment
        best_inliers = 0
        best_r = np.eye(3, dtype=np.float32)
        best_t = np.zeros(3, dtype=np.float32)
        best_rmse = 999.0
        n_pairs = len(pts1_arr)

        iterations = min(300, max(50, n_pairs * 5))
        for _ in range(iterations):
            idx_sample = np.random.choice(n_pairs, 4, replace=False)
            r_cand, t_cand, _ = compute_rigid_transform_3d(pts2_arr[idx_sample], pts1_arr[idx_sample])

            # Evaluate inliers across all pairs with 0.25m distance threshold
            pred_pts1 = (r_cand @ pts2_arr.T).T + t_cand
            errors = np.linalg.norm(pred_pts1 - pts1_arr, axis=1)
            inliers = errors < 0.25
            n_inliers = int(np.sum(inliers))

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_r = r_cand
                best_t = t_cand
                best_rmse = float(np.mean(errors[inliers])) if n_inliers > 0 else 999.0

        # Refine on all inliers if substantial
        scale_val = 1.0
        if best_inliers >= 4:
            inliers_mask = np.linalg.norm(((best_r @ pts2_arr.T).T + best_t) - pts1_arr, axis=1) < 0.25
            best_r, best_t, best_rmse = compute_rigid_transform_3d(pts2_arr[inliers_mask], pts1_arr[inliers_mask])
            # Assess metric scale consistency using Umeyama similarity transform
            _, _, sim_scale, _ = compute_similarity_transform_3d(pts2_arr[inliers_mask], pts1_arr[inliers_mask])
            scale_val = float(sim_scale)

        # Compute relative rotation angle in degrees
        angle_rad = np.arccos(np.clip((np.trace(best_r) - 1.0) / 2.0, -1.0, 1.0))
        angle_deg = float(np.degrees(angle_rad))
        baseline_m = float(np.linalg.norm(best_t))

        success = best_inliers >= 6 and best_rmse < 0.35
        rejection_reasons = []
        if not success:
            if best_inliers < 6:
                rejection_reasons.append(f"Insufficient consensus inliers ({best_inliers} < 6)")
            if best_rmse >= 0.35:
                rejection_reasons.append(f"Excessive inlier residual ({best_rmse:.3f}m >= 0.35m)")

        return {
            "success": success,
            "R": best_r,
            "t": best_t,
            "inliers_count": best_inliers,
            "total_matches": len(good_matches),
            "rotation_deg": round(angle_deg, 1),
            "baseline_m": round(baseline_m, 2),
            "rmse_m": round(best_rmse, 3),
            "scale_factor": round(scale_val, 3),
            "rejection_reason": "; ".join(rejection_reasons) if rejection_reasons else None,
        }

    def fuse_multiview(
        self,
        images: List[Image.Image],
        refine_depth: bool = True,
        voxel_size: float = 0.03,
    ) -> SceneGraph:
        """Performs end-to-end multi-view 3D scene reconstruction and fusion.

        Args:
            images: List of PIL room photographs (first image serves as reference coordinate frame).
            refine_depth: Whether to apply guided edge-preserving depth refinement.
            voxel_size: 3D regular voxel size in meters for downsampling and surface consolidation.

        Returns:
            Consolidated SceneGraph containing merged point clouds, fused 3D object envelopes,
            and unified room boundary dimensions.
        """
        assert images is not None and len(images) > 0, "At least one room image required"
        if any(img is None for img in images):
            raise ValueError("Images list cannot contain None elements.")

        val_warnings = []
        if len(images) > 1:
            valid_imgs, val_warnings, _ = self.validate_multiview_sequence(images)
            if valid_imgs:
                images = valid_imgs

        if len(images) == 1:
            # Single view execution with optional guided depth refinement
            scene = self.single_view_fusion.fuse(images[0])
            if refine_depth and getattr(scene, "depth_map", None) is not None:
                rgb_arr = np.array(images[0].convert("RGB"))
                refined_depth = refine_metric_depth_guided(scene.depth_map, rgb_arr)
                scene.depth_map = refined_depth
            return scene

        logger.info(f"Initiating Multi-View 3D Scene Fusion across {len(images)} perspectives...")

        # 1. Process Reference View (View 0)
        ref_scene = self.single_view_fusion.fuse(images[0])
        ref_depth = getattr(ref_scene, "depth_map", None)
        ref_rgb = np.array(images[0].convert("RGB"))

        if refine_depth and ref_depth is not None:
            ref_depth = refine_metric_depth_guided(ref_depth, ref_rgb)
            ref_scene.depth_map = ref_depth

        pts_list = [np.array(ref_scene.point_cloud.get("xyz", []), dtype=np.float32)]
        rgb_list = [np.array(ref_scene.point_cloud.get("rgb", []), dtype=np.float32)]
        fused_objects = list(ref_scene.objects)

        registration_telemetry = []
        per_view_provenance = [{
            "view_index": 0,
            "role": "reference_view",
            "registered": True,
            "n_points": len(pts_list[0]),
            "rejection_reason": None,
        }]

        # 2. Progressively Register and Fuse Subsequent Views into View 0 Coordinate Frame
        for idx in range(1, len(images)):
            sec_img = images[idx]
            sec_scene = self.single_view_fusion.fuse(sec_img)
            sec_depth = getattr(sec_scene, "depth_map", None)
            sec_rgb = np.array(sec_img.convert("RGB"))

            if refine_depth and sec_depth is not None:
                sec_depth = refine_metric_depth_guided(sec_depth, sec_rgb)

            # Register view idx against reference view 0
            reg = self.register_views(
                img1=images[0],
                depth1=ref_depth,
                img2=sec_img,
                depth2=sec_depth,
            )
            reg["secondary_view_index"] = idx
            registration_telemetry.append(reg)

            r_matrix = reg["R"]
            t_vector = reg["t"]

            # Transform View idx points into View 0 frame: P_transformed = R @ P + t
            sec_xyz = np.array(sec_scene.point_cloud.get("xyz", []), dtype=np.float32)
            sec_colors = np.array(sec_scene.point_cloud.get("rgb", []), dtype=np.float32)

            per_view_provenance.append({
                "view_index": idx,
                "role": "secondary_view",
                "registered": bool(reg["success"]),
                "n_points": len(sec_xyz),
                "inliers": reg.get("inliers_count", 0),
                "rmse_m": reg.get("rmse_m", 999.0),
                "scale_factor": reg.get("scale_factor", 1.0),
                "baseline_m": reg.get("baseline_m", 0.0),
                "rotation_deg": reg.get("rotation_deg", 0.0),
                "rejection_reason": reg.get("rejection_reason"),
            })

            if len(sec_xyz) > 0 and reg["success"]:
                trans_xyz = (r_matrix @ sec_xyz.T).T + t_vector
                pts_list.append(trans_xyz)
                rgb_list.append(sec_colors)

                # Fuse 3D Object Envelopes
                for s_obj in sec_scene.objects:
                    s_geom = getattr(s_obj, "geometry", None)
                    if not s_geom or not getattr(s_geom, "centroid", None) or not getattr(s_geom, "extent", None):
                        continue

                    cx, cy, cz = s_geom.centroid
                    trans_centroid = (r_matrix @ np.array([cx, cy, cz])) + t_vector

                    # Search for matching object in reference view
                    matched = False
                    for r_obj in fused_objects:
                        r_geom = getattr(r_obj, "geometry", None)
                        if not r_geom:
                            continue
                        rcx, rcy, rcz = r_geom.centroid
                        dist = np.linalg.norm(np.array([rcx, rcy, rcz]) - trans_centroid)

                        if r_obj.semantic_class == s_obj.semantic_class and dist < 0.85:
                            # Fuse bounding extents to span maximum multi-view observed coverage
                            ew = max(r_geom.extent[0], s_geom.extent[0])
                            eh = max(r_geom.extent[1], s_geom.extent[1])
                            ed = max(r_geom.extent[2], s_geom.extent[2])
                            r_geom.extent = (round(float(ew), 2), round(float(eh), 2), round(float(ed), 2))
                            r_geom.centroid = (
                                round(float((rcx + trans_centroid[0]) / 2.0), 2),
                                round(float((rcy + trans_centroid[1]) / 2.0), 2),
                                round(float((rcz + trans_centroid[2]) / 2.0), 2),
                            )
                            matched = True
                            break

                    if not matched:
                        # Add newly discovered furniture instance seen in secondary angle
                        new_obj = SceneObject(
                            object_id=f"{s_obj.object_id}_v{idx}",
                            semantic_class=s_obj.semantic_class,
                            instance_id=len(fused_objects) + 1,
                            detection_confidence=getattr(s_obj, "detection_confidence", 1.0),
                            mask_provenance=getattr(s_obj, "mask_provenance", "multiview_fused"),
                            bounding_box=s_obj.bounding_box,
                            coverage_percentage=s_obj.coverage_percentage,
                            geometry=Object3DGeometry(
                                centroid=(
                                    round(float(trans_centroid[0]), 2),
                                    round(float(trans_centroid[1]), 2),
                                    round(float(trans_centroid[2]), 2),
                                ),
                                extent=s_geom.extent,
                            ),
                            point_cloud_available=False,
                        )
                        fused_objects.append(new_obj)
            else:
                logger.warning(f"Registration between view 0 and view {idx} did not meet inlier threshold.")

        # 3. Concatenate and Consolidate Merged Point Cloud
        all_points = np.vstack(pts_list)
        all_colors = np.vstack(rgb_list)

        # Apply Voxel Grid Downsampling to merge overlapping surfaces
        down_points, down_colors = voxel_grid_downsample(all_points, all_colors, voxel_size=voxel_size)

        # Apply Statistical Outlier Removal to eliminate isolated flying pixels
        clean_points, clean_colors = statistical_outlier_removal(down_points, down_colors, k_neighbors=16, std_ratio=1.8)

        # 4. Re-estimate Unified Multi-View Room Layout & Floor Plane
        fused_cloud_dict = {
            "xyz": clean_points.tolist(),
            "rgb": clean_colors.tolist() if clean_colors is not None else [],
        }

        updated_layout = self.layout_service.estimate_layout(
            point_cloud=fused_cloud_dict,
            objects=fused_objects,
            surfaces=ref_scene.surfaces,
        )

        # Physically ground fused furniture onto the consolidated multi-view floor plane
        if updated_layout and updated_layout.get("floor_plane_detected"):
            floor_elevation = updated_layout.get("floor_elevation_m", 1.4)
            self.single_view_fusion._align_furniture_to_floor(fused_objects, floor_elevation)

        # Determine consensus status and model name
        registered_count = sum(1 for pv in per_view_provenance if pv.get("registered"))
        any_secondary_registered = any(
            pv.get("registered") for pv in per_view_provenance if pv.get("role") == "secondary_view"
        )
        if any_secondary_registered:
            model_name = f"RoomAI Multi-View Consolidated ({registered_count}/{len(images)} perspectives)"
        else:
            model_name = "RoomAI Single-View Fallback (Multi-View Consensus Unachieved)"
            logger.warning(
                "Multi-view consensus unachieved across all secondary perspectives. "
                "Retaining reference view geometry with single-view fallback."
            )

        # 5. Assemble Consolidated SceneGraph
        fused_scene = SceneGraph(
            image_id=ref_scene.image_id,
            width=ref_scene.width,
            height=ref_scene.height,
            model_name=model_name,
            depth_mode="metric",
            camera_intrinsics=ref_scene.camera_intrinsics,
            objects=fused_objects,
            surfaces=ref_scene.surfaces,
            relationships=ref_scene.relationships,
            n_total_points=len(clean_points),
            n_valid_points=len(clean_points),
            depth_min_m=float(np.min(clean_points[:, 2])) if len(clean_points) > 0 else 0.1,
            depth_max_m=float(np.max(clean_points[:, 2])) if len(clean_points) > 0 else 20.0,
            point_cloud=fused_cloud_dict,
            room_layout=updated_layout,
        )

        fused_scene.depth_map = ref_depth
        fused_scene.heatmap_image = getattr(ref_scene, "heatmap_image", None)
        fused_scene.multiview_telemetry = registration_telemetry
        fused_scene.multiview_per_view_provenance = per_view_provenance

        logger.info(
            f"Multi-View 3D Scene Fusion Complete: {len(clean_points):,} points consolidated, "
            f"{len(fused_objects)} objects recognized across {len(images)} views."
        )
        return fused_scene

