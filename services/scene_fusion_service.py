"""Scene Fusion Service — combines semantic, instance, and depth modalities into a unified 3D scene representation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from models.scene3d_model import CustomScene3DModel, MockScene3DModel
from models.segmentation_model import CustomRoomSegmenter
from models.detection_model import CustomObjectDetector, MockObjectDetector
from schemas.scene import (
    SceneGraph, SceneObject, RoomSurface3D, SpatialRelationship,
    Object3DGeometry, Object3DPointStatistics, Scene3DEvaluation
)
from utils.logging_config import get_logger

logger = get_logger("SceneFusion")


class SceneFusionService:
    """Fuses detection, segmentation, and depth into a unified 3D scene representation."""

    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        self.mock_mode = mock_mode
        if config is None:
            cfg_p = Path("config/model_config.yaml")
            if cfg_p.exists():
                try:
                    import yaml
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        self.config = yaml.safe_load(f) or {}
                except Exception:
                    self.config = {}
            else:
                self.config = {}
        else:
            self.config = config
        self._scene3d_disabled = False

        # Check if scene3d is explicitly disabled
        scene3d_config = self.config.get("scene3d_metric", self.config.get("scene3d", {}))
        if scene3d_config is not None:
            self._scene3d_disabled = scene3d_config.get("enabled", True) is False

        if mock_mode or self._scene3d_disabled:
            self.scene3d_model = MockScene3DModel(scene3d_config if self._scene3d_disabled else self.config.get("scene3d_metric", self.config.get("scene3d", {})))
            self.segmenter = None  # Not needed in mock mode with detection fallback
            self.detector = MockObjectDetector(self.config.get("detection", {}))
        else:
            self.scene3d_model: Optional[CustomScene3DModel] = None
            self.segmenter: Optional[CustomRoomSegmenter] = None
            self.detector: Optional[CustomObjectDetector] = None
        self._ready = False

    def load_models(self) -> None:
        """Lazily loads required models on first use."""
        if self.mock_mode:
            self._ready = True
            return

        scene_config = self.config.get("scene3d_metric", self.config.get("scene3d", {}))
        self.scene3d_model = CustomScene3DModel(config=scene_config)
        if not self.scene3d_model.is_ready():
            self.scene3d_model.load_model()

        seg_config = self.config.get("segmentation", {})
        self.segmenter = CustomRoomSegmenter(config=seg_config)
        self.segmenter.load_model()

        det_config = self.config.get("indoor_detection", self.config.get("detection", {}))
        self.detector = CustomObjectDetector(config=det_config)
        self.detector.load_model()
        self._ready = True

    def is_ready(self) -> bool:
        return self.mock_mode or self._ready

    def fuse(self, image: Image.Image, camera_intrinsics: Optional[Dict[str, float]] = None) -> SceneGraph:
        """Fuses all modalities into a unified 3D scene representation."""
        if image is None:
            raise ValueError("Input image cannot be None.")

        if not self.is_ready():
            self.load_models()

        w, h = image.size
        if camera_intrinsics is not None:
            intrinsics = camera_intrinsics
        else:
            focal = float(max(w, h))
            intrinsics = {"fx": focal, "fy": focal, "cx": (w - 1) / 2.0, "cy": (h - 1) / 2.0}

        # Determine depth mode from model or config
        depth_mode = "relative"
        if self.scene3d_model and hasattr(self.scene3d_model, 'depth_mode'):
            depth_mode = self.scene3d_model.depth_mode

        scene = SceneGraph(
            image_id=None,
            width=w,
            height=h,
            model_name="YOLO11s A3 + YOLO11n-Seg + Depth Anything V2 Metric Indoor Small",
            depth_mode=depth_mode,
            camera_intrinsics=intrinsics,
        )

        # 1. Run semantic segmentation and instance segmentation
        sem_resp = None
        sem_mask = None
        if self.mock_mode:
            # Mock mode: use detection for objects, create synthetic surfaces
            det_resp = self.detector.predict(image, confidence_threshold=0.35)
            scene.objects = self._extract_objects_from_detection(det_resp, w, h, image)
            scene.surfaces = self._create_mock_surfaces(w, h)
            scene.n_total_points = w * h
        else:
            # Real mode: run perception modules
            # Prioritize fine-tuned A3 indoor detector for accurate indoor taxonomy
            detected_objects = []
            sem_resp = None
            sem_mask = None
            if self.detector:
                det_resp = self.detector.predict(image, confidence_threshold=0.28)
                detected_objects = self._extract_objects_from_detection(det_resp, w, h, image)
                detected_objects = self._suppress_duplicate_objects(detected_objects, iou_threshold=0.55)
                scene.n_total_points = w * h

            if self.segmenter:
                sem_resp, sem_mask = self.segmenter.predict(image)
                scene.surfaces = self._extract_surfaces(sem_resp, sem_mask, w, h)
                scene.n_total_points = w * h

                if detected_objects:
                    scene.objects = detected_objects
                else:
                    # Fallback to instance segmentation if detector found 0 objects
                    seg_mode = getattr(
                        self.segmenter,
                        "segmentation_mode",
                        self.config.get("segmentation", {}).get("mode", "instance")
                    )
                    if seg_mode == "instance" and sem_resp.masks:
                        scene.objects = self._extract_objects_from_instance(
                            sem_resp, sem_mask, w, h, image
                        )
                    else:
                        scene.objects = []
            else:
                scene.objects = detected_objects

        # 2. Run depth estimation
        if self.scene3d_model and not self._scene3d_disabled:
            # The depth map is aligned to this RGB frame. Keep focal lengths in pixels
            # and pass the same K through the adapter's provisional point generation.
            self.scene3d_model.camera_intrinsics = intrinsics
            norm_depth, depth_resp = self.scene3d_model.reconstruct(image)
            metric_depth = getattr(self.scene3d_model, "last_raw_depth", None)
            if metric_depth is None and depth_resp.raw_depth_map is not None:
                metric_depth = np.array(depth_resp.raw_depth_map, dtype=np.float32)
            if metric_depth is None:
                metric_depth = norm_depth

            scene.n_valid_points = int(np.count_nonzero(metric_depth > 0)) if metric_depth is not None else 0
            scene.depth_min_m = float(depth_resp.depth_min)
            scene.depth_max_m = float(depth_resp.depth_max)
            scene.point_cloud = depth_resp.point_cloud

            # Attach metric depth map and rendered heatmap directly to scene graph
            scene.depth_map = metric_depth
            from services.image_service import create_depth_heatmap
            if metric_depth is not None:
                scene.heatmap_image = create_depth_heatmap(metric_depth)

            # Associate true metric depth with objects
            scene.objects = self._associate_depth_to_objects(scene.objects, metric_depth, w, h, intrinsics, semantic_mask=sem_mask)

            # Estimate room structural layout & boundaries
            from services.room_layout_service import RoomLayoutService
            layout_service = RoomLayoutService(config=self.config.get("layout", {}))
            scene.room_layout = layout_service.estimate_layout(
                point_cloud=scene.point_cloud or {},
                objects=scene.objects,
                surfaces=scene.surfaces,
            )

            # Physically ground floor-standing furniture onto the detected floor plane
            if scene.room_layout and scene.room_layout.get("floor_plane_detected"):
                floor_elevation = scene.room_layout.get("floor_elevation_m", 1.4)
                self._align_furniture_to_floor(scene.objects, floor_elevation)

            # Extract spatial relationships
            scene.relationships = self._compute_relationships(scene.objects, scene.room_layout)

        return scene

    def _create_mock_surfaces(self, w: int, h: int) -> List[RoomSurface3D]:
        """Create synthetic surfaces for mock mode testing."""
        return [
            RoomSurface3D(
                semantic_category="wall",
                n_points=int(w * h * 0.7),
                mean_depth_m=3.0,
            ),
            RoomSurface3D(
                semantic_category="floor",
                n_points=int(w * h * 0.25),
                mean_depth_m=0.1,
            ),
        ]

    def _extract_surfaces(
        self, sem_resp, sem_mask: np.ndarray, w: int, h: int
    ) -> List[RoomSurface3D]:
        """Extract semantic surfaces from segmentation mask."""
        surfaces = []
        seen_categories = set()

        for idx, m in enumerate(sem_resp.masks):
            cat = m.category.lower()
            if cat in seen_categories:
                continue
            seen_categories.add(cat)

            mask_id = getattr(m, "instance_id", idx + 1)
            pixel_mask = (sem_mask == mask_id) if sem_mask is not None else np.zeros((h, w), dtype=bool)
            n_points = int(np.count_nonzero(pixel_mask))

            surfaces.append(RoomSurface3D(
                semantic_category=cat,
                pixel_region=None,
                n_points=n_points,
                mean_depth_m=0.0,
                centroid=None,
            ))

        return surfaces

    def _extract_objects_from_instance(
        self, ins_resp, ins_mask: np.ndarray, w: int, h: int, image: Image.Image
    ) -> List[SceneObject]:
        """Create SceneObjects from instance segmentation results."""
        objects = []

        for idx, m in enumerate(ins_resp.masks):
            inst_id = getattr(m, 'instance_id', idx + 1)

            # Pixel coverage
            obj_mask = (ins_mask == inst_id) if ins_mask is not None else np.zeros((h, w), dtype=bool)
            n_pixels = int(np.count_nonzero(obj_mask))
            if n_pixels == 0:
                continue

            # 3D point extraction
            xyz_coords = []
            rgb_colors = []
            if getattr(m, 'mask_points_3d', None):
                xyz_coords = m.mask_points_3d[:10000]
                rgb_colors = m.mask_points_colors[:10000] if getattr(m, 'mask_points_colors', None) else []

            # 2D centroid and normalized bounding box
            ys, xs = np.where(obj_mask)
            if len(ys) > 0:
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                bbox = (
                    float(np.min(xs)) / w,
                    float(np.min(ys)) / h,
                    float(np.max(xs)) / w,
                    float(np.max(ys)) / h,
                )
            else:
                cx, cy = w / 2, h / 2
                bbox = (0.0, 0.0, 1.0, 1.0)

            objects.append(SceneObject(
                object_id=f"{m.category}_{inst_id:03d}",
                semantic_class=m.category,
                instance_id=inst_id,
                detection_confidence=getattr(m, 'confidence', 1.0) or 1.0,
                mask_provenance="yolo_seg" if "seg" in str(getattr(ins_resp, 'model_name', '')).lower() else "a3_fastsam",
                bounding_box=bbox,
                coverage_percentage=round((n_pixels / (w * h)) * 100, 2),
                geometry=None,  # Will be populated later with depth
                point_cloud_available=len(xyz_coords) > 0,
                xyz_coords=xyz_coords[:10000] if xyz_coords else None,
                rgb_colors=rgb_colors[:10000] if rgb_colors else None,
            ))

        return objects

    def _extract_objects_from_detection(
        self, det_resp, w: int, h: int, image: Image.Image
    ) -> List[SceneObject]:
        """Fallback: Create SceneObjects from 2D detections only."""
        objects = []

        for item in det_resp.detections:
            n_pixels = int((item.box.x_max - item.box.x_min) * (item.box.y_max - item.box.y_min) * w * h)

            objects.append(SceneObject(
                object_id=f"{item.label}_{len(objects)+1:03d}",
                semantic_class=item.label,
                instance_id=len(objects) + 1,
                detection_confidence=item.confidence,
                mask_provenance="detection_only",
                bounding_box=(item.box.x_min, item.box.y_min, item.box.x_max, item.box.y_max),
                coverage_percentage=round((n_pixels / (w * h)) * 100, 2),
                geometry=None,
                point_cloud_available=False,
                xyz_coords=None,
                rgb_colors=None,
            ))

        return objects

    def _suppress_duplicate_objects(
        self, objects: List[SceneObject], iou_threshold: float = 0.55
    ) -> List[SceneObject]:
        """Applies non-maximum suppression to eliminate overlapping duplicate detections."""
        if len(objects) <= 1:
            return objects

        sorted_objs = sorted(
            objects,
            key=lambda o: getattr(o, "detection_confidence", 0.0) or 0.0,
            reverse=True,
        )
        kept = []
        for obj in sorted_objs:
            b1 = obj.bounding_box
            overlap = False
            for k in kept:
                b2 = k.bounding_box
                # Compute 2D IoU
                x_left = max(b1[0], b2[0])
                y_top = max(b1[1], b2[1])
                x_right = min(b1[2], b2[2])
                y_bottom = min(b1[3], b2[3])
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    union = area1 + area2 - intersection
                    iou = intersection / max(union, 1e-6)
                    if iou > iou_threshold:
                        overlap = True
                        break
            if not overlap:
                kept.append(obj)
        return kept

    def _align_furniture_to_floor(
        self, objects: List[SceneObject], floor_elevation_m: float
    ) -> None:
        """Physically grounds floor-standing furniture pieces onto the estimated floor plane."""
        floor_standing_categories = {
            "chair#stool", "chair", "table", "sofa", "couch", "bed",
            "cabinet#shelf", "desk", "bench", "refrigerator", "bookcase",
        }
        for obj in objects:
            if not obj.geometry or not obj.geometry.centroid or not obj.geometry.extent:
                continue
            cat = (obj.semantic_class or "").lower()
            if any(f_cat in cat for f_cat in floor_standing_categories):
                cx, cy, cz = obj.geometry.centroid
                w, h, d = obj.geometry.extent
                # In camera coordinates (+Y is down), bottom of bounding box is cy + h / 2.0
                current_bottom = cy + (h / 2.0)
                raw_delta = abs(current_bottom - floor_elevation_m)
                obj.geometry.raw_floor_contact_delta_m = round(raw_delta, 3)

                # If bottom deviates from floor plane, snap it to rest directly on the floor
                if raw_delta > 0.02:
                    grounded_cy = floor_elevation_m - (h / 2.0)
                    obj.geometry.centroid = (cx, round(grounded_cy, 3), cz)
                    post_delta = abs(grounded_cy + (h / 2.0) - floor_elevation_m)
                else:
                    post_delta = raw_delta
                obj.geometry.post_floor_contact_delta_m = round(post_delta, 3)

    def _associate_depth_to_objects(
        self,
        objects: List[SceneObject],
        depth_map: np.ndarray,
        w: int,
        h: int,
        intrinsics: Optional[Dict[str, float]] = None,
        semantic_mask: Optional[np.ndarray] = None,
    ) -> List[SceneObject]:
        """Associate depth values with objects using interior sampling and robust clustering."""
        if depth_map is None:
            return objects

        if intrinsics is None:
            intrinsics = {"fx": 1000, "fy": 1000, "cx": 0.5, "cy": 0.5}

        fx = intrinsics.get("fx", 1000)
        fy = intrinsics.get("fy", 1000)
        cx = intrinsics.get("cx", 0.5) * w if intrinsics.get("cx", 0.5) <= 1.0 else intrinsics.get("cx", 0.5)
        cy = intrinsics.get("cy", 0.5) * h if intrinsics.get("cy", 0.5) <= 1.0 else intrinsics.get("cy", 0.5)

        for obj in objects:
            if obj.bounding_box == (0, 0, 1, 1):
                continue  # No bbox available

            x1, y1, x2, y2 = [int(b * wh) for b, wh in zip(obj.bounding_box, [w, h, w, h])]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 1. Interior depth sampling to reject boundary edge bleed into surrounding walls/floors
            pad_x = int(0.12 * (x2 - x1))
            pad_y = int(0.12 * (y2 - y1))
            in_x1, in_x2 = x1 + pad_x, x2 - pad_x
            in_y1, in_y2 = y1 + pad_y, y2 - pad_y

            if in_x2 > in_x1 and in_y2 > in_y1:
                interior_depth = depth_map[in_y1:in_y2, in_x1:in_x2].flatten()
            else:
                interior_depth = depth_map[y1:y2, x1:x2].flatten()

            valid_depths = interior_depth[(interior_depth > 0.1) & (interior_depth < 30.0)]
            if len(valid_depths) == 0:
                full_box = depth_map[y1:y2, x1:x2].flatten()
                valid_depths = full_box[(full_box > 0.1) & (full_box < 30.0)]

            if len(valid_depths) > 0:
                # 2. Robust depth clustering & Quantile Filtering
                clean_depths = valid_depths[(valid_depths >= 0.2) & (valid_depths <= 25.0)]
                if len(clean_depths) < 5:
                    clean_depths = valid_depths

                # Foreground object cluster: leading front surface cutoff avoids background wall pull
                foreground_cutoff = float(np.percentile(clean_depths, 35))
                fg_depths = clean_depths[clean_depths <= foreground_cutoff + 0.35]
                if len(fg_depths) < 5:
                    fg_depths = clean_depths

                mean_d = float(np.mean(fg_depths))
                median_d = float(np.median(fg_depths))

                bbox_center_x = (obj.bounding_box[0] + obj.bounding_box[2]) * w / 2
                bbox_center_y = (obj.bounding_box[1] + obj.bounding_box[3]) * h / 2

                z = median_d
                x = (bbox_center_x - cx) * z / fx
                y = (bbox_center_y - cy) * z / fy

                # 3. Unproject physical 3D extents in meters (retaining raw measurements)
                phys_w = float(max(0.15, (x2 - x1) * z / fx))
                phys_h = float(max(0.15, (y2 - y1) * z / fy))
                depth_span = float(np.percentile(fg_depths, 90) - np.percentile(fg_depths, 10)) if len(fg_depths) > 5 else 0.4
                phys_d = float(max(0.20, depth_span if depth_span >= 0.25 else phys_w * 0.7))

                raw_extent = (round(phys_w, 3), round(phys_h, 3), round(phys_d, 3))

                # 4. Check dimension reliability against physical residential envelopes (do NOT artificially clamp)
                cat_lower = (obj.semantic_class or "").lower()
                is_reliable = True
                warning_msg = None

                if "table" in cat_lower and (phys_h > 1.30 or phys_h < 0.25):
                    is_reliable = False
                    warning_msg = f"Table height ({phys_h:.2f}m) exceeds domestic expectation [0.35-1.15m]; check depth/FOV."
                elif ("chair" in cat_lower or "stool" in cat_lower) and (phys_h > 1.55 or phys_h < 0.30):
                    is_reliable = False
                    warning_msg = f"Chair height ({phys_h:.2f}m) exceeds domestic expectation [0.35-1.35m]."
                elif ("sofa" in cat_lower or "couch" in cat_lower) and (phys_h > 1.45 or phys_w > 4.5):
                    is_reliable = False
                    warning_msg = f"Sofa dimensions ({phys_w:.2f}x{phys_h:.2f}m) exceed domestic expectation."
                elif "bed" in cat_lower and (phys_h > 1.50 or phys_w > 3.8):
                    is_reliable = False
                    warning_msg = f"Bed dimensions ({phys_w:.2f}x{phys_h:.2f}m) exceed domestic expectation."
                elif z > 7.0:
                    is_reliable = False
                    warning_msg = f"Object distance ({z:.2f}m) is far beyond residential room range (1-5m)."

                obj.geometry = Object3DGeometry(
                    centroid=(round(x, 3), round(y, 3), round(z, 3)),
                    extent=raw_extent,
                    orientation_yaw=0.0,
                    is_dimension_reliable=is_reliable,
                    dimension_warning=warning_msg,
                    raw_extent=raw_extent,
                    point_stats=Object3DPointStatistics(
                        n_points=len(valid_depths),
                        mean_depth_m=round(mean_d, 3),
                        median_depth_m=round(median_d, 3),
                        depth_std_m=round(float(np.std(valid_depths)), 3),
                        depth_min_m=round(float(np.min(valid_depths)), 3),
                        depth_max_m=round(float(np.max(valid_depths)), 3),
                    ),
                )

        return objects

    def _compute_relationships(
        self,
        objects: List[SceneObject],
        room_layout: Optional[Dict[str, Any]] = None,
    ) -> List[SpatialRelationship]:
        """Compute physically-grounded spatial relationships between objects and room boundaries."""
        relationships = []

        floor_elevation = None
        if room_layout and room_layout.get("floor_plane_detected"):
            floor_elevation = room_layout.get("floor_elevation_m")

        for i, obj_a in enumerate(objects):
            if obj_a.geometry is None or obj_a.geometry.centroid is None:
                continue

            # Check floor placement
            if floor_elevation is not None:
                ay = obj_a.geometry.centroid[1]
                if abs(ay - floor_elevation) < 0.5:
                    relationships.append(SpatialRelationship(
                        source_object_id=obj_a.object_id,
                        relationship_type="on_floor",
                        target_object_id="room_floor",
                        evidence=f"Object base near floor elevation ({floor_elevation:.2f}m)",
                        confidence=0.90,
                    ))

            for j, obj_b in enumerate(objects):
                if i == j or obj_b.geometry is None or obj_b.geometry.centroid is None:
                    continue

                centroid_a = np.array(obj_a.geometry.centroid)
                centroid_b = np.array(obj_b.geometry.centroid)

                dist = float(np.linalg.norm(centroid_a - centroid_b))

                # Simple proximity check
                if dist < 1.5:
                    relationships.append(SpatialRelationship(
                        source_object_id=obj_a.object_id,
                        relationship_type="near",
                        target_object_id=obj_b.object_id,
                        evidence=f"3D centroid Euclidean distance: {dist:.2f}m",
                        confidence=round(max(0.2, 1.0 - dist / 2.0), 2),
                    ))

                # Depth ordering (in_front_of)
                if centroid_a[2] < (centroid_b[2] - 0.4) and dist < 2.5:
                    relationships.append(SpatialRelationship(
                        source_object_id=obj_a.object_id,
                        relationship_type="in_front_of",
                        target_object_id=obj_b.object_id,
                        evidence=f"Depth delta: {centroid_b[2] - centroid_a[2]:.2f}m closer to camera",
                        confidence=0.85,
                    ))

        return relationships


def get_scene_fusion_service(mock_mode: bool = False, config: Optional[Dict[str, Any]] = None) -> SceneFusionService:
    """Factory function for SceneFusionService."""
    return SceneFusionService(mock_mode=mock_mode, config=config)
