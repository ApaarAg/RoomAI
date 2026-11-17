"""RoomAI Phase 7.6 — Complete Subsystem Audit and Real-Inference Smoke Test."""

import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import yaml

# Set environment variables
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.registry import GLOBAL_MODEL_REGISTRY
from models.detection_model import CustomObjectDetector
from models.segmentation_model import CustomRoomSegmenter
from models.scene3d_model import CustomScene3DModel
from services.detection_service import DetectionService
from services.segmentation_service import SegmentationService
from services.scene3d_service import Scene3DService
from services.scene_fusion_service import SceneFusionService
from services.multiview_fusion_service import MultiViewFusionService
from services.classification_service import ClassificationService
from services.vision_language_service import VisionLanguageService
from services.generation_service import GenerationService
from utils.export_3d import export_point_cloud_ply, export_scene_glb

def main():
    print("=" * 70)
    print("ROOMAI PHASE 7.6 — SUBSYSTEM AUDIT & REAL-INFERENCE SMOKE TEST")
    print("=" * 70)
    
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"CUDA Available: {cuda_avail} ({device_name})")
    if cuda_avail:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"Total VRAM: {total_vram:.1f} MB")
        torch.cuda.reset_peak_memory_stats(0)

    # Load sample image
    sample_path = REPO_ROOT / "artifacts" / "sample_room.jpg"
    assert sample_path.exists(), f"Missing {sample_path}"
    test_img = Image.open(sample_path).convert("RGB")
    print(f"Loaded test image: {test_img.size} {test_img.mode}")

    # Load configs
    with open(REPO_ROOT / "config" / "model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    audit_results = {}

    # -------------------------------------------------------------
    # 1. INDOOR DETECTION (Model A3)
    # -------------------------------------------------------------
    print("\n--- [1] Indoor Detection: Model A3 YOLO11s 800px ---")
    try:
        t0 = time.perf_counter()
        det_svc = DetectionService(mock_mode=False, config=model_cfg.get("detection", {}))
        res_a3 = det_svc.run(test_img, confidence_threshold=0.25)
        dt = (time.perf_counter() - t0) * 1000
        assert res_a3["success"], f"A3 failed: {res_a3.get('error_message')}"
        resp = res_a3["response"]
        print(f"SUCCESS: {resp.total_objects} objects detected in {dt:.1f}ms (device: {resp.device})")
        print(f"Model Name: {resp.model_name} | Checkpoint: {resp.checkpoint}")
        for item in resp.detections[:4]:
            print(f"  - {item.label}: conf={item.confidence:.2f}, bbox=[{item.box.x_min:.2f}, {item.box.y_min:.2f}, {item.box.x_max:.2f}, {item.box.y_max:.2f}]")
        audit_results["indoor_a3"] = {"status": "COMPLETE", "objects": resp.total_objects, "device": resp.device, "latency_ms": dt}
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["indoor_a3"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 2. EXTERIOR DETECTION (Model B2)
    # -------------------------------------------------------------
    print("\n--- [2] Exterior Detection: Model B2 YOLO11s ---")
    try:
        t0 = time.perf_counter()
        ext_svc = DetectionService(mock_mode=False, config=model_cfg.get("exterior_detection", {}))
        res_b2 = ext_svc.run(test_img, confidence_threshold=0.25)
        dt = (time.perf_counter() - t0) * 1000
        assert res_b2["success"], f"B2 failed: {res_b2.get('error_message')}"
        resp_b2 = res_b2["response"]
        print(f"SUCCESS: {resp_b2.total_objects} objects detected in {dt:.1f}ms (device: {resp_b2.device})")
        print(f"Model Name: {resp_b2.model_name} | Checkpoint: {resp_b2.checkpoint}")
        audit_results["exterior_b2"] = {"status": "COMPLETE", "objects": resp_b2.total_objects, "device": resp_b2.device, "latency_ms": dt}
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["exterior_b2"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 3. PARKING DETECTION (Model C2)
    # -------------------------------------------------------------
    print("\n--- [3] Parking Detection: Model C2 YOLO11s ---")
    try:
        t0 = time.perf_counter()
        pkg_svc = DetectionService(mock_mode=False, config=model_cfg.get("parking_detection", {}))
        res_c2 = pkg_svc.run(test_img, confidence_threshold=0.25)
        dt = (time.perf_counter() - t0) * 1000
        assert res_c2["success"], f"C2 failed: {res_c2.get('error_message')}"
        resp_c2 = res_c2["response"]
        print(f"SUCCESS: {resp_c2.total_objects} objects detected in {dt:.1f}ms (device: {resp_c2.device})")
        print(f"Model Name: {resp_c2.model_name} | Checkpoint: {resp_c2.checkpoint}")
        audit_results["parking_c2"] = {"status": "COMPLETE", "objects": resp_c2.total_objects, "device": resp_c2.device, "latency_ms": dt}
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["parking_c2"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 4. SEGMENTATION (YOLO11-Seg)
    # -------------------------------------------------------------
    print("\n--- [4] Segmentation: YOLO11-Seg ---")
    try:
        t0 = time.perf_counter()
        seg_svc = SegmentationService(mock_mode=False, config=model_cfg.get("segmentation", {}))
        res_seg = seg_svc.run(test_img)
        dt = (time.perf_counter() - t0) * 1000
        assert res_seg["success"], f"Seg failed: {res_seg.get('error_message')}"
        resp_seg = res_seg["response"]
        print(f"SUCCESS: {len(resp_seg.masks)} masks found across categories {resp_seg.categories_found} in {dt:.1f}ms")
        audit_results["segmentation"] = {"status": "COMPLETE", "masks": len(resp_seg.masks), "categories": resp_seg.categories_found, "latency_ms": dt}
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["segmentation"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 5. METRIC DEPTH ESTIMATION
    # -------------------------------------------------------------
    print("\n--- [5] Metric Depth Estimation: Depth Anything V2 Metric Indoor Small ---")
    try:
        t0 = time.perf_counter()
        scene3d_svc = Scene3DService(mock_mode=False, config=model_cfg.get("scene3d_metric", {}))
        res_depth = scene3d_svc.run(test_img)
        dt = (time.perf_counter() - t0) * 1000
        assert res_depth["success"], f"Metric depth failed: {res_depth.get('error_message')}"
        resp_d = res_depth["response"]
        print(f"SUCCESS: Depth Range: [{resp_d.depth_min:.2f}m - {resp_d.depth_max:.2f}m] | Unit: {resp_d.depth_unit} | Relative-Only: {resp_d.relative_depth_only} in {dt:.1f}ms")
        assert not resp_d.relative_depth_only, "Metric depth should not be marked as relative_depth_only"
        assert resp_d.depth_unit == "meters", f"Depth unit should be 'meters', got '{resp_d.depth_unit}'"
        audit_results["metric_depth"] = {
            "status": "COMPLETE",
            "range": [resp_d.depth_min, resp_d.depth_max],
            "unit": resp_d.depth_unit,
            "relative_only": resp_d.relative_depth_only,
            "latency_ms": dt,
        }
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["metric_depth"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 6. RELATIVE DEPTH ESTIMATION
    # -------------------------------------------------------------
    print("\n--- [6] Relative Depth Estimation: Depth Anything V2 Small ---")
    try:
        t0 = time.perf_counter()
        scene3d_rel = Scene3DService(mock_mode=False, config=model_cfg.get("scene3d", {}))
        res_rel = scene3d_rel.run(test_img)
        dt = (time.perf_counter() - t0) * 1000
        assert res_rel["success"], f"Relative depth failed: {res_rel.get('error_message')}"
        resp_r = res_rel["response"]
        print(f"SUCCESS: Relative Depth Range: [{resp_r.depth_min:.2f} - {resp_r.depth_max:.2f}] | Unit: {resp_r.depth_unit} | Relative-Only: {resp_r.relative_depth_only} in {dt:.1f}ms")
        assert resp_r.relative_depth_only, "Relative depth should be marked as relative_depth_only"
        audit_results["relative_depth"] = {
            "status": "COMPLETE",
            "range": [resp_r.depth_min, resp_r.depth_max],
            "unit": resp_r.depth_unit,
            "relative_only": resp_r.relative_depth_only,
            "latency_ms": dt,
        }
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["relative_depth"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 7. SINGLE-VIEW 3D SCENE FUSION & LAYOUT
    # -------------------------------------------------------------
    print("\n--- [7] Single-View 3D Scene Fusion: SceneFusionService ---")
    try:
        t0 = time.perf_counter()
        fusion_svc = SceneFusionService(mock_mode=False, config=model_cfg)
        scene = fusion_svc.fuse(test_img)
        dt = (time.perf_counter() - t0) * 1000
        print(f"SUCCESS: Fused Scene in {dt:.1f}ms")
        print(f"  - Objects: {len(scene.objects)}")
        print(f"  - Surfaces: {len(scene.surfaces)}")
        print(f"  - Relationships: {len(scene.relationships)}")
        print(f"  - Floor Plane Detected: {scene.room_layout.get('floor_plane_detected') if scene.room_layout else False}")
        print(f"  - Camera Height: {scene.room_layout.get('camera_height_m') if scene.room_layout else None}m")
        print(f"  - Room Dimensions: {scene.room_layout.get('room_width_m')}m x {scene.room_layout.get('room_length_m')}m x {scene.room_layout.get('room_height_m')}m")
        print(f"  - Valid 3D Points: {scene.n_valid_points}")
        
        # Verify furniture geometry
        for obj in scene.objects[:3]:
            geom = obj.geometry
            if geom:
                print(f"    * {obj.semantic_class} ({obj.object_id}): centroid={geom.centroid}, extent={geom.extent}, reliable={geom.is_dimension_reliable}, raw_floor_delta={geom.raw_floor_contact_delta_m}m, post_floor_delta={geom.post_floor_contact_delta_m}m")
        
        audit_results["single_view_fusion"] = {
            "status": "COMPLETE",
            "objects_count": len(scene.objects),
            "surfaces_count": len(scene.surfaces),
            "valid_points": scene.n_valid_points,
            "floor_detected": scene.room_layout.get("floor_plane_detected") if scene.room_layout else False,
            "latency_ms": dt,
        }
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        audit_results["single_view_fusion"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 8. MULTI-VIEW 3D SCENE FUSION
    # -------------------------------------------------------------
    print("\n--- [8] Multi-View 3D Scene Fusion: MultiViewFusionService ---")
    try:
        t0 = time.perf_counter()
        mv_svc = MultiViewFusionService(mock_mode=False, config=model_cfg)
        # Test validation with single image
        v_imgs, warnings, metrics = mv_svc.validate_multiview_sequence([test_img])
        print(f"Validation Single Image: valid={len(v_imgs)}, warnings={warnings}")
        
        # Test fuse_multiview with 1 image
        mv_scene = mv_svc.fuse_multiview([test_img], refine_depth=True)
        dt = (time.perf_counter() - t0) * 1000
        print(f"SUCCESS: Multi-View (1 image) Fused in {dt:.1f}ms: {len(mv_scene.objects)} objects, {mv_scene.n_valid_points} points")
        audit_results["multi_view_fusion"] = {
            "status": "COMPLETE",
            "objects_count": len(mv_scene.objects),
            "valid_points": mv_scene.n_valid_points,
            "latency_ms": dt,
        }
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["multi_view_fusion"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 9. EXPORTERS (GLB & PLY)
    # -------------------------------------------------------------
    print("\n--- [9] Exporters: GLB and PLY ---")
    try:
        pts = scene.point_cloud.get("xyz", [])
        rgbs = scene.point_cloud.get("rgb", [])
        ply_bytes = export_point_cloud_ply(pts[:5000], rgbs[:5000] if rgbs else None)
        glb_bytes = export_scene_glb(point_cloud={"xyz": pts[:5000]}, objects=scene.objects)
        print(f"SUCCESS: PLY exported {len(ply_bytes):,} bytes | GLB exported {len(glb_bytes):,} bytes")
        audit_results["exporters"] = {
            "status": "COMPLETE",
            "ply_size_bytes": len(ply_bytes),
            "glb_size_bytes": len(glb_bytes),
        }
    except Exception as e:
        print(f"FAILED: {e}")
        audit_results["exporters"] = {"status": "BROKEN", "error": str(e)}

    # -------------------------------------------------------------
    # 10. OPTIONAL / OFFLINE SUBSYSTEMS (VLM, Classifier, Generator)
    # -------------------------------------------------------------
    print("\n--- [10] Graceful Degradation / Availability Audit ---")
    # VLM
    vlm_svc = VisionLanguageService(mock_mode=False, config=model_cfg.get("vision_language", {}))
    vlm_res = vlm_svc.run(test_img, "What furniture is in the room?")
    print(f"VLM Real Mode: success={vlm_res['success']}, is_implemented={vlm_res['is_implemented']}, error_msg='{vlm_res['error_message'][:60]}...'")
    audit_results["vlm"] = {"status": "BLOCKED BY EXTERNAL REQUIREMENT" if not vlm_res['is_implemented'] else "COMPLETE", "is_implemented": vlm_res['is_implemented']}

    # Classifier
    cls_svc = ClassificationService(mock_mode=False, config=model_cfg.get("classification", {}))
    cls_res = cls_svc.run(test_img)
    print(f"Classifier Real Mode: success={cls_res['success']}, is_implemented={cls_res['is_implemented']}")
    audit_results["classification"] = {"status": "NOT IMPLEMENTED" if not cls_res['is_implemented'] else "COMPLETE", "is_implemented": cls_res['is_implemented']}

    # Generator
    gen_svc = GenerationService(mock_mode=False, config=model_cfg.get("generation", {}))
    from schemas.requests import DesignRequest
    gen_res = gen_svc.run(test_img, DesignRequest(target_style="Modern"))
    print(f"Generator Real Mode: success={gen_res['success']}, is_implemented={gen_res['is_implemented']}")
    audit_results["generation"] = {"status": "NOT IMPLEMENTED" if not gen_res['is_implemented'] else "COMPLETE", "is_implemented": gen_res['is_implemented']}

    # Model Registry Spec Audit
    print("\n--- [11] ModelRegistry ModelSpec Audit ---")
    reg_specs = GLOBAL_MODEL_REGISTRY._registry
    print(f"Registered model specs count: {len(reg_specs)}")
    for mid, spec in reg_specs.items():
        exists_disk = spec.exists()
        print(f"  - [{mid}] task={spec.task_id}, primary={spec.is_primary}, exists_on_disk={exists_disk}")

    # Peak VRAM
    if cuda_avail:
        peak_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"\nAudit Peak GPU Allocated VRAM: {peak_mb:.1f} MB (Headroom: {total_vram - peak_mb:.1f} MB)")

    print("\nAudit Summary Complete.")
    return audit_results

if __name__ == "__main__":
    main()

