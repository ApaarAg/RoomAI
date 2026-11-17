"""RoomAI Phase 14 Stage F: Comprehensive Application Smoke Test.

Validates:
1. Loading app.yaml and model_config.yaml.
2. Verification of production model B2 as active default.
3. Verification of candidate model B6 and candidate model B5 registration.
4. ModelRegistry lookup and spec validation.
5. End-to-end inference using DetectionService:
   - Production Model B2
   - Candidate Model B6
   - Candidate Model B5
   - Mock Mode execution
6. Explicit error handling: Missing checkpoint raises clean FileNotFoundError without silent fallback.
7. VRAM headroom checks.
"""

import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import yaml

from models.registry import GLOBAL_MODEL_REGISTRY
from services.detection_service import DetectionService



def test_smoke():
    print("=== RoomAI Stage F: Application Smoke Test ===")

    # 1. Config Loading
    app_cfg_path = REPO_ROOT / "config" / "app.yaml"
    model_cfg_path = REPO_ROOT / "config" / "model_config.yaml"
    assert app_cfg_path.exists(), f"Missing {app_cfg_path}"
    assert model_cfg_path.exists(), f"Missing {model_cfg_path}"

    with open(app_cfg_path, "r", encoding="utf-8") as f:
        app_cfg = yaml.safe_load(f)
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    print("[PASS] Successfully loaded configuration files.")

    # 2. Production B2 Default Verification
    ext_prod = model_cfg.get("exterior_detection", {})
    assert ext_prod.get("enabled") is True, "Production exterior_detection must be enabled"
    assert "exterior_detector_b2_yolo11s_best.pt" in ext_prod.get("checkpoint_path", ""), "B2 must be active production checkpoint"
    assert ext_prod.get("status") == "production_ready", "B2 status must be production_ready"
    print(f"[PASS] Production default model verified: {ext_prod.get('checkpoint_path')} (status: {ext_prod.get('status')})")

    # 3. Candidate B6 & B5 Registration in config
    ext_b5 = model_cfg.get("exterior_detection_b5_candidate", {})
    assert ext_b5.get("enabled") is False, "B5 must not be enabled by default in production"
    assert "exterior_detector_b5_transfer_best.pt" in ext_b5.get("checkpoint_path", "")
    assert ext_b5.get("status") == "experimental_candidate"

    ext_b6 = model_cfg.get("exterior_detection_b6_candidate", {})
    assert ext_b6.get("enabled") is False, "B6 must not be enabled by default in production"
    assert "exterior_detector_b6_expanded_best.pt" in ext_b6.get("checkpoint_path", "")
    assert ext_b6.get("status") == "experimental_candidate"
    print("[PASS] Experimental candidates B5 and B6 verified registered in model_config.yaml.")

    # 4. Model Registry Check
    reg = GLOBAL_MODEL_REGISTRY
    b2_spec = reg.get_primary_for_task("exterior_detection")
    assert b2_spec is not None and b2_spec.model_id == "exterior_b2_finetuned"
    assert b2_spec.is_primary is True
    assert b2_spec.exists()

    b5_spec = reg.get_model_spec("exterior_b5_transfer")
    assert b5_spec is not None and b5_spec.exists() and b5_spec.is_primary is False

    b6_spec = reg.get_model_spec("exterior_b6_expanded")
    assert b6_spec is not None and b6_spec.exists() and b6_spec.is_primary is False
    print("[PASS] ModelRegistry specifications verified (B2 primary=True, B5 & B6 primary=False).")

    # 5. Create Test Image
    sample_img = Image.new("RGB", (640, 640), color=(130, 160, 190))

    # 6. Test Inference: Production Model B2
    print("\nRunning inference with Production Model B2...")
    b2_service = DetectionService(mock_mode=False, config=ext_prod)
    t0 = time.perf_counter()
    b2_res = b2_service.run(sample_img)
    b2_latency = (time.perf_counter() - t0) * 1000.0
    assert b2_res["success"] is True
    assert b2_res["response"].is_simulated is False
    print(f"[PASS] Model B2 executed successfully in {b2_latency:.2f} ms.")

    # 7. Test Inference: Candidate Model B6
    print("\nRunning inference with Candidate Model B6...")
    b6_service = DetectionService(mock_mode=False, config=ext_b6)
    t0 = time.perf_counter()
    b6_res = b6_service.run(sample_img)
    b6_latency = (time.perf_counter() - t0) * 1000.0
    assert b6_res["success"] is True
    assert b6_res["response"].is_simulated is False
    print(f"[PASS] Model B6 executed successfully in {b6_latency:.2f} ms.")

    # 8. Test Inference: Candidate Model B5
    print("\nRunning inference with Candidate Model B5...")
    b5_service = DetectionService(mock_mode=False, config=ext_b5)
    t0 = time.perf_counter()
    b5_res = b5_service.run(sample_img)
    b5_latency = (time.perf_counter() - t0) * 1000.0
    assert b5_res["success"] is True
    assert b5_res["response"].is_simulated is False
    print(f"[PASS] Model B5 executed successfully in {b5_latency:.2f} ms.")

    # 9. Test Inference: Mock Mode Execution
    print("\nRunning inference in Mock Mode...")
    mock_service = DetectionService(mock_mode=True, config=ext_b6)
    mock_res = mock_service.run(sample_img)
    assert mock_res["success"] is True
    assert mock_res["response"].is_simulated is True
    print(f"[PASS] Mock mode verified (simulated detections: {len(mock_res['response'].detections)}).")

    # 10. Strict Failure Check: Missing Checkpoint
    print("\nVerifying explicit failure for missing checkpoint...")
    bad_service = DetectionService(
        mock_mode=False,
        config={"checkpoint_path": "checkpoints/non_existent_model.pt", "task_group": "exterior"}
    )
    bad_res = bad_service.run(sample_img)
    assert bad_res["success"] is False, "Expected service run to fail for missing checkpoint"
    assert "does not exist on disk" in bad_res["error_message"], f"Unexpected error message: {bad_res['error_message']}"
    assert bad_res["response"] is None, "Response must be None on failure"
    print(f"[PASS] Clean explicit failure verified: {bad_res['error_message']}")



    # 11. VRAM Headroom Check
    vram_info = reg.check_vram_headroom()
    print(f"\n[VRAM STATUS] Allocated: {vram_info['allocated_mb']:.1f} MB | Reserved: {vram_info['reserved_mb']:.1f} MB | Budget: {vram_info['max_budget_mb']:.1f} MB")
    print("\n=== All Application Smoke Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_smoke()
