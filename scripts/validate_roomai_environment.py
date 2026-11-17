"""RoomAI Environment & Production Artifact Pre-flight Validation.

Validates:
1. Python runtime (>= 3.10)
2. PyTorch and CUDA availability on NVIDIA RTX 4060 GPU
3. Memory and VRAM readiness
4. Required dependencies
5. Production checkpoint existence and SHA-256 integrity
6. Application configurations (mock_mode == False default)
"""

import hashlib
from pathlib import Path
import sys
import PIL.Image
import yaml

CHECKPOINTS_TO_VERIFY = {
    "Indoor Detector A3 (YOLO11s 800px)": {
        "path": Path("checkpoints/indoor_detector_a3_yolo11s_800px_best.pt"),
        "sha256": "6cbc9e58f87156e07dd0e4e712b8aa96097aeb514a473621139a54cb83cec576",
        "expected_bytes": 19173786,
    },
    "Exterior Detector B2 (YOLO11s)": {
        "path": Path("checkpoints/exterior_detector_b2_yolo11s_best.pt"),
        "sha256": "c599c19241b8c60390ac0c4914de6cef449bc1129f73b0db7f8f04fc62a0459f",
        "expected_bytes": 19163034,
    },
    "Parking Detector C2 (YOLO11s)": {
        "path": Path("checkpoints/parking_detector_c2_yolo11s_pilot_best.pt"),
        "sha256": "4abab963ab0becfef4a03c4e7b0039cc048d7621b3456bc5435deece6eed1470",
        "expected_bytes": 19157978,
    },
    "Instance Segmentation (YOLO11n-seg)": {
        "path": Path("checkpoints/yolo11n-seg.pt"),
        "sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
        "expected_bytes": 6182636,
    },
    "Metric Depth Estimation (Depth Anything V2 Metric Small)": {
        "path": Path("checkpoints/depth_anything_v2_metric_indoor_small/model.safetensors"),
        "sha256": "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3",
        "expected_bytes": 99173660,
    },
    "Relative Depth Estimation (Depth Anything V2 Small)": {
        "path": Path("checkpoints/depth_anything_v2_small/model.safetensors"),
        "sha256": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
        "expected_bytes": 99173660,
    },
}

REQUIRED_PACKAGES = [
    "PIL",
    "numpy",
    "scipy",
    "yaml",
    "pydantic",
    "plotly",
    "streamlit",
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_environment() -> bool:
    print("=" * 70)
    print("ROOMAI LOCAL WINDOWS ENVIRONMENT PRE-FLIGHT VERIFICATION")
    print("=" * 70)
    all_ok = True

    # 1. Python version
    py_ver = sys.version_info
    print(f"\n[1] Python Runtime: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 10):
        print("  ❌ ERROR: Python 3.10+ required.")
        all_ok = False
    else:
        print("  ✅ Python version compatible.")

    # 2. Package imports
    print("\n[2] Required Dependencies:")
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg:<16} : Available")
        except ImportError as e:
            print(f"  ❌ {pkg:<16} : MISSING ({e})")
            all_ok = False

    # 3. PyTorch & CUDA
    print("\n[3] PyTorch & Hardware Acceleration:")
    import torch
    print(f"  PyTorch Version : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available  : {cuda_avail}")
    if not cuda_avail:
        print("  ❌ ERROR: CUDA acceleration not available. RTX 4060 GPU required.")
        all_ok = False
    else:
        device_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"  Device Name     : {device_name}")
        print(f"  Total VRAM      : {vram_mb:.1f} MB ({vram_mb / 1024:.2f} GB)")
        if vram_mb < 7000:
            print("  ⚠️ WARNING: VRAM below 7 GB target.")
        else:
            print("  ✅ Target GPU hardware verified.")

    # 4. Checkpoints & SHA-256 Hashes
    print("\n[4] Production Checkpoints Integrity:")
    for name, meta in CHECKPOINTS_TO_VERIFY.items():
        p = meta["path"]
        if not p.exists():
            print(f"  ❌ {name:<45} : MISSING at {p}")
            all_ok = False
            continue
        actual_bytes = p.stat().st_size
        actual_hash = compute_sha256(p)
        expected_hash = meta["sha256"]
        if actual_hash != expected_hash:
            print(f"  ❌ {name:<45} : HASH MISMATCH!")
            print(f"     Expected: {expected_hash}")
            print(f"     Actual:   {actual_hash}")
            all_ok = False
        else:
            print(f"  ✅ {name:<45} : VERIFIED ({actual_bytes / 1e6:.1f} MB, {actual_hash[:12]}...)")

    # 5. Configuration Defaults
    print("\n[5] Application Configuration Verification:")
    app_cfg_path = Path("config/app.yaml")
    if not app_cfg_path.exists():
        print(f"  ❌ Config missing: {app_cfg_path}")
        all_ok = False
    else:
        with open(app_cfg_path, "r", encoding="utf-8") as f:
            app_cfg = yaml.safe_load(f)
        mock_mode = app_cfg.get("mock_mode")
        if mock_mode is not False:
            print(f"  ❌ ERROR: mock_mode in config/app.yaml is {mock_mode}. Must be false.")
            all_ok = False
        else:
            print("  ✅ config/app.yaml: mock_mode is strictly False (Real inference default).")

    print("\n" + "=" * 70)
    if all_ok:
        print("PRE-FLIGHT VALIDATION PASSED: System is ready for local deployment.")
        print("=" * 70)
        return True
    else:
        print("PRE-FLIGHT VALIDATION FAILED: Correct the above errors before deploying.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
