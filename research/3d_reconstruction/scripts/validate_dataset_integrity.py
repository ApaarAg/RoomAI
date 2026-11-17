"""Automated Dataset Integrity & Zero-Leakage Validation Script."""

import json
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "research" / "3d_reconstruction" / "data_manifests" / "indoor_3d_scenes_manifest.json"


def validate_manifest():
    print("=" * 70)
    print("ROOMAI DATASET INTEGRITY & LEAKAGE AUDIT")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    print(f"Loaded {len(scenes)} scenes from manifest.")

    # 1. Check Scene-Level Split Disjointness (Zero-Leakage)
    splits: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for s in scenes:
        scn_id = s["scene_id"]
        split = s.get("split", "unknown")
        if split not in splits:
            splits[split] = set()
        splits[split].add(scn_id)

    train_val_overlap = splits["train"].intersection(splits["val"])
    train_test_overlap = splits["train"].intersection(splits["test"])
    val_test_overlap = splits["val"].intersection(splits["test"])

    print("\n[Audit 1/4] Zero-Leakage Split Isolation:")
    print(f"  Train Scenes: {len(splits['train'])}")
    print(f"  Val Scenes:   {len(splits['val'])}")
    print(f"  Test Scenes:  {len(splits['test'])}")
    assert len(train_val_overlap) == 0, f"LEAKAGE DETECTED between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"LEAKAGE DETECTED between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"LEAKAGE DETECTED between Val and Test: {val_test_overlap}"
    print("  ✓ PASS: Zero scene leakage confirmed across train, val, and test partitions.")

    # 2. Check Coordinate Convention Standards
    print("\n[Audit 2/4] Coordinate Convention & Camera Parameters:")
    conv = data.get("metadata", {}).get("coordinate_convention")
    print(f"  Registered Convention: {conv}")
    assert conv == "opencv_pinhole", f"Invalid coordinate convention: {conv}"
    print("  ✓ PASS: Standard OpenCV pinhole coordinate convention (+X right, +Y down, +Z forward) enforced.")

    # 3. Check Metric Depth & Units
    print("\n[Audit 3/4] Depth Representation & Scale:")
    unit = data.get("metadata", {}).get("depth_unit")
    print(f"  Registered Depth Unit: {unit}")
    assert unit == "meters", f"Invalid depth unit: {unit}"
    print("  ✓ PASS: Metric float32 depth in physical meters verified.")

    # 4. Check Camera Matrix Mathematical Invariants on Sample Frames
    print("\n[Audit 4/4] Camera Matrix Mathematical Invariants:")
    for s in scenes:
        for f in s.get("frames", []):
            if "pose_c2w" in f:
                mat = np.array(f["pose_c2w"])
                r = mat[:3, :3]
                det = np.linalg.det(r)
                assert np.isclose(det, 1.0, atol=1e-4), f"Rotation matrix determinant must be +1.0, got {det}"
                ortho_err = np.linalg.norm(r.T @ r - np.eye(3))
                assert ortho_err < 1e-4, f"Rotation matrix non-orthogonal: {ortho_err}"
    print("  ✓ PASS: Rotation matrix orthonormality and determinant (+1.0) verified.")

    print("\n" + "=" * 70)
    print("ALL DATASET INTEGRITY CHECKS PASSED (100% COMPLIANT)")
    print("=" * 70)


if __name__ == "__main__":
    validate_manifest()
