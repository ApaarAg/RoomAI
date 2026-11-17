"""Re-export an existing canonical reconstruction to both GLB tiers and run Khronos validation.

The input NPZ must be a preserved RoomAI point-cloud artifact; this script does not
create or infer geometry. Outputs are additive and the destination must be empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.export_3d import export_colored_point_cloud_glb, extract_glb_point_cloud, validate_glb_bytes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Existing canonical point_cloud_canonical.npz")
    parser.add_argument("--output", type=Path, required=True, help="New empty output directory")
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"Input reconstruction artifact does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cloud = np.load(source, allow_pickle=False)
    xyz, rgb = cloud["xyz"], cloud["rgb"]
    tiers = {
        "desktop": (40000, output / "scene_reconstruction.glb"),
        "mobile": (15000, output / "scene_reconstruction_mobile.glb"),
    }
    node = shutil.which("node")
    validator_script = ROOT / "scripts" / "validate_glb_khronos.mjs"
    if not node or not validator_script.is_file():
        raise RuntimeError("External Khronos validation unavailable; exports are not accepted.")

    results = {}
    for name, (limit, path) in tiers.items():
        payload = export_colored_point_cloud_glb(xyz, rgb, path, max_points=limit, sampling_seed=0, units="m")
        internal = validate_glb_bytes(payload)
        if not internal.get("is_valid"):
            raise RuntimeError(f"Internal GLB validation failed for {name}: {internal.get('error')}")
        gltf = internal["json_data"]
        primitive = gltf["meshes"][0]["primitives"][0]
        attributes = primitive["attributes"]
        position_accessor = gltf["accessors"][attributes["POSITION"]]
        color_accessor = gltf["accessors"][attributes["COLOR_0"]]
        position_view = gltf["bufferViews"][position_accessor["bufferView"]]
        color_view = gltf["bufferViews"][color_accessor["bufferView"]]
        node_entry = gltf["nodes"][gltf["scenes"][gltf["scene"]]["nodes"][0]]
        run = subprocess.run([node, str(validator_script), str(path)], capture_output=True, text=True, timeout=60)
        if not run.stdout.strip():
            raise RuntimeError(f"Khronos validation produced no report for {name}: {run.stderr[-2000:]}")
        external = json.loads(run.stdout)
        (output / f"khronos_{name}.json").write_text(json.dumps(external, indent=2), encoding="utf-8")
        positions, colors = extract_glb_point_cloud(payload)
        passed = run.returncode == 0 and external["errors"] == 0
        results[name] = {
            "status": "PASS" if passed else "FAIL",
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "point_count": int(len(positions)),
            "rgb_count": int(len(colors)),
            "glb_structure": {
                "primitive_mode": primitive["mode"],
                "attributes": sorted(attributes),
                "position_accessor": {key: position_accessor.get(key) for key in ("componentType", "count", "type", "min", "max")},
                "color_accessor": {key: color_accessor.get(key) for key in ("componentType", "normalized", "count", "type")},
                "position_buffer_view": {key: position_view.get(key) for key in ("buffer", "byteOffset", "byteLength", "byteStride", "target")},
                "color_buffer_view": {key: color_view.get(key) for key in ("buffer", "byteOffset", "byteLength", "byteStride", "target")},
                "buffer_byte_length": gltf["buffers"][0]["byteLength"],
                "node_transform": {key: node_entry.get(key) for key in ("matrix", "translation", "rotation", "scale")},
                "roomai_metadata": gltf.get("extras", {}).get("roomai"),
            },
            "position_bounds": {"min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            "internal_validation": {k: v for k, v in internal.items() if k not in ("binary", "json_data")},
            "khronos_report": f"khronos_{name}.json",
            "khronos_errors": external["errors"],
            "khronos_warnings": external["warnings"],
        }
    report = {
        "schema": "roomai.phase4_3_glb_validation",
        "validator": "Khronos glTF Validator",
        "validator_version": results["desktop"]["khronos_report"] and json.loads((output / results["desktop"]["khronos_report"]).read_text())["validatorVersion"],
        "source_point_cloud": str(source),
        "source_point_cloud_sha256": sha256(source),
        "tiers": results,
        "overall": "PASS" if all(tier["status"] == "PASS" for tier in results.values()) else "FAIL",
    }
    (output / "glb_validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
