"""Create the browser texture for the verified ROOMAI raw-depth exhibit.

The source .npy remains authoritative. This exporter only writes a contiguous,
little-endian Float32 payload that the browser loads into a Three.js DataTexture,
plus a provenance record beside it. No depth values are normalized, inverted, or
otherwise modified here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/jobs/job_1794499397_4aa219ea/depth/view_00_raw.npy"
OUTPUT_DIR = ROOT / "web/public/assets/roomai/exhibit-01/depth"
OUTPUT = OUTPUT_DIR / "view_00_raw.f32"
METADATA = OUTPUT_DIR / "view_00_raw.metadata.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    depth = np.load(SOURCE, allow_pickle=False)
    if depth.shape != (518, 518) or depth.dtype != np.float32:
        raise ValueError(f"Expected a 518x518 float32 depth map, received {depth.shape} {depth.dtype}")
    if not np.isfinite(depth).all():
        raise ValueError("Verified raw depth contains non-finite values")

    # This changes storage format only: NPY's Float32 samples become a browser
    # DataTexture's row-major, little-endian Float32 samples.
    browser_values = np.ascontiguousarray(depth.astype("<f4", copy=False))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(browser_values.tobytes(order="C"))
    restored = np.frombuffer(OUTPUT.read_bytes(), dtype="<f4").reshape(depth.shape)
    if not np.array_equal(depth, restored):
        raise RuntimeError("Browser depth payload did not round-trip exactly")

    METADATA.write_text(json.dumps({
        "source": {
            "upload_id": "upl_cee8234c7f93",
            "model": "worldmirror1_baseline",
            "job": "job_1794499397_4aa219ea",
            "artifact": "depth/view_00_raw.npy",
            "sha256": sha256(SOURCE),
        },
        "browser_asset": {
            "path": "depth/view_00_raw.f32",
            "format": "row-major little-endian Float32",
            "dimensions": [518, 518],
            "byte_length": OUTPUT.stat().st_size,
            "sha256": sha256(OUTPUT),
        },
        "conversion": {
            "operation": "identity Float32 sample export from NumPy NPY payload",
            "normalization": "none; rendering normalization is performed in the GPU shader",
            "vertical_orientation": "DataTexture flipY=true aligns NPY row zero with RGB texture top row",
        },
        "depth": {
            "dtype": "float32",
            "range": [float(depth.min()), float(depth.max())],
            "semantics": "raw model-native depth",
            "units": "UNKNOWN",
        },
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
