"""Expose the verified ROOMAI PLY as an unchanged browser asset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/jobs/job_1794499397_4aa219ea/scene_reconstruction.ply"
OUTPUT_DIR = ROOT / "web/public/assets/roomai/exhibit-01/point-cloud"
OUTPUT = OUTPUT_DIR / "scene_reconstruction.ply"
METADATA = OUTPUT_DIR / "scene_reconstruction.metadata.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = SOURCE.read_bytes()
    text = payload.decode("ascii")
    lines = text.strip().splitlines()
    header_end = lines.index("end_header")
    header = lines[:header_end + 1]
    vertex_count = int(next(line.split()[2] for line in header if line.startswith("element vertex ")))
    properties = [line.removeprefix("property ") for line in header if line.startswith("property ")]
    if vertex_count != 61 or len(lines) - header_end - 1 != vertex_count:
        raise ValueError("Verified job PLY no longer has its expected 61 vertices")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(payload)
    if SOURCE.read_bytes() != OUTPUT.read_bytes():
        raise RuntimeError("Browser PLY copy differs from the authoritative artifact")

    METADATA.write_text(json.dumps({
        "source": {
            "upload_id": "upl_cee8234c7f93",
            "model": "worldmirror1_baseline",
            "job": "job_1794499397_4aa219ea",
            "artifact": "scene_reconstruction.ply",
            "sha256": digest(SOURCE),
        },
        "browser_asset": {
            "path": "point-cloud/scene_reconstruction.ply",
            "format": "ASCII PLY 1.0",
            "vertex_count": vertex_count,
            "properties": properties,
            "sha256": digest(OUTPUT),
        },
        "coordinate_system": "RoomAI coordinates already transformed from model coordinates by diag(1, -1, -1); no browser conversion is applied.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
