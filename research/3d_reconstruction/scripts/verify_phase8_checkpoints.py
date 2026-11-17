"""Verify and record SHA-256 hashes for all preserved checkpoints in Phase 8."""
import hashlib
import json
import os
from pathlib import Path

checkpoints = {
    "production_model": "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt",
    "best_stage1": "research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt",
    "stage2_real_verified": "research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt",
}

results = {}
for name, p in checkpoints.items():
    path = Path(p)
    if not path.exists():
        print(f"MISSING: {p}")
        continue
    h = hashlib.sha256()
    size_bytes = path.stat().st_size
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    sha256_hash = h.hexdigest()
    results[name] = {
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "sha256": sha256_hash,
    }
    print(f"{name}: {sha256_hash} ({results[name]['size_mb']} MB)")

exp_dir = Path("research/3d_reconstruction/experiments/phase8_corrected")
exp_dir.mkdir(parents=True, exist_ok=True)
manifest_path = exp_dir / "preserved_checkpoints_phase8.json"
with open(manifest_path, "w") as f:
    json.dump(results, f, indent=2)
print("Saved preserved manifest to:", manifest_path)

