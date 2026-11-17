"""
Phase 9 Checkpoint Preservation & Verification.
Audits SHA-256 hashes of all baseline, historical, and Phase 8 checkpoints.
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path("f:/RoomAI")
CHECKPOINTS = {
    "production_model": REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt",
    "best_stage1": REPO_ROOT / "research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt",
    "stage2_real_verified": REPO_ROOT / "research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt",
    "phase8_exp_a": REPO_ROOT / "research/3d_reconstruction/experiments/phase8_corrected/experiment_a_hypersim/experiment_a_hypersim_final.ckpt",
    "phase8_exp_b": REPO_ROOT / "research/3d_reconstruction/experiments/phase8_corrected/experiment_b_multiview/experiment_b_multiview_final.ckpt",
}

results = {}
for name, p in CHECKPOINTS.items():
    if not p.exists():
        print(f"MISSING: {p}")
        continue
    h = hashlib.sha256()
    size_bytes = p.stat().st_size
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    digest = h.hexdigest()
    results[name] = {
        "path": str(p),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "sha256": digest,
    }
    print(f"[{name:<20}] {digest} ({results[name]['size_mb']} MB)")

phase9_dir = REPO_ROOT / "research/3d_reconstruction/experiments/phase9"
phase9_dir.mkdir(parents=True, exist_ok=True)
manifest_path = phase9_dir / "preserved_checkpoints_phase9.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nPreserved manifest saved to: {manifest_path}")

