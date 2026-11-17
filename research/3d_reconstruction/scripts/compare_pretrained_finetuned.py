"""
ROOMAI — PRETRAINED VS FINE-TUNED COMPARISON & PRODUCTION PROMOTION ENGINE
Implements mandatory Section 11 criteria to select and promote the winning checkpoint.
"""
import sys
import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    print(f"\n{'='*70}\nROOMAI: FINAL MODEL COMPARISON & PRODUCTION PROMOTION GATE\n{'='*70}")

    val_history_dir = Path("research/3d_reconstruction/experiments/validation_history")
    val_files = list(val_history_dir.glob("*.json"))
    if not val_files:
        print("Error: No validation history files found!")
        return

    latest_val = sorted(val_files, key=lambda f: f.stat().st_mtime)[-1]
    with open(latest_val, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    test_file = Path("research/3d_reconstruction/experiments/test_results.json")
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    else:
        test_data = {}

    print(f"Candidate Checkpoint: {val_data['checkpoint']}")
    print(f"Validation Point-Map Score: {val_data['pointmap_score']:.4f} (Baseline: 1.0000)")
    print(f"Validation Guardrails Passed: {val_data['guardrails_passed']}")

    # Promotion evaluation
    candidate_wins = (val_data["guardrails_passed"] and val_data["pointmap_score"] < 1.0)

    prod_dir = Path("research/3d_reconstruction/checkpoints/production_candidate")
    prod_dir.mkdir(parents=True, exist_ok=True)
    winner_ckpt_path = prod_dir / "production_model.ckpt"

    if candidate_wins:
        print("\nDECISION: FINE-TUNED MODEL WINS PROMOTION GATE!")
        src_ckpt = Path(val_data["checkpoint"])
        shutil.copy2(src_ckpt, winner_ckpt_path)
        winner_sha256 = compute_sha256(winner_ckpt_path)
        winner_status = "FINE_TUNED_PROMOTED"
    else:
        print("\nDECISION: PRETRAINED MODEL PRESERVED AS WINNER!")
        winner_sha256 = "PRETRAINED_BASELINE"
        winner_status = "PRETRAINED_BASELINE_PRESERVED"

    manifest = {
        "model_version": "WorldMirror-1.0 (ICML 2026)",
        "promotion_status": winner_status,
        "winner_checkpoint": str(winner_ckpt_path) if candidate_wins else "pretrained/model.safetensors",
        "sha256": winner_sha256,
        "validation_score": val_data.get("pointmap_score"),
        "validation_guardrails_passed": val_data.get("guardrails_passed"),
        "test_metrics": test_data,
        "promoted_at": datetime.utcnow().isoformat() + "Z",
        "limitations": [
            "Trained for indoor feed-forward single and multi-view reconstruction.",
            "Local RTX 4060 has 8GB VRAM limit; fine-tuning requires 12GB peak VRAM (cloud compute required for full runs).",
            "Inference is fully supported locally on RTX 4060 at 3.4GB VRAM in BF16."
        ]
    }

    manifest_path = prod_dir / "candidate_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Production candidate manifest generated at: {manifest_path}")

if __name__ == "__main__":
    main()

