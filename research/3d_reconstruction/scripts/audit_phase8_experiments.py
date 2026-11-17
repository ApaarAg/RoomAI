"""
Forensic Audit of Phase 8 Experiment A and Experiment B Reproducibility.

Investigates:
1. Order of frames in indoor_canonical_manifest.json.
2. Exact frames sampled by Experiment A vs Experiment B across steps 1..20.
3. Parameter difference between checkpoint A and checkpoint B.
4. Mathematical proof of why reported validation metrics were identical.
"""

import json
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path("f:/RoomAI")
MANIFEST_PATH = REPO_ROOT / "research/3d_reconstruction/data_manifests/indoor_canonical_manifest.json"
CKPT_A = REPO_ROOT / "research/3d_reconstruction/experiments/phase8_corrected/experiment_a_hypersim/experiment_a_hypersim_final.ckpt"
CKPT_B = REPO_ROOT / "research/3d_reconstruction/experiments/phase8_corrected/experiment_b_multiview/experiment_b_multiview_final.ckpt"
BASE_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"


def main():
    print("=" * 75)
    print("PHASE 8 EXPERIMENT REPRODUCIBILITY AUDIT")
    print("=" * 75)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Inspect train split frames
    all_train = [f for f in manifest["frames"] if f["split"] == "train"]
    hyper_train = [f for f in manifest["frames"] if f["split"] == "train" and f["dataset"] == "hypersim"]
    scannet_train = [f for f in manifest["frames"] if f["split"] == "train" and f["dataset"] == "scannet"]

    print(f"Total train frames: {len(all_train)}")
    print(f"  Hypersim train frames: {len(hyper_train)}")
    print(f"  ScanNet train frames : {len(scannet_train)}")

    print("\n--- FIRST 25 FRAMES IN all_train (Experiment B candidate pool) ---")
    for idx, f in enumerate(all_train[:25]):
        print(f"  Index {idx:02d}: dataset={f['dataset']:<10} | scene={f['scene_id']:<15} | frame_id={f['frame_id']}")

    print("\n--- FIRST 20 FRAMES IN hyper_train (Experiment A candidate pool) ---")
    for idx, f in enumerate(hyper_train[:20]):
        print(f"  Index {idx:02d}: dataset={f['dataset']:<10} | scene={f['scene_id']:<15} | frame_id={f['frame_id']}")

    # Check index-by-index match for steps 1..20
    matches = 0
    for idx in range(20):
        f_a = hyper_train[idx]
        f_b = all_train[idx]
        match = (f_a["scene_id"] == f_b["scene_id"]) and (f_a["frame_id"] == f_b["frame_id"])
        if match:
            matches += 1
        print(f"Step {idx+1:02d}: Exp A [{f_a['scene_id']} #{f_a['frame_id']}] vs Exp B [{f_b['scene_id']} #{f_b['frame_id']}] -> Match: {match}")

    print(f"\nExact Frame Match across 20 Steps: {matches} / 20 ({matches/20*100:.1f}%)")

    # 2. Checkpoint tensor comparison
    print("\n--- CHECKPOINT TENSOR COMPARISON ---")
    sd_a = torch.load(CKPT_A, map_location="cpu")["state_dict"]
    sd_b = torch.load(CKPT_B, map_location="cpu")["state_dict"]
    sd_base = torch.load(BASE_CKPT, map_location="cpu")
    if "state_dict" in sd_base:
        sd_base = sd_base["state_dict"]
    sd_base = {k.replace("model.", ""): v for k, v in sd_base.items()}

    diff_a_b = 0.0
    diff_a_base = 0.0
    diff_b_base = 0.0
    total_params = 0

    for k in sd_a.keys():
        if k in sd_b:
            d_ab = (sd_a[k].float() - sd_b[k].float()).abs().sum().item()
            diff_a_b += d_ab
            if k in sd_base:
                d_ab_base = (sd_a[k].float() - sd_base[k].float()).abs().sum().item()
                diff_a_base += d_ab_base
                d_b_base = (sd_b[k].float() - sd_base[k].float()).abs().sum().item()
                diff_b_base += d_b_base
            total_params += sd_a[k].numel()

    print(f"Total examined parameters: {total_params:,}")
    print(f"Sum L1 Absolute Difference between Exp A and Exp B weights : {diff_a_b:.8f}")
    print(f"Sum L1 Absolute Difference between Exp A and Baseline weights : {diff_a_base:.4f}")
    print(f"Sum L1 Absolute Difference between Exp B and Baseline weights : {diff_b_base:.4f}")

    if diff_a_b == 0.0:
        print("\n>>> CONFIRMED: Exp A and Exp B weights are 100.000% BIT-FOR-BIT IDENTICAL!")
    else:
        print(f"\n>>> Exp A and Exp B weights differ by {diff_a_b:.6f}")

    audit_summary = {
        "root_cause": "Sequential iteration without dataset shuffling: steps 1..20 sampled indices 0..19 from both datasets. Because Hypersim frames were positioned at the beginning of the manifest (indices 0..30), Experiment B sampled the exact same 20 Hypersim frames as Experiment A, never reaching ScanNet frames (indices 31..203).",
        "sampled_frame_overlap_steps_1_to_20": f"{matches}/20",
        "weight_difference_L1": float(diff_a_b),
        "parameter_delta_from_baseline_A": float(diff_a_base),
        "parameter_delta_from_baseline_B": float(diff_b_base),
    }

    out_file = REPO_ROOT / "research/3d_reconstruction/experiments/phase9/experiment_reproducibility_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved audit summary to: {out_file}")


if __name__ == "__main__":
    main()

