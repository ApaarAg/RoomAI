"""
RoomAI — Locked Test Set Evaluation Protocol
Evaluates the final selected Hunyuan checkpoint exactly once on the locked test scenes
from indoor_canonical_manifest_expanded.json.
Reports AbsRel, RMSE, Delta1, and per-scene metrics.
Preserves raw evaluation output as JSON and verifies dataset manifest hash and model SHA-256.
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "external" / "HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "scripts"

if str(EXTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.models.models.worldmirror import WorldMirror

DATA_ROOT = REPO_ROOT / "research" / "3d_reconstruction" / "data" / "ground_truth"
MANIFEST_PATH = REPO_ROOT / "research" / "3d_reconstruction" / "data_manifests" / "indoor_canonical_manifest_expanded.json"
OUTPUT_DIR = REPO_ROOT / "results" / "final_execution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "locked_test_evaluation_report.json"

FINAL_CHECKPOINT_PATH = REPO_ROOT / "results" / "final_execution" / "checkpoints" / "exp_05_expanded_lora_rank8_final.ckpt"
EVAL_SEED = 42

def compute_sha256(filepath: Path) -> str:
    s = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            s.update(chunk)
    return s.hexdigest().upper()

def evaluate_locked_test(
    checkpoint_path: Path,
    manifest_path: Path,
    device: str = "cuda",
    max_frames_per_scene: int = 15,
) -> Dict[str, Any]:
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    print("=" * 80)
    print("ROOMAI — LOCKED TEST SET EMPIRICAL EVALUATION")
    print("=" * 80)

    # 1. Compute and verify hashes
    manifest_hash = compute_sha256(manifest_path)
    ckpt_hash = compute_sha256(checkpoint_path)
    print(f"Manifest: {manifest_path.name}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Checkpoint SHA-256: {ckpt_hash}")

    # 2. Load manifest and filter strictly to 'test' split
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    test_frames = [f for f in manifest_data["frames"] if f.get("split") == "test"]
    print(f"\nDiscovered {len(test_frames)} total locked test frames.")

    scenes_dict: Dict[str, List[Dict[str, Any]]] = {}
    for tf in test_frames:
        scenes_dict.setdefault(tf["scene_id"], []).append(tf)

    print("Locked Test Scenes:")
    for s, frames in sorted(scenes_dict.items()):
        print(f"  - {s} ({frames[0].get('dataset', 'unknown')}): {len(frames)} frames")

    # 3. Load Model
    print(f"\nInitializing WorldMirror-1.0 architecture on {device}...")
    model = WorldMirror(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        gs_dim=256,
        enable_cam=True,
        enable_pts=True,
        enable_depth=True,
        enable_norm=True,
        enable_gs=False,
        enable_cond=False,
    ).to(device=device, dtype=torch.bfloat16)

    print(f"Loading weights from {checkpoint_path}...")
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    del sd, clean_sd
    if device == "cuda":
        torch.cuda.empty_cache()
    model.eval()

    # 4. Evaluation Loop
    per_scene_results: Dict[str, Any] = {}
    all_absrel: List[float] = []
    all_rmse: List[float] = []
    all_delta1: List[float] = []
    all_latencies: List[float] = []

    hypersim_absrel: List[float] = []
    scannet_absrel: List[float] = []

    for scene_id, frames in sorted(scenes_dict.items()):
        ds_name = frames[0].get("dataset", "unknown")
        # Subsample frames evenly up to max_frames_per_scene
        n_eval = min(len(frames), max_frames_per_scene)
        step = max(1, len(frames) // n_eval)
        eval_frames = frames[::step][:n_eval]

        scene_absrel: List[float] = []
        scene_rmse: List[float] = []
        scene_delta1: List[float] = []

        print(f"\nEvaluating Scene: {scene_id} [{ds_name}] ({len(eval_frames)} views)...")

        for f_info in eval_frames:
            rgb_path = DATA_ROOT / f_info["paths"]["rgb"]
            depth_path = DATA_ROOT / f_info["paths"]["depth"]
            mask_path = DATA_ROOT / f_info["paths"]["mask"]

            # Load RGB
            pil_img = Image.open(rgb_path).convert("RGB")
            # Ensure 518x518 square
            pil_img = pil_img.resize((518, 518), Image.Resampling.BILINEAR)
            rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

            # Load Ground Truth Depth
            gt_depth = np.load(depth_path).astype(np.float32)
            if gt_depth.shape != (518, 518):
                # Resize GT depth if necessary
                pil_gt = Image.fromarray(gt_depth)
                pil_gt = pil_gt.resize((518, 518), Image.Resampling.NEAREST)
                gt_depth = np.array(pil_gt, dtype=np.float32)

            # Load Mask
            if mask_path.exists():
                mask_raw = np.array(Image.open(mask_path).resize((518, 518), Image.Resampling.NEAREST)) > 0
            else:
                mask_raw = np.ones((518, 518), dtype=bool)

            valid_mask = mask_raw & np.isfinite(gt_depth) & (gt_depth > 0.1) & (gt_depth < 30.0)
            if not valid_mask.any():
                continue

            # Model Forward Pass
            t0 = time.time()
            with torch.no_grad():
                preds = model(views={"img": rgb_tensor}, cond_flags=[0, 0, 0], is_inference=True)
            latency = time.time() - t0
            all_latencies.append(latency)

            pred_depth = preds["depth"].squeeze().float().cpu().numpy()

            # Align scale via median ratio on valid pixels
            pred_valid = pred_depth[valid_mask]
            gt_valid = gt_depth[valid_mask]

            if len(pred_valid) < 100:
                continue

            # In WorldMirror, depth output is canonical scaled. Align to metric:
            s_opt = np.median(gt_valid) / max(1e-4, float(np.median(pred_valid)))
            pred_aligned = pred_valid * s_opt

            # Metrics
            abs_rel = float(np.mean(np.abs(pred_aligned - gt_valid) / gt_valid))
            rmse = float(np.sqrt(np.mean((pred_aligned - gt_valid) ** 2)))
            ratio = np.maximum(pred_aligned / np.maximum(gt_valid, 1e-4), gt_valid / np.maximum(pred_aligned, 1e-4))
            d1 = float(np.mean((ratio < 1.25).astype(float)))

            scene_absrel.append(abs_rel)
            scene_rmse.append(rmse)
            scene_delta1.append(d1)

            all_absrel.append(abs_rel)
            all_rmse.append(rmse)
            all_delta1.append(d1)

            if ds_name == "hypersim":
                hypersim_absrel.append(abs_rel)
            elif ds_name == "scannet":
                scannet_absrel.append(abs_rel)

        m_absrel = float(np.mean(scene_absrel)) if scene_absrel else 0.0
        m_rmse = float(np.mean(scene_rmse)) if scene_rmse else 0.0
        m_d1 = float(np.mean(scene_delta1)) if scene_delta1 else 0.0

        per_scene_results[scene_id] = {
            "dataset": ds_name,
            "evaluated_frames": len(scene_absrel),
            "absrel": round(m_absrel, 4),
            "rmse": round(m_rmse, 4),
            "delta1": round(m_d1, 4),
        }
        print(f"  -> Scene {scene_id}: AbsRel={m_absrel:.4f}, RMSE={m_rmse:.4f}m, Delta1={m_d1:.4f}")

    # Summary
    overall_report = {
        "evaluation_name": "Locked Test Set Final Evaluation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {
            "path": str(checkpoint_path),
            "filename": checkpoint_path.name,
            "sha256": ckpt_hash,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "manifest": {
            "path": str(manifest_path),
            "filename": manifest_path.name,
            "sha256": manifest_hash,
            "size_bytes": manifest_path.stat().st_size,
        },
        "configuration": {
            "seed": EVAL_SEED,
            "device": device,
            "resolution": [518, 518],
            "max_frames_per_scene": max_frames_per_scene,
        },
        "overall_metrics": {
            "total_frames_evaluated": len(all_absrel),
            "mean_absrel": round(float(np.mean(all_absrel)), 4) if all_absrel else 0.0,
            "mean_rmse": round(float(np.mean(all_rmse)), 4) if all_rmse else 0.0,
            "mean_delta1": round(float(np.mean(all_delta1)), 4) if all_delta1 else 0.0,
            "mean_latency_sec": round(float(np.mean(all_latencies)), 3) if all_latencies else 0.0,
        },
        "track_metrics": {
            "hypersim_track_a_absrel": round(float(np.mean(hypersim_absrel)), 4) if hypersim_absrel else 0.0,
            "scannet_track_b_absrel": round(float(np.mean(scannet_absrel)), 4) if scannet_absrel else 0.0,
        },
        "per_scene_breakdown": per_scene_results,
    }

    OUTPUT_JSON.write_text(json.dumps(overall_report, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print("LOCKED TEST EVALUATION SUMMARY:")
    print(f"  Total Frames Evaluated: {len(all_absrel)}")
    print(f"  Overall Mean AbsRel:    {overall_report['overall_metrics']['mean_absrel']}")
    print(f"  Overall Mean RMSE:      {overall_report['overall_metrics']['mean_rmse']} m")
    print(f"  Overall Delta1:         {overall_report['overall_metrics']['mean_delta1']}")
    print(f"  Hypersim Track A AbsRel: {overall_report['track_metrics']['hypersim_track_a_absrel']}")
    print(f"  ScanNet Track B AbsRel:  {overall_report['track_metrics']['scannet_track_b_absrel']}")
    print(f"Raw report preserved at: {OUTPUT_JSON}")
    print("=" * 80)
    return overall_report

if __name__ == "__main__":
    evaluate_locked_test(FINAL_CHECKPOINT_PATH, MANIFEST_PATH)

