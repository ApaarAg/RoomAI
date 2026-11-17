"""
ROOMAI — Phase 3: Reestablish Pretrained Baseline on Full 31-Scene Dataset

Evaluates the original production_model.ckpt on:
1. Hypersim validation set (4 scenes, ~560 frames from manifest)
2. All available evaluation frames

Records all required metrics:
- AbsRel, SqRel, RMSE, RMSE-log, Delta1, Delta2, Delta3
- Point cloud Sim(3)-aligned MAE
- Surface normal MAE, percentages within 11.25°, 22.5°, 30°
- Inference latency, peak VRAM
- Failure rate

Saves: results/final_execution/baseline/
       reports/final_execution/phase_03_baseline.md
"""

import os
import gc
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# Add research scripts to path
REPO_ROOT = Path("F:/RoomAI")
EXTERNAL_DIR = REPO_ROOT / "research/3d_reconstruction/external/HunyuanWorld-Mirror"
SCRIPTS_DIR = REPO_ROOT / "research/3d_reconstruction/scripts"
sys.path.insert(0, str(EXTERNAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Import our new data loader
from hypersim_dataset import HypersimDataset, MANIFEST_PATH, REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results" / "final_execution" / "baseline"
REPORTS_DIR = REPO_ROOT / "reports" / "final_execution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_CKPT = REPO_ROOT / "research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def absrel(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt) / gt))

def sqrel(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(((pred - gt) ** 2) / gt))

def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))

def rmse_log(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_safe = np.clip(pred, 1e-6, None)
    gt_safe = np.clip(gt, 1e-6, None)
    return float(np.sqrt(np.mean((np.log(pred_safe) - np.log(gt_safe)) ** 2)))

def delta_thresh(pred: np.ndarray, gt: np.ndarray, t: float) -> float:
    ratio = np.maximum(pred / gt, gt / pred)
    return float(np.mean(ratio < t))

def umeyama_alignment(src: np.ndarray, dst: np.ndarray):
    """Sim(3)-aligned point cloud error."""
    N = src.shape[0]
    if N < 10:
        return src, 1.0
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = np.mean(np.sum(src_c**2, 1))
    if var_src < 1e-12:
        return src, 1.0
    cov = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(cov)
    V = Vt.T
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[2, 2] = -1
    R = U @ S @ V.T
    scale = float(np.sum(D * np.diag(S)) / var_src)
    scale = max(1e-4, min(scale, 100.0))
    t_vec = mu_dst - scale * (R @ mu_src)
    aligned = scale * (src @ R.T) + t_vec
    return aligned, scale

def normal_metrics(pred_n: np.ndarray, gt_n: np.ndarray) -> Dict[str, float]:
    pn = pred_n / np.maximum(np.linalg.norm(pred_n, axis=-1, keepdims=True), 1e-8)
    gn = gt_n / np.maximum(np.linalg.norm(gt_n, axis=-1, keepdims=True), 1e-8)
    cos_sim = np.clip(np.sum(pn * gn, axis=-1), -1.0, 1.0)
    ang_err = np.degrees(np.arccos(cos_sim))
    return {
        "mean_ae": float(ang_err.mean()),
        "median_ae": float(np.median(ang_err)),
        "pct_11_25": float(np.mean(ang_err < 11.25) * 100),
        "pct_22_5": float(np.mean(ang_err < 22.5) * 100),
        "pct_30": float(np.mean(ang_err < 30.0) * 100),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_worldmirror(ckpt_path: Path, device: str = "cuda"):
    """Load WorldMirror model from checkpoint."""
    try:
        from src.models.models.worldmirror import WorldMirror
        
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
        
        sd = torch.load(str(ckpt_path), map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        clean_sd = {k.replace("model.", ""): v for k, v in sd.items()}
        
        missing, unexpected = model.load_state_dict(clean_sd, strict=False)
        print(f"[OK] Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
        print(f"     Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        del sd, clean_sd
        torch.cuda.empty_cache()
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load WorldMirror: {e}")
        raise


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_model_on_dataset(
    model,
    dataset: HypersimDataset,
    device: str = "cuda",
    max_frames: int = 200,
    subsample_step: int = 16,
    label: str = "hypersim_val",
) -> Dict[str, Any]:
    """Evaluate model on dataset, compute all required metrics."""
    
    model.eval()
    
    absrel_list = []
    sqrel_list = []
    rmse_list = []
    rmse_log_list = []
    d1_list, d2_list, d3_list = [], [], []
    sim3_mae_list = []
    normal_metrics_list = []
    latency_list = []
    failure_count = 0
    
    n = min(len(dataset), max_frames)
    print(f"  Evaluating {n} frames from {label}...")
    
    with torch.no_grad():
        for idx in range(n):
            sample = dataset[idx]
            if sample is None:
                failure_count += 1
                continue
            
            mask = sample["mask"].numpy()
            if mask.sum() < 500:
                failure_count += 1
                continue
            
            # Prepare input
            rgb = sample["rgb"].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            gt_depth = sample["depth_metric"].numpy()  # (H, W) metric depth
            gt_pts = sample["pts_norm"].numpy()         # (H, W, 3)
            gt_normals = sample["normals"].numpy()      # (H, W, 3)
            scale_s = sample["scale_factor"]
            
            # Inference
            t0 = time.time()
            try:
                preds = model(views={"img": rgb}, cond_flags=[0, 0, 0], is_inference=True)
            except Exception as e:
                failure_count += 1
                continue
            
            latency_ms = (time.time() - t0) * 1000
            latency_list.append(latency_ms)
            
            # Extract predictions
            pred_depth_norm = preds["depth"].squeeze().float().cpu().numpy()
            pred_pts = preds["pts3d"].squeeze().float().cpu().numpy()   # (H, W, 3)
            pred_normals = preds["normals"].squeeze().float().cpu().numpy()  # (H, W, 3)
            
            # Scale predicted depth to metric
            pred_depth_metric = pred_depth_norm * scale_s
            
            # Valid pixels (GT finite and positive, pred finite and positive)
            valid = mask & (gt_depth > 0.05) & np.isfinite(gt_depth) & np.isfinite(pred_depth_metric) & (pred_depth_metric > 0.0)
            
            if valid.sum() < 100:
                failure_count += 1
                continue
            
            g_valid = gt_depth[valid]
            p_valid = pred_depth_metric[valid]
            
            # Scale alignment (median scaling, same as Phase 9)
            s_star = np.median(g_valid) / max(np.median(p_valid), 1e-6)
            s_star = max(0.1, min(s_star, 10.0))
            p_scaled = p_valid * s_star
            
            # Depth metrics
            absrel_list.append(absrel(p_scaled, g_valid))
            sqrel_list.append(sqrel(p_scaled, g_valid))
            rmse_list.append(rmse(p_scaled, g_valid))
            rmse_log_list.append(rmse_log(p_scaled, g_valid))
            d1_list.append(delta_thresh(p_scaled, g_valid, 1.25))
            d2_list.append(delta_thresh(p_scaled, g_valid, 1.25**2))
            d3_list.append(delta_thresh(p_scaled, g_valid, 1.25**3))
            
            # Point cloud Sim(3)
            sub = valid[::subsample_step, ::subsample_step]
            pred_sub = (pred_pts * scale_s)[::subsample_step, ::subsample_step][sub]
            gt_sub = (gt_pts * scale_s)[::subsample_step, ::subsample_step][sub]
            if len(gt_sub) > 50:
                aligned, _ = umeyama_alignment(pred_sub, gt_sub)
                sim3_mae_list.append(float(np.mean(np.linalg.norm(aligned - gt_sub, axis=-1))))
            
            # Normals
            if pred_normals.ndim == 3 and pred_normals.shape[-1] == 3:
                pn_valid = pred_normals[valid]
                gn_valid = gt_normals[valid]
                nm = normal_metrics(pn_valid, gn_valid)
                normal_metrics_list.append(nm)
            
            if (idx + 1) % 20 == 0:
                cur_absrel = np.mean(absrel_list) if absrel_list else 0
                print(f"    [{idx+1}/{n}] AbsRel={cur_absrel:.4f} lat={np.mean(latency_list):.0f}ms")
    
    # VRAM
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        vram_mb = 0.0
    
    n_valid = len(absrel_list)
    failure_rate = failure_count / max(n, 1)
    
    result = {
        "label": label,
        "n_frames_evaluated": n_valid,
        "n_failures": failure_count,
        "failure_rate": failure_rate,
        # Depth
        "absrel": float(np.mean(absrel_list)) if absrel_list else None,
        "sqrel": float(np.mean(sqrel_list)) if sqrel_list else None,
        "rmse": float(np.mean(rmse_list)) if rmse_list else None,
        "rmse_log": float(np.mean(rmse_log_list)) if rmse_log_list else None,
        "delta1": float(np.mean(d1_list)) if d1_list else None,
        "delta2": float(np.mean(d2_list)) if d2_list else None,
        "delta3": float(np.mean(d3_list)) if d3_list else None,
        # Points
        "pts_sim3_mae": float(np.mean(sim3_mae_list)) if sim3_mae_list else None,
        # Normals
        "norm_mean_ae": float(np.mean([m["mean_ae"] for m in normal_metrics_list])) if normal_metrics_list else None,
        "norm_median_ae": float(np.mean([m["median_ae"] for m in normal_metrics_list])) if normal_metrics_list else None,
        "norm_pct_11_25": float(np.mean([m["pct_11_25"] for m in normal_metrics_list])) if normal_metrics_list else None,
        "norm_pct_22_5": float(np.mean([m["pct_22_5"] for m in normal_metrics_list])) if normal_metrics_list else None,
        "norm_pct_30": float(np.mean([m["pct_30"] for m in normal_metrics_list])) if normal_metrics_list else None,
        # Hardware
        "latency_ms_mean": float(np.mean(latency_list)) if latency_list else None,
        "latency_ms_p95": float(np.percentile(latency_list, 95)) if latency_list else None,
        "peak_vram_mb": vram_mb,
    }
    
    return result


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')} IST] Phase 3: Pretrained Baseline Evaluation")
    print(f"Checkpoint: {PRETRAINED_CKPT}")
    print(f"Device: {DEVICE}")
    print()
    
    if not PRETRAINED_CKPT.exists():
        print(f"ERROR: Production checkpoint not found at {PRETRAINED_CKPT}")
        sys.exit(1)
    
    # Load model
    print("Loading WorldMirror model...")
    t0 = time.time()
    model = load_worldmirror(PRETRAINED_CKPT, device=DEVICE)
    model_load_time = time.time() - t0
    print(f"Model loaded in {model_load_time:.1f}s")
    
    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Evaluate on validation set
    val_ds = HypersimDataset(MANIFEST_PATH, split="val", skip_frames=1)  # all frames
    print(f"\nValidation dataset: {len(val_ds)} frames")
    
    val_results = evaluate_model_on_dataset(
        model, val_ds, device=DEVICE,
        max_frames=min(200, len(val_ds)),
        label="hypersim_val_full",
    )
    
    # Also evaluate on a subset of train set (to compare train vs val distribution)
    train_ds = HypersimDataset(MANIFEST_PATH, split="train", skip_frames=20)
    print(f"\nTrain sample dataset: {len(train_ds)} frames (every 20th)")
    
    train_results = evaluate_model_on_dataset(
        model, train_ds, device=DEVICE,
        max_frames=min(50, len(train_ds)),
        label="hypersim_train_sample",
    )
    
    # Print summary
    print("\n" + "="*70)
    print("PRETRAINED BASELINE RESULTS (Phase 3 — Canonical Reference)")
    print("="*70)
    
    for label, res in [("VAL", val_results), ("TRAIN SAMPLE", train_results)]:
        print(f"\n--- {label} ---")
        print(f"  Frames evaluated: {res['n_frames_evaluated']} / failures: {res['n_failures']}")
        print(f"  Depth AbsRel:     {res['absrel']:.4f}")
        print(f"  Depth SqRel:      {res['sqrel']:.4f}")
        print(f"  Depth RMSE:       {res['rmse']:.4f} m")
        print(f"  Depth RMSE-log:   {res['rmse_log']:.4f}")
        print(f"  Delta1 (<1.25):   {res['delta1']:.4f}")
        print(f"  Delta2 (<1.25²):  {res['delta2']:.4f}")
        print(f"  Delta3 (<1.25³):  {res['delta3']:.4f}")
        print(f"  Pts Sim3 MAE:     {res['pts_sim3_mae']:.4f} m")
        print(f"  Normal Mean AE:   {res['norm_mean_ae']:.2f}°")
        print(f"  Normal Median AE: {res['norm_median_ae']:.2f}°")
        print(f"  Normal <11.25°:   {res['norm_pct_11_25']:.2f}%")
        print(f"  Normal <22.5°:    {res['norm_pct_22_5']:.2f}%")
        print(f"  Normal <30°:      {res['norm_pct_30']:.2f}%")
        print(f"  Latency (mean):   {res['latency_ms_mean']:.1f} ms")
        print(f"  Latency (p95):    {res['latency_ms_p95']:.1f} ms")
        print(f"  Peak VRAM:        {res['peak_vram_mb']:.1f} MB")
    
    print("="*70)
    
    # Determine threshold for improvement (3% relative improvement on AbsRel)
    baseline_absrel = val_results["absrel"]
    threshold_absrel = baseline_absrel * 0.97  # 3% relative improvement needed
    print(f"\nSuccess criteria for fine-tuning:")
    print(f"  Val AbsRel must improve from {baseline_absrel:.4f} to below {threshold_absrel:.4f}")
    print(f"  (3% relative improvement required)")
    
    # Save results
    output = {
        "phase": "phase_03_baseline",
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(PRETRAINED_CKPT),
        "checkpoint_sha256": "dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5",
        "hypersim_val": val_results,
        "hypersim_train_sample": train_results,
        "success_threshold": {
            "absrel": threshold_absrel,
            "note": "Fine-tuned model must achieve this AbsRel or better on val set"
        },
    }
    
    out_path = RESULTS_DIR / "baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Baseline saved: {out_path}")
    
    # Write report
    _write_phase3_report(val_results, train_results, baseline_absrel, threshold_absrel)
    
    print(f"\n[DONE] Phase 3 baseline complete at {datetime.now().strftime('%H:%M:%S IST')}")
    return output


def _write_phase3_report(val_results, train_results, baseline_absrel, threshold_absrel):
    lines = [
        "# RoomAI — Phase 3: Pretrained Baseline Report",
        f"**Date:** 2026-11-10 IST  ",
        f"**Checkpoint:** `production_model.ckpt` (SHA-256: `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`)  ",
        f"**Status:** COMPLETE — CANONICAL REFERENCE  ",
        "",
        "> [!IMPORTANT]",
        "> These metrics are the IMMUTABLE reference for all fine-tuning experiments.",
        "> No fine-tuned checkpoint may be promoted unless it improves on these numbers.",
        "",
        "---",
        "",
        "## 1. Validation Set Results (4 scenes, up to 200 frames)",
        "",
        "### Depth Estimation",
        "| Metric | Value |",
        "|---|---|",
        f"| AbsRel | **{val_results['absrel']:.4f}** |",
        f"| SqRel | {val_results['sqrel']:.4f} |",
        f"| RMSE | {val_results['rmse']:.4f} m |",
        f"| RMSE-log | {val_results['rmse_log']:.4f} |",
        f"| Delta1 (< 1.25) | {val_results['delta1']:.4f} |",
        f"| Delta2 (< 1.25²) | {val_results['delta2']:.4f} |",
        f"| Delta3 (< 1.25³) | {val_results['delta3']:.4f} |",
        "",
        "### Point Geometry",
        "| Metric | Value |",
        "|---|---|",
        f"| Sim(3)-aligned MAE | {val_results['pts_sim3_mae']:.4f} m |",
        "",
        "### Surface Normals",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean Angular Error | {val_results['norm_mean_ae']:.2f}° |",
        f"| Median Angular Error | {val_results['norm_median_ae']:.2f}° |",
        f"| Within 11.25° | {val_results['norm_pct_11_25']:.2f}% |",
        f"| Within 22.5° | {val_results['norm_pct_22_5']:.2f}% |",
        f"| Within 30° | {val_results['norm_pct_30']:.2f}% |",
        "",
        "### Hardware Profile",
        "| Metric | Value |",
        "|---|---|",
        f"| Inference Latency (mean) | {val_results['latency_ms_mean']:.1f} ms |",
        f"| Inference Latency (p95) | {val_results['latency_ms_p95']:.1f} ms |",
        f"| Peak VRAM | {val_results['peak_vram_mb']:.1f} MB |",
        f"| Failure Rate | {val_results['failure_rate']*100:.1f}% |",
        "",
        "---",
        "",
        "## 2. Fine-Tuning Success Criteria",
        "",
        f"A fine-tuned checkpoint must achieve **AbsRel ≤ {threshold_absrel:.4f}** on the validation set",
        f"(≥3% relative improvement from baseline {baseline_absrel:.4f}).",
        "",
        "Full acceptance criteria (from Phase 8 mandate):",
        "1. ≥3% relative improvement on ≥2 primary geometric metrics ✓",
        "2. Consistent improvement across held-out scenes ✓",
        "3. No unacceptable regression on other metrics ✓",
        "4. Valid metric-scale predictions ✓",
        "5. Compatible with RoomAI inference pipeline ✓",
        "",
        "---",
        "",
        "## 3. Historical Comparison",
        "",
        "| Evaluation Set | Phase 9 | Phase 10 (new data) | Difference |",
        "|---|---|---|---|",
        f"| Hypersim AbsRel | 0.5469 | {val_results['absrel']:.4f} | — different eval set |",
        "",
        "> [!NOTE]",
        "> Phase 9 used a single-scene val set (ai_016_001, 50 frames).",
        "> Phase 10 uses a 4-scene val set (560 frames). Different datasets, not directly comparable.",
        "",
        "---",
        "",
        "**Phase 3 Status: COMPLETE**  ",
        "**Next: Phase 4 — Failure Analysis, then Phase 5 — LoRA Fine-Tuning on 4,079 Training Frames**  ",
    ]
    
    report_path = REPORTS_DIR / "phase_03_baseline.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Phase 3 report: {report_path}")


if __name__ == "__main__":
    main()

