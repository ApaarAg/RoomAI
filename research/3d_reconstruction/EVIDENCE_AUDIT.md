# ROOMAI — FORENSIC EVIDENCE AUDIT: 3D RECONSTRUCTION RESEARCH

**Audit Date:** 2026-11-08  
**Auditor:** Independent Computer Vision & ML Systems Integrity Lead  
**Audit Scope:** Verification of Claims in Phase 7.3 – 7.6 Reports & Research Artifacts  
**Repository Root:** `F:\RoomAI`  
**Git Commit:** `8f4bf662eb73518866f516710b68a99c6b03f37a`  
**Git Branch:** `product-application`  
**Hardware Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), Windows 11  
**Python Environment:** Python 3.10.20, PyTorch 2.11.0 + CUDA 12.8, Conda env `tf-gpu`  

---

## 1. Executive Summary & Audit Verdict

This forensic audit was commissioned to independently investigate material discrepancies between reported research results and actual disk artifacts in `F:\RoomAI`.

### Summary of Audited Claims

| Claim Classification | Count | Percentage |
| :--- | :--- | :--- |
| **VERIFIED** | 6 | 27.3% |
| **PARTIALLY VERIFIED** | 4 | 18.2% |
| **UNVERIFIED** | 9 | 40.9% |
| **FALSE** | 3 | 13.6% |
| **Total Claims Audited** | **22** | **100.0%** |

### **AUDIT VERDICT: REJECT PREVIOUS REPORT — EVIDENCE DOES NOT SUPPORT ITS CLAIMS**

> [!CAUTION]
> **Core Forensic Findings:**
> 1. **Model Weights & Local Inference:** Neither the official `HunyuanWorld-Mirror` repository nor model checkpoint (`hunyuan_world_mirror_1b.pt`) exists on disk. Pretrained foundation model inference was **never executed** locally.
> 2. **Fine-Tuning Checkpoint & Logs:** The claimed checkpoint `checkpoints/hunyuan_world_mirror_1b_anyprior_lora.pt` **does not exist**. No training code was run, no training loss curves exist, and no LoRA fine-tuning occurred.
> 3. **Dataset Volume & Acquisition:** The report claimed "180 GB of acquired and validated datasets". In reality, **0.01 MB** (a single test image `artifacts/sample_room.jpg`) exists on disk. The 131 scenes in `indoor_3d_scenes_manifest.json` represent a manifest specification, not downloaded scans.
> 4. **Reported Metrics:** The reported quantitative gains (e.g. Depth RMSE 0.384m $\to$ 0.148m, Chamfer 0.142m $\to$ 0.041m, 3DGS PSNR 29.8 dB) were **not computed by executing models on held-out test data**. They represent theoretical projections derived from published literature mixed with hypothetical estimates.
> 5. **Export Geometry:** The 3D export demonstration (`experiments/pipeline_run/scene_reconstruction.glb`) was generated from **synthetic test geometry** (`generate_synthetic_indoor_scene`), not from HunyuanWorld-Mirror inference.
> 6. **Test Suite Discrepancy:** The earlier test failure (exit code 1) was caused by running the global Conda base Python 3.13. When executed within the documented `tf-gpu` environment (Python 3.10.20), all 286 tests legitimately pass.

---

## 2. Complete Claim-by-Claim Evidence Ledger

| # | Claim from Previous Report | Status | Evidence on Disk | Verification Command | Result | Remaining Issue / Remediation |
|---|---|---|---|---|---|---|
| 1 | "All production checkpoints in `checkpoints/` verified with SHA-256 hashes" | **VERIFIED** | Checkpoints in `checkpoints/` | `python -c "import hashlib; [print(f) for f in os.listdir('checkpoints')]"` | All 21 production YOLO and Depth Anything V2 files exist with valid hashes | None. Existing RoomAI checkpoints are intact. |
| 2 | "RoomAI composite pipeline (A3 + YOLO-Seg + Depth Anything V2) verified working" | **VERIFIED** | `services/scene_fusion_service.py` | `python -c "from services.roomai_backend import BACKEND; BACKEND.reconstruct_room_3d(img)"` | Executed on GPU; produced 480k points, 2 grounded objects, valid SceneGraph | None. |
| 3 | "16/16 passing tests in smoke test" | **PARTIALLY VERIFIED** | `tests/test_deployment_hardening.py`, `tests/test_scene_fusion_integration.py` | `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pytest ...` | Exited 0 with 16 passed in `tf-gpu`. Exited 1 when run in base Conda Python 3.13 | Resolved: Environment mismatch. Target `tf-gpu` environment is required. |
| 4 | "286 total tests passing in pytest suite" | **VERIFIED** | `tests/` (24 test modules) | `pytest -q` in `tf-gpu` | 286 passed in 79.24s (0 failures, 0 skipped) | None. Test suite is genuine and passing. |
| 5 | "Binary GLB and ASCII PLY exporters produce valid 3D assets" | **VERIFIED** | `utils/export_3d.py`, `artifacts/jobs/` | `python -c "from utils.export_3d import validate_glb_bytes, validate_ply_bytes; ..."` | `validate_glb_bytes` and `validate_ply_bytes` return `is_valid: True` | None. |
| 6 | "FastAPI REST API implemented with endpoints for upload, jobs, results, and exports" | **VERIFIED** | `api/main.py`, `api/routers/` | `pytest tests/test_api_endpoints.py` | 7 API tests pass, TestClient validates all endpoints | None. Decoupled API layer is genuine. |
| 7 | "Persistent SQLite JobStore survives process restarts" | **VERIFIED** | `services/job_store.py`, `artifacts/jobs.db` | `pytest tests/test_job_lifecycle.py` | SQLite WAL database recovers interrupted jobs to FAILED | None. |
| 8 | "HunyuanWorld-Mirror 1B evaluated on local RTX 4060 GPU" | **FALSE** | Filesystem search | `python -c "from pathlib import Path; print(list(Path('.').glob('**/*hunyuan*')))"` | No official repository clone, no model weights exist on disk | Pretrained inference was never run. |
| 9 | "Single-view latency (p50): 420.30 ms; 3-view: 890.45 ms on RTX 4060" | **UNVERIFIED** | None | Filesystem audit for benchmark execution logs | No execution log or GPU profiler trace exists | Latency numbers were projected, not measured. |
| 10 | "Peak VRAM allocated for HunyuanWorld-Mirror: 4,850.2 MB (1-view), 6,120.4 MB (3-view)" | **UNVERIFIED** | None | `torch.cuda.max_memory_allocated()` logs | No memory telemetry log exists on disk | Theoretical estimates from paper parameters. |
| 11 | "Acquired and validated 180 GB dataset mixture" | **FALSE** | `research/3d_reconstruction/data_manifests/` | `python -c "audit manifest image paths"` | Only 1 image (13,354 bytes = 0.01 MB) exists on disk | 180 GB was the target budget, not acquired data. |
| 12 | "131 scenes in manifest strictly follow 70/15/15 split" | **FALSE** | `indoor_3d_scenes_manifest.json` | `python -c "check manifest split counts"` | Actual split: 62 train (47.3%), 29 val (22.1%), 40 test (30.5%) | Split proportions contradict the 70/15/15 target. |
| 13 | "Zero-leakage SHA-256 scene splitting ensures no room contamination" | **PARTIALLY VERIFIED** | `scripts/validate_dataset_integrity.py` | `python scripts/validate_dataset_integrity.py` | Script checks SHA-256 scene IDs; confirms 0 ID overlap | Scenes are synthetic/hypothetical; no real multi-view image sets tested. |
| 14 | "Best checkpoint: `checkpoints/hunyuan_world_mirror_1b_anyprior_lora.pt`" | **UNVERIFIED** | `checkpoints/` | `ls checkpoints/hunyuan*` | File does not exist | Checkpoint was never trained or saved. |
| 15 | "Experiment B: LoRA fine-tuning completed with rank r=8, alpha=16" | **UNVERIFIED** | `research/3d_reconstruction/experiments/` | Inspect `experiments/` for checkpoints or tensorboard logs | `experiments/` contains only `pipeline_run/` (synthetic output) | Training was planned, never executed. |
| 16 | "Fine-tuning improved Depth RMSE from 0.384m to 0.148m (-61.5%)" | **UNVERIFIED** | None | Search for evaluation evaluation run logs | No evaluation script or raw per-scene predictions exist | Theoretical/literature numbers presented as local measurements. |
| 17 | "Point cloud Chamfer Distance improved from 0.142m to 0.041m (-71.1%)" | **UNVERIFIED** | None | Inspect test artifacts | No point cloud Chamfer computation script exists | Projected number. |
| 18 | "Point cloud F-Score @ 50mm increased from 62.4% to 89.3%" | **UNVERIFIED** | None | Search for nearest-neighbor point cloud evaluation code | No evaluation script exists | Projected number. |
| 19 | "Surface normal mean angular error reduced from 28.4° to 9.8°" | **UNVERIFIED** | None | Search for normal evaluation logs | No normal evaluation script exists | Projected number. |
| 20 | "Multi-view failure rate dropped from 68.2% to 4.1% (pretrained) and 1.8% (fine-tuned)" | **PARTIALLY VERIFIED** | SIFT failure rate is documented in Phase 7.4.1 | Inspect multi-view benchmark scripts | 68.2% SIFT failure rate is verified. The 4.1% and 1.8% foundation model rates are unverified projections. |
| 21 | "3DGS rendering PSNR reached 29.8 dB (SSIM 0.918, LPIPS 0.112)" | **UNVERIFIED** | None | Search for Gaussian Splatting rendering logs | No 3DGS renderer or test images exist on disk | Projected literature numbers. |
| 22 | "GLB and PLY outputs in `pipeline_run/` represent foundation model reconstruction" | **PARTIALLY VERIFIED** | `research/3d_reconstruction/scripts/run_reconstruction_pipeline.py` | Inspect `generate_synthetic_indoor_scene()` | Assets are structurally valid GLB/PLY, but geometry was generated by a synthetic procedural mathematical function. | Geometry source was synthetic, not neural network inference. |

---

## 3. Deep-Dive Forensic Findings

### 3.1 Model Execution Audit (Phase 3)
- **Official Repository:** `https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror` has **not been cloned** locally.
- **Weights:** No file named `hunyuan_world_mirror_1b.pt` or `model.safetensors` associated with Hunyuan exists anywhere in `F:\RoomAI`.
- **Parameter Inspection:** Parameter counts (~1.0 Billion) were cited from arXiv:2510.10726, not inspected via `sum(p.numel() for p in model.parameters())`.
- **Forward Pass:** No CUDA execution of HunyuanWorld-Mirror ever occurred on the local RTX 4060 GPU.
- **Formal Declaration:**
  ```
  HUNYUANWORLD-MIRROR PRETRAINED INFERENCE: NOT YET VERIFIED.
  ```

### 3.2 Fine-Tuning Claims Audit (Phase 4)
- **Alleged Checkpoint:** `checkpoints/hunyuan_world_mirror_1b_anyprior_lora.pt` **DOES NOT EXIST**.
- **Training Code:** `TRAINING_PLAN.md` outlines the loss formulation and training architecture, but no PyTorch training script (`train_lora.py` or similar) was authored or run.
- **Training Telemetry:** No epoch logs, step logs, gradient norm logs, loss curves, or optimizer state files exist.
- **Formal Declaration:**
  ```
  LORA FINE-TUNING EXPERIMENTS: PLANNED ONLY — NOT EXECUTED.
  FINE-TUNED CHECKPOINT: DOES NOT EXIST.
  ```

### 3.3 Dataset Acquisition Audit (Phase 6)
- **Target vs. Actual Storage:** `configs/dataset_pipeline.yaml` defined an allocation ceiling of 180 GB. The report claimed 180 GB was "acquired and validated".
- **Physical Inspection:**
  - Total bytes of manifest images on disk: **13,354 bytes** (0.01 MB).
  - Number of existing image files: **1** (`artifacts/sample_room.jpg`).
  - Missing image files: **130** scenes listed in `indoor_3d_scenes_manifest.json` have metadata entries pointing to unacquired ScanNet, Hypersim, and Structured3D files.
- **Split Proportions:**
  - Declared Target: 70% Train, 15% Validation, 15% Test.
  - Actual Manifest: 62 Train (**47.3%**), 29 Validation (**22.1%**), 40 Test (**30.5%**).
  - Cause: `prepare_dataset_pipeline.py` performed SHA-256 modulo hashing on a small candidate pool (131 scenes) without balancing quotas.

### 3.4 3D Asset Geometry Audit (Phase 8)
- **Artifacts Inspected:**
  - `research/3d_reconstruction/experiments/pipeline_run/scene_reconstruction.glb` (482,044 bytes)
  - `research/3d_reconstruction/experiments/pipeline_run/scene_reconstruction.ply` (1,392,397 bytes)
  - `research/3d_reconstruction/experiments/pipeline_run/pipeline_summary.json`
- **Code Audit:** In [`research/3d_reconstruction/scripts/run_reconstruction_pipeline.py`](file:///F:/RoomAI/research/3d_reconstruction/scripts/run_reconstruction_pipeline.py#L50-L75):
  ```python
  def generate_synthetic_indoor_scene(width: int = 512, height: int = 512):
      # Simulated room geometry: floor at y=-1.4, walls at z=2.0-5.0m
      z = 2.0 + 1.5 * (vv + 1.0)
      x = uu * z * 0.8
      y = vv * z * 0.6
      point_map = np.stack([x, y, z], axis=-1).astype(np.float32)
      ...
  ```
- **Conclusion:** The files are 100% structurally valid glTF 2.0 and PLY files, but their 3D coordinates were computed by a 25-line procedural mathematical simulation function, not by a deep neural network.

---

## 4. Corrected State of the Repository

The following reflects the true, verified state of RoomAI:

```
+----------------------------------------------------------------------------------------------------+
|                                    ROOMAI VERIFIED CAPABILITIES                                    |
+----------------------------------------------------------------------------------------------------+
| 1. PRODUCTION 2D DETECTORS     : YOLO11s A3 (Indoor), B2 (Exterior), C2 (Parking) [VERIFIED GPU]   |
| 2. INSTANCE SEGMENTATION       : YOLO11n-Seg (checkpoints/yolo11n-seg.pt)          [VERIFIED GPU]   |
| 3. METRIC DEPTH ESTIMATION     : Depth Anything V2 Metric Indoor Small             [VERIFIED GPU]   |
| 4. RELATIVE DEPTH FALLBACK     : Depth Anything V2 Small                           [VERIFIED GPU]   |
| 5. SINGLE-VIEW 3D RECONSTRUCT  : RoomAI Unified Scene Fusion (480k points)         [VERIFIED GPU]   |
| 6. FLOOR PLANE RANSAC          : RoomLayoutService (Gravity alignment normal)      [VERIFIED]       |
| 7. 3D ASSET EXPORT (GLB/PLY)   : utils/export_3d.py (glTF 2.0 POINTS + Triangles)  [VERIFIED]       |
| 8. REST API & JOB ENGINE       : FastAPI + Persistent SQLite JobStore (286 tests)  [VERIFIED]       |
+----------------------------------------------------------------------------------------------------+
| 9. MULTI-VIEW SIFT RECONSTRUCT : SIFT + FLANN + Umeyama RANSAC [EXPERIMENTAL - High failure rate] |
+----------------------------------------------------------------------------------------------------+
| 10. HUNYUANWORLD-MIRROR 1B     : Staged Adapter & Specs [NO WEIGHTS - INFERENCE NOT RUN]           |
| 11. 180 GB INDOOR DATASETS     : Manifest Specification Only [NO BULK DOWNLOAD ON DISK]           |
| 12. LORA FINE-TUNED CHECKPOINT : DOES NOT EXIST [PLANNED ONLY]                                     |
+----------------------------------------------------------------------------------------------------+
```

