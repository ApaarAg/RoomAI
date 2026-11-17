# RoomAI Production Deployment Checklist & Operations Manual

**Document Version:** 1.0.0 (Release Candidate)  
**Date:** 2026-10-29  
**Application Target:** RoomAI Computer Vision & Spatial Intelligence System  
**Verified Platform:** Windows 11 / Linux (Ubuntu 22.04 LTS compatible), NVIDIA RTX GPU (8 GB VRAM min)

---

## 1. Pre-Deployment Verification Matrix

Before commissioning RoomAI into production or beta environments, ensure that all pre-flight gates below have passed.

| # | Inspection Item | Verification Command | Expected Output | Status |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | Conda Environment | `conda info --envs` | Active env is `tf-gpu` | **VERIFIED** |
| **G2** | Python Version | `python --version` | `Python 3.10.20` | **VERIFIED** |
| **G3** | PyTorch & CUDA | `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` | `2.11.0+cu128 True` | **VERIFIED** |
| **G4** | Ultralytics YOLO | `python -c "import ultralytics; print(ultralytics.__version__)"` | `8.4.66` | **VERIFIED** |
| **G5** | Full Test Suite | `pytest -q` | `129 passed in ~40s` | **VERIFIED** |
| **G6** | Smoke Inference | `python scripts/smoke_test_application.py` | `All Smoke Tests Passed` | **VERIFIED** |
| **G7** | Streamlit Health | `python scripts/test_streamlit_startup.py` | `Health response: 'ok'` | **VERIFIED** |
| **G8** | Default Detector | Inspect `config/model_config.yaml` | `checkpoint_path: ...b2...` | **VERIFIED** |

---

## 2. Checkpoint Manifest & Verification

All production models must reside in the `checkpoints/` root directory. Verify file existence and byte sizes prior to launching the service:

| Role | Checkpoint Path | Architecture | Byte Size | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Exterior Production Default** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | YOLO11s | 19,002,403 B | **PRODUCTION READY** |
| **Indoor Detection Specialist** | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | YOLO11s (800px) | 19,004,611 B | **PRODUCTION READY** |
| **Parking & Grounds Specialist** | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | YOLO11s | 19,004,675 B | **PRODUCTION READY** |
| **Segmentation Model** | `checkpoints/yolo11n-seg.pt` | YOLO11n-Seg | 5,910,256 B | **PRETRAINED BENCHMARK** |
| **Relative Depth Model** | `checkpoints/depth_anything_v2_small/` | Depth Anything V2 | ~98.5 MB | **PRODUCTION READY** |
| *Quarantined Model B5* | `checkpoints/exterior_detector_b5_transfer_best.pt` | YOLO11s | 19,004,675 B | *Quarantined (enabled: false)* |
| *Quarantined Model B6* | `checkpoints/exterior_detector_b6_expanded_best.pt` | YOLO11s | 19,004,675 B | *Quarantined (enabled: false)* |
| *Quarantined Model B7* | `checkpoints/pilot_b7_best.pt` | YOLO11s | 19,004,675 B | *Quarantined (enabled: false)* |
| *Quarantined Model C1* | `checkpoints/parking_detector_c1_yolo11s_baseline_best.pt` | YOLO11s | 19,002,403 B | *Quarantined (enabled: false)* |

---

## 3. Installation & Setup Runbook

### Step 1: Environment Activation
Always activate the verified `tf-gpu` Conda environment:
```powershell
conda activate tf-gpu
```
*Note:* Do not use the base Conda environment, as its package versions conflict with Transformers depth estimation.

### Step 2: Configuration Review
Ensure `config/model_config.yaml` has Model B2 as the primary detector:
```yaml
exterior_detection:
  model_name: "Exterior Building & Architecture Detector (Model B2 - Baseline Production)"
  checkpoint_path: "checkpoints/exterior_detector_b2_yolo11s_best.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.45
  status: "production_ready"
  primary: true
  enabled: true
```

Ensure `config/app.yaml` has `mock_mode: false` for real inference:
```yaml
app:
  mock_mode: false
  debug: false
```

### Step 3: Run Smoke Test
Execute the automated smoke verification:
```powershell
python scripts/smoke_test_application.py
```
Expected output:
```
=== All Application Smoke Tests Passed Successfully! ===
```

### Step 4: Launch Web Application
Launch the Streamlit service in headless production mode:
```powershell
streamlit run app.py --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
```

### Step 5: Verify Health Endpoint
From a separate shell or HTTP probe:
```powershell
curl http://localhost:8501/_stcore/health
```
Expected response:
```
ok
```

---

## 4. Operational Monitoring & Maintenance

### 4.1 Log Monitoring
Application events are written with ISO timestamps to stdout and tracked in standard logging format:
- Level `INFO`: Model loads, cache hits, inference timing, artifact save locations.
- Level `WARNING`: Non-fatal issues (e.g. low-confidence detections, fallback triggers).
- Level `ERROR`: Catch-and-recover exceptions (e.g. corrupted image file, decompression bomb).

### 4.2 GPU & VRAM Headroom
Monitor live GPU resources during peak load:
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 2
```
- **Threshold Limit:** VRAM usage must remain $< 4,500\text{ MB}$.
- **Normal Operating Range:** $250\text{ MB} - 450\text{ MB}$.

### 4.3 Disk Artifact Purging
Images processed through the detection service generate artifacts in `artifacts/detections/`. For long-running servers, configure a daily cron or scheduled task to prune files older than 48 hours:
```powershell
Get-ChildItem -Path "artifacts\detections" -Filter "*.png" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-2) } | Remove-Item -Force
```

---

## 5. Disaster Recovery & Rollback Procedures

### Scenario A: GPU Hardware Failure / CUDA OOM
If GPU driver crashes or CUDA memory is exhausted:
1. Edit `config/model_config.yaml`:
   ```yaml
   exterior_detection:
     device: "cpu"
   indoor_detection:
     device: "cpu"
   parking_detection:
     device: "cpu"
   scene3d:
     device: "cpu"
   ```
2. Restart Streamlit: CPU fallback will automatically handle inference without crashing.

### Scenario B: Model Weights Corruption / Missing Weights
If a model checkpoint file is missing or corrupted on disk:
1. Set `mock_mode: true` in `config/app.yaml`:
   ```yaml
   app:
     mock_mode: true
   ```
2. Restart Streamlit: The UI will immediately switch to simulated mock responses with prominent UI badges, maintaining service availability while models are re-downloaded.

### Scenario C: Unintended Candidate Promotion
If an experimental model (B5, B6, or B7) is inadvertently promoted:
1. Restore production default configuration from backup:
   ```powershell
   Copy-Item "config\model_config.yaml.backup_hardening_step1" "config\model_config.yaml" -Force
   ```
2. Run `pytest -q` to verify that B2 is reinstated as the active primary model.

