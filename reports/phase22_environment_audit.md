# RoomAI Phase 22: Stage Zero Environmental Audit & Baseline Inventory

**Date:** 2026-10-29  
**Local Time:** 01:04 AM IST  
**Role:** Senior Computer Vision Research Engineer, ML Systems Engineer, and MLOps Lead  
**Working Directory:** `C:\Users\apaar\Desktop\RoomAI`  
**Execution Phase:** Phase 22 — Stage Zero (Preflight Recovery, Hardware Inspection, Storage Budgeting & Audit)

---

## 1. Executive Summary

Stage Zero of the Phase 22 Overnight Multi-Dataset Training Campaign has been executed in full compliance with all project non-negotiables:
- **No training was initiated.**
- **No datasets were downloaded.**
- **No existing checkpoints were modified, overwritten, or deleted.**
- **The Phase 17 held-out test set remains strictly quarantined and retired from model selection.**
- **Model B2 (`checkpoints/exterior_detector_b2_yolo11s_best.pt`) remains the active production default.**
- **Tool execution and environment health have been fully restored and verified.**
- **Storage and VRAM headroom confirm that targeted expansion of 1,000–3,000 images is safe and well within capacity.**

---

## 2. Hardware and Environment Inspection

### 2.1 Hardware Metrics
| Resource | Measured Value | Operational Assessment |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 4060 Laptop GPU (1 Device) | Active and ready for PyTorch acceleration |
| **Total VRAM** | **8.00 GB** | Target batch size 4–8 requires 2.1–2.85 GB; ample headroom |
| **Allocated VRAM** | **0.00 GB** | GPU is currently 100% idle |
| **System RAM** | **23.73 GB Total** (1.87 GB Available, 92.1% used) | Windows system standby/cache; conservative worker settings required |
| **Storage (Drive C:)** | **474.87 GB Total**, 401.64 GB Used, **73.23 GB Free (15.4%)** | Sufficient capacity for planned multi-dataset expansion |

### 2.2 Software Environments
- **Primary Execution Environment (`miniconda3` base):**
  - Python: `3.13.5` (64-bit AMD64)
  - PyTorch: `2.6.0+cu124` (CUDA 12.4 Available: True)
  - Ultralytics: `8.4.104`
- **Designated Project Environment (`miniconda3/envs/tf-gpu`):**
  - Path: `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`
  - Python: `3.10.20`
  - PyTorch: `2.11.0+cu128` (CUDA Available: True)
  - Ultralytics: `8.4.66`
  - Note: Both environments feature verified CUDA acceleration with compatible Ultralytics distributions.

### 2.3 Process Audit
- **Running Python Processes:**
  - PIDs 41356, 42636, 44456, 45504: MCP server processes (`graphify.serve`) and helper utilities.
  - Previous background consistency script (PID 31624) has terminated.
  - **Zero training processes are currently active.** No competing GPU workloads exist.

---

## 3. Recovery of Phase 20 & Model B7 Status

### 3.1 Training Execution & History
- In `runs/train/exterior_detector_b7_pilot/results.csv`, Model B7 Pilot completed all **20 of 20 scheduled epochs** in 1,574.06 seconds (26.23 minutes) on the RTX 4060 Laptop GPU.
- Peak VRAM observed: **2.85 GB** (at batch size 8, 640px).
- Best validation epoch was Epoch 20: mAP50 reached **0.5072**, mAP50–95 reached **0.3062**, with box loss 1.0069 and classification loss 0.8999.
- Early stopping was not triggered (`patience=6` not exceeded due to consistent validation gains).

### 3.2 Checkpoint Inventory & Integrity
All key model checkpoints were inspected, loaded, and verified against recorded cryptographic digests:

| Model Identifier | File Path | File Size | MD5 Hash | SHA-256 Hash | Operational Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model B2 Default** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | 19,163,034 B | `14a931dd...` | `C599C19241B8C60390AC0C4914DE6CEF449BC1129F73B0DB7F8F04FC62A0459F` | **Active Production Default** |
| **Model B6 Baseline**| `checkpoints/exterior_detector_b6_expanded_best.pt` | 19,167,194 B | `f5ceea56...` | `009144B272998EC9447F649E8D850E0FDB8CAE5FC973C5E0CB9EEBE717675AC9` | **Principal Research Baseline** |
| **Model B7 Pilot**   | `checkpoints/exterior_detector_b7_pilot_best.pt`    | 19,167,194 B | `41433e92...` | `0B04B97E66F033D2037BC1483D4AD4F484A00213DEFEE66A7D32AD83199E2C3F` | **Completed Pilot Candidate** |
| **Porch Specialist** | `checkpoints/exterior_specialist_porch_yolo11s_best.pt` | 19,159,130 B | `4261fc1a...` | `43228FD11E26605273C8CEF06CEFD25310C199042BD99048B8544B36A7A61E34` | Specialist Checkpoint |

- Model B7 loads successfully via Ultralytics as a 10-class `DetectionModel`.
- Model B2 and B6 have not been altered or overwritten.
- `config/model_config.yaml` explicitly retains Model B2 as `status: "production_ready"`.

### 3.3 Loss Masking Verification
- Masked loss implementation in `utils/masked_detection_loss.py` operates on `batch["im_file"]` to apply per-image class masks.
- Mathematically verified: On fully annotated images, the loss tensor is mathematically identical to standard `v8DetectionLoss` (`max abs diff = 0.0`). Missing classes produce 0 BCE loss and 0 gradient.

---

## 4. Local Dataset Manifest & Inventory Audit

### 4.1 Benchmark & Validation Splits (Locked & Protected)
- **Authorized Validation Split:** `data/exterior_b6_expanded/images/val` (154 scenes, 8,504 ground-truth annotations). Locked and untouched.
- **Retired Test Split:** `data/exterior_b6_expanded/images/test` (154 scenes). Quarantined and permanently retired from future model selection or iterative tuning.

### 4.2 Existing Training Datasets on Disk
| Dataset Directory | Total Scenes | Annotations / Notes |
| :--- | :---: | :--- |
| `data/pilot_b7_dataset` | 1,756 scenes | 1,602 train (1,302 legacy + 300 pilot), 154 val |
| `data/exterior_b6_expanded` | 1,610 scenes | 1,302 train, 154 val, 154 test |
| `data/exterior_canonical` | 726 scenes | 418 train, 154 val, 154 test |
| `data/cache_pilot_b7_candidates` | 781 images | 481 uncurated remaining candidates from Open Images V7 |
| `data/cache_oi_b6_candidates` | 399 images | Pre-downloaded Open Images candidate images |
| `data/cache_oi_b5_images` | 433 images | Pre-downloaded transfer candidates |
| `C:\Users\apaar\fiftyone\open-images-v7` | Local | Contains `train\labels\detections.csv` for Open Images V7 |

### 4.3 Git Hygiene & Working Tree State
- `git status --short`: No files staged. 13 modified tracked files (application UI, services, models, config).
- `.gitignore`: Correctly excludes checkpoints (`*.pt`), runs (`runs/`), datasets (`data/`), and cache files.
- `artifacts/`: Contains 112 MB of generated diagnostic crops (recommended for future `.gitignore` inclusion).
- No destructive git operations were executed.

---

## 5. Storage Budget & Training Feasibility Analysis

### 5.1 Storage Budget Breakdown
- **Current Available Free Disk Space:** **73.23 GB**
- **Recommended Safe Storage Ceiling for Phase 22:** **15.00 GB**  
  (Leaves $>58$ GB reserved for operating system swap, pagefiles, and general safety).
- **Target Incremental Dataset Expansion:** **1,000 to 3,000 images**.
  - Average architectural scene image size: ~250–350 KB.
  - 1,000 images: $\approx 0.35$ GB uncompressed.
  - 3,000 images: $\approx 1.05$ GB uncompressed.
  - Associated YOLO label files (`.txt`): $< 0.05$ GB.
  - Cached preprocessed arrays / dataset manifests: $\approx 1.0$ GB.
  - Experiment runs & checkpoints (3 experiments $\times$ 5 checkpoints $\times$ 19 MB): $\approx 0.30$ GB.
  - **Total Projected Storage Consumption:** **$\approx 2.5$ to $4.0$ GB**.
  - **Verdict:** Highly safe. Consumes only $\approx 5\%$ of available free disk space.

### 5.2 Training Memory (VRAM) Budget Breakdown
- **Available Dedicated VRAM:** 8.00 GB.
- **Measured VRAM Profile (YOLO11s at 640px):**
  - Batch size 4: $\approx 2.10$ GB VRAM.
  - Batch size 8: $\approx 2.85$ GB VRAM.
  - Mixed precision (`amp=True`): stable and verified.
  - DataLoader workers: `workers=0` (conservative, prevents Windows multi-process shared-memory contention).
- **Verdict:** Batch size 4–8 operates at $<36\%$ of total VRAM, leaving $\ge 5.15$ GB of safety headroom.

---

## 6. Stop Confirmation (Stage Zero Complete)

As instructed:
- **Stage Zero is complete.**
- **No data downloads have been initiated.**
- **No training jobs have been started.**
- The environment is confirmed safe, reproducible, and ready for Stage 1 dataset curation and sequential experiment design upon instruction.

