# ROOMAI — FINAL BACKEND SETUP & REPRODUCIBILITY GUIDE

**Date:** 2026-11-06  
**Target Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), 24 GB RAM, Windows 11  
**Runtime:** Python 3.10.20, Conda Environment `tf-gpu`  

---

## 1. Environment & Hardware Prerequisites

### 1.1 Hardware
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM)
- **Driver:** NVIDIA Game Ready / Studio Driver supporting CUDA 12.8+
- **Host RAM:** 16 GB minimum, 24 GB recommended

### 1.2 Python Environment
- **Conda Environment:** `tf-gpu`
- **Python Binary Absolute Path:**
  ```
  C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe
  ```

> [!CAUTION]
> **CRITICAL ENVIRONMENT DIRECTIVE:**  
> On Windows PowerShell, running bare `python` resolves to the base Miniconda environment (`C:\Users\apaar\miniconda3\python.exe`), which has conflicting Hugging Face Hub / Transformers versions.  
> **Always invoke Python using the explicit path:**  
> `& "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe"`

---

## 2. Environment Variables

Set the following environment variables to ensure pure PyTorch execution and prevent TensorFlow library initialization conflicts:

```powershell
$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"
```

---

## 3. Production Model Checkpoints & SHA-256 Signatures

The following production checkpoints must be present in the repository under `checkpoints/`. Their cryptographic SHA-256 hashes must match the verified audit values:

| Model Description | Relative Path | Size | SHA-256 Signature |
| :--- | :--- | :--- | :--- |
| **Indoor Detector (A3 800px)** | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | 18.29 MB | `6cbc9e58f87156e07dd0e4e712b8aa96097aeb514a473621139a54cb83cec576` |
| **Indoor Detector (A2 Finetune)** | `checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt` | 18.27 MB | `42113b2ddb5abffa4830bde024cea13ec5be99b1c6493cb2a01b4b52a7ea08c8` |
| **Exterior Detector (B2)** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | 18.28 MB | `c599c19241b8c60390ac0c4914de6cef449bc1129f73b0db7f8f04fc62a0459f` |
| **Parking Detector (C2 Pilot)** | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | 18.27 MB | `4abab963ab0becfef4a03c4e7b0039cc048d7621b3456bc5435deece6eed1470` |
| **Instance Segmentation** | `checkpoints/yolo11n-seg.pt` | 5.90 MB | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |
| **Metric Depth Small** | `checkpoints/depth_anything_v2_metric_indoor_small/model.safetensors` | 94.58 MB | `e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3` |
| **Relative Depth Small** | `checkpoints/depth_anything_v2_small/model.safetensors` | 94.58 MB | `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70` |

---

## 4. Verification & Operational Commands

### 4.1 Run the Full Test Suite
To verify that all 270 automated tests pass without regressions:
```powershell
& "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" -m pytest -q
```
*Expected output: `270 passed in ~64s`*

### 4.2 Run the Real GPU Performance Benchmark
To run real inference on `artifacts/sample_room.jpg` and measure VRAM / latency:
```powershell
& "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" scripts/benchmark_final_backend.py
```
*Expected output: All 8 components benchmarked with 0 errors; verification artifacts saved to `artifacts/final_backend_verification/`.*

### 4.3 Launch the Application Frontend
To launch the Streamlit frontend interface:
```powershell
& "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" -m streamlit run app.py --server.port=8501 --server.address=127.0.0.1
```

---

## 5. Troubleshooting Common Issues

### Issue 1: `huggingface-hub>=0.34.0,<1.0 is required... found 1.24.0`
- **Cause:** Command was executed using bare `python` or base Conda environment.
- **Fix:** Prefix command with `& "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe"`.

### Issue 2: `RuntimeError: CUDA out of memory`
- **Cause:** Another heavy GPU process (e.g. game, external LLM, or secondary Python script) is holding VRAM.
- **Fix:** Check VRAM usage with `nvidia-smi`. The RoomAI pipeline requires only ~500 MB of VRAM. Close background GPU tasks.

### Issue 3: `ValueError: Input image cannot be None`
- **Cause:** Frontend attempted to call a service without an uploaded image.
- **Fix:** Ensure the file uploader has completed before calling `BACKEND` methods.

