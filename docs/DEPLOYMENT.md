# ROOMAI — PRODUCTION DEPLOYMENT & OPERATIONS GUIDE

**Document Version:** 1.0.0  
**Status:** Approved for Production Deployment  
**Author:** Staff Software Engineer, Full-Stack Architect & Product Engineer  
**System Target:** Windows 11 / Linux (Ubuntu 22.04 LTS), NVIDIA GPU (>= 8 GB VRAM), CUDA 12.x  

---

## 1. System Requirements & Hardware Prerequisites

### 1.1 Minimum Hardware Specification
- **GPU:** NVIDIA GPU with $\ge 8.0\text{ GB}$ dedicated VRAM (e.g. RTX 4060, RTX 3070, A4000).
- **Compute Capability:** $\ge 8.0$ (Ampere, Ada Lovelace, Hopper).
- **System Memory (RAM):** $\ge 16\text{ GB}$ (24 GB recommended).
- **Disk Storage:** $\ge 20\text{ GB}$ free NVMe space for models, SQLite database, and temporary uploads.

### 1.2 Software Prerequisites
- **Operating System:** Windows 11 (64-bit) or Ubuntu Linux 22.04 LTS.
- **Python Version:** Python 3.10.x (verified on Python 3.10.20 in Conda env `tf-gpu`).
- **NVIDIA CUDA Toolkit:** Version 12.1 to 12.8.
- **PyTorch:** Version 2.1.x+ with CUDA support (`torch>=2.1.0+cu12x`).

---

## 2. Environment Setup

### 2.1 Activating the Production Conda Environment
```powershell
# Windows PowerShell
conda activate tf-gpu
```
```bash
# Linux Bash
source activate tf-gpu
```

### 2.2 Verifying Hardware & Python Environment
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### 2.3 Configuration Setup
Copy the environment template:
```bash
cp .env.example .env
```

---

## 3. Starting Application Services

RoomAI runs as two decoupled, independently scalable processes:
1. **The Backend REST API Server:** FastAPI running under Uvicorn ASGI on port `8000`.
2. **The Frontend Presentation Studio:** Streamlit interactive UI running on port `8501`.

### 3.1 Starting the FastAPI REST API Server

```bash
# Development Mode (with hot-reloading)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production Mode (Dedicated ASGI Worker)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```
> [!NOTE]
> Run the backend with `--workers 1` on single-GPU hardware to ensure all requests share the same process-level `GPU_GATE` semaphore and in-memory background worker queue.

### 3.2 Starting the Streamlit Frontend

In a separate terminal or service manager:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false
```

### 3.3 Quick-Start Scripts
For convenience on Windows:
```powershell
# Start Streamlit Studio directly
.\scripts\run_roomai.ps1
```

---

## 4. Operational Health & Readiness Verification

Once services are running, verify service liveness and hardware readiness via HTTP curl:

```bash
# 1. Check Liveness
curl -s http://localhost:8000/health
# Expected: {"status":"healthy","version":"1.4.0",...}

# 2. Check GPU & Model Readiness
curl -s http://localhost:8000/ready
# Expected: {"status":"ready","cuda_available":true,"device_name":"...","checkpoints_ready":true,...}

# 3. Discover Models
curl -s http://localhost:8000/api/v1/models
# Expected: {"status":"success","models":[...]}
```

---

## 5. Security & Production Hardening Checklist

1. **Path Traversal Protection:** All file paths are strictly resolved within `artifacts/` using `os.path.basename` sanitization.
2. **Decompression Bomb Protection:** Pillow's maximum pixel threshold is clamped to 100,000,000 pixels (`Image.MAX_IMAGE_PIXELS = 100_000_000`).
3. **GPU Concurrency Gate:** Serialized via `GPUConcurrencyGate` (`max_concurrent_ops = 1`). Never run multiple concurrent GPU forward passes on an 8 GB card.
4. **Artifact Storage Quota:** Monitored by `ArtifactManager` with a 500 MB hard ceiling. Run periodic retention checks to prune expired jobs older than 24 hours.
5. **CORS Restrictions:** In production, edit `CORS_ORIGINS` in `.env` to restrict cross-origin access to your trusted domain.

---

## 6. Docker Evaluation & Cloud Notes

Docker deployment on Windows requires WSL2 with NVIDIA Container Toolkit. Given the local 8 GB RTX 4060 Laptop hardware and PyTorch CUDA 12.8 drivers, bare-metal Conda execution provides optimal latency, zero virtualization overhead, and direct access to GPU memory telemetry.

For cloud deployments (e.g. Google Cloud Vertex AI or AWS EC2 G5 instance), use the official NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:24.04-py3`) with `docker run --gpus all -p 8000:8000 -p 8501:8501`.

