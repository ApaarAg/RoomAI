# ROOMAI — FINAL BACKEND SECURITY AND DEPLOYMENT REPORT

**Date:** 2026-11-06  
**Auditor:** Principal Software Architect & MLOps Security Engineer  
**System Scope:** Input Sanitation, Resource Bounds, Dependency Isolation & Local Windows Deployment  

---

## 1. Input Validation and Denial-of-Service (DoS) Protection

### 1.1 Decompression Bomb Mitigation
- **Risk:** Malicious or oversized image files (e.g. 100,000 x 100,000 pixels compressed into a tiny JPEG or PNG) can expand to dozens of gigabytes in RAM upon decoding, crashing the host operating system.
- **Protection Implemented:**
  - PIL global safety ceiling is enforced across the backend facade:
    ```python
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 100_000_000  # Caps image dimensions at ~100 megapixels
    ```
  - Upfront dimension verification: Any image with width or height exceeding **8,192 pixels** is rejected with a structured error before tensor memory allocation:
    ```python
    valid, err_msg = validate_input_image(image, max_dim=8192)
    ```

### 1.2 Type and Null-Value Safety
- **Risk:** Missing or `None` image arguments causing unhandled `AttributeError` crashes in deep neural network inference loops.
- **Protection Implemented:** All perception endpoints (`detect_objects`, `segment_scene`, `estimate_depth`, `reconstruct_room_3d`) perform strict upfront type and null checking, returning standardized error envelopes:
  ```json
  {
    "status": "error",
    "capability_status": "PRODUCTION_READY",
    "error": {
      "code": "INVALID_INPUT_IMAGE",
      "message": "Input image cannot be None.",
      "remediation": "Provide a valid RGB PIL Image instance."
    }
  }
  ```

### 1.3 3D Asset Export Bounds
- **Point Cloud Capping:** Point cloud exports are bounded by default to `max_points=40000` to prevent memory exhaustion and browser WebGL tab crashes when loading scenes into Three.js or Plotly.
- **Structural Validation:** Exported binary GLB and ASCII PLY payloads are verified before delivery via `validate_glb_bytes` and `validate_ply_bytes`.

---

## 2. Dependency & Environment Isolation

### 2.1 Conda Environment Isolation
- **Environment Name:** `tf-gpu`
- **Python Binary:** `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`
- **Notice on Base Conda:** The base Conda environment on this workstation contains conflicting Hugging Face Hub / Transformers packages. **All execution must explicitly target the `tf-gpu` environment binary**.

### 2.2 Framework Import Conflict Guards
- To prevent NumPy 2.x ABI conflicts between TensorFlow and PyTorch inside Hugging Face transformers, the backend enforces:
  ```python
  import os
  os.environ["USE_TF"] = "0"
  os.environ["TRANSFORMERS_NO_TF"] = "1"
  ```
- This ensures pure PyTorch execution without triggering unwanted TensorFlow library initializations.

### 2.3 Model Checkpoint Safety
- Hugging Face depth models are loaded exclusively from `.safetensors` files (`model.safetensors`), eliminating arbitrary code execution vulnerabilities associated with legacy Python `pickle` deserialization.

---

## 3. Resource Allocation and VRAM Management

| Resource | Safe Operational Limit | RoomAI Peak Consumption | Headroom Available | Status |
| :--- | :--- | :--- | :--- | :--- |
| **GPU VRAM (Allocated)** | 7,000 MB (of 8,187 MB) | **367.7 MB** | **7,819.8 MB** | **SAFE (>95% free)** |
| **GPU VRAM (Reserved)** | 7,500 MB (of 8,187 MB) | **486.0 MB** | **7,701.5 MB** | **SAFE (>94% free)** |
| **Host System RAM** | 20,000 MB (of 24,000 MB)| **2,292.7 MB** | **21,707.3 MB** | **SAFE (>90% free)** |
| **Export File Size** | 25 MB | **~1.35 MB (PLY), 0.48 MB (GLB)**| N/A | **LIGHTWEIGHT** |

---

## 4. Local Windows Deployment Topology

### 4.1 Architecture Diagram
```
┌────────────────────────────────────────────────────────┐
│                   Windows 11 Host                      │
│                                                        │
│  ┌───────────────────────┐   HTTP / IPC   ┌──────────┐ │
│  │ Streamlit / Frontend  │ ─────────────▶ │ RoomAI   │ │
│  │ (Port 8501)           │                │ Backend  │ │
│  └───────────────────────┘ ◀───────────── │ Facade   │ │
│                                           └──────────┘ │
│                                                 │      │
│                                           CUDA  │ 12.8 │
│                                                 ▼      │
│                                           ┌──────────┐ │
│                                           │ RTX 4060 │ │
│                                           │ (8 GB)   │ │
│                                           └──────────┘ │
└────────────────────────────────────────────────────────┘
```

### 4.2 Network Binding and Security
- For local workstation demonstrations, services must bind strictly to `localhost` (`127.0.0.1`), never `0.0.0.0`, to prevent unauthorized local area network access.
- Cross-Origin Resource Sharing (CORS): If exposing via FastAPI or REST in future phases, restrict `allow_origins` to explicit frontend development origins (e.g. `http://localhost:3000`, `http://localhost:8501`).

### 4.3 Startup Script
To launch the deployment securely:
```powershell
# Set environment variables
$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"

# Execute using explicit conda environment
& "C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

