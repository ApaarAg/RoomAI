# RoomAI Local Windows Deployment & Operator Runbook

**Document Version:** 1.0.0 (Phase 7.1 Final Release)  
**Target Environment:** Local Windows 11 Workstation  
**Hardware Specification:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
**Dedicated Python Runtime:** `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe` (Python 3.10.20, PyTorch 2.11.0+cu128)  
**Local Binding Address:** `http://127.0.0.1:8501`

---

## 1. System Architecture & Checkpoint Inventory

RoomAI runs 100% locally on your NVIDIA RTX 4060 GPU with zero external cloud dependencies. All production weights reside in `checkpoints/` and have been verified against SHA-256 hashes:

| Component | Model Identifier | Checkpoint File Path | File Size | SHA-256 Checksum (Prefix) |
| :--- | :--- | :--- | :---: | :--- |
| **Indoor Detection** | Model A3 (YOLO11s 800px) | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | 19.2 MB | `6cbc9e58f871...` |
| **Exterior Detection** | Model B2 (YOLO11s) | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | 19.2 MB | `c599c19241b8...` |
| **Parking Detection** | Model C2 (YOLO11s) | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | 19.2 MB | `4abab963ab0b...` |
| **Segmentation** | YOLO11n-seg | `checkpoints/yolo11n-seg.pt` | 6.2 MB | `55ed65c56c91...` |
| **Metric 3D Depth** | Depth Anything V2 Metric Small | `checkpoints/depth_anything_v2_metric_indoor_small` | 99.2 MB | `e990eb82fbf1...` |
| **Relative 3D Depth**| Depth Anything V2 Small | `checkpoints/depth_anything_v2_small` | 99.2 MB | `3152477ce0d8...` |

---

## 2. Step-by-Step Operator Launch Guide

### Step 1: Pre-flight Verification (Automated Self-Test)
Before starting the web server, execute the automated pre-flight script to verify Python packages, GPU acceleration, and checkpoint file integrity:

```powershell
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe scripts/validate_roomai_environment.py
```

**Expected Console Output:**
```
======================================================================
ROOMAI LOCAL WINDOWS ENVIRONMENT PRE-FLIGHT VERIFICATION
======================================================================
[1] Python Runtime: 3.10.20 -> ✅ Python version compatible.
[2] Required Dependencies  -> ✅ All 11 core packages Available.
[3] Hardware Acceleration  -> ✅ NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB)
[4] Checkpoints Integrity  -> ✅ All 6 production checkpoints VERIFIED.
[5] Config Verification    -> ✅ mock_mode is strictly False.
PRE-FLIGHT VALIDATION PASSED: System is ready for local deployment.
======================================================================
```

---

### Step 2: Launch the RoomAI Streamlit Application

#### Method A: Using the Automated PowerShell Script (Recommended)
From the project root (`F:\RoomAI`), execute:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_roomai.ps1
```

#### Method B: Direct Command-Line Launch
```powershell
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m streamlit run app.py `
    --server.address=127.0.0.1 `
    --server.port=8501 `
    --server.headless=false `
    --browser.serverAddress=127.0.0.1
```

---

### Step 3: Access the User Interface
Open your web browser (Google Chrome, Microsoft Edge, or Firefox) and navigate to:
```
http://127.0.0.1:8501
```

---

## 3. Production UI Navigation & Features

The user interface exposes three streamlined production capabilities:

### Tab 1: 🔍 Object Detection
- **Primary Detector:** Model A3 (Higher-Resolution YOLO11s 800px) active by default for indoor room images.
- **Domain Selector:** Switch to Exterior Architecture (Model B2) or Parking & Outdoor (Model C2) if analyzing building facades or driveways.
- **Visualization:** Draws bounding boxes with class labels and confidence scores.

### Tab 2: 💬 Vision Q&A
- Interactive conversational assistant powered by local vision-language features to answer questions about spatial arrangements, lighting, and interior styling.

### Tab 3: 🧊 3D Scene Reconstruction
- **One-Click Reconstruction:** Click `Reconstruct 3D Room Scene`.
- **Metric Depth Heatmap:** Visualizes true distances from camera in meters.
- **Room Architecture Metrics:** Displays:
  - `Est. Room Width (Envelope)`: Horizontal bounding span across camera field of view.
  - `Est. Room Length (Depth)`: Longitudinal span along optical depth axis.
  - `Est. Ceiling Height`: Vertical bounding span between floor and visible upper bounds.
  - `Est. Cam Elevation`: Estimated camera height above fitted floor plane.
- **Interactive 3D Viewer:** Orbit, pan, and zoom in WebGL Plotly 3D space with room boundary wireframe and 3D furniture boxes.
- **Download 3D Exports:**
  - `Download GLB (.glb)`: Complete binary glTF 2.0 scene with separated point clouds and bounding boxes.
  - `Download PLY (.ply)`: Standard Stanford ASCII 1.0 point cloud with RGB vertex coloring.

---

## 4. Operational Troubleshooting

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| `Port 8501 is already in use` | A lingering Streamlit process from a prior session is still running. | In PowerShell run: `Get-Process python \| Stop-Process -Force` then re-launch. |
| `Mock Mode Active` badge displayed | User toggled checkbox or loaded test mock mode. | In the sidebar under "Application Settings", uncheck "Enable Mock Mode". Real inference runs on GPU by default. |
| Browser shows black screen in 3D viewer | Hardware acceleration disabled in browser. | In Chrome/Edge settings, ensure "Use graphics acceleration when available" is turned ON. |
| Out of Memory (OOM) error | Multiple GPU-heavy desktop games or apps running. | Peak RoomAI VRAM is only 318.8 MB. Close background GPU-heavy games or rendering software. |

---

## 5. Security & Isolation Safeguards

- **Strict Localhost Binding:** The server is bound strictly to `127.0.0.1`, preventing any unauthorized incoming network connections from the local LAN.
- **Zero Cloud Leakage:** All images, model weights, and 3D exports are processed purely in local system memory and on-disk NVMe storage. No telemetry or images are sent to external APIs.

