# RoomAI — Restricted Private Beta Deployment Guide

**Date:** 2026-10-30
**Target:** a small, known group of trusted users via an authenticated secure tunnel to a
local Streamlit app on the RTX 4060 laptop.
**Not for:** public beta or production (see "Before public beta" below).

---

## 1. Prerequisites

- Windows 11 machine with the `tf-gpu` conda environment already set up
  (`C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe`, Python 3.10.20).
- NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), driver supporting CUDA 12.8.
- Production checkpoints present under `checkpoints/`:
  - `indoor_detector_a3_yolo11s_800px_best.pt` (Indoor A3)
  - `exterior_detector_b2_yolo11s_best.pt` (Exterior B2)
  - `parking_detector_c2_yolo11s_pilot_best.pt` (Parking C2)
  - `yolo11n-seg.pt` (segmentation)
- Dataset images present under `data/` (gitignored) for the sample picker.

Verify the environment:
```powershell
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -c "import streamlit,torch,ultralytics,transformers; print(torch.cuda.get_device_name(0))"
```
Expected: `NVIDIA GeForce RTX 4060 Laptop GPU`.

---

## 2. Reproduce the environment (from `requirements.txt`)

```powershell
cd f:\RoomAI
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pip install -r requirements.txt
# GPU runtimes (already installed in tf-gpu; if reinstalling on a fresh env):
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pip install ultralytics==8.4.66 transformers==4.49.0
```

Do **not** upgrade torch/ultralytics/transformers major versions without re-verifying
checkpoint compatibility — the production YOLO11 weights were trained with the versions above.

### CPU-only fallback (optional)
On a machine without CUDA, install the CPU torch wheel and the same ultralytics/transformers
pins; inference runs on `device='cpu'` with the same production checkpoints (slower,
correct).

---

## 3. Run the test suite (verification gate)

```powershell
cd f:\RoomAI
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pytest -q -p no:cacheprovider
```
Expected: **143 passed** (this cycle). Do not deploy if any test fails.

---

## 4. Start the application (local, loopback only)

The provided `.streamlit/config.toml` binds Streamlit to `127.0.0.1:8501`. Start it with:

```powershell
cd f:\RoomAI
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

- `--server.address 127.0.0.1` — loopback only; the app is **not** reachable from the LAN or
  the public internet directly.
- `--server.headless true` — does not auto-open a browser (appropriate for a server).

Check locally: open `http://127.0.0.1:8501` on the laptop.

---

## 5. Expose to trusted external users — authenticated tunnel

**Do NOT use router port forwarding. Do NOT bind to 0.0.0.0.** Use a secure tunnel with an
authentication layer. Two supported options:

### Option A — Cloudflare Tunnel + Cloudflare Access (recommended)
1. Install `cloudflared` on the laptop:
   ```powershell
   winget install --id Cloudflare.cloudflared
   ```
2. Authenticate and create a tunnel (run as the user who owns the Cloudflare account):
   ```powershell
   cloudflared tunnel login
   cloudflared tunnel create roomai-beta
   ```
3. Create a Cloudflare Access application in the Zero Trust dashboard:
   - **Application domain:** e.g. `roomai-beta.<your-team>.cfargotunnel.com`
   - **Identity provider:** your SSO (Google/Entra/GitHub/OIDC).
   - **Allowed emails/groups:** only the trusted beta users.
   - This is the **authentication layer**. Streamlit itself has no login; Access enforces it.
4. Map the tunnel to the local Streamlit port:
   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8501 run roomai-beta
   ```
5. Users browse to the Access-protected hostname; they must authenticate via your IdP
   before reaching Streamlit.

### Option B — Tailscale Funnel (simpler, identity-based)
1. Install Tailscale on the laptop and `tailscale up`.
2. Share the machine with your tailnet's trusted users.
3. Expose the local port:
   ```powershell
   tailscale serve --bg https://127.0.0.1:8501
   ```
   (or use `tailscale funnel` for internet-reachable access — still identity-scoped to your tailnet).
4. Users must be members of your tailnet; access is denied to anyone outside it.

### What the tunnel guarantees
- TLS termination.
- Identity (SSO / tailnet membership).
- No direct exposure of `127.0.0.1:8501` to the internet.

### What the application guarantees
- Loopback binding (`config.toml`).
- Upload validation (size/format/bomb/corruption/dimension clamp).
- GPU concurrency serialisation (no VRAM OOM under concurrency).
- Session-scoped artifacts (no cross-session collision).
- Generic error messages (no stack traces to users).

---

## 6. Revoking access

- **Cloudflare Access:** in the Zero Trust dashboard, remove the user from the Access
  application's allow-list (or delete the application). The next request is blocked at the
  edge.
- **Tailscale:** remove the user from the tailnet (`tailscale` admin console → Devices/Users).
- **Emergency:** stop the Streamlit process (`Ctrl+C`) and/or `cloudflared tunnel delete
  roomai-beta`. The app is then only reachable from the laptop's own `127.0.0.1`.

---

## 7. Artifact retention (scheduled cleanup)

The application writes session-scoped detection artifacts to `artifacts/detections/<random>/`.
Set up an hourly cleanup so the disk does not grow unbounded:

### Windows Task Scheduler (hourly)
Action:
```powershell
C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -c "from utils.artifact_manager import ArtifactManager; from pathlib import Path; s=ArtifactManager().cleanup_expired(Path('artifacts/detections')); print(s)"
```
Tune the policy in `config/app.yaml` under `artifact_retention` (defaults: 24 h retention,
2 GiB cap, 120 s active-grace). The sweep is scoped to `artifacts/detections/` only; it never
touches your existing user artifacts under `artifacts/audit_galleries/`,
`artifacts/indoor_error_analysis/`, etc.

---

## 8. Operational checklist before each demo

- [ ] `pytest -q` passes (143).
- [ ] Production checkpoints exist (`checkpoints/indoor_detector_a3_*`,
      `exterior_detector_b2_*`, `parking_detector_c2_*`, `yolo11n-seg.pt`).
- [ ] `.streamlit/config.toml` binds to `127.0.0.1`.
- [ ] Tunnel + Access policy active; only trusted users allowed.
- [ ] Scheduled retention cleanup configured.
- [ ] Laptop is plugged in and the GPU is free of other heavy workloads.

---

## 9. Known limitations (do not exceed)

- Single 8 GB GPU; the GPU gate is capacity-1. Do not raise `gpu_concurrency.max_concurrent`
  without re-measuring VRAM headroom with all production checkpoints loaded.
- No in-app rate limiting (concurrency is bounded, request rate is not). For a private beta
  with trusted users this is acceptable.
- VLM and generative-redesign features are stubbed in real mode
  (`is_implemented=False`); do not present them as working.
- Segmentation uses generic COCO YOLO11n-seg (not room-architecture-aware).
- Depth Anything V2 Small returns **relative** inverse depth, not metric distance.

---

## 10. Before public beta / production

Add (out of scope for this cycle — documented, not implemented):
1. In-app authentication / per-user authorisation (do not rely solely on the tunnel).
2. In-app request **rate limiting** (e.g. `slowapi` with a per-user token bucket).
3. User-attributed **audit log** (who ran which model on which image, when).
4. Multi-process / HA deployment (or an external inference worker queue) if more than a
   handful of concurrent users are expected.

Until those exist, stay at **RESTRICTED PRIVATE BETA**.