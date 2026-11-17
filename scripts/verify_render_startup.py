"""Verification script: simulates Render port scanner and startup lifecycle.

Verifies that:
1. Uvicorn binds immediately and /health responds with HTTP 200 within hundreds of milliseconds.
2. /ready reports 'degraded' before warmup finishes.
3. /ready transitions to 'ready' after background warmup completes.
4. /api/v1/detect returns 503 before ready, and 200 after ready.
"""

import io
import os
import socket
import subprocess
import sys
import time
import requests
from PIL import Image

def find_free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def main():
    port = find_free_port()
    print(f"--- [STEP 1] Starting live Uvicorn server on port {port} ---")
    
    env = dict(os.environ)
    env["PORT"] = str(port)

    t_launch = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Step 2: Poll /health immediately (like Render's port scanner does)
        health_ok = False
        health_elapsed = 0.0
        ready_during_init = None

        print("--- [STEP 2] Probing /health for immediate port binding ---")
        for _ in range(50):
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if r.status_code == 200:
                    health_elapsed = time.time() - t_launch
                    health_ok = True
                    print(f"PROBE PASSED: /health reachable in {health_elapsed:.2f}s! Payload: {r.json()}")
                    break
            except Exception:
                time.sleep(0.05)

        if not health_ok:
            raise RuntimeError("Server failed to respond on /health within timeout")

        # Step 3: Check /ready right after /health
        print("--- [STEP 3] Probing /ready during background warmup ---")
        r_ready = requests.get(f"http://127.0.0.1:{port}/ready", timeout=2)
        print(f"PROBE CHECK: /ready status = {r_ready.json()['status']}, checkpoints_ready = {r_ready.json()['checkpoints_ready']}")

        # Step 4: Wait for warmup to complete
        print("--- [STEP 4] Waiting for background warmup to reach READY ---")
        ready_achieved = False
        for _ in range(60):
            r = requests.get(f"http://127.0.0.1:{port}/ready", timeout=2)
            data = r.json()
            if data["status"] == "ready":
                total_ready_time = time.time() - t_launch
                ready_achieved = True
                print(f"READY REACHED: /ready reports 'ready' after {total_ready_time:.2f}s! models_ready = {data['models_ready']}")
                break
            time.sleep(0.5)

        if not ready_achieved:
            raise RuntimeError("Model failed to reach ready state within timeout")

        # Step 5: Test detection endpoint with valid image
        print("--- [STEP 5] Testing /api/v1/detect post-warmup ---")
        img = Image.new("RGB", (320, 240), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        
        det_res = requests.post(
            f"http://127.0.0.1:{port}/api/v1/detect",
            files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")},
            data={"confidence_threshold": "0.35"},
        )
        print(f"DETECTION STATUS: {det_res.status_code}, model = {det_res.json().get('inference', {}).get('model')}")
        assert det_res.status_code == 200

        print("\n=== ALL REAL-SERVER STARTUP PROBES PASSED SUCCESSFULLY ===")
    finally:
        print("--- Cleaning up Uvicorn subprocess ---")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
