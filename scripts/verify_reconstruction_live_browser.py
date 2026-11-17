import asyncio
import base64
import json
import os
import subprocess
import time
from pathlib import Path
import requests
import websockets

ARTIFACTS_DIR = Path(r"C:\Users\apaar\.gemini\antigravity\brain\b87cbd12-026c-4200-aa90-7132427794d0")
PORT = 9555
USER_DATA_DIR = r"F:\RoomAI\.chrome-test-recon-9555"

async def main():
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR, exist_ok=True)

    print("Launching headless Chrome on port", PORT)
    proc = subprocess.Popen([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={USER_DATA_DIR}",
        "--window-size=1440,960",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--no-first-run",
        "--no-default-browser-check"
    ])
    time.sleep(2.5)

    console_errors = []

    try:
        tabs = requests.get(f"http://127.0.0.1:{PORT}/json/list").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        print("Connected to CDP:", ws_url)

        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            msg_id = 0

            async def call(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("method") == "Runtime.consoleAPICalled":
                        args = resp.get("params", {}).get("args", [])
                        text = " ".join(str(a.get("value", a.get("description", ""))) for a in args)
                        t = resp.get("params", {}).get("type")
                        if t == "error":
                            console_errors.append(text)
                            print("[Console Error]:", text)
                    if resp.get("id") == msg_id:
                        return resp.get("result", {})

            async def eval_js(expression):
                res = await call("Runtime.evaluate", {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True
                })
                return res.get("result", {}).get("value")

            await call("Page.enable")
            await call("Runtime.enable")

            # -------------------------------------------------------------
            # STEP 1: Verify /model page loads with NO network errors
            # -------------------------------------------------------------
            print("\n[STEP 1] Navigating to http://localhost:3000/model ...")
            await call("Page.navigate", {"url": "http://localhost:3000/model"})
            await asyncio.sleep(4)

            # Check for error banners
            error_check = await eval_js("""
            (() => {
                const errBanner = document.querySelector('.error-banner');
                const errDetail = errBanner ? errBanner.innerText : null;
                const badge = document.querySelector('.live-status-pill')?.innerText;
                const title = document.querySelector('h1')?.innerText;
                return { hasError: !!errBanner, errDetail, badge, title };
            })()
            """)
            print("Page Initial State:", json.dumps(error_check, indent=2))
            assert not error_check["hasError"], f"Unexpected error banner found: {error_check['errDetail']}"

            snap = await call("Page.captureScreenshot", {"format": "png"})
            p1 = ARTIFACTS_DIR / "model_page_healthy_no_error.png"
            p1.write_bytes(base64.b64decode(snap["data"]))
            print("Saved screenshot:", p1)

            # -------------------------------------------------------------
            # STEP 2: Run Live Inference with Model A3 in 2D Optical View
            # -------------------------------------------------------------
            print("\n[STEP 2] Switching to 2D Optical View and Running Model A3 Live Inference ...")
            await eval_js("""
            (() => {
                const btn2d = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('2D OPTICAL VIEW'));
                if (btn2d) btn2d.click();
            })()
            """)
            await asyncio.sleep(1.0)

            await eval_js("""
            (() => {
                const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('RUN INFERENCE'));
                if (btn) btn.click();
            })()
            """)
            await asyncio.sleep(4.0)

            infer_res = await eval_js("""
            (() => {
                const boxes = document.querySelectorAll('.studio-box-group').length;
                const errBanner = document.querySelector('.error-banner');
                const readout = document.querySelector('.telemetry-readout')?.innerText;
                const detCount = document.querySelectorAll('.detection-card').length;
                return { boxes, hasError: !!errBanner, errDetail: errBanner ? errBanner.innerText : null, readout, detCount };
            })()
            """)
            print("Inference Result:", json.dumps(infer_res, indent=2))
            assert not infer_res["hasError"], f"Inference failed with error: {infer_res['errDetail']}"
            assert infer_res["boxes"] > 0, "No bounding boxes detected!"

            snap2 = await call("Page.captureScreenshot", {"format": "png"})
            p2 = ARTIFACTS_DIR / "model_inference_success.png"
            p2.write_bytes(base64.b64decode(snap2["data"]))
            print("Saved screenshot:", p2)

            # -------------------------------------------------------------
            # STEP 3: Switch to 3D INTERACTIVE ROOM View
            # -------------------------------------------------------------
            print("\n[STEP 3] Switching to 3D INTERACTIVE ROOM Mode ...")
            await eval_js("""
            (() => {
                const btn3d = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('3D INTERACTIVE ROOM'));
                if (btn3d) btn3d.click();
            })()
            """)
            await asyncio.sleep(2.0)

            mode_check = await eval_js("""
            (() => {
                const canvas = !!document.querySelector('.interactive-3d-studio-container canvas');
                const toolbar = !!document.querySelector('.studio-3d-toolbar');
                // Ensure no bounding boxes or detection text in 3D viewport
                const boxesIn3d = document.querySelectorAll('.studio-box-group').length;
                return { canvas, toolbar, boxesIn3d };
            })()
            """)
            print("3D Viewport Check:", json.dumps(mode_check, indent=2))
            assert mode_check["canvas"], "3D Canvas not found in 3D mode!"
            assert mode_check["boxesIn3d"] == 0, "Boxes should NOT be rendered in pure 3D view!"

            snap3 = await call("Page.captureScreenshot", {"format": "png"})
            p3 = ARTIFACTS_DIR / "model_3d_viewport_pure_geometry.png"
            p3.write_bytes(base64.b64decode(snap3["data"]))
            print("Saved screenshot:", p3)

            # -------------------------------------------------------------
            # STEP 4: Trigger Full 3D Room Reconstruction Pipeline
            # -------------------------------------------------------------
            print("\n[STEP 4] Triggering Full 3D Room Reconstruction Pipeline ...")
            await eval_js("""
            (() => {
                const reconBtn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('RECONSTRUCT 3D ROOM'));
                if (reconBtn) reconBtn.click();
            })()
            """)
            
            # Poll for completion up to 25 seconds
            t0 = time.time()
            completed = False
            while time.time() - t0 < 25:
                await asyncio.sleep(2)
                recon_status = await eval_js("""
                (() => {
                    const dlGlb = !!document.querySelector('a[download=\"roomai_scene.glb\"]');
                    const dlPly = !!document.querySelector('a[download=\"roomai_point_cloud.ply\"]');
                    const err = document.querySelector('.telemetry-error-box')?.innerText;
                    return { dlGlb, dlPly, err };
                })()
                """)
                print(f"Reconstruction polling ({time.time()-t0:.1f}s):", json.dumps(recon_status))
                if recon_status and recon_status.get("dlGlb"):
                    completed = True
                    break
                if recon_status and recon_status.get("err"):
                    raise RuntimeError(f"Reconstruction failed: {recon_status['err']}")

            assert completed, "3D reconstruction did not finish in time!"

            snap4 = await call("Page.captureScreenshot", {"format": "png"})
            p4 = ARTIFACTS_DIR / "model_3d_room_reconstructed_glb_active.png"
            p4.write_bytes(base64.b64decode(snap4["data"]))
            print("Saved screenshot:", p4)

            # -------------------------------------------------------------
            # STEP 5: Verify Explore Page Section 03 Room Construction Experience
            # -------------------------------------------------------------
            print("\n[STEP 5] Navigating to http://localhost:3000/explore#construction ...")
            await call("Page.navigate", {"url": "http://localhost:3000/explore#construction"})
            await asyncio.sleep(4)

            sec3_check = await eval_js("""
            (() => {
                const sec = document.getElementById('construction');
                if (!sec) return { found: false };
                const canvas = !!sec.querySelector('.room-3d-webgl-canvas');
                const pills = Array.from(sec.querySelectorAll('.stage-pill-button')).map(p => p.innerText.trim());
                return { found: true, canvas, pillsCount: pills.length, pills };
            })()
            """)
            print("Section 03 Check:", json.dumps(sec3_check, indent=2))
            assert sec3_check["found"], "Section 03 #construction not found!"
            assert sec3_check["pillsCount"] == 6, f"Expected 6 pills, got {sec3_check['pillsCount']}"

            # Jump to Stage 06 (Interactive 3D Room)
            await eval_js("""
            (() => {
                const pills = document.querySelectorAll('.stage-pill-button');
                if (pills.length >= 6) pills[5].click();
            })()
            """)
            await asyncio.sleep(2.5)

            snap5 = await call("Page.captureScreenshot", {"format": "png"})
            p5 = ARTIFACTS_DIR / "explore_section03_3d_room_active.png"
            p5.write_bytes(base64.b64decode(snap5["data"]))
            print("Saved screenshot:", p5)

            print("\n=== ALL BROWSER & 3D RECONSTRUCTION VERIFICATIONS PASSED ===")
            print("Total console errors:", len(console_errors))

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(main())
