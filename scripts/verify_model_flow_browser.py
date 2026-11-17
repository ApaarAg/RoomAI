import asyncio
import base64
import json
import os
import subprocess
import time
from pathlib import Path
import requests
import websockets

ARTIFACTS_DIR = Path(r"C:\Users\apaar\.gemini\antigravity\brain\7c6e77c3-465e-4e84-bb87-4e80db4528d5")
PORT = 9333

async def main():
    print("Launching headless Chrome on port", PORT)
    proc = subprocess.Popen([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        r"--user-data-dir=F:\RoomAI\.chrome-debug-9333",
        "--window-size=1440,1080",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check"
    ])
    time.sleep(2.5)

    console_errors = []
    failed_requests = []

    try:
        tabs = requests.get(f"http://127.0.0.1:{PORT}/json/list").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        print("Connected to CDP:", ws_url)

        async with websockets.connect(ws_url, max_size=100 * 1024 * 1024) as ws:
            msg_id = 0
            pending_responses = {}

            async def call(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    resp = json.loads(await ws.recv())
                    # Check for network or console events
                    if resp.get("method") == "Log.entryAdded":
                        entry = resp["params"]["entry"]
                        if entry.get("level") == "error":
                            console_errors.append(entry)
                            print("[Console Error Event]:", json.dumps(entry))
                    elif resp.get("method") == "Network.responseReceived":
                        resp_params = resp["params"]
                        status = resp_params["response"]["status"]
                        url = resp_params["response"]["url"]
                        if status >= 400 and not url.endswith("/favicon.ico"):
                            failed_requests.append({"url": url, "status": status})
                    
                    if resp.get("id") == msg_id:
                        return resp.get("result", {})

            await call("Page.enable")
            await call("DOM.enable")
            await call("Runtime.enable")
            await call("Network.enable")
            await call("Log.enable")

            # -----------------------------------------------------------------
            # Step 1: Landing Page (/) Navigation & CTA Discovery
            # -----------------------------------------------------------------
            print("\n[Step 1] Navigating to http://localhost:3000/ ...")
            await call("Page.navigate", {"url": "http://localhost:3000/"})
            await asyncio.sleep(4)

            # Check CTA link target
            eval_cta = await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const cta = document.querySelector('.cta-button-primary');
                    const nav = Array.from(document.querySelectorAll('.nav-link')).map(el => ({ text: el.innerText.trim(), href: el.getAttribute('href') }));
                    return {
                        ctaText: cta ? cta.innerText.trim() : null,
                        ctaHref: cta ? cta.getAttribute('href') : null,
                        navLinks: nav
                    };
                })()
                """,
                "returnByValue": True
            })
            cta_info = eval_cta.get("result", {}).get("value", {})
            print("Landing Page CTA Discovery Info:", json.dumps(cta_info, indent=2))
            assert cta_info.get("ctaHref") == "/explore#detect", f"Expected /explore#detect, got {cta_info.get('ctaHref')}"

            # Capture Landing Page Screenshot
            snap_landing = await call("Page.captureScreenshot", {"format": "png"})
            p_landing = ARTIFACTS_DIR / "model_access_01_landing_hero.png"
            with open(p_landing, "wb") as f:
                f.write(base64.b64decode(snap_landing["data"]))
            print("Saved Landing Page screenshot to:", p_landing)

            # -----------------------------------------------------------------
            # Step 2: Navigate / Click into /explore#detect
            # -----------------------------------------------------------------
            print("\n[Step 2] Navigating to http://localhost:3000/explore#detect ...")
            await call("Page.navigate", {"url": "http://localhost:3000/explore#detect"})
            await asyncio.sleep(4)

            # Center on the Result Viewer frame
            await call("Runtime.evaluate", {
                "expression": "document.querySelector('#explore .result-viewer-frame').scrollIntoView({ behavior: 'instant', block: 'center' });"
            })
            await asyncio.sleep(2)

            # Verify Model A3 detector is active and inspect the Technical Model Panel
            eval_panel = await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const panel = document.querySelector('.model-technical-strip');
                    const items = Array.from(panel ? panel.querySelectorAll('.spec-item') : []).map(el => {
                        const lbl = el.querySelector('.spec-label')?.innerText.trim();
                        const val = el.querySelector('.spec-value')?.innerText.trim();
                        return { label: lbl, value: val };
                    });
                    const activeModeBtn = document.querySelector('.mode-control[aria-pressed="true"]');
                    const isLiveDetect = document.querySelector('.live-inference-panel') !== null;
                    return {
                        hasStrip: panel !== null,
                        specs: items,
                        activeMode: activeModeBtn ? activeModeBtn.innerText.trim() : null,
                        isLiveDetect: isLiveDetect
                    };
                })()
                """,
                "returnByValue": True
            })
            panel_info = eval_panel.get("result", {}).get("value", {})
            print("Model A3 Panel Inspection:", json.dumps(panel_info, indent=2))
            assert panel_info.get("isLiveDetect"), "Live inference panel must be rendered in #detect mode"
            assert len(panel_info.get("specs", [])) == 5, f"Expected 5 spec items, found {len(panel_info.get('specs', []))}"

            # Capture initial detector screenshot
            snap_detect_initial = await call("Page.captureScreenshot", {"format": "png"})
            p_detect_initial = ARTIFACTS_DIR / "model_access_02_detector_idle.png"
            with open(p_detect_initial, "wb") as f:
                f.write(base64.b64decode(snap_detect_initial["data"]))
            print("Saved Detector Idle screenshot to:", p_detect_initial)

            # -----------------------------------------------------------------
            # Step 3: Run Model A3 Inference
            # -----------------------------------------------------------------
            print("\n[Step 3] Clicking 'RUN MODEL A3' button...")
            await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const btn = document.querySelector('.action-pill.primary');
                    if (btn && btn.innerText.includes('RUN MODEL A3')) {
                        btn.click();
                        return true;
                    }
                    return false;
                })()
                """
            })

            # Wait for inference completion (polling up to 10s)
            print("Waiting for Model A3 inference response...")
            detections_rendered = False
            telemetry_data = {}
            for attempt in range(25):
                await asyncio.sleep(0.5)
                eval_det = await call("Runtime.evaluate", {
                    "expression": """
                    (() => {
                        const boxes = document.querySelectorAll('.detection-box-group');
                        const telemetryItems = Array.from(document.querySelectorAll('.telemetry-item')).map(el => ({
                            label: el.querySelector('span')?.innerText.trim(),
                            val: el.querySelector('strong')?.innerText.trim()
                        }));
                        return {
                            boxCount: boxes.length,
                            telemetry: telemetryItems,
                            hasSvg: document.querySelector('.inference-overlay') !== null
                        };
                    })()
                    """,
                    "returnByValue": True
                })
                res = eval_det.get("result", {}).get("value", {})
                if res.get("boxCount", 0) > 0:
                    detections_rendered = True
                    telemetry_data = res
                    break

            print("Inference outcome:", json.dumps(telemetry_data, indent=2))
            assert detections_rendered, "Model A3 inference must render bounding boxes on success!"

            # Capture screenshot of live detections
            snap_detect_success = await call("Page.captureScreenshot", {"format": "png"})
            p_detect_success = ARTIFACTS_DIR / "model_access_03_detections_rendered.png"
            with open(p_detect_success, "wb") as f:
                f.write(base64.b64decode(snap_detect_success["data"]))
            print("Saved Detections Rendered screenshot to:", p_detect_success)

            # -----------------------------------------------------------------
            # Step 4: Inspect Provisional 3D Ray Metrics
            # -----------------------------------------------------------------
            print("\n[Step 4] Clicking 'INSPECT PROVISIONAL 3D RAY METRICS'...")
            await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const btn = document.querySelector('.telemetry-toggle-btn');
                    if (btn) {
                        btn.click();
                        return true;
                    }
                    return false;
                })()
                """
            })
            await asyncio.sleep(1.5)

            # Scroll spatial inspector into view
            await call("Runtime.evaluate", {
                "expression": "document.querySelector('.spatial-inspector-panel').scrollIntoView({ behavior: 'instant', block: 'center' });"
            })
            await asyncio.sleep(1.5)

            eval_spatial = await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const panel = document.querySelector('.spatial-inspector-panel');
                    const header = panel ? panel.querySelector('.inspector-header')?.innerText.trim() : null;
                    const rows = Array.from(panel ? panel.querySelectorAll('.spatial-table tbody tr') : []).map(r => {
                        const cells = Array.from(r.querySelectorAll('td')).map(c => c.innerText.trim());
                        return cells;
                    });
                    return {
                        hasPanel: panel !== null,
                        header: header,
                        rowCount: rows.length,
                        sampleRow: rows[0] || null
                    };
                })()
                """,
                "returnByValue": True
            })
            spatial_info = eval_spatial.get("result", {}).get("value", {})
            print("Spatial Inspector Info:", json.dumps(spatial_info, indent=2))
            assert spatial_info.get("hasPanel"), "Spatial inspector panel must be visible"
            assert "PROVISIONAL 3D CAMERA-RAY PROJECTION" in spatial_info.get("header", ""), "Must state provisional 3D projection"
            assert spatial_info.get("rowCount", 0) > 0, "Must have unprojected spatial coordinate rows"

            # Capture screenshot of Provisional 3D Table
            snap_spatial = await call("Page.captureScreenshot", {"format": "png"})
            p_spatial = ARTIFACTS_DIR / "model_access_04_provisional_3d_table.png"
            with open(p_spatial, "wb") as f:
                f.write(base64.b64decode(snap_spatial["data"]))
            print("Saved Provisional 3D Table screenshot to:", p_spatial)

            # -----------------------------------------------------------------
            # Step 5: Verify Visual Asset Mapping Integrity in Same Session
            # -----------------------------------------------------------------
            print("\n[Step 5] Checking asset mapping integrity on /explore ...")
            # Switch back to PHOTO mode to inspect the result viewer image layer
            await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const photoBtn = Array.from(document.querySelectorAll('.mode-control')).find(el => el.innerText.trim() === 'PHOTO');
                    if (photoBtn) photoBtn.click();
                })()
                """
            })
            await asyncio.sleep(1)

            eval_assets = await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const resultViewerImg = document.querySelector('.result-image-layer[data-stage="0"]')?.getAttribute('src');
                    const pipelineImgs = Array.from(document.querySelectorAll('.pipeline-image-layer')).map(el => el.getAttribute('src'));
                    return {
                        resultViewerImg: resultViewerImg,
                        pipelineImgs: pipelineImgs
                    };
                })()
                """,
                "returnByValue": True
            })
            assets_info = eval_assets.get("result", {}).get("value", {})
            print("Asset mapping check:", json.dumps(assets_info, indent=2))
            assert "/assets/roomai/result-viewer/01-rgb.png" in assets_info.get("resultViewerImg", "")
            for p_img in assets_info.get("pipelineImgs", []):
                assert "/assets/roomai/pipeline/" in p_img

            # -----------------------------------------------------------------
            # Step 6: Audit Console Errors and Network Failures
            # -----------------------------------------------------------------
            print("\n[Step 6] Console and Network Audit:")
            print(f"Console errors: {len(console_errors)}")
            for ce in console_errors:
                print("  Console Error:", ce.get("text"))
            print(f"Failed HTTP requests: {len(failed_requests)}")
            for fr in failed_requests:
                print(f"  HTTP {fr['status']}: {fr['url']}")

            assert len(failed_requests) == 0, f"Found failed HTTP requests: {failed_requests}"

            print("\nALL VERIFICATION STEPS PASSED WITH 100% EMPIRICAL PROOF!")

    finally:
        proc.terminate()
        print("Chrome process terminated.")

if __name__ == "__main__":
    asyncio.run(main())
