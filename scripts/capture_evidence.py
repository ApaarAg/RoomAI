import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import requests
import websockets
from PIL import Image
import numpy as np

ARTIFACTS_DIR = Path(r"C:\Users\apaar\.gemini\antigravity\brain\7c6e77c3-465e-4e84-bb87-4e80db4528d5")
USER_DATA_DIR = Path(r"F:\RoomAI\.chrome-verify-tmp")

PORT = 9333

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

async def main():
    if USER_DATA_DIR.exists():
        shutil.rmtree(USER_DATA_DIR, ignore_errors=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Starting headless Chrome on port {PORT}...")
    proc = subprocess.Popen([
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={USER_DATA_DIR}",
        "--window-size=1440,1080",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
    ])
    
    # Wait for debug port
    for _ in range(15):
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
            if r.status_code == 200:
                log("Chrome debugger port active!")
                break
        except Exception:
            time.sleep(0.5)
    else:
        log(f"ERROR: Chrome failed to open port {PORT}")
        proc.terminate()
        return

    try:
        tabs = requests.get(f"http://127.0.0.1:{PORT}/json/list").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        log(f"Connecting to CDP: {ws_url}")

        async with websockets.connect(ws_url, max_size=35 * 1024 * 1024) as ws:
            msg_id = 0
            pending_requests = {}

            async def call(method, params=None):
                nonlocal msg_id
                msg_id += 1
                curr_id = msg_id
                payload = {"id": curr_id, "method": method}
                if params:
                    payload["params"] = params
                fut = asyncio.get_running_loop().create_future()
                pending_requests[curr_id] = fut
                await ws.send(json.dumps(payload))
                return await fut

            failed_assets = []
            console_logs = []
            hydration_issues = []

            async def reader():
                try:
                    while True:
                        raw = await ws.recv()
                        data = json.loads(raw)
                        if "id" in data and data["id"] in pending_requests:
                            fut = pending_requests.pop(data["id"])
                            if "error" in data:
                                fut.set_exception(Exception(data["error"]))
                            else:
                                fut.set_result(data.get("result", {}))
                        elif "method" in data:
                            m = data["method"]
                            p = data.get("params", {})
                            if m == "Network.responseReceived":
                                r = p.get("response", {})
                                url = r.get("url", "")
                                st = r.get("status", 200)
                                if st >= 400:
                                    log(f"HTTP ERROR: {st} for {url}")
                                    failed_assets.append({"url": url, "status": st})
                            elif m == "Network.loadingFailed":
                                url = p.get("requestId", "")
                                err = p.get("errorText", "")
                                failed_assets.append({"request": url, "error": err})
                            elif m == "Runtime.consoleAPICalled":
                                c_type = p.get("type", "")
                                args = p.get("args", [])
                                txt = " ".join(str(a.get("value", "")) for a in args)
                                console_logs.append(f"[{c_type.upper()}] {txt}")
                                if "hydration" in txt.lower():
                                    hydration_issues.append(txt)
                except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                    pass
                except Exception as e:
                    log(f"Reader exception: {e}")

            reader_task = asyncio.create_task(reader())

            log("Enabling domains...")
            await call("Page.enable")
            await call("DOM.enable")
            await call("Network.enable")
            await call("Runtime.enable")

            log("Navigating to http://localhost:3000/explore ...")
            await call("Page.navigate", {"url": "http://localhost:3000/explore"})

            # Wait for DOM ready and elements present
            log("Waiting for sections #explore and #engine to mount...")
            for attempt in range(40):
                await asyncio.sleep(0.5)
                chk = await call("Runtime.evaluate", {
                    "expression": "Boolean(document.querySelector('#explore') && document.querySelector('#engine'))",
                    "returnByValue": True
                })
                if chk.get("value") is True:
                    log(f"Sections found after {(attempt + 1) * 0.5}s!")
                    break
            else:
                log("WARNING: Timeout waiting for #explore and #engine")

            # Let fonts/images settle
            await asyncio.sleep(3)

            # Query DOM details for both sections
            dom_eval = await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const secExplore = document.querySelector('#explore');
                    const secEngine = document.querySelector('#engine');

                    const extractSec = (sec) => {
                        if (!sec) return null;
                        const h2 = sec.querySelector('h2')?.innerText?.replace(/\\s+/g, ' ').trim();
                        const imgs = Array.from(sec.querySelectorAll('img')).map(i => ({
                            src: i.src,
                            currentSrc: i.currentSrc,
                            alt: i.alt,
                            className: i.className,
                            dataStage: i.getAttribute('data-stage'),
                            naturalWidth: i.naturalWidth,
                            naturalHeight: i.naturalHeight,
                            opacity: window.getComputedStyle(i).opacity,
                            visibility: window.getComputedStyle(i).visibility,
                            display: window.getComputedStyle(i).display
                        }));
                        return { h2, imgs };
                    };

                    return {
                        explore: extractSec(secExplore),
                        engine: extractSec(secEngine)
                    };
                })()
                """,
                "returnByValue": True
            })
            dom_data = dom_eval.get("value", {})
            log(f"DOM Explore H2: {dom_data.get('explore', {}).get('h2')}")
            log(f"DOM Engine H2: {dom_data.get('engine', {}).get('h2')}")

            # Capture Section: THE RESULT IS NOT A PICTURE. (#explore)
            log("Scrolling to #explore section...")
            await call("Runtime.evaluate", {
                "expression": "document.querySelector('#explore').scrollIntoView({ behavior: 'instant', block: 'start' });"
            })
            await asyncio.sleep(2)

            log("Capturing screenshot of Explore Section...")
            snap_explore = await call("Page.captureScreenshot", {"format": "png"})
            p_explore = ARTIFACTS_DIR / "section_explore_the_result_is_not_a_picture.png"
            with open(p_explore, "wb") as f:
                f.write(base64.b64decode(snap_explore.get("data", "")))
            log(f"Saved: {p_explore}")

            # Capture Section: THE IMAGE IS ONLY THE BEGINNING. (#engine)
            log("Scrolling to #engine section...")
            await call("Runtime.evaluate", {
                "expression": "document.querySelector('#engine').scrollIntoView({ behavior: 'instant', block: 'start' });"
            })
            await asyncio.sleep(2)

            log("Capturing screenshot of Engine Section...")
            snap_engine = await call("Page.captureScreenshot", {"format": "png"})
            p_engine = ARTIFACTS_DIR / "section_engine_the_image_is_only_the_beginning.png"
            with open(p_engine, "wb") as f:
                f.write(base64.b64decode(snap_engine.get("data", "")))
            log(f"Saved: {p_engine}")

            reader_task.cancel()
            await asyncio.sleep(0.5)

            # Mathematical comparison of captured screenshots
            im_explore = Image.open(p_explore)
            im_engine = Image.open(p_engine)
            arr_explore = np.array(im_explore)
            arr_engine = np.array(im_engine)
            pixel_diff = np.abs(arr_explore.astype(float) - arr_engine.astype(float))
            mean_diff = float(pixel_diff.mean())
            max_diff = float(pixel_diff.max())
            log(f"Screenshot Pixel Difference: mean={mean_diff:.2f}, max={max_diff:.2f}")

            # Save detailed json report
            summary = {
                "dom_data": dom_data,
                "screenshot_comparison": {
                    "mean_pixel_diff": mean_diff,
                    "max_pixel_diff": max_diff,
                    "are_different": mean_diff > 1.0,
                },
                "failed_assets": failed_assets,
                "console_logs": console_logs,
                "hydration_issues": hydration_issues,
                "explore_screenshot": str(p_explore),
                "engine_screenshot": str(p_engine),
            }
            with open("scratch/evidence_summary.json", "w") as fp:
                json.dump(summary, fp, indent=2)

            log("Verification complete!")
            log(f"Explore images: {dom_data.get('explore', {}).get('imgs')}")
            log(f"Engine images: {dom_data.get('engine', {}).get('imgs')}")
            log(f"Failed assets count: {len(failed_assets)}")
            log(f"Hydration warnings count: {len(hydration_issues)}")

    finally:
        proc.terminate()
        log("Terminated Chrome.")

if __name__ == "__main__":
    asyncio.run(main())
