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

async def run_cdp_verification():
    chrome_cmd = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        "--remote-debugging-port=9222",
        "--user-data-dir=F:/RoomAI/.chrome-debug-verify",
        "--window-size=1440,1080",
        "--disable-gpu",
    ]
    proc = subprocess.Popen(chrome_cmd)
    time.sleep(2.5)

    try:
        tabs_resp = requests.get("http://127.0.0.1:9222/json/list").json()
        target_tab = tabs_resp[0]
        ws_url = target_tab["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=25 * 1024 * 1024) as ws:
            msg_id = 0
            async def send_cmd(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method}
                if params:
                    payload["params"] = params
                await ws.send(json.dumps(payload))
                while True:
                    res = json.loads(await ws.recv())
                    if res.get("id") == msg_id:
                        return res.get("result", {})

            # Enable logging, console, network, page
            await send_cmd("Page.enable")
            await send_cmd("DOM.enable")
            await send_cmd("Network.enable")
            await send_cmd("Console.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("Log.enable")

            failed_requests = []
            console_errors = []
            hydration_warnings = []
            image_404s = []

            # Background listener for events
            async def event_listener():
                try:
                    while True:
                        raw = await ws.recv()
                        evt = json.loads(raw)
                        method = evt.get("method", "")
                        params = evt.get("params", {})

                        if method == "Network.responseReceived":
                            resp = params.get("response", {})
                            status = resp.get("status", 200)
                            url = resp.get("url", "")
                            mime = resp.get("mimeType", "")
                            if status >= 400:
                                failed_requests.append({"url": url, "status": status})
                                if "image" in mime or any(ext in url for ext in [".webp", ".png", ".jpg", ".jpeg"]):
                                    image_404s.append({"url": url, "status": status})
                        elif method == "Network.loadingFailed":
                            failed_requests.append({
                                "url": params.get("requestId", ""),
                                "error": params.get("errorText", "")
                            })
                        elif method == "Runtime.consoleAPICalled":
                            c_type = params.get("type", "")
                            args = params.get("args", [])
                            text = " ".join(str(a.get("value", "")) for a in args)
                            if c_type in ["error", "warning"]:
                                console_errors.append(f"[{c_type.upper()}] {text}")
                                if "hydration" in text.lower() or "hydrat" in text.lower():
                                    hydration_warnings.append(text)
                        elif method == "Log.entryAdded":
                            entry = params.get("entry", {})
                            level = entry.get("level", "")
                            text = entry.get("text", "")
                            if level in ["error", "warning"]:
                                console_errors.append(f"[LOG_{level.upper()}] {text}")
                                if "hydration" in text.lower():
                                    hydration_warnings.append(text)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            listener_task = asyncio.create_task(event_listener())

            # Navigate to /explore
            print("Navigating to http://localhost:3000/explore ...")
            await send_cmd("Page.navigate", {"url": "http://localhost:3000/explore"})
            await asyncio.sleep(4)

            # Inspect DOM for both sections
            dom_eval_script = """
            (() => {
                const getSectionData = (selector) => {
                    const sec = document.querySelector(selector);
                    if (!sec) return null;
                    const h2 = sec.querySelector('h2')?.innerText?.replace(/\\s+/g, ' ').trim();
                    const imgs = Array.from(sec.querySelectorAll('img')).map(img => ({
                        src: img.src,
                        currentSrc: img.currentSrc,
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight,
                        className: img.className,
                        displayed: img.offsetWidth > 0 && img.offsetHeight > 0,
                        opacity: window.getComputedStyle(img).opacity,
                        visibility: window.getComputedStyle(img).visibility
                    }));
                    return { h2, imgs };
                };
                return {
                    explore: getSectionData('#explore'),
                    engine: getSectionData('#engine')
                };
            })()
            """
            dom_data = await send_cmd("Runtime.evaluate", {
                "expression": dom_eval_script,
                "returnByValue": True
            })
            sec_info = dom_data.get("result", {}).get("value", {})
            print("DOM Inspection:")
            print(json.dumps(sec_info, indent=2))

            # Scroll to #explore ("THE RESULT IS NOT A PICTURE.")
            print("Scrolling to #explore section...")
            await send_cmd("Runtime.evaluate", {
                "expression": "document.querySelector('#explore').scrollIntoView({ behavior: 'instant', block: 'start' });"
            })
            await asyncio.sleep(2)

            # Capture screenshot of Explore Section
            snap_explore = await send_cmd("Page.captureScreenshot", {"format": "png"})
            explore_png_path = ARTIFACTS_DIR / "section_explore_the_result_is_not_a_picture.png"
            with open(explore_png_path, "wb") as f:
                f.write(base64.b64decode(snap_explore.get("data", "")))
            print(f"Saved Explore Section screenshot: {explore_png_path}")

            # Scroll to #engine ("THE IMAGE IS ONLY THE BEGINNING.")
            print("Scrolling to #engine section...")
            await send_cmd("Runtime.evaluate", {
                "expression": "document.querySelector('#engine').scrollIntoView({ behavior: 'instant', block: 'start' });"
            })
            await asyncio.sleep(2)

            # Capture screenshot of Engine Section
            snap_engine = await send_cmd("Page.captureScreenshot", {"format": "png"})
            engine_png_path = ARTIFACTS_DIR / "section_engine_the_image_is_only_the_beginning.png"
            with open(engine_png_path, "wb") as f:
                f.write(base64.b64decode(snap_engine.get("data", "")))
            print(f"Saved Engine Section screenshot: {engine_png_path}")

            listener_task.cancel()
            await asyncio.sleep(0.5)

            report = {
                "dom_info": sec_info,
                "image_404s": image_404s,
                "failed_requests": failed_requests,
                "console_errors": console_errors,
                "hydration_warnings": hydration_warnings,
                "explore_screenshot": str(explore_png_path),
                "engine_screenshot": str(engine_png_path),
            }

            with open("scratch/browser_verification_report.json", "w") as fp:
                json.dump(report, fp, indent=2)
            print("\n=== Verification Report Summary ===")
            print(f"Explore H2: {sec_info.get('explore', {}).get('h2')}")
            print(f"Engine H2: {sec_info.get('engine', {}).get('h2')}")
            print(f"Image 404s: {len(image_404s)}")
            print(f"Failed requests: {len(failed_requests)}")
            print(f"Console errors: {len(console_errors)}")
            print(f"Hydration warnings: {len(hydration_warnings)}")

    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(run_cdp_verification())
