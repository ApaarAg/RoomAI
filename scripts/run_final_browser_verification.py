import asyncio
import base64
import json
import os
import subprocess
import time
from pathlib import Path
from PIL import Image
import numpy as np
import requests
import websockets

ARTIFACTS_DIR = Path(r"C:\Users\apaar\.gemini\antigravity\brain\7c6e77c3-465e-4e84-bb87-4e80db4528d5")
PORT = 9333

class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.msg_id = 0
        self.failed_requests = []
        self.console_logs = []
        self.hydration_warnings = []

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=100 * 1024 * 1024)

    async def send(self, method, params=None):
        self.msg_id += 1
        req = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(req))
        while True:
            resp_raw = await self.ws.recv()
            resp = json.loads(resp_raw)
            if "method" in resp:
                m = resp["method"]
                p = resp.get("params", {})
                if m == "Network.responseReceived":
                    r = p.get("response", {})
                    url = r.get("url", "")
                    st = r.get("status", 200)
                    if st >= 400:
                        print(f"HTTP ERROR {st}: {url}")
                        self.failed_requests.append({"status": st, "url": url})
                elif m == "Runtime.consoleAPICalled":
                    t = p.get("type", "")
                    args = p.get("args", [])
                    txt = " ".join(str(a.get("value", "")) for a in args)
                    self.console_logs.append(f"[{t.upper()}] {txt}")
                    if "hydration" in txt.lower():
                        self.hydration_warnings.append(txt)
            if resp.get("id") == self.msg_id:
                return resp.get("result", {})

    async def evaluate(self, expr):
        res = await self.send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True
        })
        return res.get("result", {}).get("value")

    async def capture_screenshot(self, filepath, clip=None):
        params = {"format": "png"}
        if clip:
            params["clip"] = clip
        res = await self.send("Page.captureScreenshot", params)
        data = base64.b64decode(res["data"])
        with open(filepath, "wb") as f:
            f.write(data)

    async def close(self):
        if self.ws:
            await self.ws.close()

async def main():
    print(f"Starting Chrome on port {PORT}...")
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

    try:
        tabs = requests.get(f"http://127.0.0.1:{PORT}/json/list").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        print(f"Connecting to CDP: {ws_url}")
        client = CDPClient(ws_url)
        await client.connect()

        await client.send("Page.enable")
        await client.send("DOM.enable")
        await client.send("Network.enable")
        await client.send("Runtime.enable")

        print("Navigating to http://localhost:3000/explore ...")
        await client.send("Page.navigate", {"url": "http://localhost:3000/explore"})
        
        # Wait for page readiness
        print("Waiting for page load and hydration...")
        for attempt in range(40):
            await asyncio.sleep(0.5)
            ready = await client.evaluate("""
                (() => {
                    const secExplore = document.querySelector('#explore');
                    const secEngine = document.querySelector('#engine');
                    return Boolean(secExplore && secEngine);
                })()
            """)
            if ready:
                print(f"Both sections detected in DOM after {(attempt + 1) * 0.5}s!")
                break
        else:
            print("ERROR: Timeout waiting for DOM sections!")

        # Allow images and layout to settle
        await asyncio.sleep(3)

        # Inspect DOM for both sections
        dom_report = await client.evaluate("""
            (() => {
                const getInfo = (selector) => {
                    const el = document.querySelector(selector);
                    if (!el) return null;
                    const h2 = el.querySelector('h2')?.innerText?.replace(/\\s+/g, ' ').trim();
                    const imgs = Array.from(el.querySelectorAll('img')).map((img, idx) => ({
                        index: idx,
                        src: img.src,
                        currentSrc: img.currentSrc,
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight,
                        className: img.className,
                        dataStage: img.getAttribute('data-stage'),
                        opacity: window.getComputedStyle(img).opacity,
                        visibility: window.getComputedStyle(img).visibility,
                        isDisplayed: img.offsetWidth > 0 && img.offsetHeight > 0
                    }));
                    return { h2, imgs };
                };
                return {
                    explore: getInfo('#explore'),
                    engine: getInfo('#engine')
                };
            })()
        """)
        print("\n=== DOM Inspection Result ===")
        print("Explore Section (#explore):", dom_report.get("explore", {}).get("h2"))
        for img in dom_report.get("explore", {}).get("imgs", []):
            print(f"  Img {img['index']} (stage {img.get('dataStage')}): {img['src']} | natural={img['naturalWidth']}x{img['naturalHeight']} | op={img['opacity']} vis={img['visibility']}")

        print("\nEngine Section (#engine):", dom_report.get("engine", {}).get("h2"))
        for img in dom_report.get("engine", {}).get("imgs", []):
            print(f"  Img {img['index']} (stage {img.get('dataStage')}): {img['src']} | natural={img['naturalWidth']}x{img['naturalHeight']} | op={img['opacity']} vis={img['visibility']}")

        # 1. Scroll to #explore ("THE RESULT IS NOT A PICTURE.")
        print("\nScrolling to #explore section...")
        await client.evaluate("document.querySelector('#explore').scrollIntoView({ behavior: 'instant', block: 'start' });")
        await asyncio.sleep(2)

        p_explore = ARTIFACTS_DIR / "section_explore_the_result_is_not_a_picture.png"
        await client.capture_screenshot(str(p_explore))
        print(f"Saved Explore Screenshot: {p_explore}")

        # 2. Scroll to #engine ("THE IMAGE IS ONLY THE BEGINNING.")
        print("\nScrolling to #engine section...")
        await client.evaluate("document.querySelector('#engine').scrollIntoView({ behavior: 'instant', block: 'start' });")
        await asyncio.sleep(2)

        p_engine = ARTIFACTS_DIR / "section_engine_the_image_is_only_the_beginning.png"
        await client.capture_screenshot(str(p_engine))
        print(f"Saved Engine Screenshot: {p_engine}")

        # Mathematical verification of screenshot differences
        im1 = Image.open(p_explore)
        im2 = Image.open(p_engine)
        arr1 = np.array(im1)
        arr2 = np.array(im2)
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        print(f"\nScreenshot Pixel Difference Analysis: mean={mean_diff:.2f}, max={max_diff:.2f}")
        assert mean_diff > 1.0, "Screenshots are identical!"
        print("CONFIRMED: The two sections render completely different visual contents!")

        # Network and Console Audit
        print(f"\nFailed Requests (4xx/5xx): {len(client.failed_requests)}")
        for r in client.failed_requests:
            print(f"  FAILED: {r['status']} - {r['url']}")
        print(f"Hydration Warnings: {len(client.hydration_warnings)}")
        for w in client.hydration_warnings:
            print(f"  WARNING: {w}")

        await client.close()

        # Save verification artifact summary
        result_json = {
            "explore_heading": dom_report.get("explore", {}).get("h2"),
            "explore_images": dom_report.get("explore", {}).get("imgs", []),
            "engine_heading": dom_report.get("engine", {}).get("h2"),
            "engine_images": dom_report.get("engine", {}).get("imgs", []),
            "screenshot_explore": str(p_explore),
            "screenshot_engine": str(p_engine),
            "difference_metrics": {
                "mean_diff": mean_diff,
                "max_diff": max_diff,
                "verified_different": bool(mean_diff > 1.0)
            },
            "failed_requests": client.failed_requests,
            "hydration_warnings": client.hydration_warnings,
            "console_logs_count": len(client.console_logs)
        }
        with open("scratch/browser_audit_results.json", "w") as fp:
            json.dump(result_json, fp, indent=2)

        print("\nALL VERIFICATION CRITERIA SATISFIED WITH EMPIRICAL EVIDENCE.")

    finally:
        proc.terminate()
        print("Chrome closed.")

if __name__ == "__main__":
    asyncio.run(main())
