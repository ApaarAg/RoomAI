import asyncio
import websockets
import requests
import json
import base64
import subprocess
import time
import os

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = r"F:\RoomAI\artifacts\transition_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.msg_id = 0

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=100*1024*1024)

    async def send(self, method, params=None):
        self.msg_id += 1
        req = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(req))
        while True:
            resp_raw = await self.ws.recv()
            resp = json.loads(resp_raw)
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

async def run_audit():
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--user-data-dir=F:/RoomAI/.chrome-debug",
        "--window-size=1440,900",
        "--disable-gpu"
    ])
    time.sleep(2)

    try:
        tabs = requests.get("http://127.0.0.1:9222/json/list").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        client = CDPClient(ws_url)
        await client.connect()

        print("Navigating to http://localhost:3000...")
        await client.send("Page.navigate", {"url": "http://localhost:3000"})
        await asyncio.sleep(4)

        # Get pipeline element bounding rect
        bounds = await client.evaluate("""
            (() => {
                const el = document.querySelector(".engine-journey") || document.getElementById("engine");
                if (!el) return null;
                const rect = el.getBoundingClientRect();
                return {
                    top: rect.top + window.scrollY,
                    height: el.offsetHeight,
                    width: el.offsetWidth
                };
            })()
        """)
        print("Engine journey bounds:", bounds)
        if not bounds:
            print("Engine journey element not found!")
            return

        journey_top = bounds["top"]
        journey_height = bounds["height"]
        # In PipelineExperience:
        # start: "top top+=84", end: "bottom bottom"
        scroll_start = journey_top - 84
        scroll_end = journey_top + journey_height - 900
        scroll_dist = max(1, scroll_end - scroll_start)
        print(f"Scroll start: {scroll_start}, end: {scroll_end}, dist: {scroll_dist}")

        steps = [
            ("01_start", 0.0),
            ("01_02_mid", 0.10),
            ("02_stage", 0.20),
            ("02_03_mid", 0.30),
            ("03_stage", 0.40),
            ("03_04_mid", 0.50),
            ("04_stage", 0.60),
            ("04_05_mid", 0.70),
            ("05_stage", 0.80),
            ("05_06_mid", 0.90),
            ("06_stage", 1.00),
        ]

        for name, prog in steps:
            target_y = scroll_start + prog * scroll_dist
            await client.evaluate(f"window.scrollTo(0, {target_y});")
            # Wait for GSAP scrub + rendering
            await asyncio.sleep(0.4)

            # Get clip bounding box for #pipeline-visual
            clip = await client.evaluate("""
                (() => {
                    const el = document.querySelector(".pipeline-visual");
                    if (!el) return null;
                    const r = el.getBoundingClientRect();
                    return {
                        x: Math.max(0, r.left),
                        y: Math.max(0, r.top),
                        width: r.width,
                        height: r.height,
                        scale: 1
                    };
                })()
            """)

            full_path = os.path.join(OUTPUT_DIR, f"{name}_stage.png")
            if clip and clip['width'] > 10 and clip['height'] > 10:
                await client.capture_screenshot(full_path, clip=clip)
            else:
                await client.capture_screenshot(full_path)
            print(f"[{name}] (prog={prog:.2f}) -> {full_path}")

        await client.close()
    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(run_audit())
