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
        async with websockets.connect(ws_url, max_size=100 * 1024 * 1024) as ws:
            msg_id = 0
            async def call(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    resp = json.loads(await ws.recv())
                    if resp.get("id") == msg_id:
                        return resp.get("result", {})

            await call("Page.enable")
            await call("DOM.enable")
            await call("Runtime.enable")

            print("Navigating to http://localhost:3000/explore ...")
            await call("Page.navigate", {"url": "http://localhost:3000/explore"})
            await asyncio.sleep(4)

            # 1. Capture Explore Section centered on stage
            print("Centering explore stage...")
            await call("Runtime.evaluate", {
                "expression": "document.querySelector('#explore .explore-frame').scrollIntoView({ behavior: 'instant', block: 'center' });"
            })
            await asyncio.sleep(2)
            snap1 = await call("Page.captureScreenshot", {"format": "png"})
            p1 = ARTIFACTS_DIR / "visible_explore_section_img2_bedroom.png"
            with open(p1, "wb") as f:
                f.write(base64.b64decode(snap1["data"]))
            print(f"Saved: {p1}")

            # 2. Capture Engine Section centered on plate
            print("Centering engine plate...")
            await call("Runtime.evaluate", {
                "expression": "document.querySelector('#engine .engine-plate').scrollIntoView({ behavior: 'instant', block: 'center' });"
            })
            await asyncio.sleep(2)
            snap2 = await call("Page.captureScreenshot", {"format": "png"})
            p2 = ARTIFACTS_DIR / "visible_engine_section_img_bedroom.png"
            with open(p2, "wb") as f:
                f.write(base64.b64decode(snap2["data"]))
            print(f"Saved: {p2}")

    finally:
        proc.terminate()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
