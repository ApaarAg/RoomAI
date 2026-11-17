import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
import requests
import websockets

PORT = 9333
TEST_IMG = Path(r"F:\RoomAI\web\public\assets\roomai\result-viewer\01-rgb.png")

async def test():
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
    time.sleep(2)

    try:
        tabs = requests.get(f"http://127.0.0.1:{PORT}/json/list").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url) as ws:
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

            await call("Page.navigate", {"url": "http://localhost:3000/explore#detect"})
            await asyncio.sleep(3)

            # Get document root
            doc = await call("DOM.getDocument")
            root_id = doc["root"]["nodeId"]

            # Query input[type="file"]
            inp = await call("DOM.querySelector", {"nodeId": root_id, "selector": "input[type='file']"})
            inp_id = inp["nodeId"]
            print(f"File input nodeId: {inp_id}")

            # Set file using CDP native method
            res = await call("DOM.setFileInputFiles", {
                "files": [str(TEST_IMG.resolve())],
                "nodeId": inp_id
            })
            print("setFileInputFiles result:", res)
            await asyncio.sleep(1)

            # Check if file was received in React state
            check = await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const inp = document.querySelector("input[type='file']");
                    return {
                        hasFiles: inp.files.length > 0,
                        fileName: inp.files[0] ? inp.files[0].name : null,
                        fileSize: inp.files[0] ? inp.files[0].size : null
                    };
                })()
                """,
                "returnByValue": True
            })
            print("Input files after CDP setFileInputFiles:", json.dumps(check, indent=2))

            # Click RUN MODEL A3
            print("Clicking RUN MODEL A3...")
            await call("Runtime.evaluate", {
                "expression": "document.querySelector('.action-pill.primary')?.click();"
            })

            for i in range(20):
                await asyncio.sleep(0.5)
                poll = await call("Runtime.evaluate", {
                    "expression": """
                    (() => {
                        const boxes = document.querySelectorAll('.detection-box-group');
                        const status = document.querySelector('.telemetry-grid') !== null;
                        return { boxCount: boxes.length, isSuccess: status };
                    })()
                    """,
                    "returnByValue": True
                })
                res_val = poll.get("result", {}).get("value", {})
                print(f"Poll {i}: {res_val}")
                if res_val.get("boxCount", 0) > 0:
                    print("SUCCESS! Real file uploaded via CDP setFileInputFiles and detected!")
                    break

    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(test())
