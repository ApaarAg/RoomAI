import asyncio
import json
import subprocess
import time
import requests
import websockets
import os

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA_DIR = r"F:\RoomAI\.chrome-verify-spatial"
PORT = 9444

async def main():
    if not os.path.exists(USER_DATA_DIR):
        os.makedirs(USER_DATA_DIR, exist_ok=True)

    print("Launching headless Chrome on port", PORT)
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={USER_DATA_DIR}",
        "--window-size=1440,900",
        "--enable-webgl",
        "--ignore-gpu-blocklist",
    ])
    time.sleep(2.5)

    console_errors = []

    try:
        tabs = requests.get(f"http://127.0.0.1:{PORT}/json/list").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        print("Connected to CDP at:", ws_url)

        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            req_id = 0

            async def send_cmd(method, params=None):
                nonlocal req_id
                req_id += 1
                msg = {"id": req_id, "method": method, "params": params or {}}
                await ws.send(json.dumps(msg))
                return req_id

            async def wait_for_id(target_id):
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("method") == "Runtime.consoleAPICalled":
                        args = data.get("params", {}).get("args", [])
                        text = " ".join(str(a.get("value", a.get("description", ""))) for a in args)
                        t = data.get("params", {}).get("type")
                        if t in ["error", "warning"] and "THREE" not in text:
                            console_errors.append(f"[{t}] {text}")
                    if data.get("id") == target_id:
                        return data

            async def eval_js(expression):
                cid = await send_cmd("Runtime.evaluate", {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True
                })
                res = await wait_for_id(cid)
                return res.get("result", {}).get("result", {}).get("value")

            # Enable events
            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")

            # 1. Navigate to /explore
            print("\n--- 1. Testing /explore navigation ---")
            cid = await send_cmd("Page.navigate", {"url": "http://localhost:3000/explore"})
            await wait_for_id(cid)
            await asyncio.sleep(3)

            # Check page title and section 03 existence
            title = await eval_js("document.title")
            print("Page Title:", title)

            sec_check = await eval_js("""
                (() => {
                    const sec = document.getElementById('construction');
                    if (!sec) return null;
                    const h2 = sec.querySelector('h2')?.innerText;
                    const label = sec.querySelector('.section-label')?.innerText;
                    const stageCount = sec.querySelectorAll('.stage-pill-button').length;
                    const canvas = !!sec.querySelector('.room-3d-webgl-canvas');
                    return { found: true, h2, label, stageCount, canvas };
                })()
            """)
            print("Section 03 Check:", json.dumps(sec_check, indent=2))
            assert sec_check and sec_check["found"], "Section 03 #construction not found!"
            assert sec_check["stageCount"] == 6, f"Expected 6 stages, got {sec_check['stageCount']}"

            # 2. Test Jumping through Stages 01 to 06 via stage pills
            print("\n--- 2. Testing Stage Transitions (Forward) ---")
            for stage_idx in range(6):
                await eval_js(f"""
                    (() => {{
                        const pills = document.querySelectorAll('.stage-pill-button');
                        if (pills[{stage_idx}]) pills[{stage_idx}].click();
                    }})()
                """)
                await asyncio.sleep(0.8)
                stage_info = await eval_js("""
                    (() => {
                        const headline = document.querySelector('.banner-headline')?.innerText;
                        const kicker = document.querySelector('.banner-kicker')?.innerText;
                        const activePill = document.querySelector('.stage-pill-button.is-active .pill-title')?.innerText;
                        const hudCategory = document.querySelector('.hud-category')?.innerText;
                        return { headline, kicker, activePill, hudCategory };
                    })()
                """)
                print(f"Stage {stage_idx + 1} State:", json.dumps(stage_info))

            # 3. Test 3D Canvas in Stage 06
            print("\n--- 3. Testing 3D Interactive Room in Stage 06 ---")
            room_3d_state = await eval_js("""
                (() => {
                    const plate = document.querySelector('.room-construction-stage-plate');
                    const canvas = document.querySelector('.room-3d-webgl-canvas');
                    const toolbar = !!document.querySelector('.topbar-3d-toolbar');
                    const hint = document.querySelector('.hint-pill')?.innerText;
                    const rect = canvas ? canvas.getBoundingClientRect() : null;
                    return {
                        isFinalRoom: plate?.classList.contains('is-final-room'),
                        canvasWidth: rect?.width,
                        canvasHeight: rect?.height,
                        toolbarPresent: toolbar,
                        hint
                    };
                })()
            """)
            print("3D Room State:", json.dumps(room_3d_state, indent=2))
            assert room_3d_state["isFinalRoom"], "Plate should have .is-final-room class in Stage 06!"
            assert room_3d_state["toolbarPresent"], "Toolbar presets should be visible in Stage 06!"

            # Test camera preset buttons
            print("\n--- 4. Testing Camera Presets ---")
            presets = ["PLAN (TOP)", "ISO 45°", "PERSPECTIVE", "RESET ⟲"]
            for p in presets:
                clicked = await eval_js(f"""
                    (() => {{
                        const btns = Array.from(document.querySelectorAll('.toolbar-btn'));
                        const btn = btns.find(b => b.innerText.includes('{p}'));
                        if (btn) {{ btn.click(); return true; }}
                        return false;
                    }})()
                """)
                await asyncio.sleep(0.5)
                print(f"Clicked preset '{p}': {clicked}")

            # 5. Test Reverse Scroll / Jump (Stage 6 -> Stage 1)
            print("\n--- 5. Testing Stage Transitions (Reverse: 6 -> 1) ---")
            for stage_idx in [4, 3, 2, 1, 0]:
                await eval_js(f"""
                    (() => {{
                        const pills = document.querySelectorAll('.stage-pill-button');
                        if (pills[{stage_idx}]) pills[{stage_idx}].click();
                    }})()
                """)
                await asyncio.sleep(0.6)
                reverse_info = await eval_js("""
                    (() => {
                        const headline = document.querySelector('.banner-headline')?.innerText;
                        const activePill = document.querySelector('.stage-pill-button.is-active .pill-title')?.innerText;
                        return { headline, activePill };
                    })()
                """)
                print(f"Reversed to Stage {stage_idx + 1}:", reverse_info)

            # 6. Verify Navigation bar links
            print("\n--- 6. Testing Navigation Links ---")
            nav_links = await eval_js("""
                (() => {
                    const links = Array.from(document.querySelectorAll('header nav a')).map(a => ({
                        href: a.getAttribute('href'),
                        text: a.innerText.trim()
                    }));
                    return links;
                })()
            """)
            print("Navigation Links:", json.dumps(nav_links, indent=2))
            has_3d_link = any("#construction" in l.get("href", "") or "3D ROOM" in l.get("text", "") for l in nav_links)
            print("Has 3D ROOM Navigation link:", has_3d_link)
            assert has_3d_link, "Navigation bar must include 03 3D ROOM!"

            # 7. Check Homepage (/)
            print("\n--- 7. Testing Homepage (/) ---")
            cid = await send_cmd("Page.navigate", {"url": "http://localhost:3000"})
            await wait_for_id(cid)
            await asyncio.sleep(2.5)

            home_3d_link = await eval_js("""
                (() => {
                    const links = Array.from(document.querySelectorAll('a')).map(a => a.getAttribute('href'));
                    return links.filter(h => h && h.includes('construction'));
                })()
            """)
            print("Homepage links to 3D construction:", home_3d_link)

            # 8. Check Model Page (/model)
            print("\n--- 8. Testing /model Page ---")
            cid = await send_cmd("Page.navigate", {"url": "http://localhost:3000/model"})
            await wait_for_id(cid)
            await asyncio.sleep(2.5)

            model_check = await eval_js("""
                (() => {
                    return {
                        title: document.title,
                        hasPresets: document.querySelectorAll('.preset-btn').length,
                        hasRunBtn: !!document.querySelector('.run-inference-btn')
                    };
                })()
            """)
            print("Model Page State:", json.dumps(model_check, indent=2))

            print("\n--- Console Errors Check ---")
            print("Errors recorded:", console_errors)
            assert len(console_errors) == 0, f"Found console errors: {console_errors}"

            print("\nALL VERIFICATION CHECKS PASSED PERFECTLY!")

    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(main())
