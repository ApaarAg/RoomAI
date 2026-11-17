import asyncio, websockets, requests, json, subprocess, time

proc = subprocess.Popen([
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    '--headless=new',
    '--remote-debugging-port=9222',
    '--user-data-dir=F:/RoomAI/.chrome-debug',
    '--window-size=1440,900',
    '--disable-gpu'
])
time.sleep(2)
try:
    tabs = requests.get('http://127.0.0.1:9222/json/list').json()
    ws_url = tabs[0]['webSocketDebuggerUrl']
    async def get_sections():
        ws = await websockets.connect(ws_url)
        await ws.send(json.dumps({'id': 1, 'method': 'Page.navigate', 'params': {'url': 'http://localhost:3000'}}))
        await asyncio.sleep(4)
        expr = 'Array.from(document.querySelectorAll("section")).map(s => ({id: s.id, cls: s.className}))'
        await ws.send(json.dumps({'id': 2, 'method': 'Runtime.evaluate', 'params': {
            'expression': expr,
            'returnByValue': True
        }}))
        while True:
            r = json.loads(await ws.recv())
            if r.get('id') == 2:
                print('Sections found:', r.get('result', {}).get('result', {}).get('value'))
                break
        await ws.close()
    asyncio.run(get_sections())
finally:
    proc.terminate()
