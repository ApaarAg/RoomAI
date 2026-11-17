const { spawn } = require("child_process");
const fs = require("fs");

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const USER_DATA_DIR = "F:\\RoomAI\\.chrome-shot-spatial";
const PORT = 9555;

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function run() {
  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  const chromeProc = spawn(CHROME_PATH, [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    "--window-size=1440,960",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
  ]);

  await sleep(2500);

  try {
    const listRes = await fetch(`http://127.0.0.1:${PORT}/json/list`);
    const tabs = await listRes.json();
    const wsUrl = tabs[0].webSocketDebuggerUrl;

    const ws = new WebSocket(wsUrl);
    let reqId = 0;
    const pending = new Map();

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && pending.has(msg.id)) {
        const resolve = pending.get(msg.id);
        pending.delete(msg.id);
        resolve(msg);
      }
    };

    await new Promise((resolve) => {
      ws.onopen = resolve;
    });

    function sendCmd(method, params = {}) {
      return new Promise((resolve) => {
        const id = ++reqId;
        pending.set(id, resolve);
        ws.send(JSON.stringify({ id, method, params }));
      });
    }

    async function evalJs(expr) {
      const res = await sendCmd("Runtime.evaluate", {
        expression: expr,
        returnByValue: true,
        awaitPromise: true,
      });
      return res.result?.result?.value;
    }

    await sendCmd("Page.enable");
    await sendCmd("Runtime.enable");

    // Navigate to /explore
    await sendCmd("Page.navigate", { url: "http://localhost:3000/explore#construction" });
    await sleep(3500);

    // Scroll to #construction
    await evalJs(`document.getElementById('construction').scrollIntoView({ behavior: 'instant', block: 'start' })`);
    await sleep(1000);

    // 1. Screenshot Stage 01
    const shot1 = await sendCmd("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync("spatial_stage_01.png", Buffer.from(shot1.result.data, "base64"));
    console.log("Saved spatial_stage_01.png");

    // 2. Click Stage 06 (3D Room)
    await evalJs(`document.querySelectorAll('.stage-pill-button')[5].click()`);
    await sleep(2000);

    // Screenshot Stage 06
    const shot6 = await sendCmd("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync("spatial_stage_06.png", Buffer.from(shot6.result.data, "base64"));
    console.log("Saved spatial_stage_06.png");

    ws.close();
  } finally {
    chromeProc.kill();
  }
}

run().catch(console.error);
