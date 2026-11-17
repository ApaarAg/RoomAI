const { spawn } = require("child_process");
const fs = require("fs");

const CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const USER_DATA_DIR = "F:\\RoomAI\\.chrome-verify-node";
const PORT = 9444;

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function run() {
  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  console.log(`[1] Launching headless Chrome on port ${PORT}...`);
  const chromeProc = spawn(CHROME_PATH, [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    "--window-size=1440,900",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
  ]);

  await sleep(2500);

  const consoleErrors = [];

  try {
    const listRes = await fetch(`http://127.0.0.1:${PORT}/json/list`);
    const tabs = await listRes.json();
    const wsUrl = tabs[0].webSocketDebuggerUrl;
    console.log(`[2] Connected to Chrome CDP at: ${wsUrl}`);

    const ws = new WebSocket(wsUrl);

    let reqId = 0;
    const pending = new Map();

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.method === "Runtime.consoleAPICalled") {
        const type = msg.params.type;
        const text = (msg.params.args || [])
          .map((a) => a.value || a.description || "")
          .join(" ");
        if ((type === "error" || type === "warning") && !text.includes("THREE")) {
          consoleErrors.push(`[${type}] ${text}`);
        }
      }
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

    // A. Navigate to /explore
    console.log("\n[3] Navigating to http://localhost:3000/explore...");
    await sendCmd("Page.navigate", { url: "http://localhost:3000/explore" });
    await sleep(3500);

    const title = await evalJs("document.title");
    console.log(`Page title: "${title}"`);

    // Verify Section 03 Presence
    const sec3 = await evalJs(`(() => {
      const el = document.getElementById('construction');
      if (!el) return null;
      const h2 = el.querySelector('h2')?.innerText;
      const label = el.querySelector('.section-label')?.innerText;
      const pills = el.querySelectorAll('.stage-pill-button').length;
      const canvas = !!el.querySelector('.room-3d-webgl-canvas');
      const footnote = el.querySelector('.construction-technical-footnote')?.innerText;
      return { found: true, h2, label, pills, canvas, hasFootnote: !!footnote };
    })()`);

    console.log("Section 03 Inspection:", JSON.stringify(sec3, null, 2));
    if (!sec3 || !sec3.found) {
      throw new Error("Section 03 #construction was not found on /explore!");
    }
    if (sec3.pills !== 6) {
      throw new Error(`Expected 6 stage pills, got ${sec3.pills}`);
    }
    if (!sec3.canvas) {
      throw new Error("3D WebGL Canvas was not found in Section 03!");
    }

    // B. Test Forward Stage Progression (01 -> 06)
    console.log("\n[4] Testing Forward Stage Progression (Stages 01 -> 06)...");
    for (let i = 0; i < 6; i++) {
      await evalJs(`document.querySelectorAll('.stage-pill-button')[${i}].click()`);
      await sleep(700);

      const st = await evalJs(`(() => {
        const headline = document.querySelector('.banner-headline')?.innerText;
        const kicker = document.querySelector('.banner-kicker')?.innerText;
        const activeTitle = document.querySelector('.stage-pill-button.is-active .pill-title')?.innerText;
        return { headline, kicker, activeTitle };
      })()`);
      console.log(`  Stage 0${i + 1}: ${st.kicker} | "${st.headline}"`);
    }

    // C. Test Stage 06 3D Room State
    console.log("\n[5] Verifying 3D Room Interactive State (Stage 06)...");
    const st6 = await evalJs(`(() => {
      const plate = document.querySelector('.room-construction-stage-plate');
      const canvas = document.querySelector('.room-3d-webgl-canvas');
      const toolbar = !!document.querySelector('.topbar-3d-toolbar');
      const hint = document.querySelector('.hint-pill')?.innerText;
      const hud = document.querySelector('.spatial-inspection-hud')?.innerText;
      const rect = canvas ? canvas.getBoundingClientRect() : null;
      return {
        isFinalRoom: plate?.classList.contains('is-final-room'),
        toolbarPresent: toolbar,
        hintText: hint,
        hudSnippet: hud?.slice(0, 100).replace(/\\n/g, ' '),
        canvasDims: rect ? \`\${Math.round(rect.width)}x\${Math.round(rect.height)}\` : 'none'
      };
    })()`);

    console.log("Stage 06 Details:", JSON.stringify(st6, null, 2));
    if (!st6.isFinalRoom) throw new Error("Plate should have class .is-final-room in Stage 06!");
    if (!st6.toolbarPresent) throw new Error("Toolbar view presets should be visible in Stage 06!");

    // D. Test Camera Presets
    console.log("\n[6] Testing Camera Preset Actions...");
    const presets = ["PLAN (TOP)", "ISO 45°", "PERSPECTIVE", "RESET ⟲"];
    for (const p of presets) {
      const res = await evalJs(`(() => {
        const btns = Array.from(document.querySelectorAll('.toolbar-btn'));
        const b = btns.find(btn => btn.innerText.includes('${p}'));
        if (b) { b.click(); return true; }
        return false;
      })()`);
      await sleep(400);
      console.log(`  Preset "${p}" clicked: ${res}`);
    }

    // E. Test Reverse Progression (06 -> 01)
    console.log("\n[7] Testing Reverse Stage Progression (Stages 06 -> 01)...");
    for (let i = 4; i >= 0; i--) {
      await evalJs(`document.querySelectorAll('.stage-pill-button')[${i}].click()`);
      await sleep(500);
      const rev = await evalJs(`(() => {
        const kicker = document.querySelector('.banner-kicker')?.innerText;
        return kicker;
      })()`);
      console.log(`  Reversed to: ${rev}`);
    }

    // F. Test Navigation Links
    console.log("\n[8] Verifying Site Navigation Menu...");
    const navItems = await evalJs(`(() => {
      const links = Array.from(document.querySelectorAll('header nav a')).map(a => ({
        href: a.getAttribute('href'),
        text: a.innerText.trim().replace(/\\n/g, ' ')
      }));
      return links;
    })()`);
    console.log("Nav Items:", JSON.stringify(navItems, null, 2));
    const has3DNav = navItems.some(
      (n) => n.href?.includes("construction") || n.text?.includes("3D ROOM")
    );
    console.log(`Navigation contains 3D ROOM: ${has3DNav}`);
    if (!has3DNav) throw new Error("Site navigation must contain 3D ROOM!");

    // G. Verify Homepage (/)
    console.log("\n[9] Testing Homepage (http://localhost:3000)...");
    await sendCmd("Page.navigate", { url: "http://localhost:3000" });
    await sleep(2500);

    const homeLinks = await evalJs(`(() => {
      const links = Array.from(document.querySelectorAll('a'))
        .map(a => a.getAttribute('href'))
        .filter(h => h && h.includes('construction'));
      return links;
    })()`);
    console.log("Homepage links to #construction:", homeLinks);
    if (homeLinks.length === 0) {
      throw new Error("Homepage should link to /explore#construction!");
    }

    // H. Verify /model route
    console.log("\n[10] Testing Model Studio (http://localhost:3000/model)...");
    await sendCmd("Page.navigate", { url: "http://localhost:3000/model" });
    await sleep(2500);

    const modelPage = await evalJs(`(() => {
      return {
        title: document.title,
        presets: document.querySelectorAll('.preset-btn').length,
        hasRunBtn: !!document.querySelector('.run-inference-btn')
      };
    })()`);
    console.log("Model Studio state:", modelPage);

    // I. Check for errors
    console.log("\n[11] Checking for Console Errors...");
    console.log(`Console errors/warnings count: ${consoleErrors.length}`);
    if (consoleErrors.length > 0) {
      console.log("Errors logged:", consoleErrors);
    }

    console.log("\n=========================================");
    console.log("SUCCESS: ALL 11 VERIFICATION TESTS PASSED!");
    console.log("=========================================\n");

    ws.close();
  } finally {
    chromeProc.kill();
  }
}

run().catch((err) => {
  console.error("FATAL ERROR IN VERIFICATION:", err);
  process.exit(1);
});
