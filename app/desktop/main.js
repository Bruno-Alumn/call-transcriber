// Call Transcriber desktop app (macOS + Windows, one codebase).
//
//  - Owns the local transcription server: starts it on launch, stops it on
//    quit (the server also watches our pid via --parent-pid).
//  - Dashboard window: the server's live dashboard.
//  - Discord window: discord.com in a built-in Chromium. page-hook.js is
//    injected before Discord's scripts run and each person's audio is relayed
//    to the server here, so no Chrome or extension is needed.
const { app, BrowserWindow, Menu, Tray, dialog, ipcMain, nativeImage, powerSaveBlocker,
  session, shell, desktopCapturer, systemPreferences } = require('electron');
const { spawn, execFile } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const ROOT = path.resolve(__dirname, '..', '..');
const SERVER_DIR = path.join(ROOT, 'server');
const IS_WIN = process.platform === 'win32';
const IS_MAC = process.platform === 'darwin';
const TEST = process.env.CT_TEST === '1';                        // test harness only
const PORT = Number(process.env.CT_PORT || 8765);
const BASE = `http://127.0.0.1:${PORT}/`;
const DISCORD_URL = (TEST && process.env.CT_DISCORD_URL) || 'https://discord.com/app';
const LOG = path.join(ROOT, TEST ? `server-test-${PORT}.log` : 'server.log');
const ICON_PNG = path.join(ROOT, 'app', 'windows', 'icon.png');
const ICON_ICO = path.join(ROOT, 'app', 'windows', 'icon.ico');
const BG = { dark: '#111110', light: '#f4f4f2' };

app.setName('Call Transcriber');
if (TEST && process.env.CT_USER_DATA) app.setPath('userData', process.env.CT_USER_DATA);
if (IS_WIN) app.setAppUserModelId('local.calltranscriber');
if (!app.requestSingleInstanceLock()) { app.quit(); process.exit(0); }

let dash = null, discord = null, tray = null;
let server = null, stoppedByUser = false, quitting = false, paused = false;
let ws = null, wsOpen = false;

// ----------------------------------------------------------------- helpers
const isDiscordUrl = (u) => {
  try {
    const x = new URL(u);
    if (TEST && process.env.CT_DISCORD_URL && x.origin === new URL(process.env.CT_DISCORD_URL).origin) return true;
    return x.protocol === 'https:' && /(^|\.)(discord\.com|discordapp\.com)$/.test(x.hostname);
  } catch { return false; }
};
const isDashUrl = (u) => typeof u === 'string' && (u.startsWith(BASE) || u.startsWith('data:text/html'));

function request(pathname, { method = 'GET', timeout = 1000, app: appHeader = false } = {}) {
  return new Promise((resolve) => {
    const req = http.request(BASE + pathname.replace(/^\//, ''), { method, timeout, headers: appHeader ? { 'X-CT': '1' } : {} }, (res) => {
      let body = ''; res.on('data', (d) => (body += d)); res.on('end', () => resolve({ status: res.statusCode, body }));
    });
    req.on('error', () => resolve(null)); req.on('timeout', () => { req.destroy(); resolve(null); });
    req.end();
  });
}
const serverUp = async () => { const r = await request('api/state'); return !!r && r.status === 200 && r.body.startsWith('{'); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };

function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { windowsHide: true, stdio: 'ignore', ...opts });
    p.on('error', () => resolve(-1)); p.on('exit', (code) => resolve(code ?? -1));
  });
}

function killTree(pid) {
  if (!pid || !alive(pid)) return Promise.resolve();
  if (IS_WIN) return run('taskkill', ['/PID', String(pid), '/T', '/F']);   // venv python.exe is a stub + child
  return (async () => {
    try { process.kill(pid, 'SIGINT'); } catch {}                    // lets the server flush pending lines
    for (let i = 0; i < 40 && alive(pid); i++) await sleep(100);
    if (alive(pid)) try { process.kill(pid, 'SIGKILL'); } catch {}
  })();
}

function listeningPids() {
  return new Promise((resolve) => {
    if (IS_WIN) {
      execFile('netstat', ['-ano', '-p', 'tcp'], { windowsHide: true }, (_e, out = '') => resolve([...new Set(out.split('\n')
        .map((l) => l.trim().split(/\s+/)).filter((c) => c[1]?.endsWith(':' + PORT) && c[3] === 'LISTENING').map((c) => Number(c[4])))]));
    } else {
      execFile('/usr/sbin/lsof', ['-nP', '-t', `-iTCP:${PORT}`, '-sTCP:LISTEN'], (_e, out = '') =>
        resolve(out.split('\n').map(Number).filter(Boolean)));
    }
  });
}

// ----------------------------------------------------------------- server
function serverPython() {
  return IS_WIN ? path.join(SERVER_DIR, '.venv', 'Scripts', 'python.exe') : path.join(SERVER_DIR, '.venv', 'bin', 'python');
}

async function stopStrays() {
  // Our own server already on the port (started from a terminal, or left over): stop it.
  if (!(await serverUp())) return;
  await request('api/shutdown', { method: 'POST', app: true });
  for (let i = 0; i < 50 && (await serverUp()); i++) await sleep(100);
  if (await serverUp()) for (const pid of await listeningPids()) if (pid !== server?.pid) await killTree(pid);
}

async function startServer() {
  stoppedByUser = false;
  if (server && server.exitCode === null) { showDashboard(); return; }
  splash('Starting the transcription server…');
  await stopStrays();
  const py = serverPython();
  if (!fs.existsSync(py) || !fs.existsSync(path.join(SERVER_DIR, 'server.py'))) {
    splash(IS_WIN ? "Couldn't find the server. Run install.cmd again." : "Couldn't find the server. See README.md (Setup).", { error: true }); return;
  }
  const log = fs.openSync(LOG, 'w');
  const extra = TEST && process.env.CT_SERVER_ARGS ? process.env.CT_SERVER_ARGS.split(' ').filter(Boolean) : [];
  const p = spawn(py, ['-u', 'server.py', '--port', String(PORT), '--parent-pid', String(process.pid), ...extra], {
    cwd: SERVER_DIR, stdio: ['ignore', log, log], windowsHide: true,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
  });
  server = p;
  p.on('exit', (code) => {
    fs.closeSync(log);
    if (server !== p) return;
    server = null;
    if (quitting) return;
    if (stoppedByUser) splash('The server is stopped. Nothing is being transcribed.', { offerStart: true });
    else splash(`The server stopped unexpectedly (exit ${code}). See the server log.`, { error: true, offerStart: true });
    updateTray();
  });
  for (let i = 0; i < 480; i++) {                         // model loading can take a while the first time
    if (server !== p || p.exitCode !== null) return;
    if (await serverUp()) { showDashboard(); updateTray(); return; }
    await sleep(500);
  }
  splash('The server is taking too long to start. See the server log.', { error: true });
}

async function stopServer({ quiet = false } = {}) {
  const p = server;
  if (!p || p.exitCode !== null) { if (!quiet) splash('The server is stopped. Nothing is being transcribed.', { offerStart: true }); return; }
  stoppedByUser = true;
  if (!quiet) splash('Stopping the server…');
  await request('api/shutdown', { method: 'POST', app: true, timeout: 1500 });   // clean stop: flushes pending lines
  for (let i = 0; i < 50 && p.exitCode === null; i++) await sleep(100);
  if (p.exitCode === null) await killTree(p.pid);
}

async function restartServer() {
  if (server && server.exitCode === null) { splash('Restarting the server…'); await stopServer({ quiet: true }); server = null; }
  return startServer();
}

// ----------------------------------------------------------------- audio relay
function connectServer() {
  if (quitting) return;
  let sock;
  try { sock = new WebSocket(`ws://127.0.0.1:${PORT}`); } catch { setTimeout(connectServer, 2000); return; }
  ws = sock;
  sock.onopen = () => { wsOpen = true; broadcast(); };
  sock.onclose = () => { if (ws === sock) { wsOpen = false; ws = null; broadcast(); } setTimeout(connectServer, 2000); };
  sock.onerror = () => {};
  sock.onmessage = () => {};          // ('reload' is for the Chrome extension; reloading here would drop the call)
}
const send = (obj) => { if (wsOpen && ws?.readyState === 1) try { ws.send(JSON.stringify(obj)); } catch {} };

function broadcast() {
  const on = wsOpen && !paused;
  if (discord && !discord.isDestroyed()) discord.webContents.send('ct:status', on);
  if (dash && !dash.isDestroyed()) dash.webContents.send('ct:app-state', appState());
  updateTray();
}
const appState = () => ({ discordOpen: !!discord && !discord.isDestroyed() && discord.isVisible(), paused, connected: wsOpen });

// Names learned in earlier calls (Discord-in-app's own memory, like the extension's storage).
const namesFile = () => path.join(app.getPath('userData'), 'names.json');
const loadNames = () => { try { return JSON.parse(fs.readFileSync(namesFile(), 'utf8')); } catch { return {}; } };

const fromDiscord = (e) => discord && !discord.isDestroyed() && e.sender === discord.webContents &&
  e.senderFrame === discord.webContents.mainFrame && isDiscordUrl(e.senderFrame.url);
const fromDash = (e) => dash && !dash.isDestroyed() && e.sender === dash.webContents &&
  e.senderFrame === dash.webContents.mainFrame && isDashUrl(e.senderFrame.url);

let hookSource = '';
ipcMain.on('ct:hook-source', (e) => {
  // Only for the Discord window's top frame (preloads only run there anyway).
  e.returnValue = discord && !discord.isDestroyed() && e.sender === discord.webContents
    ? (hookSource ||= fs.readFileSync(path.join(ROOT, 'extension', 'page-hook.js'), 'utf8')) : '';
});
ipcMain.on('ct:hello', (e) => {
  if (!fromDiscord(e)) return;
  e.sender.send('ct:status', wsOpen && !paused);
  e.sender.send('ct:names-init', loadNames());
});
ipcMain.on('ct:audio', (e, m) => {
  if (!fromDiscord(e) || paused || !m?.pcm) return;
  const pcm = Buffer.from(m.pcm instanceof ArrayBuffer ? new Uint8Array(m.pcm) : m.pcm);
  send({ type: 'track-audio', user: String(m.user), name: m.name ?? null, ch: m.ch ?? null, t: m.t, pcm: pcm.toString('base64') });
});
ipcMain.on('ct:hook-status', (e, s) => { if (fromDiscord(e)) send({ type: 'hook-status', s: { ...s, client: 'app' }, paused }); });
ipcMain.on('ct:names', (e, names) => {
  if (!fromDiscord(e) || !names || typeof names !== 'object') return;
  try { fs.writeFileSync(namesFile(), JSON.stringify({ ...loadNames(), ...names })); } catch {}
});

// ----------------------------------------------------------------- dashboard bridge
ipcMain.on('ct:server', (e, action) => {
  if (!fromDash(e)) return;
  ({ start: startServer, stop: stopServer, restart: restartServer })[action]?.();
});
ipcMain.handle('ct:save', async (e, name, content) => {
  if (!fromDash(e)) return false;
  const { canceled, filePath } = await dialog.showSaveDialog(dash, { defaultPath: path.join(app.getPath('downloads'), path.basename(name)) });
  if (canceled || !filePath) return false;
  fs.writeFileSync(filePath, content, 'utf8');
  return true;
});
ipcMain.on('ct:relaunch', (e, opts) => { if (fromDash(e)) relaunchAfterUpdate(opts || {}); });
ipcMain.on('ct:open-discord', (e) => { if (fromDash(e)) showDiscord(); });
ipcMain.on('ct:pause', (e, on) => { if (fromDash(e)) setPaused(!!on); });

function setPaused(on) { paused = on; broadcast(); buildMenu(); }

// ----------------------------------------------------------------- windows
const SPLASH = ({ text, spin, error, offerStart }) => `<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="light dark"><style>
html,body{margin:0;height:100%;overflow:hidden}
body{display:grid;place-items:center;font:15px -apple-system,"Segoe UI",system-ui,sans-serif;background:${BG.light};color:#52514e}
@media (prefers-color-scheme:dark){body{background:${BG.dark};color:#c3c2b7}}
.s{width:28px;height:28px;border-radius:50%;border:3px solid #3987e544;border-top-color:#3987e5;animation:r 1s linear infinite;margin:0 auto 16px}
@keyframes r{to{transform:rotate(360deg)}} .e{color:#e66767}
button{margin-top:16px;font:inherit;font-size:14px;padding:8px 18px;border-radius:10px;border:0;background:#3987e5;color:#fff;cursor:pointer}
</style></head><body><div style="text-align:center">${spin ? '<div class="s"></div>' : ''}<div class="${error ? 'e' : ''}">${text}</div>
${offerStart ? '<button onclick="ctApp.server(\'start\')">Start server</button>' : ''}</div></body></html>`;

function splash(text, { error = false, offerStart = false } = {}) {
  if (!dash || dash.isDestroyed()) return;
  const html = SPLASH({ text, error, offerStart, spin: !error && !offerStart });
  dash.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html)).catch(() => {});
}
function showDashboard() { if (dash && !dash.isDestroyed()) dash.loadURL(BASE).catch(() => {}); }

function windowBg() { return require('electron').nativeTheme.shouldUseDarkColors ? BG.dark : BG.light; }
const icon = () => (IS_WIN && fs.existsSync(ICON_ICO) ? ICON_ICO : fs.existsSync(ICON_PNG) ? ICON_PNG : undefined);

function createDashboard() {
  dash = new BrowserWindow({
    width: 1320, height: 860, minWidth: 980, minHeight: 620, title: 'Call Transcriber', show: false,
    backgroundColor: windowBg(), icon: icon(), autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'dashboard-preload.js'), contextIsolation: true, sandbox: true, backgroundThrottling: true },
  });
  dash.once('ready-to-show', () => dash.show());
  dash.webContents.on('did-finish-load', () => dash.webContents.send('ct:app-state', appState()));
  dash.on('close', (e) => { if (!quitting) { e.preventDefault(); dash.hide(); } });   // keeps transcribing
  dash.webContents.on('will-navigate', (e, url) => { if (!isDashUrl(url)) { e.preventDefault(); openExternal(url); } });
  dash.webContents.setWindowOpenHandler(({ url }) => { openExternal(url); return { action: 'deny' }; });
  dash.webContents.on('did-fail-load', (_e, code, _d, url, isMain) => {
    if (!isMain || code === -3 || !url.startsWith(BASE)) return;      // -3: aborted by our own navigation
    if (!server) splash(stoppedByUser ? 'The server is stopped. Nothing is being transcribed.' : 'Server not running.', { offerStart: true, error: !stoppedByUser });
    else setTimeout(async () => { if (await serverUp()) showDashboard(); }, 1000);
  });
  splash('Starting…');
}

function chromeUserAgent() {
  // Look like plain Chrome (Discord shows an "unsupported browser" notice otherwise).
  return app.userAgentFallback.split(' ').filter((t) => !/^(Electron|CallTranscriber|call-transcriber)\//i.test(t)).join(' ');
}

function setupDiscordSession() {
  const ses = session.fromPartition('persist:discord');
  ses.setUserAgent(chromeUserAgent());
  const allowed = new Set(['media', 'notifications', 'clipboard-sanitized-write', 'clipboard-read', 'fullscreen', 'speaker-selection', 'display-capture']);
  ses.setPermissionRequestHandler((_wc, perm, cb, details) => cb(allowed.has(perm) && isDiscordUrl(details.requestingUrl || '')));
  ses.setPermissionCheckHandler((_wc, perm, origin) => allowed.has(perm) && isDiscordUrl(origin || ''));
  ses.setDisplayMediaRequestHandler(async (_req, cb) => {   // screen share: the main screen
    try {
      const [src] = await desktopCapturer.getSources({ types: ['screen'] });
      cb(src ? { video: src, ...(IS_WIN ? { audio: 'loopback' } : {}) } : {});
    } catch { cb({}); }
  });
  return ses;
}

function createDiscord() {
  discord = new BrowserWindow({
    width: 1240, height: 820, minWidth: 940, minHeight: 500, title: 'Discord — Call Transcriber', show: false,
    backgroundColor: '#313338', icon: icon(), autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'discord-preload.js'), contextIsolation: true, sandbox: true, nodeIntegration: false,
      session: setupDiscordSession(),
      backgroundThrottling: false,          // keep capturing at full rate behind other windows
      spellcheck: true,
    },
  });
  const wc = discord.webContents;
  wc.on('will-navigate', (e, url) => { if (!isDiscordUrl(url)) { e.preventDefault(); openExternal(url); } });
  wc.on('will-redirect', (e, url) => { if (!isDiscordUrl(url)) { e.preventDefault(); openExternal(url); } });
  wc.setWindowOpenHandler(({ url }) => { openExternal(url); return { action: 'deny' }; });
  discord.on('close', (e) => { if (!quitting) { e.preventDefault(); discord.hide(); broadcast(); } });   // stay in the call
  discord.on('show', broadcast); discord.on('hide', broadcast);
  discord.once('ready-to-show', () => { if (!TEST || process.env.CT_SHOW === '1') discord.show(); });
  discord.loadURL(DISCORD_URL).catch(() => {});
}

async function showDiscord() {
  if (IS_MAC && systemPreferences.getMediaAccessStatus('microphone') === 'not-determined') {
    systemPreferences.askForMediaAccess('microphone').catch(() => {});
  }
  if (!discord || discord.isDestroyed()) { createDiscord(); return; }
  discord.show(); discord.focus();
}

function openExternal(url) {
  try { const u = new URL(url); if (u.protocol === 'https:' || u.protocol === 'http:') shell.openExternal(u.href); } catch {}
}

function showMain() { if (!dash || dash.isDestroyed()) createDashboard(); dash.show(); dash.focus(); }

// ----------------------------------------------------------------- update relaunch
async function relaunchAfterUpdate({ rebuild = false, deps = false, app: appDeps = false }) {
  splash('Installing the update…');
  await stopServer({ quiet: true });
  if (deps) {
    const uv = IS_WIN ? 'uv' : '/bin/zsh';
    const args = IS_WIN ? ['sync', '--inexact'] : ['-lc', 'uv sync --inexact'];
    await run(uv, args, { cwd: SERVER_DIR, shell: IS_WIN });
  }
  const desktopDir = __dirname;
  if (IS_MAC && (rebuild || appDeps)) {
    // The app bundle embeds Electron: rebuild it, then open the new copy.
    const code = await run('/bin/zsh', ['-lc', `"${path.join(ROOT, 'app', 'mac', 'build.sh')}"`]);
    if (code !== 0) { splash('The app rebuild failed. Run app/mac/build.sh in a terminal to see why.', { error: true, offerStart: true }); return; }
    spawn('/bin/sh', ['-c', `sleep 1; open -n "${path.join(process.env.CT_INSTALL_DIR || path.join(app.getPath('home'), 'Applications'), 'Call Transcriber.app')}"`], { detached: true, stdio: 'ignore' }).unref();
  } else if (IS_WIN && appDeps) {
    // electron.exe can't be replaced while it runs: refresh it after we exit, then start again.
    const exe = path.join(desktopDir, 'node_modules', 'electron', 'dist', 'electron.exe');
    const script = path.join(app.getPath('temp'), 'ct-update.cmd');
    fs.writeFileSync(script, `@echo off\r\nping -n 3 127.0.0.1 >nul\r\ncd /d "${desktopDir}"\r\ncall npm ci --no-audit --no-fund\r\nnode node_modules\\electron\\install.js\r\nstart "" "${exe}" "${desktopDir}"\r\n`);
    spawn('cmd.exe', ['/c', script], { detached: true, stdio: 'ignore', windowsHide: true }).unref();
  } else {
    app.relaunch();
  }
  quitting = true;
  app.exit(0);
}

// ----------------------------------------------------------------- menu / tray
function buildMenu() {
  const t = [];
  if (IS_MAC) t.push({ label: 'Call Transcriber', submenu: [
    { role: 'about' }, { type: 'separator' }, { role: 'hide' }, { role: 'hideOthers' }, { type: 'separator' },
    { label: 'Quit Call Transcriber (stops transcribing)', accelerator: 'Cmd+Q', click: quit },
  ] });
  t.push({ label: 'Discord', submenu: [
    { label: 'Show Discord', accelerator: 'CmdOrCtrl+D', click: showDiscord },
    { label: 'Show Dashboard', accelerator: 'CmdOrCtrl+1', click: showMain },
    { type: 'separator' },
    { label: 'Pause Transcribing', type: 'checkbox', checked: paused, click: (i) => setPaused(i.checked) },
    { label: 'Reload Discord', click: () => discord && !discord.isDestroyed() && discord.webContents.reload() },
  ] });
  t.push({ label: 'Server', submenu: [
    { label: 'Start Server', click: startServer },
    { label: 'Restart Server', accelerator: 'CmdOrCtrl+Shift+R', click: restartServer },
    { label: 'Stop Server', click: () => stopServer() },
    { type: 'separator' },
    { label: 'Show Server Log', accelerator: 'CmdOrCtrl+L', click: () => shell.openPath(LOG) },
  ] });
  const js = (code) => () => dash && !dash.isDestroyed() && dash.webContents.executeJavaScript(code).catch(() => {});
  t.push({ label: 'View', submenu: [
    { label: 'Reload Dashboard', accelerator: 'CmdOrCtrl+R', click: () => (server ? showDashboard() : splash('The server is stopped. Nothing is being transcribed.', { offerStart: true })) },
    { label: 'Show/Hide Calls List', accelerator: 'CmdOrCtrl+\\', click: js("document.getElementById('toggle-side')?.click()") },
    { label: 'Search All Calls…', accelerator: 'CmdOrCtrl+Shift+F', click: js("(()=>{const s=document.getElementById('shell');s?.classList.remove('collapsed');document.getElementById('hq')?.focus()})()") },
    { label: 'Find in Transcript…', accelerator: 'CmdOrCtrl+F', click: js("document.getElementById('q')?.focus()") },
    { type: 'separator' },
    { label: 'Open Transcripts Folder', accelerator: 'CmdOrCtrl+O', click: () => shell.openPath(path.join(ROOT, 'transcripts')) },
    { label: 'Open Dashboard in Browser', accelerator: 'CmdOrCtrl+B', click: () => shell.openExternal(BASE) },
    { type: 'separator' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'resetZoom' }, { role: 'toggleDevTools' },
  ] });
  t.push({ role: 'editMenu' });
  t.push({ label: 'Window', submenu: [{ role: 'minimize' }, { label: 'Close Window (keeps transcribing)', accelerator: 'CmdOrCtrl+W', role: 'close' },
    ...(IS_WIN ? [{ type: 'separator' }, { label: 'Quit (stops transcribing)', click: quit }] : [])] });
  Menu.setApplicationMenu(Menu.buildFromTemplate(t));
}

function updateTray() {
  if (!tray) return;
  const state = !server ? 'server stopped' : paused ? 'paused' : wsOpen ? 'ready' : 'starting';
  tray.setToolTip(`Call Transcriber — ${state}`);
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open Call Transcriber', click: showMain },
    { label: 'Open Discord', click: showDiscord },
    { type: 'separator' },
    { label: 'Pause transcribing', type: 'checkbox', checked: paused, click: (i) => setPaused(i.checked) },
    { label: 'Restart server', click: restartServer },
    { label: server ? 'Stop server' : 'Start server', click: () => (server ? stopServer() : startServer()) },
    { label: 'Open transcripts folder', click: () => shell.openPath(path.join(ROOT, 'transcripts')) },
    { type: 'separator' },
    { label: 'Quit (stops transcribing)', click: quit },
  ]));
}

async function quit() {
  if (quitting) return;
  quitting = true;
  await stopServer({ quiet: true });
  app.quit();
}

// ----------------------------------------------------------------- lifecycle
app.on('second-instance', showMain);
app.on('activate', showMain);
app.on('window-all-closed', () => {});                    // closing windows keeps transcribing
app.on('before-quit', (e) => {
  if (quitting) return;
  e.preventDefault(); quit();
});
app.on('web-contents-created', (_e, wc) => {
  wc.on('will-attach-webview', (ev) => ev.preventDefault());
});

app.whenReady().then(() => {
  buildMenu();
  powerSaveBlocker.start('prevent-app-suspension');        // keep transcribing at full speed in the background
  if (IS_WIN && fs.existsSync(ICON_PNG)) {
    tray = new Tray(nativeImage.createFromPath(ICON_PNG).resize({ width: 16, height: 16 }));
    tray.on('click', showMain);
    updateTray();
  }
  createDashboard();
  createDiscord();
  startServer().then(connectServer);
});
