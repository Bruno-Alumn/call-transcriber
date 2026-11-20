// Holds the WebSocket to the local transcription server and relays per-person
// audio from Discord tabs. Transcribing is on whenever the server is running;
// click the toolbar icon to pause/resume.
const DEFAULT_SERVER = 'ws://127.0.0.1:8765';
let ws = null, wsOpen = false, paused = false;
const ports = new Set();
const lastSeen = new Map(); // user -> ms

async function loadSettings() {
  const s = await chrome.storage.local.get(['paused', 'server']);
  paused = !!s.paused;
  return s.server || DEFAULT_SERVER;
}

function broadcast() {
  const on = wsOpen && !paused;
  for (const p of ports) { try { p.postMessage({ type: 'status', on }); } catch {} }
  updateBadge();
}

function updateBadge() {
  const now = Date.now();
  const n = [...lastSeen.values()].filter((t) => now - t < 15000).length;
  let text = '', color = '#5865f2', title;
  if (paused) { text = 'II'; color = '#777'; title = 'Paused — click to resume'; }
  else if (!wsOpen) { text = 'OFF'; color = '#777'; title = 'Server not running — start ~/discord-transcriber/start.sh'; }
  else { text = n ? String(n) : 'ON'; color = '#d93025'; title = n ? `Transcribing ${n} people — click to pause` : 'Ready — join a voice channel'; }
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
  chrome.action.setTitle({ title });
}

async function connectServer() {
  const url = await loadSettings();
  try { ws = new WebSocket(url); } catch { setTimeout(connectServer, 3000); return; }
  ws.onopen = () => { wsOpen = true; broadcast(); };
  ws.onclose = () => { wsOpen = false; broadcast(); setTimeout(connectServer, 3000); };
  ws.onerror = () => {};
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'audio') return;
  ports.add(port);
  port.postMessage({ type: 'status', on: wsOpen && !paused });
  port.onDisconnect.addListener(() => ports.delete(port));
  port.onMessage.addListener((m) => {
    if (m.type === 'hook-status') { if (wsOpen) ws.send(JSON.stringify({ type: 'hook-status', s: m.s, paused })); return; }
    if (m.type !== 'track-audio' || paused || !wsOpen) return;
    lastSeen.set(m.user, Date.now());
    ws.send(JSON.stringify(m));
  });
});

chrome.action.onClicked.addListener(async () => {
  paused = !paused;
  await chrome.storage.local.set({ paused });
  broadcast();
});

// Discord tabs that were already open when the extension (re)loaded: hook them
// now. Calls joined after this point are captured; rejoin any call in progress.
async function hookOpenTabs() {
  const tabs = await chrome.tabs.query({ url: 'https://discord.com/*' });
  for (const t of tabs) {
    await chrome.scripting.executeScript({ target: { tabId: t.id }, world: 'MAIN', files: ['page-hook.js'] }).catch(() => {});
    await chrome.scripting.executeScript({ target: { tabId: t.id }, files: ['content.js'] }).catch(() => {});
  }
}
chrome.runtime.onInstalled.addListener(async () => {
  await chrome.storage.local.set({ paused: false });   // fresh install/reload starts un-paused
  paused = false; broadcast();
  hookOpenTabs();
});
// A Discord page that navigated/reloaded: make sure the hook is in (it is a
// no-op if the manifest already injected it). Voice is joined by hand later.
chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.status === 'loading' && tab.url?.startsWith('https://discord.com/')) {
    chrome.scripting.executeScript({ target: { tabId }, world: 'MAIN', files: ['page-hook.js'], injectImmediately: true }).catch(() => {});
  }
});

setInterval(updateBadge, 5000);
connectServer();
