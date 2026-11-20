// Bridge: page-hook.js (Discord's page) -> background.js (-> local server).
(() => {
  if (window.__dtBridge) return; window.__dtBridge = true;
  let port = null, connected = false;
  const toB64 = (buf) => {
    let s = ''; const b = new Uint8Array(buf);
    for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
    return btoa(s);
  };
  const setEnabled = (on) => window.postMessage({ __dtControl: true, enabled: on }, location.origin);

  function connect() {
    try { port = chrome.runtime.connect({ name: 'audio' }); } catch { return; } // extension reloaded
    port.onMessage.addListener((m) => { if (m.type === 'status') { connected = m.on; setEnabled(m.on); } });
    port.onDisconnect.addListener(() => { port = null; connected = false; setEnabled(false); setTimeout(connect, 1000); });
  }
  connect();
  // Names learned in earlier calls (kept in the extension's own storage).
  const sendNames = () => chrome.storage.local.get('names').then(({ names }) => {
    if (names) window.postMessage({ __dtNamesInit: names }, location.origin);
  }).catch(() => {});
  sendNames(); setTimeout(sendNames, 3000);
  window.addEventListener('message', (e) => {
    if (e.source !== window || !e.data?.__dtNames) return;
    chrome.storage.local.get('names').then(({ names = {} }) =>
      chrome.storage.local.set({ names: { ...names, ...e.data.__dtNames } })).catch(() => {});
  });
  // Re-announce state so a hook that loaded late still gets it.
  setInterval(() => setEnabled(connected), 2000);

  window.addEventListener('message', (e) => {
    if (e.source === window && e.data?.__dtStatus && port) { try { port.postMessage({ type: 'hook-status', s: e.data.s }); } catch {} }
  });
  window.addEventListener('message', (e) => {
    if (e.source !== window || !e.data?.__dtAudio || !port || !connected) return;
    const { user, name, ch, t, pcm } = e.data;
    try { port.postMessage({ type: 'track-audio', user, name, ch, t, pcm: toB64(pcm) }); } catch {}
  });
})();
