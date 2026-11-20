// Preload for the dashboard window: the dashboard's only way to talk to the app.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('ctApp', {
  platform: process.platform,
  server: (action) => ipcRenderer.send('ct:server', String(action)),
  save: (name, content) => ipcRenderer.invoke('ct:save', String(name), String(content)),
  relaunch: (opts) => ipcRenderer.send('ct:relaunch', { rebuild: !!opts?.rebuild, deps: !!opts?.deps, app: !!opts?.app }),
  openDiscord: () => ipcRenderer.send('ct:open-discord'),
  setPaused: (on) => ipcRenderer.send('ct:pause', !!on),
  onState: (fn) => ipcRenderer.on('ct:app-state', (_e, s) => fn(s)),
});
