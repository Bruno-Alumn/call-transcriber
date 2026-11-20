// Preload for the screen-share picker: shows what Discord may share, returns the choice.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('picker', {
  sources: () => ipcRenderer.invoke('ct:picker-sources'),
  choose: (id) => ipcRenderer.send('ct:picker-choose', id == null ? null : String(id)),
});
