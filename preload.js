const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  readFile: (relPath) => ipcRenderer.invoke('read-file', relPath),
  listFiles: (relDir) => ipcRenderer.invoke('list-files', relDir),
});
