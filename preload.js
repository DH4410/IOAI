const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  readFile:    (relPath)         => ipcRenderer.invoke('read-file', relPath),
  listFiles:   (relDir)          => ipcRenderer.invoke('list-files', relDir),
  loadData:    ()                => ipcRenderer.invoke('load-data'),
  saveData:    (json)            => ipcRenderer.invoke('save-data', json),
  ollamaChat:  (opts)            => ipcRenderer.invoke('ollama-chat', opts),
  ollamaCheck: ()                => ipcRenderer.invoke('ollama-check'),
});
