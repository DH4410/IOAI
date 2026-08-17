const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const path = require('path');
const fs = require('fs');
const http = require('http');

function createWindow() {
  // Remove File / Edit / View / Help menu bar completely
  Menu.setApplicationMenu(null);

  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: 'IOAI Prep',
    backgroundColor: '#1E1B4B',
    autoHideMenuBar: true,
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.loadFile('renderer/index.html');

  if (process.argv.includes('--dev')) {
    win.webContents.openDevTools();
  }
}

// Read a file relative to the app root
ipcMain.handle('read-file', (_, relPath) => {
  const abs = path.join(__dirname, relPath);
  if (!fs.existsSync(abs)) return null;
  return fs.readFileSync(abs, 'utf8');
});

// List files in a content directory
ipcMain.handle('list-files', (_, relDir) => {
  const abs = path.join(__dirname, relDir);
  if (!fs.existsSync(abs)) return [];
  return fs.readdirSync(abs).filter(f => !f.startsWith('.'));
});

// Persistent file-based progress (survives app reinstalls)
const progressFile = path.join(app.getPath('userData'), 'progress.json');

ipcMain.handle('load-data', () => {
  try {
    if (fs.existsSync(progressFile)) return fs.readFileSync(progressFile, 'utf8');
  } catch (_) {}
  return null;
});

ipcMain.handle('save-data', (_, json) => {
  try {
    fs.mkdirSync(path.dirname(progressFile), { recursive: true });
    fs.writeFileSync(progressFile, json, 'utf8');
    return true;
  } catch (_) {
    return false;
  }
});

// Ollama local AI — proxied from main process so CSP is not an issue
ipcMain.handle('ollama-chat', async (_, { model, messages }) => {
  return new Promise((resolve) => {
    const body = JSON.stringify({ model: model || 'llama3.2:1b', messages, stream: false });
    const req = http.request({
      hostname: '127.0.0.1',
      port: 11434,
      path: '/api/chat',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      timeout: 30000,
    }, (res) => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ ok: true, content: parsed.message?.content || '' });
        } catch (e) {
          resolve({ ok: false, error: 'Bad response from Ollama' });
        }
      });
    });
    req.on('error', () => resolve({ ok: false, error: 'Ollama not running. Install from ollama.com and run: ollama pull llama3.2:1b' }));
    req.on('timeout', () => { req.destroy(); resolve({ ok: false, error: 'Ollama timed out (>30s)' }); });
    req.write(body);
    req.end();
  });
});

// Check if Ollama is available
ipcMain.handle('ollama-check', async () => {
  return new Promise((resolve) => {
    const req = http.request({ hostname: '127.0.0.1', port: 11434, path: '/', method: 'GET', timeout: 2000 },
      (res) => resolve(res.statusCode < 500));
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
    req.end();
  });
});

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
