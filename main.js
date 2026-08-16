const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: 'IOAI Prep',
    backgroundColor: '#F4F5FA',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
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

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
