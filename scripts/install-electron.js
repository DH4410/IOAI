// One-time script: download the Electron binary for Windows if missing
const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const electronPkg = require('../node_modules/electron/package.json');
const version = electronPkg.version;
const distDir = path.join(__dirname, '..', 'node_modules', 'electron', 'dist');
const electronExe = path.join(distDir, 'electron.exe');

if (fs.existsSync(electronExe)) {
  console.log('Electron already installed at', electronExe);
  process.exit(0);
}

const url = `https://github.com/electron/electron/releases/download/v${version}/electron-v${version}-win32-x64.zip`;
const zipPath = path.join(distDir, `electron-v${version}-win32-x64.zip`);

console.log(`Downloading Electron v${version}...`);

const file = fs.createWriteStream(zipPath);
https.get(url, res => {
  if (res.statusCode === 302 || res.statusCode === 301) {
    https.get(res.headers.location, res2 => res2.pipe(file));
  } else {
    res.pipe(file);
  }
  file.on('finish', () => {
    file.close();
    console.log('Download complete. Extracting...');
    try {
      execSync(`powershell Expand-Archive -Path "${zipPath}" -DestinationPath "${distDir}" -Force`);
      fs.unlinkSync(zipPath);
      console.log('Electron ready!');
    } catch (e) {
      console.error('Extraction failed:', e.message);
      console.log('Try running manually: Expand-Archive -Path "' + zipPath + '" -DestinationPath "' + distDir + '"');
    }
  });
}).on('error', e => {
  console.error('Download error:', e.message);
});
