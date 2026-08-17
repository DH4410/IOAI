// Pure Node.js icon generator — no extra packages required
// Generates assets/icon.png and assets/icon.ico

const zlib = require('zlib');
const fs   = require('fs');
const path = require('path');

const S = 256;

// ── CRC32 (needed for PNG chunks) ─────────────────────────────────────────
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xFFFFFFFF;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

function pngChunk(type, data) {
  const typeBuf = Buffer.from(type, 'ascii');
  const lenBuf  = Buffer.alloc(4);  lenBuf.writeUInt32BE(data.length);
  const crcBuf  = Buffer.alloc(4);  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])));
  return Buffer.concat([lenBuf, typeBuf, data, crcBuf]);
}

// ── Pixel helpers ─────────────────────────────────────────────────────────
const lerp  = (a, b, t) => a + (b - a) * t;
const clamp = (x)       => Math.max(0, Math.min(255, Math.round(x)));
const dist  = (ax, ay, bx, by) => Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);

// Rounded-rectangle SDF — positive = outside, negative = inside
function rrSDF(px, py, cx, cy, hw, hh, r) {
  const qx = Math.abs(px - cx) - hw + r;
  const qy = Math.abs(py - cy) - hh + r;
  return Math.min(Math.max(qx, qy), 0)
       + Math.sqrt(Math.max(qx, 0) ** 2 + Math.max(qy, 0) ** 2)
       - r;
}

// ── Design ────────────────────────────────────────────────────────────────
//   Dark indigo rounded square + purple radial glow
//   Neural-net: 6 outer nodes (hexagon) + 1 centre node, spokes + ring
const CORNER_R = 48;
const CX = S / 2, CY = S / 2;

// Hex ring of 6 outer nodes
const outerNodes = Array.from({ length: 6 }, (_, i) => {
  const a = (Math.PI / 3) * i - Math.PI / 6;
  return [CX + Math.cos(a) * S * 0.30, CY + Math.sin(a) * S * 0.30];
});
const centreNode = [CX, CY];
const allNodes   = [centreNode, ...outerNodes];

// Edges: centre → each outer node, and ring edges between adjacent outer nodes
const edges = [
  ...outerNodes.map((_, i) => [0, i + 1]),              // spokes
  ...outerNodes.map((_, i) => [i + 1, ((i + 1) % 6) + 1]), // ring
];

const pixels = Buffer.alloc(S * S * 4); // RGBA

for (let y = 0; y < S; y++) {
  for (let x = 0; x < S; x++) {
    // Rounded-rect mask (alpha with 1.5px anti-alias)
    const sdf = rrSDF(x, y, CX, CY, S / 2, S / 2, CORNER_R);
    let alpha = 255;
    if (sdf > 1)   { alpha = 0; }
    else if (sdf > -1) { alpha = clamp((-sdf + 1) / 2 * 255); }

    // Background gradient: #1E1B4B → #2D2060
    const gx = x / S, gy = y / S;
    let r = lerp(30, 45, gy);
    let g = lerp(27, 32, gy);
    let b = lerp(75, 96, gy);

    // Purple radial glow from centre
    const cdist = dist(x, y, CX, CY) / (S * 0.46);
    const glow  = Math.max(0, 1 - cdist * cdist);
    r = lerp(r, 110, glow * glow * 0.55);
    g = lerp(g,  70, glow * glow * 0.30);
    b = lerp(b, 220, glow * glow * 0.70);

    // Connection edges
    for (const [ai, bi] of edges) {
      const [ax, ay] = allNodes[ai];
      const [bx, by] = allNodes[bi];
      const dx = bx - ax, dy = by - ay;
      const len2 = dx * dx + dy * dy;
      const t2   = len2 < 0.001 ? 0 : Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2));
      const closX = ax + t2 * dx, closY = ay + t2 * dy;
      const ld    = dist(x, y, closX, closY);
      if (ld < 1.5) {
        const blend = Math.max(0, 1 - ld / 1.5) * 0.35;
        r = lerp(r, 170, blend);
        g = lerp(g, 150, blend);
        b = lerp(b, 255, blend);
      }
    }

    // Node circles
    const nodeR = [S * 0.075, ...Array(6).fill(S * 0.060)]; // centre bigger
    for (let ni = 0; ni < allNodes.length; ni++) {
      const [nx, ny] = allNodes[ni];
      const nr       = nodeR[ni];
      const nd       = dist(x, y, nx, ny);
      if (nd < nr) {
        const blend = (1 - nd / nr) ** 1.5;
        r = lerp(120, 240, blend);
        g = lerp(100, 220, blend);
        b = lerp(220, 255, blend);
        // Specular highlight
        const hl = dist(x, y, nx - nr * 0.25, ny - nr * 0.25);
        if (hl < nr * 0.40) {
          const hb = Math.max(0, 1 - hl / (nr * 0.40)) ** 2 * 0.55;
          r = lerp(r, 255, hb);
          g = lerp(g, 255, hb);
          b = lerp(b, 255, hb);
        }
      } else if (nd < nr + 1.5) {
        const bd = Math.max(0, 1 - (nd - nr) / 1.5) * 0.5;
        r = lerp(r, 200, bd);
        g = lerp(g, 180, bd);
        b = lerp(b, 255, bd);
      }
    }

    const i = (y * S + x) * 4;
    pixels[i]     = clamp(r);
    pixels[i + 1] = clamp(g);
    pixels[i + 2] = clamp(b);
    pixels[i + 3] = alpha;
  }
}

// ── Encode PNG ────────────────────────────────────────────────────────────
const IHDR = Buffer.alloc(13);
IHDR.writeUInt32BE(S, 0); IHDR.writeUInt32BE(S, 4);
IHDR[8] = 8; IHDR[9] = 6; // 8-bit RGBA

const rawData = Buffer.alloc(S * (1 + S * 4));
for (let row = 0; row < S; row++) {
  rawData[row * (1 + S * 4)] = 0; // filter type: None
  pixels.copy(rawData, row * (1 + S * 4) + 1, row * S * 4, (row + 1) * S * 4);
}

const compressed = zlib.deflateSync(rawData, { level: 9 });
const png = Buffer.concat([
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), // PNG signature
  pngChunk('IHDR', IHDR),
  pngChunk('IDAT', compressed),
  pngChunk('IEND', Buffer.alloc(0)),
]);

// ── Encode ICO (single 256×256 PNG entry) ─────────────────────────────────
// Windows Vista+ supports PNG images inside ICO files
function pngToIco(pngBuf) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); // reserved
  header.writeUInt16LE(1, 2); // type: icon
  header.writeUInt16LE(1, 4); // count

  const entry = Buffer.alloc(16);
  entry[0] = 0;  entry[1] = 0;  // width/height: 0 = 256
  entry[2] = 0;  entry[3] = 0;  // colorCount, reserved
  entry.writeUInt16LE(1,  4);   // planes
  entry.writeUInt16LE(32, 6);   // bit count
  entry.writeUInt32LE(pngBuf.length, 8);  // bytes in image
  entry.writeUInt32LE(22, 12);            // offset (6 + 16 = 22)

  return Buffer.concat([header, entry, pngBuf]);
}

// ── Write files ───────────────────────────────────────────────────────────
const assetsDir = path.join(__dirname, '..', 'assets');
fs.mkdirSync(assetsDir, { recursive: true });

const pngPath = path.join(assetsDir, 'icon.png');
const icoPath = path.join(assetsDir, 'icon.ico');

fs.writeFileSync(pngPath, png);
fs.writeFileSync(icoPath, pngToIco(png));

console.log(`icon.png  ${(png.length / 1024).toFixed(1)} KB → ${pngPath}`);
console.log(`icon.ico  ${(fs.statSync(icoPath).size / 1024).toFixed(1)} KB → ${icoPath}`);
