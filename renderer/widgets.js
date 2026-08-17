// ── Widget Engine ──────────────────────────────────────────────────────────────
// Parses ```widget {...} ``` fenced blocks and renders interactive components.
// Usage in lesson markdown:
//   ```widget
//   { "type": "gradient-slider", "title": "See gradient descent" }
//   ```

export function preprocessWidgets(markdown) {
  return markdown.replace(/```widget\n([\s\S]*?)```/gm, (_, json) => {
    const safe = encodeURIComponent(json.trim());
    return `<div class="widget-placeholder" data-config="${safe}"></div>`;
  });
}

export function renderWidgets(container) {
  container.querySelectorAll('.widget-placeholder').forEach(el => {
    try {
      const config = JSON.parse(decodeURIComponent(el.dataset.config));
      const widget = buildWidget(config);
      if (widget) el.replaceWith(widget);
      else el.remove();
    } catch (e) {
      const err = document.createElement('div');
      err.className = 'widget-error';
      err.textContent = `Widget error: ${e.message}`;
      el.replaceWith(err);
    }
  });
}

function buildWidget(config) {
  switch (config.type) {
    case 'gradient-slider':   return gradientSlider(config);
    case 'perceptron-demo':   return perceptronDemo(config);
    case 'overfit-curve':     return overfitCurve(config);
    case 'attention-heatmap': return attentionHeatmap(config);
    case 'tokenizer-live':    return tokenizerLive(config);
    case 'conv-stepper':      return convStepper(config);
    case 'neuron-diagram':    return neuronDiagram(config);
    default: return null;
  }
}

function isDark() {
  return document.documentElement.classList.contains('dark');
}

function makeWidget(title, icon, bodyHTML) {
  const div = document.createElement('div');
  div.className = 'widget-card';
  div.innerHTML = `
    <div class="widget-header">
      <span class="widget-icon">${icon}</span>
      <span class="widget-title">${title}</span>
      <span class="widget-badge">Interactive</span>
    </div>
    <div class="widget-body">${bodyHTML}</div>
  `;
  return div;
}

// ── 1. Gradient Descent Slider ────────────────────────────────────────────────
function gradientSlider(config) {
  const title = config.title || 'Gradient Descent in Action';
  const div = makeWidget(title, '📉', `
    <div class="grad-layout">
      <div class="grad-controls">
        <label class="ctrl-label">Learning rate: <strong class="lr-display">0.30</strong></label>
        <input type="range" class="ctrl-range lr-range" min="1" max="95" value="30">
        <div class="widget-btn-row">
          <button class="widget-btn step-btn">Step →</button>
          <button class="widget-btn auto-btn">▶ Auto</button>
          <button class="widget-btn widget-btn-sec reset-btn">Reset</button>
        </div>
        <div class="grad-stats">
          <div class="stat-row"><span>Position x:</span><strong class="s-pos">—</strong></div>
          <div class="stat-row"><span>Loss x²:</span><strong class="s-loss">—</strong></div>
          <div class="stat-row"><span>Gradient 2x:</span><strong class="s-grad">—</strong></div>
          <div class="grad-status">Press Step to begin</div>
        </div>
        <div class="grad-formula">x<sub>new</sub> = x − lr × gradient</div>
      </div>
      <canvas class="grad-canvas" width="290" height="210"></canvas>
    </div>
  `);

  const canvas = div.querySelector('.grad-canvas');
  const ctx = canvas.getContext('2d');
  const lrRange = div.querySelector('.lr-range');
  const lrDisplay = div.querySelector('.lr-display');
  const status = div.querySelector('.grad-status');

  let x = (Math.random() > 0.5 ? 1 : -1) * (2.5 + Math.random());
  let autoHandle = null, running = false;

  const loss = x => x * x;
  const grad = x => 2 * x;
  const toC = (px, py) => [10 + ((px + 4.5) / 9) * 270, 195 - (py / 20) * 180];

  function draw() {
    const W = canvas.width, H = canvas.height;
    const dark = isDark();
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = dark ? '#1a1a2e' : '#f0f1ff';
    ctx.fillRect(0, 0, W, H);

    // Axes
    const [ax] = toC(0, 0), [, ay] = toC(0, 0);
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(10, ay); ctx.lineTo(280, ay);
    ctx.moveTo(ax, 10); ctx.lineTo(ax, H - 10);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = dark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.35)';
    ctx.font = '10px system-ui';
    ctx.fillText('x', 275, ay - 6);
    ctx.fillText('0', ax + 4, ay - 4);
    ctx.fillText('loss = x²', 14, 18);

    // Parabola
    ctx.beginPath();
    ctx.strokeStyle = '#5B5BD6';
    ctx.lineWidth = 2.5;
    for (let xi = -4.5; xi <= 4.5; xi += 0.05) {
      const [cx, cy] = toC(xi, loss(xi));
      xi === -4.5 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Gradient arrow
    const g = grad(x), [bx, by] = toC(x, loss(x));
    const arrowLen = Math.min(45, Math.abs(g) * 9);
    const dir = g > 0 ? 1 : -1;
    ctx.strokeStyle = '#F97316';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx + dir * arrowLen, by - 12);
    ctx.stroke();
    ctx.fillStyle = '#F97316';
    ctx.beginPath();
    ctx.arc(bx + dir * arrowLen, by - 12, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Ball
    ctx.fillStyle = '#22C55E';
    ctx.beginPath();
    ctx.arc(bx, by, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Stats
    div.querySelector('.s-pos').textContent = x.toFixed(4);
    div.querySelector('.s-loss').textContent = loss(x).toFixed(4);
    div.querySelector('.s-grad').textContent = g.toFixed(4);
  }

  function step() {
    const lr = lrRange.value / 100;
    x = x - lr * grad(x);
    x = Math.max(-4.4, Math.min(4.4, x));
    if (Math.abs(x) < 0.004) {
      status.textContent = '🎉 Converged at minimum (x ≈ 0)!';
      status.style.color = 'var(--green)';
      stopAuto();
    } else if (Math.abs(x) > 4) {
      status.textContent = '⚠️ Diverging — try a smaller learning rate';
      status.style.color = 'var(--orange)';
    } else {
      status.textContent = `Moving toward x = 0 (step size: ${(lr * Math.abs(grad(x))).toFixed(3)})`;
      status.style.color = '';
    }
    draw();
  }

  function stopAuto() {
    if (autoHandle) { clearInterval(autoHandle); autoHandle = null; }
    running = false;
    div.querySelector('.auto-btn').textContent = '▶ Auto';
  }

  function reset() {
    x = (Math.random() > 0.5 ? 1 : -1) * (2.5 + Math.random());
    status.textContent = 'Press Step to begin';
    status.style.color = '';
    stopAuto();
    draw();
  }

  div.querySelector('.step-btn').addEventListener('click', step);
  div.querySelector('.reset-btn').addEventListener('click', reset);
  div.querySelector('.auto-btn').addEventListener('click', () => {
    if (running) { stopAuto(); return; }
    running = true;
    div.querySelector('.auto-btn').textContent = '⏸ Stop';
    autoHandle = setInterval(() => {
      step();
      if (Math.abs(x) < 0.004 || Math.abs(x) > 4) stopAuto();
    }, 280);
  });
  lrRange.addEventListener('input', () => {
    lrDisplay.textContent = (lrRange.value / 100).toFixed(2);
    draw();
  });

  requestAnimationFrame(draw);
  return div;
}

// ── 2. Perceptron Playground ──────────────────────────────────────────────────
function perceptronDemo(config) {
  const title = config.title || 'Perceptron Playground';
  const div = makeWidget(title, '🧠', `
    <div class="perc-layout">
      <div class="perc-sliders">
        <div class="perc-group-label">Inputs</div>
        ${[1,2,3].map(i => `
          <div class="perc-row">
            <span class="perc-lbl">x${i}</span>
            <input type="range" class="ctrl-range" data-xi="${i}" min="-5" max="5" value="${i===1?3:i===2?-1:2}" step="0.5">
            <span class="perc-val" data-xv="${i}">0</span>
          </div>`).join('')}
        <div class="perc-group-label" style="margin-top:10px">Weights</div>
        ${[1,2,3].map(i => `
          <div class="perc-row">
            <span class="perc-lbl weight">w${i}</span>
            <input type="range" class="ctrl-range" data-wi="${i}" min="-5" max="5" value="${i===1?2:i===2?1:-1}" step="0.5">
            <span class="perc-val" data-wv="${i}">0</span>
          </div>`).join('')}
        <div class="perc-row" style="margin-top:6px">
          <span class="perc-lbl bias">b</span>
          <input type="range" class="ctrl-range bias-r" min="-5" max="5" value="-1" step="0.5">
          <span class="perc-val bias-v">0</span>
        </div>
        <div class="perc-act-row">
          <span class="perc-lbl">σ</span>
          <select class="perc-select act-sel">
            <option value="step">Step (0 or 1)</option>
            <option value="sigmoid">Sigmoid (0→1)</option>
            <option value="relu">ReLU max(0,z)</option>
          </select>
        </div>
      </div>
      <div class="perc-right">
        <svg class="perc-svg" viewBox="0 0 240 190" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="perc-math-box">
          <div class="perc-eq-text" id="perc-eq">z = ?</div>
          <div class="perc-out-row">
            <span>Output:</span>
            <span class="perc-out-chip" id="perc-out">—</span>
          </div>
        </div>
      </div>
    </div>
  `);

  const xi = [0, 3, -1, 2], wi = [0, 2, 1, -1];
  let bias = -1;

  function vals() {
    [1,2,3].forEach(i => {
      xi[i] = parseFloat(div.querySelector(`[data-xi="${i}"]`).value);
      wi[i] = parseFloat(div.querySelector(`[data-wi="${i}"]`).value);
      div.querySelector(`[data-xv="${i}"]`).textContent = xi[i].toFixed(1);
      div.querySelector(`[data-wv="${i}"]`).textContent = wi[i].toFixed(1);
    });
    bias = parseFloat(div.querySelector('.bias-r').value);
    div.querySelector('.bias-v').textContent = bias.toFixed(1);
  }

  const activations = {
    step: z => z >= 0 ? 1 : 0,
    sigmoid: z => 1 / (1 + Math.exp(-z)),
    relu: z => Math.max(0, z),
  };

  function update() {
    vals();
    const z = xi[1]*wi[1] + xi[2]*wi[2] + xi[3]*wi[3] + bias;
    const actType = div.querySelector('.act-sel').value;
    const out = activations[actType](z);

    div.querySelector('#perc-eq').textContent =
      `z = (${wi[1]}×${xi[1]}) + (${wi[2]}×${xi[2]}) + (${wi[3]}×${xi[3]}) + ${bias} = ${z.toFixed(2)}`;

    const chip = div.querySelector('#perc-out');
    chip.textContent = out.toFixed(3);
    chip.style.background = out > 0.5 ? 'var(--green-lt)' : 'var(--red-lt)';
    chip.style.color = out > 0.5 ? 'var(--green)' : 'var(--red)';

    drawSVG(z, out);
  }

  function drawSVG(z, out) {
    const dark = isDark();
    const tc = dark ? '#ECECF4' : '#1C1F2E';
    const dim = dark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.1)';
    const ys = [38, 95, 152];
    const nodeColor = v => {
      const a = Math.min(Math.abs(v) / 5, 1);
      return v >= 0 ? `rgba(34,197,94,${0.25 + a*0.6})` : `rgba(239,68,68,${0.25 + a*0.6})`;
    };
    const lineColor = w => w >= 0 ? '#5B5BD6' : '#EF4444';
    const lineW = w => Math.max(0.8, Math.abs(w) * 0.6);

    const svg = div.querySelector('.perc-svg');
    let h = '';
    [1,2,3].forEach((i, idx) => {
      h += `<line x1="46" y1="${ys[idx]}" x2="148" y2="95" stroke="${lineColor(wi[i])}" stroke-width="${lineW(wi[i])}" opacity="0.7"/>`;
      h += `<text x="97" y="${ys[idx] < 95 ? ys[idx]+14 : ys[idx] > 95 ? ys[idx]-6 : ys[idx]+4}" text-anchor="middle" font-size="9" fill="${tc}" opacity="0.7">w${i}=${wi[i].toFixed(1)}</text>`;
      h += `<circle cx="28" cy="${ys[idx]}" r="17" fill="${nodeColor(xi[i])}" stroke="${dim}" stroke-width="1.5"/>`;
      h += `<text x="28" y="${ys[idx]+4}" text-anchor="middle" font-size="10" fill="${tc}">${xi[i].toFixed(1)}</text>`;
    });
    h += `<line x1="166" y1="95" x2="212" y2="95" stroke="#22C55E" stroke-width="2"/>`;
    h += `<circle cx="157" cy="95" r="22" fill="rgba(91,91,214,0.2)" stroke="#5B5BD6" stroke-width="2"/>`;
    h += `<text x="157" y="91" text-anchor="middle" font-size="9" fill="${tc}">Σ</text>`;
    h += `<text x="157" y="103" text-anchor="middle" font-size="8" fill="${tc}">${z.toFixed(1)}</text>`;
    h += `<circle cx="222" cy="95" r="15" fill="${out > 0.5 ? 'rgba(34,197,94,0.35)' : 'rgba(239,68,68,0.2)'}" stroke="${out > 0.5 ? '#22C55E' : '#EF4444'}" stroke-width="1.5"/>`;
    h += `<text x="222" y="99" text-anchor="middle" font-size="9" fill="${tc}">${out.toFixed(2)}</text>`;
    svg.innerHTML = h;
  }

  div.querySelectorAll('input[type=range], select').forEach(el => el.addEventListener('input', update));
  update();
  return div;
}

// ── 3. Overfitting Curve ──────────────────────────────────────────────────────
function overfitCurve(config) {
  const title = config.title || 'Overfitting vs Underfitting';
  const div = makeWidget(title, '📈', `
    <div class="overfit-layout">
      <canvas class="overfit-canvas" width="330" height="210"></canvas>
      <div class="overfit-controls">
        <label class="ctrl-label">Model complexity: <strong class="oc-val">50</strong></label>
        <input type="range" class="ctrl-range oc-range" min="1" max="100" value="50">
        <div class="overfit-legend">
          <span><span class="leg-dot" style="background:#5B5BD6"></span>Train accuracy</span>
          <span><span class="leg-dot" style="background:#22C55E"></span>Validation accuracy</span>
        </div>
        <div class="overfit-status" id="oc-status"></div>
        <div class="overfit-tip">Drag the slider and watch the gap grow</div>
      </div>
    </div>
  `);

  const canvas = div.querySelector('.overfit-canvas');
  const ctx = canvas.getContext('2d');
  const range = div.querySelector('.oc-range');
  const valEl = div.querySelector('.oc-val');
  const statusEl = div.querySelector('#oc-status');

  const trainAcc = c => 0.48 + (c / 100) * 0.5;
  const valAcc = c => {
    const peak = 0.88, pc = 43;
    if (c <= pc) return 0.45 + (c / pc) * (peak - 0.45);
    return peak - ((c - pc) / 57) * 0.38;
  };

  function draw() {
    const dark = isDark(), c = parseInt(range.value);
    const W = 330, H = 210;
    const pad = {l:38, r:18, t:16, b:32};
    const gW = W-pad.l-pad.r, gH = H-pad.t-pad.b;

    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = dark ? '#1a1a2e' : '#f0f1ff';
    ctx.fillRect(0,0,W,H);

    // Overfitting zone
    const ozX = pad.l + (43/100)*gW;
    ctx.fillStyle = dark ? 'rgba(239,68,68,0.06)' : 'rgba(239,68,68,0.05)';
    ctx.fillRect(ozX, pad.t, gW-(ozX-pad.l), gH);

    // Grid
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 1;
    for (let i=0;i<=4;i++) {
      const y = pad.t + (i/4)*gH;
      ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
      const pct = (1-i/4)*100;
      ctx.fillStyle = dark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)';
      ctx.font = '9px system-ui';
      ctx.fillText(pct+'%', 2, y+3);
    }

    const plot = (fn, color) => {
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      for (let xi=0;xi<=100;xi++) {
        const px = pad.l+(xi/100)*gW, py = pad.t+(1-fn(xi))*gH;
        xi===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
      }
      ctx.stroke();
    };
    plot(trainAcc, '#5B5BD6');
    plot(valAcc, '#22C55E');

    // Current line
    const cx2 = pad.l+(c/100)*gW;
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.2)';
    ctx.setLineDash([4,4]); ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cx2,pad.t); ctx.lineTo(cx2,pad.t+gH); ctx.stroke();
    ctx.setLineDash([]);

    const dot = (fn, color) => {
      const px=cx2, py=pad.t+(1-fn(c))*gH;
      ctx.fillStyle=color; ctx.beginPath(); ctx.arc(px,py,5.5,0,Math.PI*2); ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
      ctx.fillStyle=dark?'#ECECF4':'#1C1F2E'; ctx.font='bold 9.5px system-ui';
      ctx.fillText((fn(c)*100).toFixed(0)+'%', px+9, py+4);
    };
    dot(trainAcc, '#5B5BD6');
    dot(valAcc, '#22C55E');

    // X axis labels
    ctx.fillStyle = dark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.35)';
    ctx.font = '9.5px system-ui';
    ctx.fillText('Simple', pad.l, pad.t+gH+20);
    ctx.fillText('Complex', W-pad.r-38, pad.t+gH+20);
    ctx.fillText('↑ Overfitting zone', ozX+4, pad.t+12);

    // Status
    valEl.textContent = c;
    const gap = trainAcc(c)-valAcc(c);
    if (c<20) { statusEl.textContent='⚠️ Underfitting — model too simple'; statusEl.style.color='var(--orange)'; }
    else if (gap>0.18) { statusEl.textContent='🔴 Overfitting — memorizing training data'; statusEl.style.color='var(--red)'; }
    else { statusEl.textContent='✅ Good fit — generalizes well'; statusEl.style.color='var(--green)'; }
  }

  range.addEventListener('input', draw);
  requestAnimationFrame(draw);
  return div;
}

// ── 4. Attention Heatmap ──────────────────────────────────────────────────────
function attentionHeatmap(config) {
  const words = config.words || ['The','cat','sat','on','the','mat'];
  const title = config.title || 'Attention Mechanism';
  const weights = config.weights || [
    [0.60,0.12,0.08,0.07,0.08,0.05],
    [0.10,0.55,0.18,0.07,0.06,0.04],
    [0.06,0.28,0.42,0.09,0.10,0.05],
    [0.05,0.06,0.10,0.68,0.07,0.04],
    [0.12,0.10,0.08,0.12,0.48,0.10],
    [0.07,0.28,0.12,0.08,0.08,0.37],
  ];

  let sel = 1;
  const div = makeWidget(title, '🔍', `
    <div class="attn-layout">
      <div class="attn-instruction">Click a word to see what it pays attention to:</div>
      <div class="attn-tokens" id="attn-tokens"></div>
      <div class="attn-grid-wrap">
        <div class="attn-grid" id="attn-grid" style="grid-template-columns:repeat(${words.length},1fr)"></div>
      </div>
      <div class="attn-explain" id="attn-explain"></div>
    </div>
  `);

  function render() {
    const tokensEl = div.querySelector('#attn-tokens');
    tokensEl.innerHTML = words.map((w,i) =>
      `<span class="attn-token${i===sel?' active':''}" data-i="${i}">${w}</span>`
    ).join('');
    tokensEl.querySelectorAll('.attn-token').forEach(t =>
      t.addEventListener('click', () => { sel=parseInt(t.dataset.i); render(); })
    );

    const grid = div.querySelector('#attn-grid');
    grid.innerHTML = '';
    // Column headers
    words.forEach(w => {
      const h = document.createElement('div');
      h.className = 'attn-col-hdr';
      h.textContent = w;
      grid.appendChild(h);
    });
    // Row cells
    const row = weights[sel];
    words.forEach((w, i) => {
      const cell = document.createElement('div');
      cell.className = 'attn-cell';
      const v = row[i];
      cell.style.background = `rgba(91,91,214,${0.06 + v * 0.9})`;
      cell.textContent = v.toFixed(2);
      cell.style.color = v > 0.38 ? '#fff' : 'var(--text)';
      cell.style.fontWeight = v > 0.38 ? '700' : '';
      grid.appendChild(cell);
    });

    const maxI = row.indexOf(Math.max(...row));
    div.querySelector('#attn-explain').innerHTML =
      `<strong>"${words[sel]}"</strong> attends most to <strong>"${words[maxI]}"</strong> (${(row[maxI]*100).toFixed(0)}% weight). ` +
      `Try clicking other words to see different patterns.`;
  }

  render();
  return div;
}

// ── 5. Live Tokenizer ─────────────────────────────────────────────────────────
function tokenizerLive(config) {
  const title = config.title || 'Live Tokenizer';
  const defaultText = config.default || 'machine learning is amazing';

  const commonWords = new Set(['the','a','is','in','on','at','to','of','and','it','for',
    'machine','learning','deep','neural','network','model','train','data','loss',
    'gradient','weight','bias','epoch','batch','token','embed','class','label',
    'feature','input','output','layer','hidden','amazing','great','good','bad',
    'backprop','softmax','attention','transformer','encoder','decoder','sigmoid',
    'relu','dropout','overfit','underfit','accuracy','precision','recall']);

  const COLORS = [
    ['#EEF2FF','#4338CA'],['#FEF2F2','#BE123C'],['#F0FDF4','#15803D'],
    ['#FFFBEB','#92400E'],['#F0F9FF','#0369A1'],['#FDF2F8','#9D174D'],
    ['#FAF5FF','#6D28D9'],
  ];

  function tokenize(text) {
    if (!text.trim()) return [];
    const tokens = [];
    text.split(/(\s+)/).forEach(part => {
      if (!part.trim()) { if (part) tokens.push({text: '·', space: true}); return; }
      const lower = part.toLowerCase();
      if (commonWords.has(lower)) {
        tokens.push({text: part});
      } else if (part.length > 7) {
        const s = Math.ceil(part.length * 0.55);
        tokens.push({text: part.slice(0,s)});
        tokens.push({text: '##'+part.slice(s), sub: true});
      } else if (part.length > 4) {
        tokens.push({text: part.slice(0,-2)});
        tokens.push({text: '##'+part.slice(-2), sub: true});
      } else {
        tokens.push({text: part});
      }
    });
    return tokens;
  }

  const div = makeWidget(title, '🔤', `
    <div class="tokenizer-layout">
      <input class="tokenizer-input" type="text" placeholder="Type any sentence..." value="${defaultText}">
      <div class="tokenizer-tokens" id="tok-out"></div>
      <div class="tokenizer-stats" id="tok-stats"></div>
      <div class="tokenizer-tip">## = continuation of a word (subword token)</div>
    </div>
  `);

  function render() {
    const text = div.querySelector('.tokenizer-input').value;
    const toks = tokenize(text);
    const out = div.querySelector('#tok-out');
    const stats = div.querySelector('#tok-stats');
    let ci = 0;
    out.innerHTML = toks.filter(t => !t.space).map(t => {
      const [bg, fg] = COLORS[ci++ % COLORS.length];
      return `<span class="tok-chip" style="background:${bg};color:${fg}">${
        t.text.replace('##', '<span style="opacity:0.45">##</span>')
      }</span>`;
    }).join('');
    const realToks = toks.filter(t => !t.space);
    const words = text.trim().split(/\s+/).filter(Boolean).length;
    stats.innerHTML = `<strong>${realToks.length}</strong> tokens &nbsp;·&nbsp; <strong>${text.length}</strong> chars &nbsp;·&nbsp; <strong>${words ? (realToks.length/words).toFixed(1) : 0}</strong> tokens/word`;
  }

  div.querySelector('.tokenizer-input').addEventListener('input', render);
  render();
  return div;
}

// ── 6. Convolution Stepper ────────────────────────────────────────────────────
function convStepper(config) {
  const title = config.title || 'Convolution Step-by-Step';
  const input = config.input || [
    [1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]
  ];
  const kernel = config.kernel || [[1,0,-1],[0,1,0],[-1,0,1]];

  // Pre-compute output
  const output = [];
  for (let r=0;r<3;r++) for (let c=0;c<3;c++) {
    let s=0;
    for (let kr=0;kr<3;kr++) for (let kc=0;kc<3;kc++) s+=input[r+kr][c+kc]*kernel[kr][kc];
    output.push(s);
  }

  let step = 0;
  const div = makeWidget(title, '🔲', `
    <div class="conv-layout">
      <div class="conv-section">
        <div class="conv-label">Input (5×5)</div>
        <div class="conv-grid input-grid" id="cg-input" style="grid-template-columns:repeat(5,1fr)"></div>
      </div>
      <div class="conv-op">★</div>
      <div class="conv-section">
        <div class="conv-label">Filter (3×3)</div>
        <div class="conv-grid kernel-grid" id="cg-kernel" style="grid-template-columns:repeat(3,1fr)"></div>
      </div>
      <div class="conv-op">=</div>
      <div class="conv-section">
        <div class="conv-label">Output (3×3)</div>
        <div class="conv-grid output-grid" id="cg-output" style="grid-template-columns:repeat(3,1fr)"></div>
      </div>
    </div>
    <div class="conv-footer">
      <button class="widget-btn widget-btn-sec" id="cv-prev">← Back</button>
      <span class="conv-step-info" id="cv-info">Step 1/9 — Position (row 0, col 0)</span>
      <button class="widget-btn" id="cv-next">Next →</button>
    </div>
    <div class="conv-calc" id="cv-calc"></div>
  `);

  function render() {
    const row = Math.floor(step/3), col = step%3;
    const hi = 'rgba(91,91,214,0.22)';
    const dark = isDark();

    // Input grid
    const ig = div.querySelector('#cg-input');
    ig.innerHTML = '';
    for (let r=0;r<5;r++) for (let c=0;c<5;c++) {
      const cell = document.createElement('div');
      cell.className = 'conv-cell';
      const isHi = r>=row && r<row+3 && c>=col && c<col+3;
      cell.style.background = isHi ? hi : '';
      cell.style.fontWeight = isHi ? '700' : '';
      cell.textContent = input[r][c];
      ig.appendChild(cell);
    }

    // Kernel grid
    const kg = div.querySelector('#cg-kernel');
    kg.innerHTML = '';
    for (let r=0;r<3;r++) for (let c=0;c<3;c++) {
      const cell = document.createElement('div');
      cell.className = 'conv-cell kern-cell';
      cell.textContent = kernel[r][c];
      kg.appendChild(cell);
    }

    // Output grid
    const og = div.querySelector('#cg-output');
    og.innerHTML = '';
    for (let i=0;i<9;i++) {
      const cell = document.createElement('div');
      cell.className = 'conv-cell';
      if (i<step) { cell.textContent=output[i]; cell.style.background='rgba(34,197,94,0.18)'; }
      else if (i===step) { cell.textContent=output[i]; cell.style.background=hi; cell.style.fontWeight='700'; }
      else { cell.textContent='?'; cell.style.opacity='0.25'; }
      og.appendChild(cell);
    }

    div.querySelector('#cv-info').textContent = `Step ${step+1}/9 — Position (row ${row}, col ${col})`;

    // Calculation breakdown
    let parts = [], sum = 0;
    for (let kr=0;kr<3;kr++) for (let kc=0;kc<3;kc++) {
      const v = input[row+kr][col+kc] * kernel[kr][kc];
      parts.push(`(${input[row+kr][col+kc]}×${kernel[kr][kc]})`);
      sum += v;
    }
    div.querySelector('#cv-calc').innerHTML = `<span class="cv-parts">${parts.join(' + ')}</span> = <strong>${sum}</strong>`;
  }

  div.querySelector('#cv-prev').addEventListener('click', () => { step=Math.max(0,step-1); render(); });
  div.querySelector('#cv-next').addEventListener('click', () => { step=Math.min(8,step+1); render(); });
  render();
  return div;
}

// ── 7. Neural Network Diagram ─────────────────────────────────────────────────
function neuronDiagram(config) {
  const title = config.title || 'Neural Network Forward Pass';
  const layers = config.layers || [3, 4, 4, 2];
  const labelNames = config.labels || ['Input', ...Array(layers.length-2).fill('Hidden'), 'Output'];

  const div = makeWidget(title, '🕸️', `
    <div class="nn-wrap">
      <svg class="nn-svg" id="nn-svg" viewBox="0 0 400 220" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="nn-legend" id="nn-legend"></div>
      <div class="nn-footer-btns">
        <button class="widget-btn" id="nn-animate">▶ Animate forward pass</button>
        <button class="widget-btn widget-btn-sec" id="nn-reset-btn">Reset</button>
      </div>
    </div>
  `);

  const W = 400, H = 220;
  const numL = layers.length;
  const xStep = W / (numL + 1);

  const nodePos = layers.map((n, li) => {
    const x = xStep * (li + 1);
    const yStep = H / (n + 1);
    return Array.from({length: n}, (_, ni) => ({x, y: yStep*(ni+1), li, ni}));
  });

  let animLayer = -1, animHandle = null;

  function drawNet(highlightLayer = -1) {
    const dark = isDark();
    const tc = dark ? '#C9C9DA' : '#3A3F52';
    const dimLine = dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)';
    const svg = div.querySelector('#nn-svg');
    let h = '';

    // Connections
    for (let li=0; li<numL-1; li++) {
      const active = li === highlightLayer;
      nodePos[li].forEach(from => nodePos[li+1].forEach(to => {
        h += `<line x1="${from.x.toFixed(1)}" y1="${from.y.toFixed(1)}" x2="${to.x.toFixed(1)}" y2="${to.y.toFixed(1)}"
          stroke="${active ? '#5B5BD6' : dimLine}" stroke-width="${active ? 1.6 : 0.8}"
          opacity="${active ? 0.55 : 1}"/>`;
      }));
    }

    const layerColors = ['#22C55E', '#5B5BD6', '#7C3AED', '#F97316'];
    // Nodes
    nodePos.forEach((layer, li) => {
      const active = li === highlightLayer || li === highlightLayer+1;
      const color = layerColors[Math.min(li, layerColors.length-1)];
      layer.forEach(node => {
        h += `<circle cx="${node.x.toFixed(1)}" cy="${node.y.toFixed(1)}" r="${li===0||li===numL-1?14:12}"
          fill="${color}" fill-opacity="${active ? 0.95 : 0.28}"
          stroke="${active ? color : dimLine}" stroke-width="${active ? 2 : 1}"/>`;
        if (active && li > 0) {
          const v = (Math.random()*0.8+0.1).toFixed(2);
          h += `<text x="${node.x.toFixed(1)}" y="${(node.y+4).toFixed(1)}" text-anchor="middle" font-size="9" fill="${dark?'#fff':'#fff'}" opacity="0.9">${v}</text>`;
        }
      });
    });

    // Layer labels
    nodePos.forEach((_, li) => {
      const x = xStep*(li+1);
      h += `<text x="${x.toFixed(1)}" y="215" text-anchor="middle" font-size="10" fill="${tc}" opacity="0.6">${labelNames[li]||''}</text>`;
      h += `<text x="${x.toFixed(1)}" y="15" text-anchor="middle" font-size="9" fill="${tc}" opacity="0.5">${layers[li]} nodes</text>`;
    });

    svg.innerHTML = h;
  }

  function stopAnim() {
    if (animHandle) { clearInterval(animHandle); animHandle = null; }
    div.querySelector('#nn-animate').textContent = '▶ Animate forward pass';
  }

  div.querySelector('#nn-animate').addEventListener('click', () => {
    if (animHandle) { stopAnim(); return; }
    animLayer = -1;
    div.querySelector('#nn-animate').textContent = '⏸ Stop';
    animHandle = setInterval(() => {
      animLayer++;
      if (animLayer >= numL) {
        animLayer = numL - 1;
        drawNet(animLayer);
        stopAnim();
        return;
      }
      drawNet(animLayer);
    }, 700);
  });

  div.querySelector('#nn-reset-btn').addEventListener('click', () => {
    stopAnim();
    animLayer = -1;
    drawNet(-1);
  });

  const legend = div.querySelector('#nn-legend');
  legend.innerHTML = ['#22C55E','#5B5BD6','#F97316'].map((c,i) =>
    `<span><span class="leg-dot" style="background:${c}"></span>${['Input','Hidden','Output'][i]}</span>`
  ).join('');

  drawNet(-1);
  return div;
}
