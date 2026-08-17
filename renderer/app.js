import { preprocessWidgets, renderWidgets } from './widgets.js';

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  courses: null,
  currentLesson: null, // { trackId, lessonId }
  pyodide: null,
  pyodideLoading: false,
  pyodideReady: false,
  xp: 0,
  streak: 0,
  completed: new Set(),
  userName: '',
  darkMode: false,
  sidebarCollapsed: false,
  lastLesson: null,
  // Spaced repetition: { [questionKey]: { ef: float, interval: days, due: ISO-date, correct: int, total: int } }
  sr: {},
};

// ─── Storage ─────────────────────────────────────────────────────────────────
function progressToJSON() {
  return JSON.stringify({
    completed: [...state.completed],
    xp: state.xp,
    streak: state.streak,
    lastDate: new Date().toDateString(),
    userName: state.userName,
    darkMode: state.darkMode,
    sidebarCollapsed: state.sidebarCollapsed,
    lastLesson: state.lastLesson,
    sr: state.sr,
  });
}

function applyProgressJSON(json) {
  try {
    const d = JSON.parse(json);
    if (d.completed) state.completed = new Set(d.completed);
    if (d.xp != null) state.xp = d.xp;
    if (d.userName) state.userName = d.userName;
    const today = new Date().toDateString();
    const yesterday = new Date(Date.now() - 86400000).toDateString();
    if (d.lastDate === today || d.lastDate === yesterday) state.streak = d.streak || 0;
    else state.streak = 0;
    if (d.darkMode != null) state.darkMode = d.darkMode;
    if (d.sidebarCollapsed != null) state.sidebarCollapsed = d.sidebarCollapsed;
    if (d.lastLesson) state.lastLesson = d.lastLesson;
    if (d.sr) state.sr = d.sr;
  } catch (_) {}
}

// ─── Spaced Repetition (SM-2 simplified) ──────────────────────────────────────
function srUpdate(key, correct) {
  const now = state.sr[key] || { ef: 2.5, interval: 1, correct: 0, total: 0 };
  now.total = (now.total || 0) + 1;
  if (correct) {
    now.correct = (now.correct || 0) + 1;
    now.interval = now.interval < 1 ? 1 : now.interval < 6 ? 6 : Math.round(now.interval * now.ef);
    now.ef = Math.max(1.3, now.ef + 0.1 - (1 - 1) * (0.08 + (1 - 1) * 0.02));
  } else {
    now.interval = 1;
    now.ef = Math.max(1.3, now.ef - 0.2);
  }
  const due = new Date();
  due.setDate(due.getDate() + now.interval);
  now.due = due.toISOString().slice(0, 10);
  state.sr[key] = now;
  saveProgress();
}

function srDueCount() {
  const today = new Date().toISOString().slice(0, 10);
  return Object.values(state.sr).filter(v => v.due && v.due <= today).length;
}

function saveProgress() {
  const json = progressToJSON();
  // Write to file (persists across reinstalls)
  window.api.saveData(json);
  // Also keep localStorage as fast cache
  localStorage.setItem('ioai_progress', json);
}

async function loadProgress() {
  // File storage is source of truth; localStorage is fallback
  let json = await window.api.loadData();
  if (!json) json = localStorage.getItem('ioai_progress');
  // Legacy fallback for older installs
  if (!json) {
    const c = localStorage.getItem('ioai_completed');
    if (c) {
      state.completed = new Set(JSON.parse(c));
      state.xp = parseInt(localStorage.getItem('ioai_xp') || '0');
      state.userName = localStorage.getItem('ioai_name') || '';
      const lastDate = localStorage.getItem('ioai_last_date');
      const savedStreak = parseInt(localStorage.getItem('ioai_streak') || '0');
      const yesterday = new Date(Date.now() - 86400000).toDateString();
      const today = new Date().toDateString();
      if (lastDate === today || lastDate === yesterday) state.streak = savedStreak;
      else state.streak = 0;
      saveProgress(); // migrate to new format
    }
    return;
  }
  applyProgressJSON(json);
}

// ─── Glossary Tooltips ────────────────────────────────────────────────────────
const GLOSSARY = {
  'gradient': 'The direction of steepest increase of the loss. We move in the opposite direction during training.',
  'gradients': 'Plural of gradient - the slopes that tell us how to update each weight.',
  'backpropagation': 'Algorithm that computes gradients by working backwards through the network, layer by layer.',
  'loss function': 'Measures how wrong the model\'s predictions are. Training tries to make this as small as possible.',
  'overfitting': 'When the model memorizes training data but fails on new examples it hasn\'t seen.',
  'underfitting': 'When the model is too simple to capture the pattern - bad on both training and test data.',
  'regularization': 'A penalty added during training to keep weights small and prevent overfitting (e.g. L1, L2).',
  'learning rate': 'How big each step is during gradient descent. Too high = unstable, too low = very slow.',
  'epoch': 'One full pass through the entire training dataset.',
  'batch': 'A small chunk of training examples processed together in one forward + backward pass.',
  'weight': 'A learnable number the model adjusts during training to improve its predictions.',
  'bias': 'An extra learnable offset added to a neuron\'s output before the activation.',
  'activation function': 'Adds non-linearity to the network so it can learn complex patterns (e.g. ReLU, sigmoid).',
  'softmax': 'Converts raw scores into probabilities that sum to 1. Used in classification outputs.',
  'ReLU': 'Rectified Linear Unit: max(0, x). The most common activation - simple and effective.',
  'sigmoid': 'Maps any number to a value between 0 and 1. Useful for binary classification.',
  'tokenization': 'Breaking text into smaller pieces (tokens) that the model can process.',
  'embedding': 'A dense vector of numbers that represents a word or token in a way a model can use.',
  'embeddings': 'Dense vector representations of words or tokens that capture meaning and relationships.',
  'attention': 'A mechanism that lets the model decide which parts of the input to focus on.',
  'transformer': 'A neural network architecture built entirely on attention, with no recurrence. Powers BERT, GPT, etc.',
  'cross-entropy': 'A loss function for classification that penalizes wrong confident predictions more harshly.',
  'precision': 'Of all the positive predictions made, how many were actually positive?',
  'recall': 'Of all the actual positives in the data, how many did the model find?',
  'F1 score': 'Harmonic mean of precision and recall. Useful when classes are imbalanced.',
  'accuracy': 'The fraction of all predictions that were correct.',
  'decision tree': 'A model that makes predictions by asking a series of yes/no questions about the features.',
  'random forest': 'An ensemble of many decision trees - reduces overfitting compared to a single tree.',
  'gradient boosting': 'Builds trees one at a time, each correcting the errors of the previous ones.',
  'ensemble': 'Combining multiple models to get better predictions than any single model alone.',
  'normalization': 'Scaling values to a common range, e.g. 0 to 1.',
  'standardization': 'Rescaling data to have mean 0 and standard deviation 1.',
  'PCA': 'Principal Component Analysis - reduces the number of features while keeping as much variation as possible.',
  'dropout': 'Randomly disables neurons during training to prevent any single neuron from being over-relied on.',
  'batch normalization': 'Normalizes the activations within each mini-batch to stabilize and speed up training.',
  'fine-tuning': 'Starting from a pre-trained model and training it further on your specific task.',
  'transfer learning': 'Using knowledge a model gained from one task to help with a different task.',
  'hyperparameter': 'A setting you choose before training starts, like learning rate or number of layers.',
  'perceptron': 'The simplest neural network unit - takes inputs, multiplies by weights, sums, applies a threshold.',
  'convolution': 'A filter that slides over an image to detect local patterns like edges or textures.',
};

function addGlossaryTooltips(container) {
  const terms = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);
  const escaped = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      let el = node.parentElement;
      while (el && el !== container) {
        if (['CODE', 'PRE', 'SCRIPT', 'STYLE', 'A', 'H1', 'H2', 'H3'].includes(el.tagName)
            || el.classList.contains('gl')) return NodeFilter.FILTER_REJECT;
        el = el.parentElement;
      }
      return NodeFilter.FILTER_ACCEPT;
    }
  });

  const textNodes = [];
  let node;
  while ((node = walker.nextNode())) {
    if (pattern.test(node.textContent)) textNodes.push(node);
    pattern.lastIndex = 0;
  }

  for (const textNode of textNodes) {
    const text = textNode.textContent;
    pattern.lastIndex = 0;
    if (!pattern.test(text)) continue;
    pattern.lastIndex = 0;

    const frag = document.createDocumentFragment();
    let last = 0, match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > last) frag.appendChild(document.createTextNode(text.slice(last, match.index)));
      const key = Object.keys(GLOSSARY).find(k => k.toLowerCase() === match[0].toLowerCase()) || match[0];
      const span = document.createElement('span');
      span.className = 'gl';
      span.dataset.tip = GLOSSARY[key] || '';
      span.textContent = match[0];
      frag.appendChild(span);
      last = match.index + match[0].length;
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    textNode.parentNode.replaceChild(frag, textNode);
  }
}

// ─── Theme ────────────────────────────────────────────────────────────────────
function applyTheme() {
  document.documentElement.classList.toggle('dark', state.darkMode);
}

function toggleDarkMode() {
  state.darkMode = !state.darkMode;
  applyTheme();
  saveProgress();
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  document.getElementById('sidebar').classList.toggle('collapsed', state.sidebarCollapsed);
  saveProgress();
}

// ─── Error parsing ────────────────────────────────────────────────────────────
function parseError(raw) {
  const lines = raw.split('\n').filter(l => l.trim());
  const errorLine = [...lines].reverse().find(l =>
    /^(\w*Error|SyntaxError|IndentationError|AssertionError)/.test(l.trim())
  );
  const lineRef = [...lines].reverse().find(l => l.trim().startsWith('File') && l.includes('line'));
  const lineNum = lineRef?.match(/line (\d+)/)?.[1];
  if (errorLine) return (lineNum ? `Line ${lineNum}: ` : '') + errorLine.trim();
  return lines[lines.length - 1]?.trim() || raw;
}

// ─── Celebrations ─────────────────────────────────────────────────────────────
function celebrate(anchorEl) {
  const colors = ['#4f6ef7','#10b981','#f97316','#f59e0b','#8b5cf6','#ef4444','#06b6d4'];
  const rect = anchorEl?.getBoundingClientRect?.() ?? { left: window.innerWidth/2, top: window.innerHeight/2, width: 0, height: 0 };
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  for (let i = 0; i < 40; i++) {
    const p = document.createElement('div');
    p.style.cssText = `position:fixed;pointer-events:none;z-index:9999;border-radius:${Math.random()>.5?'50%':'3px'};width:${6+Math.random()*7}px;height:${6+Math.random()*7}px;background:${colors[i%colors.length]};left:${cx}px;top:${cy}px;`;
    document.body.appendChild(p);
    const angle = Math.random() * Math.PI * 2;
    const v = 80 + Math.random() * 220;
    p.animate([
      { transform: 'translate(0,0) rotate(0deg)', opacity: 1 },
      { transform: `translate(${Math.cos(angle)*v}px,${Math.sin(angle)*v - 60}px) rotate(${Math.random()*720}deg)`, opacity: 0 }
    ], { duration: 700 + Math.random()*600, easing: 'cubic-bezier(.25,.46,.45,.94)', fill: 'forwards' }).onfinish = () => p.remove();
  }
}

function showXPGain(amount, anchorEl) {
  const rect = anchorEl?.getBoundingClientRect?.() ?? { left: window.innerWidth/2, top: window.innerHeight/2, width: 0 };
  const el = document.createElement('div');
  el.className = 'xp-popup';
  el.textContent = `+${amount} XP`;
  el.style.left = (rect.left + (rect.width||0)/2) + 'px';
  el.style.top = rect.top + 'px';
  document.body.appendChild(el);
  el.animate([
    { transform: 'translateY(0) translateX(-50%) scale(0.8)', opacity: 0 },
    { transform: 'translateY(-10px) translateX(-50%) scale(1.1)', opacity: 1, offset: 0.2 },
    { transform: 'translateY(-55px) translateX(-50%) scale(1)', opacity: 0 }
  ], { duration: 1200, fill: 'forwards' }).onfinish = () => el.remove();
}

// ─── Line numbers ─────────────────────────────────────────────────────────────
function updateLineNums(textarea, numEl) {
  const count = textarea.value.split('\n').length;
  numEl.innerHTML = Array.from({ length: count }, (_, i) => `<div>${i + 1}</div>`).join('');
  numEl.scrollTop = textarea.scrollTop;
}

// ─── Pyodide ─────────────────────────────────────────────────────────────────
function updatePyodideStatus(status) {
  const el = document.getElementById('pyodide-status');
  const label = document.getElementById('py-label');
  el.className = status;
  if (status === 'loading') label.textContent = 'Loading Python…';
  else if (status === 'ready') label.textContent = 'Python ready';
  else label.textContent = 'Python';
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function ensurePyodide() {
  if (state.pyodideReady) return state.pyodide;
  if (state.pyodideLoading) {
    while (!state.pyodideReady) await sleep(100);
    return state.pyodide;
  }
  state.pyodideLoading = true;
  updatePyodideStatus('loading');
  try {
    const py = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/' });
    state.pyodide = py;
    state.pyodideReady = true;
    updatePyodideStatus('ready');
    return py;
  } catch (e) {
    state.pyodideLoading = false;
    updatePyodideStatus('');
    return null;
  }
}

async function runPython(code) {
  const py = await ensurePyodide();
  if (!py) return { output: '', error: 'Python engine not available. Check your internet connection.' };
  try {
    py.runPython(`
import sys, io
_cap = io.StringIO()
sys.stdout = _cap
`);
    await py.runPythonAsync(code);
    const out = py.runPython(`sys.stdout = sys.__stdout__\n_cap.getvalue()`);
    return { output: out || '(no output)', error: null };
  } catch (e) {
    try { py.runPython('import sys; sys.stdout = sys.__stdout__'); } catch (_) {}
    return { output: '', error: String(e).replace('PythonError: Traceback', 'Error') };
  }
}

// ─── Utilities ───────────────────────────────────────────────────────────────
function toast(msg, duration = 2500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), duration);
}

function diffBadgeClass(diff) {
  return { beginner: 'badge-bgn', intermediate: 'badge-int', advanced: 'badge-adv' }[diff] || 'badge-bgn';
}

function diffLabel(diff) {
  return { beginner: 'Beginner', intermediate: 'Intermediate', advanced: 'Advanced' }[diff] || (diff || 'Beginner');
}

// ─── Course Data ─────────────────────────────────────────────────────────────
async function loadCourses() {
  const raw = await window.api.readFile('content/courses.json');
  state.courses = JSON.parse(raw);
}

function getNextLesson(trackId, lessonId) {
  const track = state.courses.tracks.find(t => t.id === trackId);
  if (!track) return null;
  const idx = track.lessons.findIndex(l => l.id === lessonId);
  if (idx < 0 || idx >= track.lessons.length - 1) return null;
  return { trackId, lesson: track.lessons[idx + 1] };
}

function getPrevLesson(trackId, lessonId) {
  const track = state.courses.tracks.find(t => t.id === trackId);
  if (!track) return null;
  const idx = track.lessons.findIndex(l => l.id === lessonId);
  if (idx <= 0) return null;
  return { trackId, lesson: track.lessons[idx - 1] };
}

function getTrackProgress(trackId) {
  const track = state.courses.tracks.find(t => t.id === trackId);
  if (!track || !track.lessons.length) return 0;
  const done = track.lessons.filter(l => state.completed.has(l.id)).length;
  return Math.round((done / track.lessons.length) * 100);
}

function getOverallProgress() {
  const all = state.courses.tracks.flatMap(t => t.lessons);
  if (!all.length) return 0;
  return Math.round((state.completed.size / all.length) * 100);
}

// ─── Navigation ──────────────────────────────────────────────────────────────
function navigate(viewName) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  const view = document.getElementById(`view-${viewName}`);
  if (view) view.classList.add('active');
  document.querySelectorAll('.nav-item[data-view]').forEach(item => {
    item.classList.toggle('active', item.dataset.view === viewName);
  });
}

// ─── Markdown Config ─────────────────────────────────────────────────────────
function configureMarked() {
  marked.setOptions({
    gfm: true,
    breaks: false,
    highlight(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    }
  });
}

function renderMath(container) {
  renderMathInElement(container, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  });
}

function parseMarkdownMeta(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { meta: {}, body: raw };
  const meta = {};
  match[1].split('\n').forEach(line => {
    const [k, ...rest] = line.split(':');
    if (k) meta[k.trim()] = rest.join(':').trim();
  });
  return { meta, body: match[2] };
}

// ─── Code Block Enhancement ──────────────────────────────────────────────────
function enhanceCodeBlocks(container) {
  container.querySelectorAll('pre code').forEach(codeEl => {
    const pre = codeEl.parentElement;
    const lang = (codeEl.className.match(/language-(\w+)/) || [])[1] || 'text';
    const isPython = lang === 'python';

    const toolbar = document.createElement('div');
    toolbar.className = 'code-toolbar';
    toolbar.innerHTML = `
      <span class="code-lang-label">${lang}</span>
      <div class="code-actions">
        <button class="btn-copy">Copy</button>
        ${isPython ? '<button class="btn-run-code">▶ Run</button>' : ''}
      </div>
    `;
    pre.insertBefore(toolbar, codeEl);

    toolbar.querySelector('.btn-copy').addEventListener('click', () => {
      navigator.clipboard.writeText(codeEl.textContent).then(() => {
        const btn = toolbar.querySelector('.btn-copy');
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
      }).catch(() => toast('Copy failed'));
    });

    if (isPython) {
      const outputArea = document.createElement('div');
      outputArea.className = 'code-output-area';
      pre.appendChild(outputArea);

      toolbar.querySelector('.btn-run-code').addEventListener('click', async e => {
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = '⟳ Running…';
        outputArea.className = 'code-output-area visible';
        outputArea.textContent = 'Running…';
        const { output, error } = await runPython(codeEl.textContent);
        btn.disabled = false;
        btn.textContent = '▶ Run';
        if (error) {
          outputArea.className = 'code-output-area visible error';
          const short = parseError(error);
          outputArea.innerHTML = `<strong>${short}</strong><details style="margin-top:6px"><summary style="font-size:11px;cursor:pointer;color:#9ca3af">Full traceback</summary><pre style="margin-top:6px;font-size:11px;color:#9ca3af;white-space:pre-wrap">${error}</pre></details>`;
        } else {
          outputArea.className = 'code-output-area visible';
          outputArea.textContent = output;
        }
      });
    }
  });
}

// ─── Quiz Renderer ────────────────────────────────────────────────────────────
function renderQuiz(quizData, container) {
  if (!quizData?.questions?.length) return;

  const section = document.createElement('div');
  section.className = 'quiz-section';
  section.innerHTML = '<div class="quiz-section-title">Quick check</div>';

  quizData.questions.forEach(q => {
    const letters = ['A', 'B', 'C', 'D', 'E'];
    const card = document.createElement('div');
    card.className = 'quiz-card';
    card.innerHTML = `
      <div class="quiz-q-text">${q.question}</div>
      <div class="quiz-opts">
        ${q.options.map((opt, i) => `
          <div class="quiz-opt" data-idx="${i}">
            <span class="opt-letter">${letters[i]}</span>
            <span>${opt}</span>
          </div>`).join('')}
      </div>
      <div class="quiz-explanation">${q.explanation || ''}</div>
    `;

    const opts = card.querySelectorAll('.quiz-opt');
    const explanation = card.querySelector('.quiz-explanation');

    opts.forEach(opt => {
      opt.addEventListener('click', () => {
        if (opt.classList.contains('disabled')) return;
        const chosen = parseInt(opt.dataset.idx);
        const correct = chosen === q.correct;
        opts.forEach(o => {
          o.classList.add('disabled');
          if (parseInt(o.dataset.idx) === q.correct) o.classList.add('correct');
        });
        if (!correct) opt.classList.add('wrong');
        explanation.classList.add('visible');
        // Spaced repetition tracking
        const qKey = `quiz_${encodeURIComponent(q.question.slice(0,40))}`;
        srUpdate(qKey, correct);
        if (correct) { state.xp += 5; saveProgress(); celebrate(opt); showXPGain(5, opt); }
      });
    });

    section.appendChild(card);
  });

  container.appendChild(section);
}

// ─── Exercise Renderer ────────────────────────────────────────────────────────
function renderExercises(quizData, container, lessonId) {
  if (!quizData?.exercises?.length) return;

  const wrap = document.createElement('div');
  wrap.innerHTML = '<div class="quiz-section-title" style="margin:24px 32px 16px">Practice exercises</div>';

  quizData.exercises.forEach((ex, ei) => {
    const card = document.createElement('div');
    card.className = 'exercise-card';
    card.style.margin = '0 32px 20px';

    card.innerHTML = `
      <div class="exercise-header">
        <div class="exercise-num">Exercise ${ei + 1}</div>
        <div class="exercise-prompt">${ex.prompt}</div>
      </div>
      <div class="exercise-editor">
        <div class="ide-editor-wrap">
          <div class="ide-line-nums" aria-hidden="true"></div>
          <textarea class="editor-textarea" spellcheck="false"></textarea>
        </div>
        <div class="editor-actions">
          <span class="editor-lang-tag">Python</span>
          <button class="btn-check-ex">&#9654; Run &amp; Check</button>
          ${ex.hints?.length ? '<button class="btn-hint-ex">Hint</button>' : ''}
          <button class="btn-sol-ex">Show solution</button>
        </div>
        <div class="hint-area"></div>
        <div class="exercise-result"></div>
      </div>
    `;

    const textarea = card.querySelector('.editor-textarea');
    const numEl = card.querySelector('.ide-line-nums');
    const resultEl = card.querySelector('.exercise-result');
    const hintArea = card.querySelector('.hint-area');
    let hintIdx = 0;

    // Restore saved code or fall back to starter
    const codeKey = lessonId ? `ioai_code_${lessonId}_ex${ei}` : null;
    const savedCode = codeKey ? localStorage.getItem(codeKey) : null;
    textarea.value = savedCode !== null ? savedCode : (ex.starter || '# Write your code here\n');

    updateLineNums(textarea, numEl);
    textarea.addEventListener('input', () => {
      updateLineNums(textarea, numEl);
      if (codeKey) localStorage.setItem(codeKey, textarea.value);
    });
    textarea.addEventListener('scroll', () => { numEl.scrollTop = textarea.scrollTop; });

    textarea.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = textarea.selectionStart;
        textarea.value = textarea.value.substring(0, s) + '    ' + textarea.value.substring(textarea.selectionEnd);
        textarea.selectionStart = textarea.selectionEnd = s + 4;
        updateLineNums(textarea, numEl);
      }
    });

    card.querySelector('.btn-check-ex').addEventListener('click', async () => {
      const btn = card.querySelector('.btn-check-ex');
      btn.disabled = true;
      btn.textContent = 'Running...';
      resultEl.className = 'exercise-result visible output';
      resultEl.textContent = 'Running...';

      let fullCode = textarea.value;
      if (ex.tests?.length) {
        fullCode += '\n' + ex.tests.map(t => `assert ${t}, "Test failed: ${t.replace(/"/g, "'")}"`).join('\n');
      }

      const { output, error } = await runPython(fullCode);
      btn.disabled = false;
      btn.textContent = '▶ Run & Check';

      if (error) {
        const isFail = error.includes('AssertionError') || error.includes('Test failed');
        resultEl.className = 'exercise-result visible fail';
        const short = parseError(error);
        resultEl.innerHTML = isFail
          ? `<div class="err-headline">Not quite - check your logic and try again</div><details class="err-details"><summary>Show error</summary><pre>${error}</pre></details>`
          : `<div class="err-headline">${short}</div><details class="err-details"><summary>Full traceback</summary><pre>${error}</pre></details>`;
      } else {
        resultEl.className = 'exercise-result visible pass';
        resultEl.innerHTML = `<div class="pass-headline">Correct!</div>` + (output && output !== '(no output)' ? `<div class="pass-output">${output}</div>` : '');
        state.xp += 15;
        saveProgress();
        celebrate(resultEl);
        showXPGain(15, resultEl);
        toast('Exercise solved! +15 XP');
      }
    });

    const hintBtn = card.querySelector('.btn-hint-ex');
    if (hintBtn) {
      hintBtn.addEventListener('click', () => {
        if (!ex.hints || hintIdx >= ex.hints.length) { toast('No more hints!'); return; }
        hintArea.textContent = `Hint ${hintIdx + 1}: ${ex.hints[hintIdx]}`;
        hintArea.classList.add('visible');
        hintIdx++;
        if (hintIdx >= ex.hints.length) hintBtn.disabled = true;
      });
    }

    card.querySelector('.btn-sol-ex').addEventListener('click', () => {
      textarea.value = ex.solution;
      toast('Solution shown. Study it, then try on your own!');
    });

    wrap.appendChild(card);
  });

  container.appendChild(wrap);
}

// ─── TOC Builder ─────────────────────────────────────────────────────────────
function setLessonProgress(pct) {
  const p = Math.max(0, Math.min(100, Math.round(pct)));
  document.getElementById('lesson-progress-pct').textContent = `${p}%`;
  document.getElementById('lesson-progress-fill').style.width = `${p}%`;
}

function buildTOC(contentEl) {
  const headings = [...contentEl.querySelectorAll('h2, h3')];
  const tocList = document.getElementById('lesson-toc-list');
  const outlineEl = document.getElementById('lesson-outline');
  tocList.innerHTML = '';
  outlineEl.innerHTML = '';

  let h2Count = 0;
  headings.forEach((h, i) => {
    h.id = `sec-${i}`;
    const isH3 = h.tagName === 'H3';
    if (!isH3) h2Count++;

    const tocItem = document.createElement('div');
    tocItem.className = `toc-item${isH3 ? ' h3' : ''}`;
    if (!isH3) {
      tocItem.innerHTML = `<span class="toc-num">${h2Count}</span><span>${h.textContent}</span>`;
    } else {
      tocItem.innerHTML = `<span>${h.textContent}</span>`;
    }
    tocItem.addEventListener('click', () => h.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    tocList.appendChild(tocItem);

    const outItem = document.createElement('div');
    outItem.className = 'outline-item';
    outItem.textContent = h.textContent;
    outItem.addEventListener('click', () => h.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    outlineEl.appendChild(outItem);
  });

  // Scroll spy
  const tocItems = [...tocList.querySelectorAll('.toc-item')];
  const outItems = [...outlineEl.querySelectorAll('.outline-item')];
  const centerPanel = document.getElementById('lesson-center-panel');

  setLessonProgress(headings.length ? (1 / headings.length) * 100 : 0);

  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const idx = headings.indexOf(entry.target);
      if (idx < 0) return;
      tocItems.forEach((t, i) => {
        t.classList.toggle('active', i === idx);
        t.classList.toggle('done', i < idx);
      });
      outItems.forEach((t, i) => t.classList.toggle('active', i === idx));
      setLessonProgress(((idx + 1) / headings.length) * 100);
    });
  }, { root: centerPanel, threshold: 0.2 });

  headings.forEach(h => obs.observe(h));
}

// ─── Tab Switcher ─────────────────────────────────────────────────────────────
function switchTab(tabName) {
  document.querySelectorAll('.lesson-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `tab-${tabName}`);
  });
}

// ─── Lesson Loader ────────────────────────────────────────────────────────────
async function openLesson(trackId, lessonId) {
  const track = state.courses.tracks.find(t => t.id === trackId);
  const lesson = track?.lessons.find(l => l.id === lessonId);
  if (!track || !lesson) return;

  state.currentLesson = { trackId, lessonId };
  navigate('lesson');

  // Breadcrumb
  const breadcrumb = document.getElementById('lesson-breadcrumb');
  breadcrumb.innerHTML = `
    <span class="bc-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10L12 5 2 10l10 5 10-5z"/><path d="M6 12v5c0 1.1 2.7 2.5 6 2.5s6-1.4 6-2.5v-5"/></svg></span>
    <span class="bc-link">${track.name}</span>
    <span class="bc-sep">›</span>
    <span class="bc-current">${lesson.title}</span>
  `;
  breadcrumb.querySelectorAll('.bc-link').forEach(link => {
    link.addEventListener('click', () => { navigate('roadmap'); renderRoadmap(); });
  });

  // Prev / Next
  const prev = getPrevLesson(trackId, lessonId);
  const next = getNextLesson(trackId, lessonId);
  const btnPrev = document.getElementById('btn-prev-lesson');
  const btnNext = document.getElementById('btn-next-lesson');
  const btnMark = document.getElementById('btn-mark-complete');

  btnPrev.disabled = !prev;
  btnNext.disabled = !next;
  btnPrev.onclick = prev ? () => openLesson(trackId, prev.lesson.id) : null;
  btnNext.onclick = next ? () => openLesson(trackId, next.lesson.id) : null;

  function refreshMarkBtn() {
    const done = state.completed.has(lessonId);
    btnMark.textContent = done ? '✓ Completed' : '✓ Mark Complete';
    btnMark.className = done ? 'btn-mark-complete done' : 'btn-mark-complete';
    if (done) setLessonProgress(100);
  }
  refreshMarkBtn();

  btnMark.onclick = () => {
    if (!state.completed.has(lessonId)) {
      state.completed.add(lessonId);
      state.xp += 50;
      state.streak++;
      saveProgress();
      refreshMarkBtn();
      celebrate(btnMark);
      showXPGain(50, btnMark);
      toast('Lesson complete! +50 XP');
    }
  };

  // Reset UI
  document.getElementById('lesson-title-heading').textContent = lesson.title;
  switchTab('explanation');

  const contentEl = document.getElementById('lesson-content');
  contentEl.innerHTML = '<div style="color:var(--text4);padding:24px 0 40px">Loading…</div>';
  document.getElementById('lesson-toc-list').innerHTML = '';
  document.getElementById('lesson-outline').innerHTML = '';
  setLessonProgress(state.completed.has(lessonId) ? 100 : 0);
  document.getElementById('lesson-center-panel').scrollTop = 0;

  // Load markdown
  const mdRaw = await window.api.readFile(`content/${trackId}/${lesson.file}`);

  if (!mdRaw) {
    contentEl.innerHTML = `
      <div style="padding:48px 0;text-align:center">
        <div style="font-size:48px;margin-bottom:16px">🚧</div>
        <div style="font-size:16px;font-weight:600;color:var(--text2);margin-bottom:8px">Lesson coming soon</div>
        <div style="font-size:13px;color:var(--text4)">Content is being prepared. Check back soon.</div>
      </div>
    `;
    document.getElementById('quick-practice-card').style.display = 'none';
    return;
  }

  const { body } = parseMarkdownMeta(mdRaw);
  const processedBody = preprocessWidgets(body);
  contentEl.innerHTML = marked.parse(processedBody);
  renderWidgets(contentEl);
  enhanceCodeBlocks(contentEl);
  renderMath(contentEl);
  addGlossaryTooltips(contentEl);
  buildTOC(contentEl);

  // Save + restore scroll position per lesson
  state.lastLesson = { trackId, lessonId };
  saveProgress();
  const centerPanel = document.getElementById('lesson-center-panel');
  const savedScroll = parseInt(localStorage.getItem(`ioai_scroll_${lessonId}`) || '0');
  if (savedScroll > 0) setTimeout(() => { centerPanel.scrollTop = savedScroll; }, 120);
  if (centerPanel._scrollSaver) centerPanel.removeEventListener('scroll', centerPanel._scrollSaver);
  centerPanel._scrollSaver = () => localStorage.setItem(`ioai_scroll_${lessonId}`, centerPanel.scrollTop);
  centerPanel.addEventListener('scroll', centerPanel._scrollSaver, { passive: true });

  // Load quiz
  const quizFile = lesson.file.replace('.md', '-quiz.json');
  const quizRaw = await window.api.readFile(`content/${trackId}/${quizFile}`);
  const quizData = quizRaw ? JSON.parse(quizRaw) : null;

  // Practice tab
  const exercisesEl = document.getElementById('lesson-exercises');
  exercisesEl.innerHTML = '';
  if (quizData) {
    renderQuiz(quizData, exercisesEl);
    renderExercises(quizData, exercisesEl, lessonId);
  } else {
    exercisesEl.innerHTML = '<div style="padding:24px 32px;color:var(--text4)">No practice questions for this lesson yet.</div>';
  }

  // Quick practice card
  const qpCard = document.getElementById('quick-practice-card');
  if (quizData?.questions?.length) {
    qpCard.style.display = 'block';
    document.getElementById('qp-prompt').textContent = quizData.questions[0].question.replace(/<[^>]+>/g, '');
    document.getElementById('btn-start-practice').onclick = () => switchTab('practice');
  } else {
    qpCard.style.display = 'none';
  }

  // Notes
  const notesArea = document.getElementById('lesson-notes-area');
  notesArea.value = localStorage.getItem(`ioai_notes_${lessonId}`) || '';
  document.getElementById('btn-save-notes').onclick = () => {
    localStorage.setItem(`ioai_notes_${lessonId}`, notesArea.value);
    toast('Notes saved!');
  };

  // Pre-warm Pyodide if lesson has Python
  if (!state.pyodideReady && body.includes('```python')) ensurePyodide();

  // Wire up AI assistant panel
  setupAIPanel(track.name, lesson.title);
}

// ─── AI Assistant (Ollama) ────────────────────────────────────────────────────
let aiOnline = false;

async function setupAIPanel(trackName, lessonTitle) {
  const dot = document.getElementById('ai-dot');
  const log = document.getElementById('ai-chat-log');
  const input = document.getElementById('ai-input');
  const send = document.getElementById('ai-send');
  const hint = document.getElementById('ai-hint');

  // Clear previous messages
  log.innerHTML = '';

  // Check Ollama availability
  aiOnline = await window.api.ollamaCheck().catch(() => false);
  dot.className = `ai-status-dot ${aiOnline ? 'online' : 'offline'}`;
  hint.textContent = aiOnline
    ? 'AI ready — ask anything about this lesson'
    : 'Offline: install Ollama + run: ollama pull llama3.2:1b';

  if (aiOnline && log.children.length === 0) {
    const welcome = document.createElement('div');
    welcome.className = 'ai-msg ai';
    welcome.textContent = `Hi! I'm here to help with "${lessonTitle}". Ask me to explain any concept, give an example, or quiz you.`;
    log.appendChild(welcome);
  }

  // Remove old listeners by cloning
  const newSend = send.cloneNode(true);
  send.parentNode.replaceChild(newSend, send);
  const newInput = input.cloneNode(true);
  input.parentNode.replaceChild(newInput, input);

  const sendMsg = async () => {
    const q = newInput.value.trim();
    if (!q) return;
    if (!aiOnline) { hint.textContent = 'Ollama not running. Install from ollama.com'; return; }

    const userEl = document.createElement('div');
    userEl.className = 'ai-msg user';
    userEl.textContent = q;
    log.appendChild(userEl);
    newInput.value = '';
    newSend.disabled = true;

    const thinkEl = document.createElement('div');
    thinkEl.className = 'ai-msg ai';
    thinkEl.textContent = '…';
    log.appendChild(thinkEl);
    log.scrollTop = log.scrollHeight;

    const systemPrompt = `You are a helpful tutor for the IOAI (International Olympiad in AI) learning platform. The student is studying "${lessonTitle}" in the "${trackName}" track. Give short, clear, friendly explanations. If they ask for code, keep it concise. Max 3 paragraphs.`;

    const result = await window.api.ollamaChat({
      model: 'llama3.2:1b',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: q },
      ],
    }).catch(() => ({ ok: false, error: 'Connection failed' }));

    if (result.ok) {
      thinkEl.textContent = result.content;
    } else {
      thinkEl.className = 'ai-msg error';
      thinkEl.textContent = result.error;
    }
    newSend.disabled = false;
    log.scrollTop = log.scrollHeight;
  };

  newSend.addEventListener('click', sendMsg);
  newInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
  });
}

// ─── Home Dashboard ───────────────────────────────────────────────────────────
function renderHome() {
  const view = document.getElementById('view-home');
  const pct = getOverallProgress();
  const totalLessons = state.courses.tracks.flatMap(t => t.lessons).length;
  const doneLessons = state.completed.size;

  // Next lesson to continue
  let nextTrack = null, nextLesson = null;
  for (const track of state.courses.tracks) {
    const first = track.lessons.find(l => !state.completed.has(l.id));
    if (first) { nextTrack = track; nextLesson = first; break; }
  }

  // Resume target: last visited lesson takes priority over first incomplete
  let resumeTarget = null;
  if (state.lastLesson) {
    const rTrack = state.courses.tracks.find(t => t.id === state.lastLesson.trackId);
    const rLesson = rTrack?.lessons.find(l => l.id === state.lastLesson.lessonId);
    if (rTrack && rLesson) resumeTarget = { track: rTrack, lesson: rLesson };
  }
  if (!resumeTarget && nextTrack && nextLesson) {
    resumeTarget = { track: nextTrack, lesson: nextLesson };
  }

  // Upcoming (first 4 incomplete)
  const upcoming = [];
  for (const track of state.courses.tracks) {
    for (const lesson of track.lessons) {
      if (!state.completed.has(lesson.id) && upcoming.length < 4) {
        upcoming.push({ track, lesson });
      }
    }
    if (upcoming.length >= 4) break;
  }

  // Recommended (further along the path)
  const remaining = [];
  for (const track of state.courses.tracks) {
    for (const lesson of track.lessons) {
      if (!state.completed.has(lesson.id)) remaining.push({ track, lesson });
    }
  }
  const recReasons = ['Next up in your path', 'Builds on what you just learned', 'Strengthens your fundamentals'];
  const recTones = ['sky', 'emerald', 'amber'];
  const recIcons = ['📘', '🧠', '⚡'];
  const recommended = (remaining.length > 7 ? remaining.slice(4, 7) : remaining.slice(0, 3))
    .map((r, i) => ({ ...r, reason: recReasons[i], tone: recTones[i], icon: recIcons[i] }));

  // Daily challenge
  const challenges = [
    { title: 'Write a function to flatten a nested list', diff: 'Python · Beginner' },
    { title: 'Implement dot product without NumPy', diff: 'Math · Intermediate' },
    { title: 'Code gradient descent for linear regression', diff: 'ML · Intermediate' },
    { title: 'Build a 2-layer network from scratch', diff: 'NN · Advanced' },
    { title: 'Explain why ReLU beats sigmoid in deep nets', diff: 'NN · Intermediate' },
    { title: 'What is the vanishing gradient problem?', diff: 'NN · Advanced' },
    { title: 'Calculate cross-entropy loss manually', diff: 'Math · Intermediate' },
  ];
  const challenge = challenges[new Date().getDay()];

  // SVG rings
  const rSm = 11, circSm = 2 * Math.PI * rSm;
  const offsetSm = circSm - (pct / 100) * circSm;
  const r = 40, circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  const ICON_BOOK = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>';
  const ICON_ZAP = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>';
  const ICON_TARGET = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/></svg>';
  const ICON_MOON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  const ICON_SUN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';

  const dailyGoalDone = Math.min(state.completed.size, 3);
  const dueReviews = srDueCount();

  view.innerHTML = `
    <div class="home-wrap">
    <div class="home-topbar">
      <div class="home-greeting">
        <h2>${greeting}, ${state.userName || 'Student'} 👋</h2>
        <p>You're ${pct}% of the way to IOAI-ready.</p>
      </div>
      <div class="home-topbar-right">
        ${state.streak > 0 ? '<div class="stat-chip"><span class="chip-icon">🔥</span><span>' + state.streak + '</span><span class="stat-chip-label">day streak</span></div>' : ''}
        <div class="stat-chip">
          <div class="stat-chip-ring">
            <svg width="28" height="28" viewBox="0 0 28 28" style="transform:rotate(-90deg)"><circle cx="14" cy="14" r="${rSm}" fill="none" stroke="var(--muted)" stroke-width="3"/><circle cx="14" cy="14" r="${rSm}" fill="none" stroke="var(--primary)" stroke-width="3" stroke-linecap="round" stroke-dasharray="${circSm.toFixed(2)}" stroke-dashoffset="${offsetSm.toFixed(2)}"/></svg>
            <div class="stat-chip-ring-text">${pct}%</div>
          </div>
          <span>${doneLessons}/${totalLessons}</span>
          <span class="stat-chip-label">lessons</span>
        </div>
        <button class="icon-btn" id="btn-home-theme" title="Toggle theme">${state.darkMode ? ICON_SUN : ICON_MOON}</button>
        <button class="bell-btn" title="Notifications">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
          <span class="bell-dot"></span>
        </button>
      </div>
    </div>

    <div class="home-top-row">
      <div class="goal-card">
        <svg class="goal-mountains" viewBox="0 0 200 130" fill="none" aria-hidden="true">
          <path d="M0 130 L58 44 L92 92 L118 62 L200 130 Z" fill="#fff"/>
          <path d="M40 130 L110 34 L200 130 Z" fill="#fff" opacity="0.55"/>
        </svg>
        <div class="goal-label">🎯 Your Goal</div>
        <div class="goal-title">International Olympiad in AI</div>
        <div class="goal-desc">Go from Python basics to neural networks and compete at the world's top AI olympiad for students.</div>
        ${resumeTarget
          ? '<button class="btn-continue" id="btn-home-continue">View Roadmap →</button>'
          : '<div style="color:rgba(255,255,255,0.9);font-weight:600;font-size:14px;position:relative">🎉 All lessons complete!</div>'}
      </div>

      <div class="progress-card">
        <div class="progress-card-title">Today's Plan</div>
        <div class="plan-body">
          <div class="plan-list">
            <div class="plan-row">
              <div class="plan-icon blue">${ICON_BOOK}</div>
              <div class="plan-text">
                <div class="plan-name">Lessons</div>
                <div class="plan-val">${doneLessons} of ${totalLessons} done</div>
              </div>
            </div>
            <div class="plan-row">
              <div class="plan-icon green">${ICON_ZAP}</div>
              <div class="plan-text">
                <div class="plan-name">XP Earned</div>
                <div class="plan-val">${state.xp} XP total</div>
              </div>
            </div>
            <div class="plan-row">
              <div class="plan-icon amber">${ICON_TARGET}</div>
              <div class="plan-text">
                <div class="plan-name">Daily Goal</div>
                <div class="plan-val">${dailyGoalDone} of 3 lessons</div>
              </div>
            </div>
            ${dueReviews > 0 ? `
            <div class="plan-row review-due-row" id="btn-home-review">
              <div class="plan-icon" style="background:var(--purple-lt)"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--purple)" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg></div>
              <div class="plan-text">
                <div class="plan-name" style="color:var(--purple)">Reviews Due</div>
                <div class="plan-val">${dueReviews} question${dueReviews!==1?'s':''} to review today</div>
              </div>
            </div>` : ''}
          </div>
          <div class="progress-circle-wrap">
            <svg class="progress-circle-svg" viewBox="0 0 92 92">
              <circle class="progress-circle-bg" cx="46" cy="46" r="${r}"/>
              <circle class="progress-circle-fg" cx="46" cy="46" r="${r}"
                stroke-dasharray="${circ.toFixed(2)}"
                stroke-dashoffset="${offset.toFixed(2)}"/>
            </svg>
            <div class="progress-circle-text">${pct}%<small>COMPLETE</small></div>
          </div>
        </div>
        <button class="progress-card-btn" id="btn-home-continue-plan">Continue Learning →</button>
      </div>

      <div class="challenge-card">
        <div class="challenge-header">
          <span class="challenge-label">Daily Challenge</span>
          <span class="challenge-pts">+25 XP</span>
        </div>
        <div class="challenge-title">${challenge.title}</div>
        <div class="challenge-diff">${challenge.diff}</div>
        <button class="btn-solve" id="btn-home-challenge">Try it →</button>
        <div class="streak-bonus">
          <div class="streak-bonus-inner">
            <div class="streak-bonus-icon">🎁</div>
            <div>
              <div class="streak-bonus-title">Streak Bonus</div>
              <div class="streak-bonus-desc">Finish today's challenge to keep your ${state.streak}-day streak alive.</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="home-mid-row">
      <div class="continue-card">
        <div class="card-head">
          <div class="card-title">Continue Learning</div>
          <span class="card-link" id="link-view-all">View all</span>
        </div>
        <div class="continue-list">
          ${upcoming.map(({ track, lesson }) => {
            const tp = getTrackProgress(track.id);
            return '<div class="continue-lesson-item" data-track="' + track.id + '" data-lesson="' + lesson.id + '">'
              + '<div class="continue-lesson-meta">'
              + '<div class="continue-track">' + track.icon + ' ' + track.name + '</div>'
              + '<div class="continue-title">' + lesson.title + '</div>'
              + '<div class="continue-bar-row"><div class="continue-bar-wrap"><div class="continue-bar-fill" style="width:' + tp + '%"></div></div><div class="continue-bar-pct">' + tp + '%</div></div>'
              + '</div>'
              + '<button class="btn-continue-sm">Continue →</button>'
              + '</div>';
          }).join('')}
        </div>
      </div>

      <div class="upcoming-card">
        <div class="card-head">
          <div class="card-title">✨ Recommended for You</div>
        </div>
        ${recommended.map(rec =>
          '<div class="rec-item" data-track="' + rec.track.id + '" data-lesson="' + rec.lesson.id + '">'
          + '<div class="rec-icon ' + rec.tone + '">' + rec.icon + '</div>'
          + '<div class="rec-text"><div class="rec-title">' + rec.lesson.title + '</div>'
          + '<div class="rec-reason">' + rec.reason + '</div></div>'
          + '<span class="rec-arrow">→</span>'
          + '</div>'
        ).join('')}
        <div class="card-head" style="margin-top:18px;margin-bottom:10px"><div class="card-title">All Tracks</div></div>
        ${state.courses.tracks.map(track => {
          const tp = getTrackProgress(track.id);
          return '<div class="upcoming-item" data-track="' + track.id + '">'
            + '<div class="upcoming-dot ' + (tp === 100 ? 'orange' : 'blue') + '"></div>'
            + '<div><div class="upcoming-title">' + track.name + '</div>'
            + '<div class="upcoming-meta">' + track.lessons.filter(l => state.completed.has(l.id)).length + '/' + track.lessons.length + ' lessons · ' + tp + '%</div></div>'
            + '</div>';
        }).join('')}
      </div>
    </div>
    </div>
  `;

  // Events
  view.querySelector('#btn-home-continue')?.addEventListener('click', () => {
    navigate('roadmap'); renderRoadmap();
  });
  view.querySelector('#btn-home-continue-plan')?.addEventListener('click', () => {
    if (resumeTarget) openLesson(resumeTarget.track.id, resumeTarget.lesson.id);
  });
  view.querySelector('#btn-home-theme')?.addEventListener('click', () => {
    toggleDarkMode(); renderHome();
  });
  view.querySelector('#link-view-all')?.addEventListener('click', () => {
    navigate('lessons'); renderLessonsBrowser();
  });

  view.querySelectorAll('.continue-lesson-item').forEach(item => {
    item.addEventListener('click', () => openLesson(item.dataset.track, item.dataset.lesson));
    item.querySelector('.btn-continue-sm').addEventListener('click', e => {
      e.stopPropagation();
      openLesson(item.dataset.track, item.dataset.lesson);
    });
  });

  view.querySelectorAll('.rec-item').forEach(item => {
    item.addEventListener('click', () => openLesson(item.dataset.track, item.dataset.lesson));
  });

  view.querySelector('#btn-home-challenge')?.addEventListener('click', () => {
    navigate('practice'); renderPractice();
  });

  view.querySelectorAll('.upcoming-item').forEach(item => {
    item.addEventListener('click', () => { navigate('roadmap'); renderRoadmap(); });
  });
}

// ─── Roadmap View ─────────────────────────────────────────────────────────────
function renderRoadmap() {
  const view = document.getElementById('view-roadmap');
  const doneLessons = state.completed.size;
  const totalLessons = state.courses.tracks.flatMap(t => t.lessons).length;

  view.innerHTML = `
    <div class="view-header">
      <h1>Learning Roadmap</h1>
      <p>${doneLessons} of ${totalLessons} lessons complete · ${state.xp} XP earned</p>
    </div>
    <div class="roadmap-track-list" id="roadmap-list"></div>
  `;

  const list = view.querySelector('#roadmap-list');

  state.courses.tracks.forEach((track, i) => {
    const prev = state.courses.tracks[i - 1];
    const unlocked = !track.requiresPrev || !prev || getTrackProgress(prev.id) >= 50;
    const pct = getTrackProgress(track.id);
    const done = track.lessons.filter(l => state.completed.has(l.id)).length;
    const pctClass = pct === 100 ? 'pct-done' : (unlocked ? 'pct-progress' : 'pct-locked');
    const pctText = !unlocked ? '🔒 Locked' : (pct === 100 ? '✓ Done' : `${pct}%`);

    const card = document.createElement('div');
    card.className = 'roadmap-track-card';
    card.innerHTML = `
      <div class="roadmap-track-header">
        <div class="track-icon-box">${track.icon}</div>
        <div class="track-info">
          <div class="track-name">${track.name}</div>
          <div class="track-meta-row">
            <span>${track.lessons.length} lessons</span>
            <span>~${track.totalTime} min</span>
            <span>${done} done</span>
          </div>
        </div>
        <span class="track-pct-badge ${pctClass}">${pctText}</span>
        <button class="track-chevron-btn">›</button>
      </div>
      <div class="track-progress-bar-row">
        <div class="track-prog-bar">
          <div class="track-prog-fill" style="width:${pct}%"></div>
        </div>
        <div class="track-prog-pct">${pct}%</div>
      </div>
      <div class="track-lessons-list">
        ${track.lessons.map(l => {
          const isDone = state.completed.has(l.id);
          const isCurrent = state.currentLesson?.lessonId === l.id;
          const dotClass = isDone ? 'done' : (isCurrent ? 'active' : '');
          return `
            <div class="track-lesson-row" data-track="${track.id}" data-lesson="${l.id}" style="${!unlocked ? 'opacity:0.4;cursor:not-allowed' : ''}">
              <div class="lesson-status-dot ${dotClass}"></div>
              <div class="lesson-row-title ${isDone ? 'done' : ''}">${l.title}</div>
              <span class="lesson-row-time">${l.estimatedTime}m</span>
              <span class="lesson-row-badge ${diffBadgeClass(l.difficulty)}">${diffLabel(l.difficulty)}</span>
            </div>`;
        }).join('')}
      </div>
    `;

    card.querySelector('.roadmap-track-header').addEventListener('click', () => {
      card.classList.toggle('open');
    });

    if (unlocked) {
      card.querySelectorAll('.track-lesson-row').forEach(row => {
        row.addEventListener('click', () => openLesson(row.dataset.track, row.dataset.lesson));
      });
    }

    list.appendChild(card);
  });
}

// ─── Lessons Browser ──────────────────────────────────────────────────────────
function renderLessonsBrowser() {
  const view = document.getElementById('view-lessons');
  view.innerHTML = `
    <div class="view-header">
      <h1>All Lessons</h1>
      <p>Browse every lesson across all 7 tracks. Click any card to open.</p>
    </div>
  `;

  state.courses.tracks.forEach(track => {
    const section = document.createElement('div');
    section.style.marginBottom = '32px';
    const doneCount = track.lessons.filter(l => state.completed.has(l.id)).length;
    section.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
        <span style="font-size:20px">${track.icon}</span>
        <span style="font-size:15px;font-weight:700">${track.name}</span>
        <span style="font-size:12px;color:var(--text4);margin-left:4px">${doneCount}/${track.lessons.length} done</span>
      </div>
    `;

    const grid = document.createElement('div');
    grid.className = 'lessons-grid';

    track.lessons.forEach(lesson => {
      const isDone = state.completed.has(lesson.id);
      const card = document.createElement('div');
      card.className = `lesson-card${isDone ? ' completed' : ''}`;
      card.innerHTML = `
        <div class="lesson-card-icon">${isDone ? '✅' : track.icon}</div>
        <div class="lesson-card-track">${track.name}</div>
        <div class="lesson-card-title">${lesson.title}</div>
        <div class="lesson-card-meta">
          <span>⏱ ${lesson.estimatedTime}m</span>
          <span class="lesson-row-badge ${diffBadgeClass(lesson.difficulty)}">${diffLabel(lesson.difficulty)}</span>
        </div>
      `;
      card.addEventListener('click', () => openLesson(track.id, lesson.id));
      grid.appendChild(card);
    });

    section.appendChild(grid);
    view.appendChild(section);
  });
}

// ─── Practice View ────────────────────────────────────────────────────────────
function renderPractice() {
  const view = document.getElementById('view-practice');
  view.innerHTML = `
    <div class="view-header">
      <h1>Practice</h1>
      <p>Jump into exercises from any track. Choose a topic to start.</p>
    </div>
    <div class="ai-track-list">
      ${state.courses.tracks.map(track => {
        const tp = getTrackProgress(track.id);
        return `
          <div class="ai-track-item" data-trackid="${track.id}">
            <div class="ai-item-icon" style="font-size:20px">${track.icon}</div>
            <div style="flex:1">
              <div style="font-size:14px;font-weight:600">${track.name}</div>
              <div style="font-size:12px;color:var(--text4);margin-top:2px">${track.lessons.length} lessons · ${tp}% done</div>
            </div>
            <span class="track-pct-badge ${tp === 100 ? 'pct-done' : 'pct-progress'}">${tp}%</span>
          </div>`;
      }).join('')}
    </div>
  `;

  view.querySelectorAll('.ai-track-item').forEach(item => {
    item.addEventListener('click', () => {
      const track = state.courses.tracks.find(t => t.id === item.dataset.trackid);
      if (!track) return;
      const first = track.lessons.find(l => !state.completed.has(l.id)) || track.lessons[0];
      if (first) openLesson(track.id, first.id);
    });
  });
}

// ─── AI Track View ────────────────────────────────────────────────────────────
function renderAITrack() {
  const aiTracks = state.courses.tracks.filter(t => ['nn', 'cv', 'nlp', 'ioai'].includes(t.id));
  const view = document.getElementById('view-aitrack');
  view.innerHTML = `
    <div class="view-header">
      <h1>AI Track</h1>
      <p>Neural networks, computer vision, NLP, and IOAI competition strategy.</p>
    </div>
    <div class="ai-track-list">
      ${aiTracks.map(track => {
        const tp = getTrackProgress(track.id);
        return `
          <div class="ai-track-item" data-trackid="${track.id}">
            <div class="ai-item-icon" style="font-size:22px">${track.icon}</div>
            <div style="flex:1">
              <div style="font-size:14px;font-weight:600;margin-bottom:4px">${track.name}</div>
              <div style="font-size:12px;color:var(--text3);margin-bottom:8px">${track.description}</div>
              <div style="height:4px;background:var(--border);border-radius:2px;max-width:320px">
                <div style="height:100%;width:${tp}%;background:var(--blue);border-radius:2px"></div>
              </div>
            </div>
            <span class="track-pct-badge ${tp === 100 ? 'pct-done' : 'pct-progress'}">${tp}%</span>
          </div>`;
      }).join('')}
    </div>
  `;

  view.querySelectorAll('.ai-track-item').forEach(item => {
    item.addEventListener('click', () => {
      const track = state.courses.tracks.find(t => t.id === item.dataset.trackid);
      if (!track) return;
      const first = track.lessons.find(l => !state.completed.has(l.id)) || track.lessons[0];
      if (first) openLesson(track.id, first.id);
    });
  });
}

// ─── Math View ────────────────────────────────────────────────────────────────
function renderMathView() {
  const mathTrack = state.courses.tracks.find(t => t.id === 'math');
  const view = document.getElementById('view-math');
  if (!mathTrack) { view.innerHTML = '<div class="view-header"><h1>Math for AI</h1></div>'; return; }
  const tp = getTrackProgress('math');

  view.innerHTML = `
    <div class="view-header">
      <h1>Math for AI</h1>
      <p>Linear algebra, calculus, probability, and information theory. The foundation everything else is built on.</p>
    </div>
    <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:18px 24px;margin-bottom:20px;display:flex;align-items:center;gap:20px">
      <div style="font-size:30px">∑</div>
      <div style="flex:1">
        <div style="font-weight:600;margin-bottom:6px">Track progress: ${tp}%</div>
        <div style="height:6px;background:var(--border);border-radius:3px">
          <div style="height:100%;width:${tp}%;background:${tp === 100 ? 'var(--green)' : 'var(--blue)'};border-radius:3px"></div>
        </div>
      </div>
      <span class="track-pct-badge ${tp === 100 ? 'pct-done' : 'pct-progress'}">${tp}%</span>
    </div>
    <div class="ai-track-list">
      ${mathTrack.lessons.map((lesson, idx) => {
        const isDone = state.completed.has(lesson.id);
        return `
          <div class="ai-track-item" data-trackid="math" data-lessonid="${lesson.id}">
            <div class="ai-item-icon" style="font-size:15px;font-weight:700;background:var(--purple-lt);color:var(--purple)">${isDone ? '✓' : (idx + 1)}</div>
            <div style="flex:1">
              <div style="font-size:14px;font-weight:600;${isDone ? 'color:var(--text3)' : ''}">${lesson.title}</div>
              <div style="font-size:12px;color:var(--text4)">~${lesson.estimatedTime} min · ${diffLabel(lesson.difficulty)}</div>
            </div>
            ${isDone ? '<span style="color:var(--green);font-size:16px">✓</span>' : `<span class="lesson-row-badge ${diffBadgeClass(lesson.difficulty)}">${diffLabel(lesson.difficulty)}</span>`}
          </div>`;
      }).join('')}
    </div>
  `;

  view.querySelectorAll('[data-lessonid]').forEach(item => {
    item.addEventListener('click', () => openLesson('math', item.dataset.lessonid));
  });
}

// ─── Progress View ────────────────────────────────────────────────────────────
function renderProgress() {
  const view = document.getElementById('view-progress');
  const pct = getOverallProgress();
  const total = state.courses.tracks.flatMap(t => t.lessons).length;

  view.innerHTML = `
    <div class="view-header">
      <h1>Your Progress</h1>
      <p>Track your journey from Python basics to IOAI competition level.</p>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px">
      <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;text-align:center">
        <div style="font-size:32px;font-weight:700;color:var(--blue)">${state.completed.size}</div>
        <div style="font-size:13px;color:var(--text3);margin-top:4px">Lessons done</div>
      </div>
      <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;text-align:center">
        <div style="font-size:32px;font-weight:700;color:var(--orange)">${state.xp}</div>
        <div style="font-size:13px;color:var(--text3);margin-top:4px">XP earned</div>
      </div>
      <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;text-align:center">
        <div style="font-size:32px;font-weight:700;color:var(--green)">${state.streak}</div>
        <div style="font-size:13px;color:var(--text3);margin-top:4px">Day streak 🔥</div>
      </div>
    </div>
    <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:20px;margin-bottom:24px">
      <div style="display:flex;justify-content:space-between;margin-bottom:10px">
        <span style="font-weight:600">Overall progress</span>
        <span style="color:var(--blue);font-weight:700">${pct}%</span>
      </div>
      <div style="height:8px;background:var(--border);border-radius:4px;overflow:hidden">
        <div style="height:100%;width:${pct}%;background:var(--blue);border-radius:4px;transition:width 0.6s"></div>
      </div>
      <div style="font-size:12px;color:var(--text4);margin-top:6px">${state.completed.size} of ${total} lessons completed</div>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px">
      ${state.courses.tracks.map(track => {
        const tp = getTrackProgress(track.id);
        const done = track.lessons.filter(l => state.completed.has(l.id)).length;
        return `
          <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:16px 20px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
              <span style="font-size:18px">${track.icon}</span>
              <span style="font-weight:600;flex:1">${track.name}</span>
              <span style="font-size:13px;color:var(--text3)">${done}/${track.lessons.length}</span>
              <span style="font-weight:700;color:${tp === 100 ? 'var(--green)' : 'var(--blue)'}">${tp}%</span>
            </div>
            <div style="height:5px;background:var(--border);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${tp}%;background:${tp === 100 ? 'var(--green)' : 'var(--blue)'};border-radius:3px"></div>
            </div>
          </div>`;
      }).join('')}
    </div>
  `;
}

// ─── Settings View ────────────────────────────────────────────────────────────
function renderSettings() {
  const view = document.getElementById('view-settings');
  view.innerHTML = `
    <div class="view-header">
      <h1>Settings</h1>
      <p>Manage your profile and preferences.</p>
    </div>
    <div style="max-width:520px;display:flex;flex-direction:column;gap:16px;padding-bottom:32px">

      <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:24px">
        <div style="font-weight:700;font-size:14px;margin-bottom:16px">Profile</div>
        <label style="font-size:13px;color:var(--text3);display:block;margin-bottom:6px">Your name</label>
        <input id="settings-name" class="modal-input" type="text" value="${state.userName}" placeholder="Enter your name" maxlength="30" style="margin-bottom:12px"/>
        <button class="btn-save-notes" id="btn-save-name">Save name</button>
      </div>

      <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:24px">
        <div style="font-weight:700;font-size:14px;margin-bottom:16px">Appearance</div>

        <div class="settings-row">
          <div>
            <div style="font-size:13px;font-weight:600">Dark mode</div>
            <div style="font-size:12px;color:var(--text4);margin-top:2px">Switch to a dark background</div>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" id="toggle-dark" ${state.darkMode ? 'checked' : ''}>
            <div class="toggle-track"><div class="toggle-thumb"></div></div>
          </label>
        </div>

        <div class="settings-row" style="margin-top:14px">
          <div>
            <div style="font-size:13px;font-weight:600">Collapse sidebar</div>
            <div style="font-size:12px;color:var(--text4);margin-top:2px">Show only icons on the left</div>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" id="toggle-sidebar" ${state.sidebarCollapsed ? 'checked' : ''}>
            <div class="toggle-track"><div class="toggle-thumb"></div></div>
          </label>
        </div>
      </div>

      <div style="background:var(--white);border:1px solid var(--border);border-radius:var(--radius-lg);padding:24px">
        <div style="font-weight:700;font-size:14px;margin-bottom:6px">Reset progress</div>
        <div style="font-size:13px;color:var(--text3);margin-bottom:14px">Clear all completed lessons and XP. Cannot be undone.</div>
        <button id="btn-reset" style="background:var(--red-lt);border:1px solid var(--red);color:var(--red);padding:7px 18px;border-radius:var(--radius-sm);font-size:13px;font-weight:600;cursor:pointer">Reset all progress</button>
      </div>
    </div>
  `;

  view.querySelector('#btn-save-name').addEventListener('click', () => {
    const name = view.querySelector('#settings-name').value.trim();
    if (!name) return;
    state.userName = name;
    document.getElementById('user-name-display').textContent = name;
    document.getElementById('user-avatar').textContent = name[0].toUpperCase();
    saveProgress();
    toast('Name saved!');
  });

  view.querySelector('#toggle-dark').addEventListener('change', toggleDarkMode);
  view.querySelector('#toggle-sidebar').addEventListener('change', toggleSidebar);

  view.querySelector('#btn-reset').addEventListener('click', () => {
    if (!confirm('Reset all progress? This cannot be undone.')) return;
    state.completed.clear();
    state.xp = 0;
    state.streak = 0;
    saveProgress();
    toast('Progress reset.');
    navigate('home');
    renderHome();
  });
}

// ─── Name Modal ───────────────────────────────────────────────────────────────
function setupNameModal() {
  if (state.userName) return;
  const modal = document.getElementById('name-modal');
  modal.style.display = 'flex';

  const input = document.getElementById('name-input');
  const submitBtn = document.getElementById('name-submit');

  function submit() {
    const name = input.value.trim();
    if (!name) { input.focus(); return; }
    state.userName = name;
    localStorage.setItem('ioai_name', name);
    document.getElementById('user-name-display').textContent = name;
    document.getElementById('user-avatar').textContent = name[0].toUpperCase();
    modal.style.display = 'none';
    renderHome();
  }

  submitBtn.addEventListener('click', submit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
  setTimeout(() => input.focus(), 100);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  // Load Pyodide async
  const pyScript = document.createElement('script');
  pyScript.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js';
  document.head.appendChild(pyScript);

  await loadProgress();
  await loadCourses();
  configureMarked();

  // Apply saved theme + sidebar state
  applyTheme();
  if (state.sidebarCollapsed) document.getElementById('sidebar').classList.add('collapsed');

  // Populate user info
  if (state.userName) {
    document.getElementById('user-name-display').textContent = state.userName;
    document.getElementById('user-avatar').textContent = state.userName[0].toUpperCase();
  }

  // Sidebar collapse button
  document.getElementById('btn-sidebar-toggle').addEventListener('click', toggleSidebar);

  // Wire up sidebar nav
  document.querySelectorAll('.nav-item[data-view]').forEach(item => {
    item.addEventListener('click', () => {
      const v = item.dataset.view;
      navigate(v);
      if (v === 'home') renderHome();
      else if (v === 'roadmap') renderRoadmap();
      else if (v === 'lessons') renderLessonsBrowser();
      else if (v === 'practice') renderPractice();
      else if (v === 'aitrack') renderAITrack();
      else if (v === 'math') renderMathView();
      else if (v === 'progress') renderProgress();
      else if (v === 'settings') renderSettings();
    });
  });

  // Wire up lesson tabs
  document.querySelectorAll('.lesson-tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Initial render
  renderHome();
  navigate('home');
  setupNameModal();
}

boot();
