// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  courses: null,         // parsed courses.json
  currentLesson: null,   // { trackId, lessonId, meta }
  pyodide: null,
  pyodideLoading: false,
  pyodideReady: false,
  xp: 0,
  streak: 0,
  completed: new Set(),  // lessonIds
  hintIndex: {},         // lessonId -> hintUsed index
};

// ─── Storage ─────────────────────────────────────────────────────────────────
function saveProgress() {
  localStorage.setItem('ioai_completed', JSON.stringify([...state.completed]));
  localStorage.setItem('ioai_xp', state.xp);
  localStorage.setItem('ioai_streak', state.streak);
  localStorage.setItem('ioai_last_date', new Date().toDateString());
}

function loadProgress() {
  const c = localStorage.getItem('ioai_completed');
  if (c) state.completed = new Set(JSON.parse(c));
  state.xp = parseInt(localStorage.getItem('ioai_xp') || '0');

  const lastDate = localStorage.getItem('ioai_last_date');
  const today = new Date().toDateString();
  const yesterday = new Date(Date.now() - 86400000).toDateString();
  const savedStreak = parseInt(localStorage.getItem('ioai_streak') || '0');
  if (lastDate === today) state.streak = savedStreak;
  else if (lastDate === yesterday) state.streak = savedStreak; // continue today
  else state.streak = 0; // broke streak
}

// ─── Pyodide ─────────────────────────────────────────────────────────────────
function updatePyodideStatus(status) {
  const el = document.getElementById('pyodide-status');
  el.className = `status-${status}`;
  const dot = el.querySelector('.status-dot');
  const label = el.querySelector('.status-label');
  if (status === 'loading') { el.classList.add('loading'); label.textContent = 'Loading Python...'; }
  if (status === 'ready')   { el.classList.add('ready');   label.textContent = 'Python ready'; }
  if (status === 'idle')    { label.textContent = 'Python'; }
}

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
    await py.loadPackagesFromImports('import numpy, pandas, matplotlib');
    state.pyodide = py;
    state.pyodideReady = true;
    updatePyodideStatus('ready');
    return py;
  } catch (e) {
    updatePyodideStatus('idle');
    return null;
  }
}

async function runPython(code) {
  const py = await ensurePyodide();
  if (!py) return { output: '', error: 'Python engine not available. Check your internet connection.' };
  try {
    // Capture stdout
    py.runPython(`
import sys, io
_stdout_capture = io.StringIO()
sys.stdout = _stdout_capture
`);
    await py.runPythonAsync(code);
    const output = py.runPython(`
sys.stdout = sys.__stdout__
_stdout_capture.getvalue()
`);
    return { output: output || '(no output)', error: null };
  } catch (e) {
    try { py.runPython('import sys; sys.stdout = sys.__stdout__'); } catch (_) {}
    return { output: '', error: String(e).replace('PythonError: Traceback', 'Error') };
  }
}

// ─── Utilities ───────────────────────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));

function toast(msg, duration = 2500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), duration);
}

function updateXpDisplay() {
  document.getElementById('xp-display').textContent = `${state.xp} XP`;
}

function updateStreakDisplay() {
  const el = document.getElementById('streak-badge');
  if (state.streak > 0) el.textContent = `🔥 ${state.streak}`;
  else el.textContent = '';
}

// ─── Course Data ──────────────────────────────────────────────────────────────
async function loadCourses() {
  const raw = await window.api.readFile('content/courses.json');
  state.courses = JSON.parse(raw);
}

function getLessonIndex(trackId, lessonId) {
  const track = state.courses.tracks.find(t => t.id === trackId);
  if (!track) return -1;
  return track.lessons.findIndex(l => l.id === lessonId);
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
  if (!track || track.lessons.length === 0) return 0;
  const done = track.lessons.filter(l => state.completed.has(l.id)).length;
  return Math.round((done / track.lessons.length) * 100);
}

function getOverallProgress() {
  const all = state.courses.tracks.flatMap(t => t.lessons);
  if (all.length === 0) return 0;
  return Math.round((state.completed.size / all.length) * 100);
}

// ─── Markdown + Rendering ────────────────────────────────────────────────────
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
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
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

    const bar = document.createElement('div');
    bar.className = 'code-run-bar';
    bar.innerHTML = `
      <span class="code-lang">${lang}</span>
      ${isPython ? `<button class="btn-run" title="Run in Python">
        <span class="run-icon">▶ Run</span>
        <span class="run-spinner">⟳</span>
      </button>` : ''}
    `;
    pre.appendChild(bar);

    if (isPython) {
      const outputEl = document.createElement('div');
      outputEl.className = 'code-output';
      pre.appendChild(outputEl);

      bar.querySelector('.btn-run').addEventListener('click', async (e) => {
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.classList.add('loading');
        outputEl.classList.add('visible');
        outputEl.className = 'code-output visible';
        outputEl.textContent = 'Running…';
        const { output, error } = await runPython(codeEl.textContent);
        btn.disabled = false;
        btn.classList.remove('loading');
        if (error) { outputEl.className = 'code-output visible error'; outputEl.textContent = error; }
        else { outputEl.className = 'code-output visible'; outputEl.textContent = output; }
      });
    }
  });
}

// ─── Quiz Renderer ────────────────────────────────────────────────────────────
function renderQuiz(quizData, container) {
  if (!quizData || !quizData.questions || quizData.questions.length === 0) return;

  const section = document.createElement('div');
  section.className = 'quiz-section';
  section.innerHTML = `<div class="quiz-section-title">Quick check</div>`;

  quizData.questions.forEach((q, qi) => {
    const card = document.createElement('div');
    card.className = 'quiz-card';

    const letters = ['A', 'B', 'C', 'D', 'E'];
    card.innerHTML = `
      <div class="quiz-question">${q.question}</div>
      <div class="quiz-options">
        ${q.options.map((opt, i) => `
          <div class="quiz-option" data-idx="${i}">
            <span class="option-letter">${letters[i]}</span>
            <span>${opt}</span>
          </div>`).join('')}
      </div>
      <div class="quiz-explanation">${q.explanation || ''}</div>
    `;

    const options = card.querySelectorAll('.quiz-option');
    const explanation = card.querySelector('.quiz-explanation');

    options.forEach(opt => {
      opt.addEventListener('click', () => {
        if (opt.classList.contains('disabled')) return;
        const chosen = parseInt(opt.dataset.idx);
        options.forEach(o => {
          o.classList.add('disabled');
          if (parseInt(o.dataset.idx) === q.correct) o.classList.add('correct');
        });
        if (chosen !== q.correct) opt.classList.add('wrong');
        explanation.classList.add('visible');
        if (chosen === q.correct) {
          state.xp += 5;
          updateXpDisplay();
          saveProgress();
        }
      });
    });

    section.appendChild(card);
  });

  container.appendChild(section);
}

// ─── Exercise Renderer ────────────────────────────────────────────────────────
function renderExercises(quizData, container, lessonId) {
  if (!quizData || !quizData.exercises || quizData.exercises.length === 0) return;

  const section = document.createElement('div');
  section.className = 'exercise-section';
  section.innerHTML = `<div class="exercise-section-title">Practice exercises</div>`;

  quizData.exercises.forEach((ex, ei) => {
    const card = document.createElement('div');
    card.className = 'exercise-card';

    card.innerHTML = `
      <div class="exercise-header">
        <div class="exercise-number">Exercise ${ei + 1}</div>
        <div class="exercise-prompt">${ex.prompt}</div>
      </div>
      <div class="exercise-editor">
        <textarea class="editor-textarea" spellcheck="false" rows="${Math.max(4, (ex.starter || '').split('\n').length + 1)}">${ex.starter || '# Write your code here\n'}</textarea>
        <div class="editor-actions">
          <button class="btn-check">Run & Check</button>
          ${ex.hints && ex.hints.length ? '<button class="btn-hint">Hint</button>' : ''}
          <button class="btn-solution">Show solution</button>
        </div>
        <div class="hint-box"></div>
        <div class="exercise-result"></div>
      </div>
    `;

    const textarea = card.querySelector('.editor-textarea');
    const resultEl = card.querySelector('.exercise-result');
    const hintBox = card.querySelector('.hint-box');
    let hintIdx = 0;

    // Tab key in textarea
    textarea.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = textarea.selectionStart, end = textarea.selectionEnd;
        textarea.value = textarea.value.substring(0, s) + '    ' + textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = s + 4;
      }
    });

    // Check button
    card.querySelector('.btn-check').addEventListener('click', async () => {
      const code = textarea.value;
      const btn = card.querySelector('.btn-check');
      btn.disabled = true;
      btn.textContent = 'Running…';
      resultEl.className = 'exercise-result visible output';
      resultEl.textContent = 'Running…';

      // Run the code + tests
      let fullCode = code;
      if (ex.tests && ex.tests.length) {
        fullCode += '\n' + ex.tests.map(t => `assert ${t}, "Test failed: ${t.replace(/"/g, "'")}"`).join('\n');
      }

      const { output, error } = await runPython(fullCode);
      btn.disabled = false;
      btn.textContent = 'Run & Check';

      if (error) {
        if (error.includes('AssertionError') || error.includes('Test failed')) {
          resultEl.className = 'exercise-result visible fail';
          resultEl.textContent = '✗ Not quite right. Check your logic and try again.\n\n' + error;
        } else {
          resultEl.className = 'exercise-result visible fail';
          resultEl.textContent = '✗ Error:\n' + error;
        }
      } else {
        resultEl.className = 'exercise-result visible pass';
        resultEl.textContent = '✓ Correct! ' + (output !== '(no output)' ? '\n\nOutput:\n' + output : '');
        state.xp += 15;
        updateXpDisplay();
        saveProgress();
        toast('Exercise solved! +15 XP');
      }
    });

    // Hint button
    const hintBtn = card.querySelector('.btn-hint');
    if (hintBtn) {
      hintBtn.addEventListener('click', () => {
        if (!ex.hints || hintIdx >= ex.hints.length) { toast('No more hints!'); return; }
        hintBox.textContent = `Hint ${hintIdx + 1}: ${ex.hints[hintIdx]}`;
        hintBox.classList.add('visible');
        hintIdx++;
        if (hintIdx >= ex.hints.length) hintBtn.disabled = true;
        state.xp = Math.max(0, state.xp - 2);
        updateXpDisplay();
      });
    }

    // Solution button
    card.querySelector('.btn-solution').addEventListener('click', () => {
      textarea.value = ex.solution;
      toast('Solution revealed. Study it, then try on your own!');
    });

    section.appendChild(card);
  });

  container.appendChild(section);
}

// ─── Lesson Loader ────────────────────────────────────────────────────────────
async function openLesson(trackId, lessonId) {
  const track = state.courses.tracks.find(t => t.id === trackId);
  const lesson = track?.lessons.find(l => l.id === lessonId);
  if (!track || !lesson) return;

  state.currentLesson = { trackId, lessonId };
  showView('lesson');
  updateSidebarActive(trackId, lessonId);

  // Update topbar
  document.getElementById('btn-lesson').style.display = '';
  document.getElementById('btn-lesson').textContent = lesson.title.length > 30
    ? lesson.title.slice(0, 28) + '…'
    : lesson.title;

  const container = document.getElementById('lesson-container');
  container.innerHTML = '<div style="color:var(--text3);padding:40px 0">Loading…</div>';

  // Load markdown
  const mdRaw = await window.api.readFile(`content/${trackId}/${lesson.file}`);
  if (!mdRaw) {
    container.innerHTML = `
      <div class="lesson-header">
        <div class="lesson-breadcrumb">${track.name} <span>/ ${lesson.title}</span></div>
        <div class="lesson-title">${lesson.title}</div>
      </div>
      <div style="padding:40px 0;color:var(--text3);text-align:center">
        <div style="font-size:32px;margin-bottom:16px">🚧</div>
        <div style="font-size:15px;color:var(--text2);margin-bottom:8px">This lesson is being prepared</div>
        <div style="font-size:13px">Check back soon — content is being added regularly.</div>
      </div>`;
    return;
  }

  const { meta, body } = parseMarkdownMeta(mdRaw);

  // Load quiz data
  const quizFile = lesson.file.replace('.md', '-quiz.json');
  const quizRaw = await window.api.readFile(`content/${trackId}/${quizFile}`);
  const quizData = quizRaw ? JSON.parse(quizRaw) : null;

  // Render
  const diffBadge = {
    beginner: '<span class="badge badge-beginner">Beginner</span>',
    intermediate: '<span class="badge badge-intermediate">Intermediate</span>',
    advanced: '<span class="badge badge-advanced">Advanced</span>',
  }[lesson.difficulty || 'beginner'] || '';

  container.innerHTML = `
    <div class="lesson-header">
      <div class="lesson-breadcrumb">${track.name} <span>/ ${lesson.title}</span></div>
      <div class="lesson-title">${lesson.title}</div>
      <div class="lesson-meta">
        ${diffBadge}
        <span>⏱ ~${lesson.estimatedTime} min</span>
        ${state.completed.has(lessonId) ? '<span style="color:var(--green)">✓ Completed</span>' : ''}
      </div>
    </div>
    <div class="lesson-body" id="lesson-body"></div>
  `;

  const bodyEl = document.getElementById('lesson-body');
  bodyEl.innerHTML = marked.parse(body);
  enhanceCodeBlocks(bodyEl);
  renderMath(bodyEl);

  // Quiz + exercises
  renderQuiz(quizData, container);
  renderExercises(quizData, container, lessonId);

  // Navigation
  const prev = getPrevLesson(trackId, lessonId);
  const next = getNextLesson(trackId, lessonId);
  const isCompleted = state.completed.has(lessonId);

  const nav = document.createElement('div');
  nav.className = 'lesson-nav';
  nav.innerHTML = `
    <button class="btn-nav" id="btn-prev" ${!prev ? 'disabled' : ''}>← Previous</button>
    <button class="${isCompleted ? 'btn-nav' : 'btn-complete'}" id="btn-complete">
      ${isCompleted ? '✓ Completed' : 'Mark complete'}
    </button>
    <button class="btn-nav primary" id="btn-next" ${!next ? 'disabled' : ''}>Next →</button>
  `;
  container.appendChild(nav);

  if (prev) nav.querySelector('#btn-prev').addEventListener('click', () => openLesson(trackId, prev.lesson.id));
  if (next) nav.querySelector('#btn-next').addEventListener('click', () => openLesson(trackId, next.lesson.id));

  nav.querySelector('#btn-complete').addEventListener('click', () => {
    if (!state.completed.has(lessonId)) {
      state.completed.add(lessonId);
      state.xp += 50;
      state.streak++;
      updateXpDisplay();
      updateStreakDisplay();
      saveProgress();
      renderSidebar();
      updateOverallProgress();
      toast('Lesson complete! +50 XP 🎉');
      // Re-render nav
      openLesson(trackId, lessonId);
    }
  });

  // Scroll top
  document.getElementById('view-lesson').scrollTop = 0;

  // Pre-warm Pyodide if not ready and lesson has Python
  if (!state.pyodideReady && body.includes('```python')) {
    ensurePyodide();
  }
}

// ─── Roadmap ──────────────────────────────────────────────────────────────────
function renderRoadmap() {
  const view = document.getElementById('view-roadmap');

  const completedAll = state.courses.tracks.flatMap(t => t.lessons).filter(l => state.completed.has(l.id)).length;
  const totalAll = state.courses.tracks.flatMap(t => t.lessons).length;

  view.innerHTML = `
    <div class="roadmap-hero">
      <h1>Your IOAI journey</h1>
      <p>Go from zero to international olympiad level. Complete every track to be ready for IOAI competition.</p>
      <div style="margin-top:16px;font-size:13px;color:var(--text3)">
        ${completedAll} / ${totalAll} lessons completed &nbsp;·&nbsp; ${state.xp} XP
      </div>
    </div>
    <div class="roadmap-grid" id="roadmap-grid"></div>
  `;

  const grid = document.getElementById('roadmap-grid');
  state.courses.tracks.forEach((track, i) => {
    // Check if this track is unlocked
    const prev = state.courses.tracks[i - 1];
    const unlocked = !track.requiresPrev || !prev || getTrackProgress(prev.id) >= 50;

    const pct = getTrackProgress(track.id);
    const done = track.lessons.filter(l => state.completed.has(l.id)).length;

    const card = document.createElement('div');
    card.className = `track-card ${!unlocked ? 'locked' : ''}`;
    card.innerHTML = `
      ${!unlocked ? '<div class="lock-overlay">🔒</div>' : ''}
      <div class="track-card-icon">${track.icon}</div>
      <div class="track-card-title">${track.name}</div>
      <div class="track-card-desc">${track.description}</div>
      <div class="track-card-meta">
        <span>📚 ${track.lessons.length} lessons</span>
        <span>⏱ ${track.totalTime} min</span>
      </div>
      <div class="card-progress-bar">
        <div class="card-progress-fill" style="width:${pct}%"></div>
      </div>
      <div style="font-size:11px;color:var(--text3);margin-top:6px">${done}/${track.lessons.length} complete · ${pct}%</div>
    `;

    if (unlocked) {
      card.addEventListener('click', () => {
        // Open first incomplete lesson, or first lesson
        const firstIncomplete = track.lessons.find(l => !state.completed.has(l.id));
        const target = firstIncomplete || track.lessons[0];
        if (target) openLesson(track.id, target.id);
      });
    }

    grid.appendChild(card);
  });
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
function renderSidebar() {
  const nav = document.getElementById('track-nav');
  nav.innerHTML = '';

  state.courses.tracks.forEach(track => {
    const pct = getTrackProgress(track.id);
    const section = document.createElement('div');
    section.className = 'track-section';

    const isCurrentTrack = state.currentLesson?.trackId === track.id;

    section.innerHTML = `
      <div class="track-header ${isCurrentTrack ? 'open' : ''}" data-track="${track.id}">
        <span class="track-icon">${track.icon}</span>
        <span>${track.name}</span>
        <span class="track-chevron">›</span>
      </div>
      <div class="track-progress-mini">
        <div class="track-progress-mini-fill" style="width:${pct}%"></div>
      </div>
      <div class="lesson-list ${isCurrentTrack ? 'open' : ''}" id="lessons-${track.id}">
        ${track.lessons.map(l => `
          <div class="lesson-item ${state.completed.has(l.id) ? 'completed' : ''} ${state.currentLesson?.lessonId === l.id ? 'active' : ''}"
               data-track="${track.id}" data-lesson="${l.id}">
            <span class="lesson-dot"></span>
            <span class="lesson-title-text">${l.title}</span>
          </div>`).join('')}
      </div>
    `;

    // Toggle track expand
    section.querySelector('.track-header').addEventListener('click', () => {
      const header = section.querySelector('.track-header');
      const list = section.querySelector('.lesson-list');
      header.classList.toggle('open');
      list.classList.toggle('open');
    });

    // Lesson click
    section.querySelectorAll('.lesson-item').forEach(item => {
      item.addEventListener('click', () => {
        openLesson(item.dataset.track, item.dataset.lesson);
      });
    });

    nav.appendChild(section);
  });
}

function updateSidebarActive(trackId, lessonId) {
  document.querySelectorAll('.lesson-item').forEach(el => {
    el.classList.toggle('active', el.dataset.track === trackId && el.dataset.lesson === lessonId);
  });
  // Expand the right track
  document.querySelectorAll('.track-header').forEach(h => {
    if (h.dataset.track === trackId) {
      h.classList.add('open');
      h.nextElementSibling?.nextElementSibling?.classList.add('open'); // lesson-list
    }
  });
}

function updateOverallProgress() {
  const pct = getOverallProgress();
  document.getElementById('overall-bar').style.width = pct + '%';
  document.getElementById('overall-pct').textContent = pct + '%';
}

// ─── View Router ──────────────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(`view-${name}`).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.view === name));
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  // Inject runtime elements
  document.body.insertAdjacentHTML('beforeend', `
    <div id="pyodide-status">
      <span class="status-dot"></span>
      <span class="status-label">Python</span>
    </div>
    <div id="toast"></div>
  `);

  // Inject Pyodide loader script
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js';
  document.head.appendChild(script);

  loadProgress();
  await loadCourses();
  configureMarked();

  renderSidebar();
  renderRoadmap();
  updateXpDisplay();
  updateStreakDisplay();
  updateOverallProgress();

  // Tab buttons
  document.getElementById('btn-roadmap').addEventListener('click', () => {
    showView('roadmap');
    renderRoadmap();
  });
  document.getElementById('btn-lesson').addEventListener('click', () => showView('lesson'));
}

boot();
