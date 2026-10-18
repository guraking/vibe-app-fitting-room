// ============================================================
// 가상 피팅룸
//  - 정밀 모드: Hugging Face Space(IDM-VTON 호환)를 @gradio/client로 호출
//  - 간편 모드: MediaPipe 자세 인식 + 캔버스 합성
// ============================================================

const GRADIO_URL = 'https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js';
const MP_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const SETTINGS_KEY = 'fitting-room-settings-v1';
const DEFAULT_SETTINGS = {
  space: 'yisol/IDM-VTON',
  endpoint: '/tryon',
  token: '',
  steps: 30,
  seed: 42,
  autoMask: true,
  autoCrop: false,
};

const $ = (s) => document.querySelector(s);

// ---------- 상태 ----------
let settings = loadSettings();
let uid = 0;
const state = {
  mode: 'ai',
  person: null,    // { canvas, jpeg, url, w, h, id, pose, poseTried }
  garment: null,   // { canvas, jpeg, url, w, h, cutout }
  category: 'top',
  view: 'result',
  resultUrl: null,
  busy: false,
  job: null,
  cancelled: false,
  adj: { scale: 100, x: 0, y: 0, rot: 0, alpha: 100, bg: 30 },
};
const history = [];

// ---------- 요소 ----------
const el = {
  stageEmpty: $('#stageEmpty'), stageCanvas: $('#stageCanvas'), stageImg: $('#stageImg'),
  busy: $('#busy'), busyText: $('#busyText'), busyTime: $('#busyTime'), cancelBtn: $('#cancelBtn'),
  runBtn: $('#runBtn'), errorBox: $('#errorBox'), errorText: $('#errorText'), toQuick: $('#toQuick'),
  quickStatus: $('#quickStatus'), quickControls: $('#quickControls'),
  viewResult: $('#viewResult'), viewOriginal: $('#viewOriginal'),
  saveHistory: $('#saveHistory'), downloadBtn: $('#downloadBtn'), zoomBtn: $('#zoomBtn'),
  desc: $('#garmentDesc'), toast: $('#toast'),
};

// ---------- 유틸 ----------
function loadSettings() {
  try { return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') }; }
  catch { return { ...DEFAULT_SETTINGS }; }
}
function persistSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* 저장 불가 환경 */ }
}

let toastTimer;
function toast(msg, kind = '') {
  el.toast.textContent = msg;
  el.toast.className = 'toast ' + kind;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 4200);
}

function downloadBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

async function decode(blob) {
  if ('createImageBitmap' in window) {
    try { return await createImageBitmap(blob); } catch { /* 폴백 */ }
  }
  const img = new Image();
  img.src = URL.createObjectURL(blob);
  await img.decode();
  return img;
}

async function prepareImage(blob, maxSide) {
  const src = await decode(blob);
  const k = Math.min(1, maxSide / Math.max(src.width, src.height));
  const w = Math.round(src.width * k);
  const h = Math.round(src.height * k);

  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  canvas.getContext('2d').drawImage(src, 0, 0, w, h);

  // JPEG는 투명 영역이 검게 변하므로 흰 배경으로 평탄화
  const flat = document.createElement('canvas');
  flat.width = w; flat.height = h;
  const fctx = flat.getContext('2d');
  fctx.fillStyle = '#fff';
  fctx.fillRect(0, 0, w, h);
  fctx.drawImage(canvas, 0, 0);
  const jpeg = await new Promise((r) => flat.toBlob(r, 'image/jpeg', 0.92));
  return { canvas, jpeg, url: URL.createObjectURL(jpeg), w, h };
}

// ---------- 배경 제거 (옷 이미지) ----------
function removeBackground(imgData, w, h, tol) {
  const d = imgData.data;
  const corners = [0, w - 1, (h - 1) * w, h * w - 1].map((p) => p * 4);
  let r = 0, g = 0, b = 0;
  corners.forEach((i) => { r += d[i]; g += d[i + 1]; b += d[i + 2]; });
  r /= 4; g /= 4; b /= 4;
  const th = tol * 1.5;
  const th2 = th * th;

  const visited = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const match = (p) => {
    const i = p * 4;
    const dr = d[i] - r, dg = d[i + 1] - g, db = d[i + 2] - b;
    return dr * dr + dg * dg + db * db <= th2 || d[i + 3] < 10;
  };
  const push = (p) => { if (!visited[p] && match(p)) { visited[p] = 1; stack[sp++] = p; } };

  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (sp) {
    const p = stack[--sp];
    d[p * 4 + 3] = 0;
    const x = p % w;
    if (x > 0) push(p - 1);
    if (x < w - 1) push(p + 1);
    if (p >= w) push(p - w);
    if (p < w * (h - 1)) push(p + w);
  }

  // 경계 부드럽게
  const edge = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const p = y * w + x;
      if (d[p * 4 + 3] === 0) continue;
      if (d[(p - 1) * 4 + 3] === 0 || d[(p + 1) * 4 + 3] === 0 || d[(p - w) * 4 + 3] === 0 || d[(p + w) * 4 + 3] === 0) edge.push(p);
    }
  }
  edge.forEach((p) => { d[p * 4 + 3] = 150; });
}

function cropToContent(canvas) {
  const w = canvas.width, h = canvas.height;
  const d = canvas.getContext('2d').getImageData(0, 0, w, h).data;
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (d[(y * w + x) * 4 + 3] > 10) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  const cw = maxX - minX + 1, ch = maxY - minY + 1;
  const out = document.createElement('canvas');
  out.width = cw; out.height = ch;
  out.getContext('2d').drawImage(canvas, minX, minY, cw, ch, 0, 0, cw, ch);
  return out;
}

function buildCutout() {
  const g = state.garment;
  if (!g) return;
  const w = g.canvas.width, h = g.canvas.height;
  const work = document.createElement('canvas');
  work.width = w; work.height = h;
  const ctx = work.getContext('2d');
  ctx.drawImage(g.canvas, 0, 0);
  const data = ctx.getImageData(0, 0, w, h);

  // 이미 투명 배경(PNG)이면 그대로 사용
  let transparent = 0;
  for (let i = 3; i < data.data.length; i += 4 * 40) if (data.data[i] < 250) transparent++;
  if (transparent < 20) {
    removeBackground(data, w, h, state.adj.bg);
    ctx.putImageData(data, 0, 0);
  }
  const cropped = cropToContent(work);
  if (!cropped) {
    toast('배경 제거가 너무 강해 옷이 사라졌어요. "배경 제거" 값을 낮춰주세요.', 'error');
    g.cutout = cropToContent(g.canvas) || g.canvas;
  } else {
    g.cutout = cropped;
  }
}

// ---------- 자세 인식 ----------
let landmarkerPromise = null;
function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      const make = (delegate) => vision.PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate },
        runningMode: 'IMAGE',
        numPoses: 1,
      });
      try { return await make('GPU'); } catch { return await make('CPU'); }
    })();
    landmarkerPromise.catch(() => { landmarkerPromise = null; });
  }
  return landmarkerPromise;
}

function setQuickStatus(msg, kind = '') {
  el.quickStatus.textContent = msg;
  el.quickStatus.className = 'status ' + kind;
}

async function ensurePose() {
  const p = state.person;
  if (!p || p.poseTried) return;
  p.poseTried = true;
  setQuickStatus('자세 인식 모델을 불러오는 중… (처음 한 번만 몇 초 걸려요)');
  try {
    const lm = await getLandmarker();
    const res = lm.detect(p.canvas);
    const pts = res.landmarks && res.landmarks[0];
    if (!pts) throw new Error('자세를 찾지 못함');
    const P = (i) => ({ x: pts[i].x * p.w, y: pts[i].y * p.h, v: pts[i].visibility ?? 1 });
    p.pose = { ls: P(11), rs: P(12), lh: P(23), rh: P(24), la: P(27), ra: P(28) };
    setQuickStatus('자세 인식 완료 — 슬라이더로 위치를 미세 조정하세요.', 'ok');
  } catch (e) {
    console.warn('pose failed', e);
    p.pose = null;
    setQuickStatus('자세를 찾지 못해 기본 위치에 배치했어요. 슬라이더로 맞춰주세요.', 'warn');
  }
  if (state.person === p) refresh();
}

// ---------- 간편 모드 배치 ----------
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function bodyMetrics(p) {
  const W = p.w, H = p.h, P = p.pose;
  if (!P) {
    return {
      sMid: { x: W / 2, y: H * 0.22 }, hMid: { x: W / 2, y: H * 0.52 },
      shoulderW: W * 0.3, hipW: W * 0.2, ankleY: H * 0.93, angle: 0,
    };
  }
  const sMid = mid(P.ls, P.rs);
  const shoulderW = dist(P.ls, P.rs);
  let hMid, hipW;
  if (P.lh.v > 0.4 && P.rh.v > 0.4) { hMid = mid(P.lh, P.rh); hipW = dist(P.lh, P.rh); }
  else { hMid = { x: sMid.x, y: sMid.y + shoulderW * 1.3 }; hipW = shoulderW * 0.7; }
  const torso = dist(sMid, hMid);
  const ankleY = (P.la.v > 0.4 && P.ra.v > 0.4) ? (P.la.y + P.ra.y) / 2 : hMid.y + torso * 1.85;
  const [a, b] = P.ls.x < P.rs.x ? [P.ls, P.rs] : [P.rs, P.ls];
  let angle = Math.atan2(b.y - a.y, b.x - a.x);
  angle = Math.max(-0.4, Math.min(0.4, angle));
  return { sMid, hMid, shoulderW, hipW, ankleY, angle };
}

function layout(p, g, cat) {
  const m = bodyMetrics(p);
  const torso = dist(m.sMid, m.hMid);
  let targetH, targetW, wH, ax, ay;

  if (cat === 'bottom') {
    ax = m.hMid.x; ay = m.hMid.y - torso * 0.12;
    targetH = (m.ankleY - m.hMid.y) + torso * 0.12;
    targetW = m.hipW * 2.2; wH = 0.75;
  } else if (cat === 'dress') {
    ax = m.sMid.x; ay = m.sMid.y - torso * 0.12;
    const kneeY = m.hMid.y + (m.ankleY - m.hMid.y) * 0.5;
    targetH = (kneeY - m.sMid.y) + torso * 0.12;
    targetW = m.shoulderW * 1.8; wH = 0.6;
  } else {
    ax = m.sMid.x; ay = m.sMid.y - torso * 0.12;
    targetH = torso * 1.3;
    targetW = m.shoulderW * 1.9; wH = 0.5;
  }

  const base = wH * (targetH / g.height) + (1 - wH) * (targetW / g.width);
  const a = state.adj;
  return {
    x: ax + (a.x / 100) * p.w,
    y: ay + (a.y / 100) * p.h,
    s: base * (a.scale / 100),
    rot: m.angle + (a.rot * Math.PI) / 180,
  };
}

function drawGarment(ctx) {
  const p = state.person, g = state.garment.cutout;
  const L = layout(p, g, state.category);
  ctx.save();
  ctx.translate(L.x, L.y);
  ctx.rotate(L.rot);
  ctx.scale(L.s, L.s);
  ctx.globalAlpha = state.adj.alpha / 100;
  ctx.shadowColor = 'rgba(0,0,0,.28)';
  ctx.shadowBlur = 14;
  ctx.shadowOffsetY = 5;
  ctx.drawImage(g, -g.width / 2, 0);
  ctx.restore();
}

function renderQuick() {
  const p = state.person, c = el.stageCanvas;
  if (c.width !== p.w || c.height !== p.h) { c.width = p.w; c.height = p.h; }
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, p.w, p.h);
  ctx.drawImage(p.canvas, 0, 0);
  if (state.garment && state.garment.cutout && effectiveView() === 'result') drawGarment(ctx);
}

let rafId = 0;
function scheduleRender() {
  if (rafId) return;
  rafId = requestAnimationFrame(() => {
    rafId = 0;
    if (state.mode === 'quick' && state.person) renderQuick();
  });
}

// ---------- 화면 갱신 ----------
function hasResult() {
  return state.mode === 'ai' ? !!state.resultUrl : !!(state.garment && state.garment.cutout && state.person);
}
function effectiveView() {
  return hasResult() ? state.view : 'original';
}

function refresh() {
  const hasP = !!state.person, hasG = !!state.garment;
  document.body.dataset.mode = state.mode;

  el.stageEmpty.hidden = hasP;
  el.stageCanvas.hidden = !(hasP && state.mode === 'quick');
  el.stageImg.hidden = !(hasP && state.mode === 'ai');

  if (hasP && state.mode === 'quick') renderQuick();
  if (hasP && state.mode === 'ai') {
    const src = (state.resultUrl && effectiveView() === 'result') ? state.resultUrl : state.person.url;
    if (el.stageImg.getAttribute('src') !== src) el.stageImg.src = src;
  }

  const res = hasResult();
  const ev = effectiveView();
  el.viewResult.disabled = !res;
  el.viewOriginal.disabled = !hasP;
  el.viewResult.classList.toggle('is-active', ev === 'result');
  el.viewOriginal.classList.toggle('is-active', ev === 'original');
  el.viewResult.setAttribute('aria-pressed', String(ev === 'result'));
  el.viewOriginal.setAttribute('aria-pressed', String(ev === 'original'));

  el.zoomBtn.disabled = !hasP;
  el.saveHistory.disabled = !res;
  el.downloadBtn.disabled = !res;
  el.runBtn.disabled = !(hasP && hasG) || state.busy;
  el.quickControls.hidden = !(state.mode === 'quick' && hasP && hasG);

  document.querySelectorAll('.mode-btn').forEach((b) => {
    const on = b.dataset.mode === state.mode;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-pressed', String(on));
  });
}

// ---------- 이미지 입력 ----------
async function setPerson(blob) {
  try {
    const prep = await prepareImage(blob, 1280);
    if (state.person) URL.revokeObjectURL(state.person.url);
    state.person = { ...prep, id: ++uid, pose: undefined, poseTried: false };
    state.resultUrl = null;
    state.view = 'result';
    showPreview('Person', prep.url);
    el.errorBox.hidden = true;
    refresh();
    if (state.mode === 'quick') ensurePose();
  } catch (e) {
    console.warn(e);
    toast('전신 사진을 읽을 수 없어요. 다른 이미지로 시도해 주세요.', 'error');
  }
}

async function setGarment(blob) {
  try {
    const prep = await prepareImage(blob, 1024);
    if (state.garment) URL.revokeObjectURL(state.garment.url);
    state.garment = { ...prep, cutout: null };
    state.resultUrl = null;
    state.view = 'result';
    buildCutout();
    showPreview('Garment', prep.url);
    el.errorBox.hidden = true;
    refresh();
  } catch (e) {
    console.warn(e);
    toast('옷 이미지를 읽을 수 없어요. 다른 이미지로 시도해 주세요.', 'error');
  }
}

function showPreview(kind, url) {
  $('#prev' + kind).src = url;
  $('#prev' + kind).hidden = false;
  $('#hint' + kind).hidden = true;
  $('#clear' + kind).hidden = false;
  $('#drop' + kind).classList.add('has-image');
}
function clearSlot(kind) {
  const key = kind === 'Person' ? 'person' : 'garment';
  if (state[key]) URL.revokeObjectURL(state[key].url);
  state[key] = null;
  state.resultUrl = null;
  $('#prev' + kind).hidden = true;
  $('#prev' + kind).removeAttribute('src');
  $('#hint' + kind).hidden = false;
  $('#clear' + kind).hidden = true;
  $('#drop' + kind).classList.remove('has-image');
  $('#file' + kind).value = '';
  refresh();
}

function bindDrop(kind, handler) {
  const drop = $('#drop' + kind), input = $('#file' + kind);
  drop.addEventListener('click', (e) => { if (e.target.closest('.drop-clear')) return; input.click(); });
  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => { if (input.files[0]) handler(input.files[0]); });
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('is-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('is-over');
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('image/'));
    if (f) handler(f); else toast('이미지 파일만 올릴 수 있어요.', 'error');
  });
  $('#clear' + kind).addEventListener('click', () => clearSlot(kind));
}
bindDrop('Person', setPerson);
bindDrop('Garment', setGarment);

// 붙여넣기: 전신 사진이 없으면 사진으로, 있으면 옷으로
document.addEventListener('paste', (e) => {
  if (e.target.matches('input[type="text"], input[type="url"], input[type="password"], input[type="number"]')) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (!item) return;
  const blob = item.getAsFile();
  if (!blob) return;
  e.preventDefault();
  if (!state.person) { setPerson(blob); toast('전신 사진으로 붙여넣었어요.'); }
  else { setGarment(blob); toast('옷 이미지로 붙여넣었어요.'); }
});

// URL 불러오기
$('#urlBtn').addEventListener('click', async () => {
  const url = $('#urlInput').value.trim();
  if (!url) return;
  try {
    const res = await fetch(url, { mode: 'cors' });
    if (!res.ok) throw new Error(res.status);
    const blob = await res.blob();
    if (!blob.type.startsWith('image/')) throw new Error('이미지가 아님');
    await setGarment(blob);
  } catch {
    toast('이 주소는 브라우저 보안(CORS) 때문에 불러올 수 없어요. 이미지를 복사해서 붙여넣거나 저장 후 올려주세요.', 'error');
  }
});

// ---------- 모드 / 보기 / 옵션 ----------
document.querySelectorAll('.mode-btn').forEach((b) => {
  b.addEventListener('click', () => {
    if (state.busy) return;
    state.mode = b.dataset.mode;
    state.view = 'result';
    el.stageImg.removeAttribute('src');
    refresh();
    if (state.mode === 'quick') ensurePose();
  });
});
el.toQuick.addEventListener('click', () => {
  state.mode = 'quick';
  el.errorBox.hidden = true;
  el.stageImg.removeAttribute('src');
  refresh();
  ensurePose();
});

el.viewResult.addEventListener('click', () => { state.view = 'result'; refresh(); });
el.viewOriginal.addEventListener('click', () => { state.view = 'original'; refresh(); });

document.querySelectorAll('input[name="category"]').forEach((r) => {
  r.addEventListener('change', () => {
    state.category = r.value;
    state.resultUrl = state.mode === 'ai' ? null : state.resultUrl;
    refresh();
  });
});

// 슬라이더
const sliders = [
  ['Scale', 'scale', (v) => v + '%'],
  ['X', 'x', (v) => v],
  ['Y', 'y', (v) => v],
  ['Rot', 'rot', (v) => v + '°'],
  ['Alpha', 'alpha', (v) => v + '%'],
  ['Bg', 'bg', (v) => v],
];
let bgTimer;
sliders.forEach(([id, key, fmt]) => {
  const input = $('#rng' + id), out = $('#out' + id);
  input.addEventListener('input', () => {
    const v = Number(input.value);
    state.adj[key] = v;
    out.textContent = fmt(v);
    if (key === 'bg') {
      clearTimeout(bgTimer);
      bgTimer = setTimeout(() => { buildCutout(); scheduleRender(); }, 120);
    } else {
      scheduleRender();
    }
  });
});
$('#resetAdj').addEventListener('click', () => {
  const bg = state.adj.bg;
  state.adj = { scale: 100, x: 0, y: 0, rot: 0, alpha: 100, bg };
  sliders.forEach(([id, key, fmt]) => {
    $('#rng' + id).value = state.adj[key];
    $('#out' + id).textContent = fmt(state.adj[key]);
  });
  scheduleRender();
});

// ---------- 정밀 모드 (Hugging Face Space) ----------
function statusText(ev) {
  if (ev.stage === 'pending') {
    if (typeof ev.position === 'number') {
      const size = ev.size ? ` / 총 ${ev.size}명` : '';
      const eta = ev.eta ? ` · 약 ${Math.ceil(ev.eta)}초` : '';
      return `대기열 ${ev.position + 1}번째${size}${eta}`;
    }
    return '서버에 요청을 보내는 중…';
  }
  if (ev.stage === 'generating') return '이미지를 생성하는 중…';
  return null;
}

function friendlyError(e) {
  const msg = (e && (e.message || e.detail || e.statusText)) || String(e);
  if (/quota|exceeded|zerogpu/i.test(msg)) {
    return '무료 GPU 사용량 한도에 도달했어요. 설정에서 Hugging Face 토큰을 넣거나, 잠시 후 다시 시도하거나, 간편 모드를 이용해 보세요.';
  }
  if (/not found|404|sleep|build|pause|runtime/i.test(msg)) {
    return '이 Space가 현재 꺼져 있거나 주소가 바뀐 것 같아요. 설정에서 다른 Space 주소를 확인해 주세요.';
  }
  return `AI 서버 호출에 실패했어요. (${String(msg).slice(0, 160)})`;
}

let timerId;
function setBusy(on, text) {
  state.busy = on;
  el.busy.hidden = !on;
  clearInterval(timerId);
  if (on) {
    el.busyText.textContent = text || '준비 중…';
    const t0 = Date.now();
    el.busyTime.textContent = '0초';
    timerId = setInterval(() => {
      el.busyTime.textContent = `${Math.floor((Date.now() - t0) / 1000)}초`;
    }, 1000);
  }
  refresh();
}

async function runAI() {
  if (!state.person || !state.garment || state.busy) return;
  el.errorBox.hidden = true;
  state.cancelled = false;
  setBusy(true, '서버에 연결하는 중…');

  try {
    const mod = await import(GRADIO_URL);
    const { Client } = mod;
    const wrap = (blob) => (mod.handle_file ? mod.handle_file(blob) : blob);

    const opts = settings.token ? { hf_token: settings.token } : {};
    const client = await Client.connect(settings.space, opts);

    const job = client.submit(settings.endpoint, {
      dict: { background: wrap(state.person.jpeg), layers: [], composite: null },
      garm_img: wrap(state.garment.jpeg),
      garment_des: el.desc.value.trim() || 'a garment',
      is_checked: settings.autoMask,
      is_checked_crop: settings.autoCrop,
      denoise_steps: Number(settings.steps) || 30,
      seed: Number(settings.seed) || 42,
    });
    state.job = job;

    let output = null;
    for await (const ev of job) {
      if (ev.type === 'status') {
        if (ev.stage === 'error') throw new Error(ev.message || '서버 오류');
        const t = statusText(ev);
        if (t) el.busyText.textContent = t;
      } else if (ev.type === 'data') {
        output = ev.data;
      }
    }
    if (state.cancelled) return;

    const first = Array.isArray(output) ? output[0] : null;
    const url = first && (first.url || first.path);
    if (!url) throw new Error('결과 이미지를 받지 못했어요.');

    state.resultUrl = url;
    state.view = 'result';
    el.stageImg.removeAttribute('src');
    toast('완성! "원본" 버튼으로 비교해 보세요.');
  } catch (e) {
    if (!state.cancelled) {
      console.warn(e);
      el.errorText.textContent = friendlyError(e);
      el.errorBox.hidden = false;
    }
  } finally {
    state.job = null;
    setBusy(false);
  }
}
el.runBtn.addEventListener('click', runAI);
el.cancelBtn.addEventListener('click', () => {
  state.cancelled = true;
  try { state.job && state.job.cancel && state.job.cancel(); } catch { /* 무시 */ }
  setBusy(false);
  toast('취소했어요.');
});

// ---------- 저장 / 기록 ----------
async function currentBlob() {
  if (state.mode === 'quick') {
    renderQuick();
    return new Promise((r) => el.stageCanvas.toBlob(r, 'image/png'));
  }
  const res = await fetch(state.resultUrl);
  if (!res.ok) throw new Error('다운로드 실패');
  return res.blob();
}

el.downloadBtn.addEventListener('click', async () => {
  if (!hasResult()) return;
  try {
    const blob = await currentBlob();
    const ext = blob.type.includes('webp') ? 'webp' : blob.type.includes('jpeg') ? 'jpg' : 'png';
    downloadBlob(blob, `가상피팅_${state.mode === 'quick' ? '간편' : 'AI'}.${ext}`);
  } catch {
    if (state.resultUrl) {
      window.open(state.resultUrl, '_blank', 'noopener');
      toast('직접 저장이 막혀서 새 탭으로 열었어요. 이미지를 우클릭해 저장하세요.');
    } else {
      toast('저장에 실패했어요.', 'error');
    }
  }
});

const catNames = { top: '상의', bottom: '하의', dress: '원피스' };
el.saveHistory.addEventListener('click', () => {
  if (!hasResult()) return;
  let src;
  if (state.mode === 'quick') {
    renderQuick();
    src = el.stageCanvas.toDataURL('image/jpeg', 0.88);
  } else {
    src = state.resultUrl;
  }
  history.unshift({ src, label: `${state.mode === 'quick' ? '간편' : 'AI'} · ${catNames[state.category]} ${history.length + 1}` });
  if (history.length > 12) history.pop();
  renderHistory();
  toast('기록에 담았어요.');
});

function openLightbox(src, alt) {
  $('#lightboxImg').src = src;
  $('#lightboxImg').alt = alt || '확대한 이미지';
  $('#lightbox').showModal();
}

function renderHistory() {
  const list = $('#historyList');
  list.innerHTML = '';
  $('#historyEmpty').hidden = history.length > 0;
  history.forEach((h) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'history-item';
    b.title = h.label;
    const img = document.createElement('img');
    img.src = h.src;
    img.alt = h.label;
    const s = document.createElement('span');
    s.textContent = h.label;
    b.append(img, s);
    b.addEventListener('click', () => openLightbox(h.src, h.label));
    list.appendChild(b);
  });
}
$('#closeLightbox').addEventListener('click', () => $('#lightbox').close());

// ---------- 확대 보기 ----------
function openZoom() {
  if (!state.person) return;
  let src;
  if (state.mode === 'quick') {
    renderQuick();
    src = el.stageCanvas.toDataURL('image/png');
  } else {
    src = el.stageImg.getAttribute('src') || state.person.url;
  }
  openLightbox(src, '확대한 결과');
}
el.zoomBtn.addEventListener('click', openZoom);
el.stageCanvas.addEventListener('click', openZoom);
el.stageImg.addEventListener('click', openZoom);

// ---------- 설정 ----------
const dlg = $('#settingsDialog');
function fillSettings() {
  $('#setSpace').value = settings.space;
  $('#setEndpoint').value = settings.endpoint;
  $('#setToken').value = settings.token;
  $('#setSteps').value = settings.steps;
  $('#setSeed').value = settings.seed;
  $('#setMask').checked = settings.autoMask;
  $('#setCrop').checked = settings.autoCrop;
}
$('#openSettings').addEventListener('click', () => { fillSettings(); dlg.showModal(); });
$('#closeSettings').addEventListener('click', () => dlg.close());
$('#resetSettings').addEventListener('click', () => { settings = { ...DEFAULT_SETTINGS }; fillSettings(); });
$('#settingsForm').addEventListener('submit', () => {
  settings = {
    space: $('#setSpace').value.trim() || DEFAULT_SETTINGS.space,
    endpoint: $('#setEndpoint').value.trim() || DEFAULT_SETTINGS.endpoint,
    token: $('#setToken').value.trim(),
    steps: Math.min(40, Math.max(20, Number($('#setSteps').value) || 30)),
    seed: Number($('#setSeed').value) || 42,
    autoMask: $('#setMask').checked,
    autoCrop: $('#setCrop').checked,
  };
  persistSettings();
  toast('설정을 저장했어요.');
});

// ---------- 시작 ----------
renderHistory();
refresh();