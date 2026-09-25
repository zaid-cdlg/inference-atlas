// Page wiring. Every external string is rendered with textContent or an attribute setter,
// never innerHTML.
import { parseState, serializeState, urlSyncer } from './state.js';
import { plan, usd, shortGpu, ctxLabel } from './planner.js';
import {
  USE_CASES, precisionOptions, fmtCount, selfHostPerDay, CHART_MIN_TPD, CHART_MAX_TPD,
} from './math.js';

const $ = (id) => document.getElementById(id);
const el = (tag, text, cls) => {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (cls) e.className = cls;
  return e;
};
const safeUrl = (u) => (typeof u === 'string' && u.startsWith('https://') ? u : null);
const link = (text, href) => {
  const a = el('a', text);
  const url = safeUrl(href);
  if (url) a.href = url;
  return a;
};

const PREC_LABEL = { fp16: 'FP16', fp8: 'FP8', int4: 'INT4', mxfp4: 'MXFP4' };
const NOTICE = {
  model_removed: 'That model is no longer listed. Showing the example instead.',
  gpu_unknown: "That GPU isn't in our list. Showing the H100 instead.",
  prec_unavailable: "That precision isn't available for this model and GPU, so the best available one is shown.",
  link_invalid: "Part of that link wasn't valid, so defaults are used for it.",
};
const USE_HINT = {
  chat: 'Replies within 50 ms per token, 4K-token conversations.',
  agents: '16K-token tasks that reuse a long prompt, 150 ms per token.',
  batch: 'Offline jobs: no speed target, as much throughput as possible.',
};

let data;
let state;
let ctx;
let last;
let played = false;
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const sync = urlSyncer(window);

const displayName = (m) => m.name.replace(/^[^:]+:\s*/, '') || m.hf_id;
const int4For = (id) => data.quality.quantized_repos[id]?.int4 ?? null;

async function load() {
  try {
    const get = async (p) => {
      const r = await fetch(p);
      if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
      return r.json();
    };
    const [models, gpus, quality] = await Promise.all(
      ['data/models.json', 'data/gpus.json', 'data/quality.json'].map(get),
    );
    const byId = new Map(models.models.map((m) => [m.hf_id, m]));
    const featured = models.featured.filter((id) => byId.has(id));
    data = { models, byId, featured, gpus: gpus.gpus, gpuMeta: gpus._meta, quality };
  } catch {
    showLoadError();
    return;
  }
  ctx = {
    models: data.featured,
    gpus: Object.fromEntries(data.gpus.map((g) => [g.id, g])),
    precOk: (id, gpuId, prec) => {
      const arch = data.byId.get(id).arch;
      const list = gpuId ? [ctx.gpus[gpuId]] : data.gpus;
      return list.some((g) => precisionOptions(arch, g, Boolean(int4For(id)))[prec] === null);
    },
  };
  const parsed = parseState(new URLSearchParams(location.search), ctx);
  state = parsed.state;
  initControls();
  showNotices(parsed.notices);
  render();
}

function showLoadError() {
  const box = $('notices');
  box.replaceChildren();
  const p = el('p', "Couldn't load the model list. ", 'notice error');
  const retry = el('button', 'Retry', 'btn secondary');
  retry.type = 'button';
  retry.addEventListener('click', () => location.reload());
  p.append(retry);
  box.append(p);
  $('model').placeholder = 'Model list unavailable';
  $('verdict').textContent = '';
}

function showNotices(keys) {
  $('notices').replaceChildren(...keys.map((k) => el('p', NOTICE[k], 'notice')));
}

// Every edit goes back through parseState, so the page and the link share one set of rules.
function update(changes) {
  const qs = new URLSearchParams(serializeState(state));
  for (const [k, v] of Object.entries(changes)) {
    if (v === null || v === '') qs.delete(k);
    else qs.set(k, String(v));
  }
  state = parseState(qs, ctx).state;
  render();
}

function initControls() {
  const list = $('model-list');
  for (const id of data.featured) {
    const o = document.createElement('option');
    o.value = displayName(data.byId.get(id));
    o.label = id;
    list.append(o);
  }
  const input = $('model');
  input.disabled = false;
  input.placeholder = 'Search models, e.g. qwen';
  const pick = () => {
    const q = input.value.trim().toLowerCase();
    if (!q) return;
    const id = data.featured.find((f) => f.toLowerCase() === q || displayName(data.byId.get(f)).toLowerCase() === q);
    if (!id) {
      $('model-hint').textContent = `No model matches '${input.value.trim()}'. Try a family name like 'qwen'.`;
      return;
    }
    $('model-hint').textContent = '';
    update({ model: id, prec: ctx.precOk(id, state.gpu, state.prec ?? 'fp16') ? state.prec : null });
  };
  input.addEventListener('change', pick);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') pick(); });
  input.addEventListener('focus', () => input.select());

  for (const b of document.querySelectorAll('[data-use]')) {
    b.addEventListener('click', () => update({
      use: b.dataset.use, max_ctx: null, avg_ctx: null, r: null, cache: null, batch: null,
    }));
  }

  const gpu = $('gpu');
  gpu.append(new Option('Cheapest that fits (auto)', ''));
  for (const g of data.gpus) {
    gpu.append(new Option(`${shortGpu(g)} · $${g.usd_per_hr.toFixed(2)}/hr${g.verify ? ' (estimate)' : ''}`, g.id));
  }
  gpu.addEventListener('change', () => {
    const id = gpu.value || null;
    update({ gpu: id, prec: state.prec && !ctx.precOk(state.model, id, state.prec) ? null : state.prec });
  });
  $('prec').addEventListener('change', (e) => update({ prec: e.target.value || null }));

  for (const f of document.querySelectorAll('[data-param]')) {
    f.addEventListener('change', () => update({ [f.dataset.param]: f.value.trim() }));
  }
  $('avg').addEventListener('input', (e) => update({ avg_ctx: e.target.value }));

  $('copy-cmd').addEventListener('click', () => copy($('command').textContent, $('copy-cmd'), $('command')));
  $('copy-link').addEventListener('click', () => {
    const qs = serializeState(state);
    copy(`${location.origin}${location.pathname}${qs ? `?${qs}` : ''}`, $('copy-link'));
  });
  $('how-link').addEventListener('click', () => { $('how').open = true; });

  $('asof').textContent = data.models.as_of;
  $('gpudate').textContent = data.gpuMeta.checked_date;
  renderQuality();

  new ResizeObserver(() => last && drawChart(last)).observe($('chart'));
}

async function copy(text, button, selectable) {
  const status = $('copy-status');
  try {
    await navigator.clipboard.writeText(text);
    const label = button.textContent;
    button.textContent = 'Copied';
    status.textContent = '';
    setTimeout(() => { button.textContent = label; }, 2000);
  } catch {
    status.textContent = 'Select and copy manually.';
    if (selectable) getSelection().selectAllChildren(selectable);
  }
}

function render() {
  const model = data.byId.get(state.model);
  const p = plan({ model, gpus: data.gpus, state, int4: int4For(state.model) });
  last = { p, model };
  sync(state);

  // Controls
  $('model').value = displayName(model);
  for (const b of document.querySelectorAll('[data-use]')) b.setAttribute('aria-pressed', String(b.dataset.use === state.use));
  $('use-hint').textContent = USE_HINT[state.use];
  $('gpu').value = state.gpu ?? '';
  const gpuNotes = [];
  if (!state.gpu) gpuNotes.push(`Picked ${shortGpu(p.gpu)}: lowest cost per token that fits.`);
  if (p.amdHint) {
    const name = p.amdHint.gpu.name.replace(/^AMD Instinct /, '').replace(/ \d+GB$/, '');
    gpuNotes.push(`An AMD ${name} may cost less here (about ${usd(p.amdHint.selfPerM)} per 1M tokens). Pick it under GPU. It needs vLLM's ROCm build.`);
  }
  if (p.gpu.verify) gpuNotes.push(`Estimate: ${p.gpu.verify_reason}`);
  $('gpu-hint').textContent = gpuNotes.join(' ');
  renderPrecision(p, model);
  for (const f of document.querySelectorAll('[data-param]')) {
    if (document.activeElement !== f) f.value = state[f.dataset.param] ?? '';
  }

  // Verdict
  const verdict = $('verdict');
  verdict.textContent = p.verdict;
  verdict.classList.toggle('long', p.verdict.length > 70);
  $('slo').hidden = !p.slo;
  $('slo').textContent = p.slo ?? '';
  const ok = !p.error;
  $('stats').textContent = ok
    ? `${p.tp}× ${shortGpu(p.gpu)} · ${p.users} users at once · ${usd(p.selfPerM)}${p.api ? ` vs ${usd(p.api.perM)}` : ''} per 1M tokens`
    : '';
  $('explain-box').hidden = !ok;
  if (ok) renderExplain(p, model);

  // Chart
  const showChart = ok && Boolean(p.api);
  $('chart-fig').hidden = !showChart;
  const note = !ok || !p.api ? '' : p.be.kind === 'api' ? 'API is cheaper across 100K to 10B tokens/day.'
    : p.be.kind === 'self' ? 'Self-hosting is cheaper from the first GPU.' : '';
  $('chart-note').hidden = !note;
  $('chart-note').textContent = note;
  if (showChart) drawChart(last);

  // Command
  $('command-card').hidden = !ok;
  $('cmd-error').hidden = ok;
  $('cmd-error').textContent = ok ? '' : p.error;
  if (ok) {
    $('command').replaceChildren(el('code', p.command));
    const notes = [];
    const q = int4For(state.model);
    if (p.prec === 'int4' && q) notes.push(`Uses ${q.repo}${q.publisher === 'community' ? ' (community quant)' : ''}.`);
    if (p.gpu.vendor === 'amd') notes.push("This runs on AMD. Install vLLM's ROCm build, not the default CUDA one.");
    if (model.arch.quant) notes.push(`The weights are published in ${PREC_LABEL[model.arch.quant]}, so vLLM loads them as they are.`);
    $('cmd-note').textContent = notes.join(' ');
  }

  // KV sim
  $('kv').hidden = !ok;
  if (ok) renderKv(p, model);
}

function renderPrecision(p, model) {
  const sel = $('prec');
  const q = int4For(state.model);
  sel.replaceChildren(new Option(`Best available (${PREC_LABEL[p.prec]})`, ''));
  const reasons = [];
  for (const k of ['fp16', 'fp8', 'int4', 'mxfp4']) {
    const reason = p.precOptions[k];
    if (k === 'mxfp4' && reason) continue;
    let label = PREC_LABEL[k];
    if (k === 'int4' && q) label += q.publisher === 'community' ? ' (community quant)' : ` (${q.method.toUpperCase()})`;
    const o = new Option(label, k);
    o.disabled = Boolean(reason);
    sel.append(o);
    if (reason && !reasons.includes(reason)) reasons.push(k === 'fp16' || model.arch.quant ? reason : `${PREC_LABEL[k]}: ${reason}`);
  }
  sel.value = state.prec ?? '';
  $('prec-hint').textContent = reasons.join(' ');
}

function renderExplain(p, model) {
  const rows = [
    ['GPUs', `${p.tp}× ${shortGpu(p.gpu)}`, `One copy of the model needs ${p.tp === 1 ? 'one GPU' : `${p.tp} GPUs working together`}: the weights plus one conversation of the longest length must fit in 90% of each GPU's memory.`],
    ['Users at once', `${p.users}`, `How many average conversations (${fmtCount(p.avgCtx)} tokens) fit in memory at the same time. The server runs ${p.batch} at once${p.batch < p.users ? ' to stay under the speed target' : ''}.`],
    ['Speed', `${Math.round(p.itl * 1000)} ms per token`, `Time to write one token for each user while ${p.batch} share the GPU${p.tp > 1 ? 's' : ''}: about ${Math.max(1, Math.round(1 / p.itl))} tokens per second each. Reading is roughly 4 words per second.`],
    ['Throughput', `${fmtCount(p.tps)} tokens/s`, 'All tokens one copy of the model serves per second, prompts and answers together.'],
    ['Self-host price', `${usd(p.selfPerM)} per 1M tokens`, `GPU rent (${p.tp} × $${p.gpu.usd_per_hr.toFixed(2)}/hr) divided by the tokens served, with the GPUs busy ${state.util}% of the time.`],
  ];
  if (p.api) {
    rows.push(['API price', `${usd(p.api.perM)} per 1M tokens`, `OpenRouter's cheapest paid price for ${displayName(model)}, mixing ${state.r} input tokens per output token.${state.cache > 0 && !p.api.cachePriced ? ' This API lists no cached-input price, so cached tokens are charged at the full input price.' : ''}`]);
  }
  if (model.arch.moe) rows.push(['Approximate', 'mixture of experts', 'Only some experts run per token. Speed assumes each step reads just those, which is optimistic at large batches.']);
  if (model.arch.attn?.approx) rows.push(['Approximate', 'attention layout', 'We could not read which layers keep the full conversation, so we assume all do. Real memory use may be lower.']);
  $('explain').replaceChildren(...rows.flatMap(([k, v, why]) => {
    const dt = el('dt', k);
    const dd = el('dd');
    dd.append(el('span', `${v}. `, 'num'), why);
    return [dt, dd];
  }));
}

// Break-even chart: log x (tokens/day), log y ($/day), inline SVG.
const NS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text != null) e.textContent = text;
  return e;
};

function drawChart({ p }) {
  const box = $('chart');
  if (!p.api || p.error || box.clientWidth === 0) return;
  const W = box.clientWidth;
  const H = box.clientHeight;
  const phone = W < 600;
  const m = { l: 56, r: 16, t: 28, b: 28 };
  const selfDay = (t) => selfHostPerDay(t, p.capacity, p.tp, p.gpu.usd_per_hr);
  const apiDay = (t) => (p.api.perM * t) / 1e6;
  const lx0 = Math.log10(CHART_MIN_TPD);
  const lx1 = Math.log10(CHART_MAX_TPD);
  const N = 400;
  const ts = Array.from({ length: N + 1 }, (_, i) => 10 ** (lx0 + ((lx1 - lx0) * i) / N));
  const ys = ts.flatMap((t) => [selfDay(t), apiDay(t)]).filter((v) => v > 0);
  const ly0 = Math.floor(Math.log10(Math.min(...ys)));
  const ly1 = Math.ceil(Math.log10(Math.max(...ys)));
  const x = (t) => m.l + ((Math.log10(t) - lx0) / (lx1 - lx0)) * (W - m.l - m.r);
  const y = (v) => H - m.b - ((Math.log10(v) - ly0) / (ly1 - ly0)) * (H - m.t - m.b);

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${p.verdict}. Chart of cost per day: API line against self-hosting steps, from 100K to 10B tokens per day.` });
  const labels = phone ? [1e5, 1e7, 1e10] : [1e5, 1e6, 1e7, 1e8, 1e9, 1e10];
  for (const t of labels) svg.append(svgEl('text', { x: x(t), y: H - 8, 'text-anchor': 'middle' }, fmtCount(t)));
  const yStep = Math.max(1, Math.ceil((ly1 - ly0) / (phone ? 3 : 5)));
  for (let e = ly0; e <= ly1; e += yStep) {
    svg.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: y(10 ** e), y2: y(10 ** e), class: 'axis' }));
    svg.append(svgEl('text', { x: m.l - 8, y: y(10 ** e) + 4, 'text-anchor': 'end' }, `$${fmtCount(10 ** e)}`));
  }
  svg.append(svgEl('text', { x: 0, y: 12 }, '$ per day'));

  let d = '';
  ts.forEach((t, i) => {
    const px = x(t).toFixed(1);
    const py = y(selfDay(t)).toFixed(1);
    d += i === 0 ? `M${px},${py}` : `H${px}V${py}`;
  });
  svg.append(svgEl('path', { d: `${d}V${H - m.b}H${m.l}Z`, class: 'self-area' }));
  svg.append(svgEl('path', { d, class: 'self' }));
  svg.append(svgEl('path', { d: `M${x(ts[0])},${y(apiDay(ts[0]))}L${x(ts[N])},${y(apiDay(ts[N]))}`, class: 'api' }));

  const endT = ts[Math.round(N * 0.97)];
  svg.append(svgEl('text', { x: x(endT), y: y(apiDay(endT)) - 8, 'text-anchor': 'end', class: 'label' }, 'API'));
  const selfLabelT = ts[Math.round(N * 0.08)];
  svg.append(svgEl('text', { x: x(selfLabelT), y: y(selfDay(selfLabelT)) - 8, class: 'label' }, 'Self-host'));

  if (state.tpd >= CHART_MIN_TPD && state.tpd <= CHART_MAX_TPD) {
    svg.append(svgEl('line', { x1: x(state.tpd), x2: x(state.tpd), y1: m.t, y2: H - m.b, class: 'you' }));
    svg.append(svgEl('text', { x: x(state.tpd) + 4, y: m.t + 12 }, 'you'));
  }
  if (p.be.kind === 'cross') {
    svg.append(svgEl('circle', { cx: x(p.be.tpd), cy: y(apiDay(p.be.tpd)), r: 6, class: 'cross' }));
  }
  const probe = svgEl('line', { y1: m.t, y2: H - m.b, class: 'probe', visibility: 'hidden' });
  svg.append(probe);

  const read = (clientX) => {
    const r = svg.getBoundingClientRect();
    const px = Math.min(Math.max(clientX - r.left, m.l), W - m.r);
    const t = 10 ** (lx0 + ((px - m.l) / (W - m.l - m.r)) * (lx1 - lx0));
    probe.setAttribute('x1', px);
    probe.setAttribute('x2', px);
    probe.setAttribute('visibility', 'visible');
    $('chart-readout').textContent = `At ${fmtCount(t)} tokens/day: self-hosting ${usd(selfDay(t))}/day, API ${usd(apiDay(t))}/day.`;
  };
  svg.addEventListener('pointermove', (e) => read(e.clientX));
  svg.addEventListener('pointerdown', (e) => read(e.clientX));
  box.replaceChildren(svg);

  // Hidden data table for screen readers
  const rows = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10].map((t) => {
    const tr = el('tr');
    tr.append(el('td', `${fmtCount(t)} tokens/day`), el('td', usd(selfDay(t))), el('td', usd(apiDay(t))));
    return tr;
  });
  const head = el('tr');
  for (const h of ['Tokens per day', 'Self-hosting per day', 'API per day']) {
    const th = el('th', h);
    th.scope = 'col';
    head.append(th);
  }
  $('chart-table').replaceChildren(el('caption', 'Cost per day by volume'), head, ...rows);
}

function renderKv(p, model) {
  const { usable, weights, perUser, pages } = p.kv;
  const gb = (b) => `${Number((b / 1e9).toPrecision(2))} GB`;
  const kvTotal = perUser * p.users;
  $('kv-h').textContent = `Why ${p.users} users fit`;
  const cap = [`One ${shortGpu(p.gpu)}'s usable memory: ${p.gpu.vram_gb} GB × 0.9 = ${gb(usable)}.`];
  if (p.tp > 1) cap.push(`The model is split across ${p.tp} GPUs, so each holds 1/${p.tp} of the weights and its share of every user's cache.`);
  cap.push(`Each coral block is one user's conversation (${ctxLabel(p.avgCtx)}), stored in ${pages} pages of 16 tokens.`);
  const a = model.arch.attn;
  if (a && (a.sliding || a.linear)) cap.push(`This model keeps the whole conversation on only ${a.full} of its ${model.arch.layers} layers, so long chats cost less memory.`);
  $('kv-caption').textContent = cap.join(' ');

  const wPct = Math.min(100, (weights / usable) * 100);
  const uPct = Math.min(100 - wPct, (kvTotal / usable) * 100);
  const w = $('kv-weights');
  w.style.width = `${wPct}%`;
  w.textContent = wPct > 18 ? `Weights ${gb(weights)}` : '';
  const users = $('kv-users');
  users.style.width = `${uPct}%`;
  users.classList.toggle('dense', p.users > 200);
  const blocks = [];
  if (p.users <= 200) {
    for (let i = 0; i < p.users; i++) {
      const b = el('span', null, `u ${pages > 64 ? 'dense' : 'pages'}`);
      if (pages <= 64) b.style.backgroundSize = `${100 / pages}% 100%`;
      blocks.push(b);
    }
  }
  if (!played) blocks.push(el('span', null, 'kv-cover'));
  users.replaceChildren(...blocks);
  $('kvbar').setAttribute('aria-label', `${p.users} users fit: weights ${gb(weights)}, conversations ${gb(kvTotal)}, free ${gb(usable - weights - kvTotal)} of ${gb(usable)}.`);

  const legend = $('kv-legend');
  const fill = () => {
    legend.replaceChildren(el('strong', `${p.users} users fit.`), ` Weights ${gb(weights)} · conversations ${gb(kvTotal)} · free ${gb(Math.max(0, usable - weights - kvTotal))}`);
  };
  if (played || reduceMotion.matches) fill();
  else legend.textContent = '';
  playOnce(fill);

  const slider = $('avg');
  slider.max = p.maxCtx;
  if (document.activeElement !== slider) slider.value = p.avgCtx;
  slider.setAttribute('aria-valuetext', ctxLabel(p.avgCtx));
  $('avg-out').textContent = ctxLabel(p.avgCtx);
}

// The one authored animation: fill once on first view, never loop.
let observer;
function playOnce(done) {
  if (played) return;
  if (reduceMotion.matches || !('IntersectionObserver' in window)) {
    played = true;
    done();
    return;
  }
  observer?.disconnect();
  observer = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    observer.disconnect();
    const bar = $('kvbar');
    bar.classList.add('played');
    const cover = bar.querySelector('.kv-cover');
    const finish = () => {
      played = true;
      cover?.remove();
      done();
    };
    if (cover) cover.addEventListener('transitionend', finish, { once: true });
    else finish();
  }, { threshold: 0.6 });
  observer.observe($('kvbar'));
}

function renderQuality() {
  const rows = data.quality.quant_penalty.map((q) => {
    const tr = el('tr');
    const src = el('td');
    q.sources.forEach((s, i) => {
      if (i) src.append(el('br'));
      src.append(link(s.title.split(',')[0].split(' (')[0], s.url));
    });
    const lose = el('td');
    lose.append(el('strong', `${q.label}. `), q.quote);
    tr.append(el('td', `${q.name}${q.selectable ? '' : ' (reference only)'}`), lose, src);
    return tr;
  });
  $('quality').tBodies[0].replaceChildren(...rows);
}

load();
