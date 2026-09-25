// Permalink state: every user-editable input lives in the URL. Params are untrusted,
// so enums and the model id must match known values, and numbers are clamped.
// Nothing from the URL is ever echoed back into the page.
import { USE_CASES } from './math.js';

export const DEFAULT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct';
export const FALLBACK_GPU = 'h100';

const PRECS = ['fp16', 'fp8', 'int4', 'mxfp4'];
const KV_DTYPES = ['fp16', 'fp8'];

// [min, max, integer]. cache and util are percentages.
const RANGES = {
  max_ctx: [512, 1048576, true],
  avg_ctx: [1, 1048576, true],
  r: [0.1, 100, false],
  cache: [0, 95, false],
  batch: [1, 4096, true],
  util: [5, 100, false],
  tpd: [1e3, 1e12, true],
};
const KEYS = ['model', 'gpu', 'prec', 'kv', 'use', ...Object.keys(RANGES)];

// Defaults for one use case, in URL units. gpu, prec and batch default to null, which
// means "auto": the page picks the cheapest GPU that fits, FP8 where supported, and the
// batch from the latency pin.
function defaults(use) {
  const u = USE_CASES[use];
  return {
    model: DEFAULT_MODEL, gpu: null, prec: null, kv: 'fp16', use,
    max_ctx: u.maxCtx, avg_ctx: u.avgCtx, r: u.r, cache: u.cache * 100, batch: null, util: 60, tpd: 1e7,
  };
}

function num(raw, [min, max, int]) {
  if (raw === null || raw.trim() === '') return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  const c = Math.min(max, Math.max(min, n));
  return int ? Math.round(c) : c;
}

// ctx = { models: [hf_id], gpus: { id: { fp8 } }, precOk(model, gpu, prec) -> bool }, where
// precOk comes from math.js precisionOptions (gpu is null while the page picks it).
// Returns { state, notices }, where notices are keys the page turns into copy.
export function parseState(params, ctx) {
  const notices = new Set();
  const get = (k) => params.get(k);
  const pick = (k, allowed) => {
    const v = get(k);
    if (v === null) return undefined;
    if (allowed.includes(v)) return v;
    notices.add('link_invalid');
    return undefined;
  };

  let model = get('model');
  if (model !== null && !ctx.models.includes(model)) {
    notices.add('model_removed');
    model = null;
  }
  let gpu = get('gpu');
  if (gpu !== null && !Object.hasOwn(ctx.gpus, gpu)) {
    notices.add('gpu_unknown');
    gpu = FALLBACK_GPU;
  }
  const use = pick('use', Object.keys(USE_CASES)) ?? 'chat';
  const s = { ...defaults(use), model: model ?? DEFAULT_MODEL, gpu };

  const kv = pick('kv', KV_DTYPES);
  if (kv) s.kv = kv;
  const prec = pick('prec', PRECS);
  if (prec) {
    // Not available for this model and GPU: fall back to auto, which picks a valid one.
    if (ctx.precOk(s.model, gpu, prec)) s.prec = prec;
    else notices.add('prec_unavailable');
  }

  for (const k of Object.keys(RANGES)) {
    const v = num(get(k), RANGES[k]);
    if (v !== null) s[k] = v;
  }
  s.avg_ctx = Math.min(s.avg_ctx, s.max_ctx);
  return { state: s, notices: [...notices] };
}

// Only values that differ from the use-case defaults go into the link.
export function serializeState(state) {
  const d = { ...defaults(state.use), use: 'chat' };
  const out = new URLSearchParams();
  for (const k of KEYS) {
    if (state[k] !== null && state[k] !== d[k]) out.set(k, String(state[k]));
  }
  return out.toString();
}

// Keeps the address bar in sync without growing history: one replaceState per burst.
export function urlSyncer(win, ms = 300) {
  let timer;
  return (state) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const qs = serializeState(state);
      win.history.replaceState(null, '', win.location.pathname + (qs ? `?${qs}` : ''));
    }, ms);
  };
}
