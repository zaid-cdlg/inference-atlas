// Turns one page state into everything the page shows. Pure: no DOM, no fetch.
import {
  USE_CASES, GPU_MEM_UTIL, chooseTP, maxUsers, operatingBatch, splitTokens, prefillTime, decodeItl,
  tokensPerSec, selfHostPerM, blendedApiPrice, breakEven, vllmCommand, precisionOptions, weightBytes,
  kvPerGpuPerSeq, fmtCount,
} from './math.js';
import { FALLBACK_GPU } from './state.js';

// 2 significant figures, as the verdict never claims more precision than that.
export function usd(x) {
  if (x < 0.01) return '<$0.01';
  if (x < 100) return `$${x.toPrecision(2)}`;
  return `$${Number(x.toPrecision(2)).toLocaleString('en-US')}`;
}

// Context length the way people say it (4096 -> "4k tokens"), for the slider and its
// screen-reader value.
export function ctxLabel(t) {
  if (t < 1024) return `${t} tokens`;
  const k = t / 1024;
  return `${k >= 10 ? Math.round(k) : Number(k.toFixed(1))}k tokens`;
}

export const shortGpu = (g) => g.name.replace(/^(NVIDIA|AMD Instinct) /, '');

// Precision the page uses when the reader has not picked one: the published format of a
// pre-quantized model, else FP8 where the GPU supports it, else FP16.
const autoPrec = (arch, gpu) => arch.quant ?? (gpu.fp8 ? 'fp8' : 'fp16');

function evaluate(model, gpu, state, int4) {
  const arch = model.arch;
  const uc = USE_CASES[state.use];
  const precOptions = precisionOptions(arch, gpu, Boolean(int4));
  const prec = state.prec && precOptions[state.prec] === null ? state.prec : autoPrec(arch, gpu);
  const maxCtx = Math.min(state.max_ctx, arch.max_ctx ?? Infinity);
  const avgCtx = Math.min(state.avg_ctx, maxCtx);
  const base = { gpu, prec, precOptions, maxCtx, avgCtx, command: null, be: null, slo: null };

  const fit = chooseTP(arch, gpu, prec, state.kv, maxCtx);
  if (fit.error === 'multi_node') {
    return { ...base, error: `This model doesn't fit on ${shortGpu(gpu)}, even 8 of them. It needs more than 8 GPUs (multi-node), which is out of scope. Try FP8, a bigger GPU, or a smaller model.` };
  }
  if (fit.error === 'heads') {
    return { ...base, error: `This model needs ${fit.fitTp} GPUs to fit, but its attention heads can't be split ${fit.fitTp} ways. Try FP8 or a bigger GPU.` };
  }
  const { tp } = fit;
  const users = maxUsers(arch, gpu, prec, state.kv, tp, avgCtx);
  if (users < 1) {
    return { ...base, tp, error: `This model doesn't fit on ${shortGpu(gpu)}. Try FP8, a bigger GPU, or a smaller model.` };
  }

  const cache = state.cache / 100;
  const util = state.util / 100;
  const { batch, sloMiss } = operatingBatch(arch, gpu, prec, state.kv, tp, avgCtx, users, uc.itlPin, state.batch);
  const { inTok, outTok } = splitTokens(avgCtx, state.r);
  const prefill = prefillTime(arch, gpu, prec, tp, inTok, cache);
  const itl = decodeItl(arch, gpu, prec, state.kv, tp, batch, avgCtx);
  const tps = tokensPerSec(batch, inTok, outTok, prefill, itl);
  const selfPerM = selfHostPerM(tp, gpu.usd_per_hr, tps, util);
  const api = model.pricing ? blendedApiPrice(model.pricing, state.r, cache) : null;
  const be = api ? breakEven({ tps, util, tp, usdHr: gpu.usd_per_hr, apiPerM: api.perM }) : null;

  const usable = gpu.vram_gb * 1e9 * GPU_MEM_UTIL;
  return {
    ...base,
    error: null,
    tp, users, batch, itl, tps, prefill, selfPerM, api, be,
    capacity: tps * 86400 * util,
    slo: sloMiss ? `Misses the ${Math.round(uc.itlPin * 1000)} ms target even at 1 user. Try a faster GPU or FP8.` : null,
    command: vllmCommand({
      hfId: prec === 'int4' ? int4.repo : model.hf_id, tp, maxCtx, batch, prec, kvDtype: state.kv, native: arch.quant,
    }),
    kv: {
      usable,
      weights: weightBytes(arch, prec) / tp,
      perUser: kvPerGpuPerSeq(arch, state.kv, tp, avgCtx),
      pages: Math.ceil(avgCtx / 16),
    },
  };
}

function verdictFor(r) {
  if (r.error) return r.error;
  if (!r.api) {
    return `No paid API for this model yet, so there's no break-even. Self-hosting costs ${usd(r.selfPerM)} per 1M tokens.`;
  }
  if (r.be.kind === 'cross') return `Self-hosting likely wins above ~${fmtCount(r.be.tpd)} tokens/day`;
  if (r.be.kind === 'self') return 'Self-hosting likely wins from the first GPU';
  return 'The API is likely cheaper at any volume up to 10B tokens/day';
}

// state.gpu null = auto: the GPU with the lowest self-host $/1M tokens among those where
// the model fits (and the chosen precision is available). None fits: show the H100's error.
export function plan({ model, gpus, state, int4 = null }) {
  let r;
  if (state.gpu) {
    r = evaluate(model, gpus.find((g) => g.id === state.gpu), state, int4);
  } else {
    const ok = gpus
      .filter((g) => !state.prec || precisionOptions(model.arch, g, Boolean(int4))[state.prec] === null)
      .map((g) => evaluate(model, g, state, int4))
      .filter((x) => !x.error);
    r = ok.length
      ? ok.reduce((a, b) => (b.selfPerM < a.selfPerM ? b : a))
      : evaluate(model, gpus.find((g) => g.id === FALLBACK_GPU) ?? gpus[0], state, int4);
  }
  return { ...r, verdict: verdictFor(r) };
}
