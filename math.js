// inference-atlas math. Pure functions only, no DOM. Every estimate here is a
// first-order estimate and usually optimistic: verify with `vllm bench serve`.
//
// Units: bytes, seconds, FLOPs. GPU VRAM is in GB (1e9 bytes), bandwidth in GB/s,
// peak compute in dense TFLOPS, as vendor spec sheets list them.
//
// Model shape (normalized by scripts/refresh.mjs from Hugging Face config.json):
//   { params, active_params, layers, heads, kv_heads, head_dim,
//     mla: null | { kv_lora_rank, qk_rope_head_dim },
//     attn: { full, sliding, window, linear, approx } (optional; absent = all full) }
// GPU shape (data/gpus.json): { vram_gb, bandwidth_gbs, tflops: { fp16, fp8 }, fp8, usd_hr }

// vLLM's --gpu-memory-utilization 0.9: the other 10% of VRAM covers activations,
// CUDA graphs and fragmentation. A fixed constant, not modelled per model.
export const GPU_MEM_UTIL = 0.9;
// Share of bandwidth left after tensor-parallel all-reduce traffic (TP > 1 only).
export const TP_COMM_EFFICIENCY = 0.85;
// Share of peak FLOPS that real kernels reach (model FLOPs utilization).
export const MFU = 0.5;
export const TP_OPTIONS = [1, 2, 4, 8];
// Break-even chart range, tokens per day (log x).
export const CHART_MIN_TPD = 1e5;
export const CHART_MAX_TPD = 1e10;

// Use-case defaults, shared by the v0.1 selector and the v1.0 wizard. itlPin is the
// decode latency target in seconds (null = none); cache is the cached prefix share.
export const USE_CASES = {
  chat: { maxCtx: 8192, avgCtx: 4096, r: 3, itlPin: 0.05, cache: 0 },
  agents: { maxCtx: 32768, avgCtx: 16384, r: 10, itlPin: 0.15, cache: 0.7 },
  batch: { maxCtx: 8192, avgCtx: 2048, r: 5, itlPin: null, cache: 0 },
};

const BYTES = { fp16: 2, fp8: 1, int4: 0.5 };

export const bytesPer = (dtype) => BYTES[dtype];

export const weightBytes = (model, prec) => model.params * BYTES[prec];

// Per token, all layers. GQA/MHA stores K and V per KV head. MLA stores one
// compressed latent plus the rope key per layer (approximate).
export function kvBytesPerToken(model, kvDtype) {
  const b = BYTES[kvDtype];
  if (model.mla) return (model.mla.kv_lora_rank + model.mla.qk_rope_head_dim) * model.layers * b;
  return 2 * model.kv_heads * model.head_dim * model.layers * b;
}

// vLLM shards KV heads across ranks, and replicates them once TP > kv_heads.
// The MLA latent is not split by heads, so every rank holds all of it.
export function kvPerGpuPerToken(model, kvDtype, tp) {
  const kv = kvBytesPerToken(model, kvDtype);
  if (model.mla) return kv;
  return (kv * Math.max(1, tp / model.kv_heads)) / tp;
}

// vLLM refuses to start unless attention heads split evenly and KV heads either
// split evenly (TP <= kv_heads) or replicate evenly (TP > kv_heads).
export function headsSplitOk(model, tp) {
  if (model.heads % tp !== 0) return false;
  return tp <= model.kv_heads ? model.kv_heads % tp === 0 : tp % model.kv_heads === 0;
}

const usableBytes = (gpu) => gpu.vram_gb * 1e9 * GPU_MEM_UTIL;

// KV bytes one sequence of ctx tokens holds on each GPU. Full-attention layers keep every
// token, sliding-window layers keep at most the window, and linear-attention layers keep a
// small constant state, which is ignored.
export function kvPerGpuPerSeq(model, kvDtype, tp, ctx) {
  const perToken = kvPerGpuPerToken(model, kvDtype, tp);
  const a = model.attn;
  if (!a) return perToken * ctx;
  return (perToken / model.layers) * (a.full * ctx + a.sliding * Math.min(ctx, a.window));
}

// Smallest power-of-two TP where weights plus one max-length sequence of KV fit
// per GPU and the head count splits. Returns { tp } or { error, ... }.
export function chooseTP(model, gpu, prec, kvDtype, maxCtx) {
  const fits = (tp) =>
    weightBytes(model, prec) / tp + kvPerGpuPerSeq(model, kvDtype, tp, maxCtx) <= usableBytes(gpu);
  const fitTp = TP_OPTIONS.find(fits);
  if (fitTp === undefined) return { error: 'multi_node' };
  // A head split that fails at fitTp also fails at every larger power of two (tested),
  // so there is no "next valid TP" to fall back to.
  if (headsSplitOk(model, fitTp)) return { tp: fitTp };
  const maxValidTp = TP_OPTIONS.filter((t) => headsSplitOk(model, t)).pop();
  return { error: 'heads', fitTp, maxValidTp };
}

// INT4 AWQ/GPTQ kernels dequantize and compute in fp16.
const peakFlops = (gpu, prec) => gpu.tflops[prec === 'fp8' ? 'fp8' : 'fp16'] * 1e12;

// Seconds per decode step for a batch of B sequences, each holding avgCtx tokens.
// Roofline: the slower of streaming bytes and doing the matmul FLOPs. MoE streams
// active expert weights only, which is approximate at large batch.
export function decodeItl(model, gpu, prec, kvDtype, tp, batch, avgCtx) {
  const bw = gpu.bandwidth_gbs * 1e9 * (tp > 1 ? TP_COMM_EFFICIENCY : 1);
  const streamed = (model.active_params * BYTES[prec]) / tp
    + batch * kvPerGpuPerSeq(model, kvDtype, tp, avgCtx);
  const memory = streamed / bw;
  const compute = (2 * model.active_params * batch) / (tp * peakFlops(gpu, prec) * MFU);
  return Math.max(memory, compute);
}

// Split one request's context into prompt and output tokens by the in:out ratio r.
export const splitTokens = (avgCtx, r) => ({ inTok: (avgCtx * r) / (r + 1), outTok: avgCtx / (r + 1) });

// Seconds to prefill one request. Compute-bound; tokens in the cached prefix are
// served from vLLM's prefix cache and cost no FLOPs.
export const prefillTime = (model, gpu, prec, tp, inTok, cachedFrac) =>
  (2 * model.active_params * inTok * (1 - cachedFrac)) / (tp * peakFlops(gpu, prec) * MFU);

// Served tokens/s for one replica. The B prefills of a wave share the same GPUs,
// so a wave of B requests takes B x prefill plus the decode steps.
export const tokensPerSec = (batch, inTok, outTok, prefill, itl) =>
  (batch * (inTok + outTok)) / (batch * prefill + outTok * itl);

// Blended API price in USD per 1M tokens (pricing is per 1M, converted by refresh.mjs).
// The cached prompt share uses the cache-read price when the listing has one.
export function blendedApiPrice(pricing, r, cachedFrac) {
  const cachePriced = pricing.cache_read != null;
  const cacheRead = cachePriced ? pricing.cache_read : pricing.prompt;
  const prompt = cachedFrac * cacheRead + (1 - cachedFrac) * pricing.prompt;
  return { perM: (r * prompt + pricing.completion) / (r + 1), cachePriced };
}

// Users whose average context fits in the KV space left on each GPU. 0 = does not fit.
export function maxUsers(model, gpu, prec, kvDtype, tp, avgCtx) {
  const free = usableBytes(gpu) - weightBytes(model, prec) / tp;
  return Math.max(0, Math.floor(free / kvPerGpuPerSeq(model, kvDtype, tp, avgCtx)));
}

// Batch the server runs at: the user override, else the largest batch whose ITL meets
// the pin, never above maxUsers or below 1. If batch 1 misses the pin, run at 1 and flag it.
export function operatingBatch(model, gpu, prec, kvDtype, tp, avgCtx, users, itlPin, override) {
  const cap = Math.max(1, users);
  if (override) return { batch: Math.min(Math.max(1, Math.floor(override)), cap), sloMiss: false };
  if (itlPin == null) return { batch: cap, sloMiss: false };
  const itl = (b) => decodeItl(model, gpu, prec, kvDtype, tp, b, avgCtx);
  if (itl(1) > itlPin) return { batch: 1, sloMiss: true };
  // ITL grows with batch, so binary-search the largest batch under the pin.
  let lo = 1;
  let hi = cap;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (itl(mid) <= itlPin) lo = mid; else hi = mid - 1;
  }
  return { batch: lo, sloMiss: false };
}

// USD per 1M served tokens for one replica of TP GPUs at the given utilization.
export const selfHostPerM = (tp, usdHr, tps, util) => ((tp * usdHr) / 3600 / (tps * util)) * 1e6;

// USD per day to serve tpd tokens/day: whole replicas, each TP GPUs for 24 h.
export const selfHostPerDay = (tpd, capacityPerDay, tp, usdHr) =>
  Math.ceil(tpd / capacityPerDay) * tp * usdHr * 24;

// First tokens/day where self-hosting costs no more than the API. Within one replica the
// self-host cost is flat and the API cost is linear, so they cross at the volume one
// replica-day of the API costs. Past one replica's capacity every later step is the same
// ratio, so there is no crossing at all. Returns { kind: 'cross', tpd } | { kind: 'api' | 'self' }.
export function breakEven({ tps, util, tp, usdHr, apiPerM }) {
  const capacity = tps * 86400 * util;
  const tpd = (tp * usdHr * 24 * 1e6) / apiPerM;
  if (tpd > capacity || tpd > CHART_MAX_TPD) return { kind: 'api' };
  if (tpd <= CHART_MIN_TPD) return { kind: 'self' };
  return { kind: 'cross', tpd };
}

// The command to run. For INT4 the caller passes the AWQ/GPTQ repo as hfId; vLLM reads
// the quantization method from that repo's config, so no flag is needed.
export function vllmCommand({ hfId, tp, maxCtx, batch, prec, kvDtype }) {
  const parts = [
    `vllm serve ${hfId}`, `--tensor-parallel-size ${tp}`, `--max-model-len ${maxCtx}`,
    `--gpu-memory-utilization ${GPU_MEM_UTIL}`, `--max-num-seqs ${batch}`,
  ];
  if (prec === 'fp8') parts.push('--quantization fp8');
  if (kvDtype === 'fp8') parts.push('--kv-cache-dtype fp8');
  return parts.join(' ');
}

// 2 significant figures with a K/M/B/T suffix: 42.3e6 -> "42M". Never claim more precision.
export function fmtCount(n) {
  const r = Number(n.toPrecision(2));
  for (const [v, s] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) {
    if (r >= v) return `${Number((r / v).toPrecision(2))}${s}`;
  }
  return String(r);
}
