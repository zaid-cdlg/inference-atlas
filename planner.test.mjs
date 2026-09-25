// Page logic: auto GPU and precision, verdict copy, error states, command.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { plan, usd, ctxLabel } from './planner.js';
import { parseState, DEFAULT_MODEL } from './state.js';
import { fmtCount } from './math.js';

// Hand-made data in the data/ shapes. Llama 3.1 8B config: 32 layers, 32 heads, 8 KV heads.
const arch8b = {
  params: 8e9, active_params: 8e9, layers: 32, heads: 32, kv_heads: 8, head_dim: 128, mla: null,
  moe: false, max_ctx: 131072, attn: { full: 32, sliding: 0, window: null, linear: 0, approx: false },
  quant: null, native_bytes: null,
};
const model = (over = {}) => ({
  hf_id: 'x/Model-8B', name: 'X: Model 8B', pricing: { prompt: 0.6, completion: 2.4, cache_read: null }, arch: arch8b, ...over,
});
const gpu = (id, over = {}) => ({
  id, name: `GPU ${id}`, vendor: 'nvidia', vram_gb: 80, bandwidth_gbs: 3350, peak_tflops: { fp16: 989, fp8: 1979 }, fp8: true, usd_per_hr: 4, ...over,
});
const chat = { use: 'chat', kv: 'fp16', max_ctx: 8192, avg_ctx: 4096, r: 3, cache: 0, batch: null, util: 60, tpd: 1e7, gpu: null, prec: null };

test('auto picks the GPU with the earliest break-even, and FP8 where the GPU has it', () => {
  const p = plan({ model: model(), gpus: [gpu('dear', { usd_per_hr: 9 }), gpu('cheap', { usd_per_hr: 2 })], state: chat });
  assert.equal(p.gpu.id, 'cheap');
  assert.equal(p.prec, 'fp8');
  assert.equal(p.error, null);
});

test('a chosen GPU without FP8 runs FP16 by default; a native format always wins', () => {
  const noFp8 = gpu('a100', { fp8: false, peak_tflops: { fp16: 312, fp8: null } });
  assert.equal(plan({ model: model(), gpus: [noFp8], state: { ...chat, gpu: 'a100' } }).prec, 'fp16');
  const mx = model({ arch: { ...arch8b, quant: 'mxfp4', native_bytes: 5e9 } });
  assert.equal(plan({ model: mx, gpus: [gpu('h100')], state: chat }).prec, 'mxfp4');
});

test('verdict: calibrated break-even sentence, 2 significant figures', () => {
  const p = plan({ model: model(), gpus: [gpu('h100')], state: { ...chat, gpu: 'h100' } });
  assert.equal(p.be.kind, 'cross');
  assert.match(p.verdict, /^Self-hosting likely wins above ~\d+(\.\d)?[KMB] tokens\/day$/);
});

test('no paid API: no break-even, the self-host price instead, and no chart', () => {
  const p = plan({ model: model({ pricing: null }), gpus: [gpu('h100')], state: chat });
  assert.equal(p.be, null);
  assert.equal(p.verdict, `No paid API for this model yet, so there's no break-even. Self-hosting costs ${usd(p.selfPerM)} per 1M tokens.`);
});

test('multi-node: no command, and the exact copy', () => {
  const huge = model({ arch: { ...arch8b, params: 700e9, active_params: 700e9 } });
  const p = plan({ model: huge, gpus: [gpu('h100', { name: 'H100' })], state: { ...chat, gpu: 'h100', prec: 'fp16' } });
  assert.equal(p.command, null);
  assert.equal(p.error, "This model doesn't fit on H100, even 8 of them. It needs more than 8 GPUs (multi-node), which is out of scope. Try FP8, a bigger GPU, or a smaller model.");
});

test('head split: the model fits on N GPUs but its heads cannot split N ways', () => {
  // 24 heads, 6 KV heads, 100e9 params FP16: fits first at TP 4, which 6 KV heads cannot split
  const odd = model({ arch: { ...arch8b, params: 100e9, active_params: 100e9, heads: 24, kv_heads: 6, layers: 40 } });
  const p = plan({ model: odd, gpus: [gpu('h100')], state: { ...chat, gpu: 'h100', prec: 'fp16' } });
  assert.equal(p.error, "This model needs 4 GPUs to fit, but its attention heads can't be split 4 ways. Try FP8 or a bigger GPU.");
});

test('SLO miss: batch 1 with a red line, costs still shown', () => {
  const slow = gpu('slow', { bandwidth_gbs: 100 });
  const p = plan({ model: model(), gpus: [slow], state: { ...chat, gpu: 'slow', prec: 'fp16' } });
  assert.equal(p.batch, 1);
  assert.equal(p.slo, 'Misses the 50 ms target even at 1 user. Try a faster GPU or FP8.');
  assert.ok(p.selfPerM > 0);
});

test('INT4 swaps in the quantized repo; FP8 checkpoints get no --quantization flag', () => {
  const int4 = { repo: 'q/Model-8B-AWQ', url: 'https://huggingface.co/q/Model-8B-AWQ', method: 'awq', publisher: 'community' };
  const p = plan({ model: model(), gpus: [gpu('h100')], state: { ...chat, gpu: 'h100', prec: 'int4' }, int4 });
  assert.match(p.command, /^vllm serve q\/Model-8B-AWQ /);
  const f8 = model({ arch: { ...arch8b, quant: 'fp8', native_bytes: 8e9 } });
  assert.doesNotMatch(plan({ model: f8, gpus: [gpu('h100')], state: chat }).command, /--quantization/);
});

test('max context is capped at the model limit', () => {
  const short = model({ arch: { ...arch8b, max_ctx: 4096 } });
  const p = plan({ model: short, gpus: [gpu('h100')], state: { ...chat, max_ctx: 32768, avg_ctx: 16384 } });
  assert.match(p.command, /--max-model-len 4096/);
  assert.equal(p.avgCtx, 4096);
});

test('usd: 2 significant figures', () => {
  assert.equal(usd(0.3149), '$0.31');
  assert.equal(usd(12.49), '$12');
  assert.equal(usd(1234), '$1,200');
  assert.equal(usd(0.001), '<$0.01');
});

test('context length reads the way people say it: 4k tokens', () => {
  assert.equal(ctxLabel(4096), '4k tokens');
  assert.equal(ctxLabel(16384), '16k tokens');
  assert.equal(ctxLabel(4352), '4.3k tokens');
  assert.equal(ctxLabel(131072), '128k tokens');
  assert.equal(ctxLabel(512), '512 tokens');
});

test('KV sim markup: the slider is named by its label and the heading says users fit', () => {
  const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
  const label = html.match(/<label[^>]*for="avg"[^>]*>(.*?)<\/label>/s);
  assert.ok(label, 'the slider has a <label for="avg">');
  assert.match(label[1], /Average conversation length/);
  assert.match(html, /<input type="range" id="avg"/);
  assert.match(html, /<h2 id="kv-h">[^<]*users fit<\/h2>/);
  // app.js keeps aria-valuetext in step with the visible value
  assert.match(readFileSync(new URL('./app.js', import.meta.url), 'utf8'), /setAttribute\('aria-valuetext', ctxLabel\(/);
});

// The real data: auto mode must never hand a beginner an AMD command.
const read = (p) => JSON.parse(readFileSync(new URL(`./data/${p}`, import.meta.url)));
const models = read('models.json');
const { gpus } = read('gpus.json');
const quality = read('quality.json');
const byId = new Map(models.models.map((m) => [m.hf_id, m]));
const planFor = (qs) => {
  const ctx = { models: models.featured, gpus: Object.fromEntries(gpus.map((g) => [g.id, g])), precOk: () => true };
  const { state } = parseState(new URLSearchParams(qs), ctx);
  return plan({ model: byId.get(state.model), gpus, state, int4: quality.quantized_repos[state.model]?.int4 ?? null });
};

test('auto never picks an AMD GPU, for any featured model or use case', () => {
  for (const id of models.featured) {
    for (const use of ['chat', 'agents', 'batch']) {
      assert.equal(planFor(`model=${encodeURIComponent(id)}&use=${use}`).gpu.vendor, 'nvidia', `${id} ${use}`);
    }
  }
});

test('amdHint: a cheaper MI300X is offered as a hint, not picked', () => {
  const p = planFor('model=openai%2Fgpt-oss-120b');
  assert.equal(p.amdHint.gpu.id, 'mi300x');
  assert.ok(p.amdHint.selfPerM < p.selfPerM);
});

test('MI300X picked by hand works and gets no hint', () => {
  const p = planFor('model=openai%2Fgpt-oss-120b&gpu=mi300x');
  assert.equal(p.gpu.id, 'mi300x');
  assert.equal(p.error, null);
  assert.equal(p.amdHint, null);
  assert.match(p.command, /^vllm serve openai\/gpt-oss-120b /);
});

test('default example: gpt-oss-120b in MXFP4 on one NVIDIA GPU, with a break-even to show', () => {
  assert.equal(DEFAULT_MODEL, 'openai/gpt-oss-120b');
  const p = planFor('');
  assert.equal(p.gpu.vendor, 'nvidia');
  assert.equal(p.tp, 1);
  assert.equal(p.prec, 'mxfp4');
  // Prices change weekly, so check the shape and the rounding rather than the number
  assert.equal(p.be.kind, 'cross');
  assert.equal(p.verdict, `Self-hosting likely wins above ~${fmtCount(p.be.tpd)} tokens/day`);
  assert.match(p.verdict, /~\d{1,3}(\.\d)?[KMB] tokens/);
});

test('a natively INT4 checkpoint runs as itself (no separate quantized repo needed)', () => {
  const awq = model({ hf_id: 'x/Model-8B-AWQ', arch: { ...arch8b, quant: 'int4', native_bytes: 5e9 } });
  const p = plan({ model: awq, gpus: [gpu('h100')], state: chat });
  assert.equal(p.prec, 'int4');
  assert.match(p.command, /^vllm serve x\/Model-8B-AWQ /);
});

test('a manual batch that misses the latency target is flagged with its own copy', () => {
  const p = plan({ model: model(), gpus: [gpu('slow', { bandwidth_gbs: 300 })], state: { ...chat, gpu: 'slow', prec: 'fp16', batch: 60 } });
  assert.equal(p.batch, 60);
  assert.match(p.slo, /^Misses the 50 ms target with 60 users at once\. Lower "Users served at once" or leave it empty\.$/);
});

test('auto reports the earliest crossing, not the cheapest GPU per token at full load', () => {
  // Every NVIDIA GPU picked by hand: the auto verdict must be the lowest break-even of them all
  const auto = planFor('model=openai%2Fgpt-oss-120b');
  const crossings = gpus.filter((g) => g.vendor === 'nvidia')
    .map((g) => planFor(`model=openai%2Fgpt-oss-120b&gpu=${g.id}`))
    .filter((p) => !p.error && p.be?.kind === 'cross');
  assert.equal(auto.be.tpd, Math.min(...crossings.map((p) => p.be.tpd)));
});

test('auto falls back to the cheapest per token when no GPU crosses (the API wins everywhere)', () => {
  const p = planFor('model=meta-llama%2FLlama-3.3-70B-Instruct');
  assert.equal(p.be.kind, 'api');
  const perToken = gpus.filter((g) => g.vendor === 'nvidia')
    .map((g) => planFor(`model=meta-llama%2FLlama-3.3-70B-Instruct&gpu=${g.id}`)).filter((x) => !x.error);
  assert.equal(p.selfPerM, Math.min(...perToken.map((x) => x.selfPerM)));
});

test('a model whose only format cannot run on the chosen GPU gets a clear error and no command', () => {
  const p = planFor('model=openai%2Fgpt-oss-20b&gpu=t4');
  assert.equal(p.command, null);
  assert.equal(p.error, 'vLLM has no MXFP4 kernel for Turing GPUs such as the T4. Pick an A10G, L4 or newer GPU.');
  // and auto never lands there
  assert.notEqual(planFor('model=openai%2Fgpt-oss-20b').gpu.generation, 'turing');
});

test('a manual batch of 1 that misses the target gets the "even at 1 user" advice', () => {
  const p = plan({ model: model(), gpus: [gpu('slow', { bandwidth_gbs: 100 })], state: { ...chat, gpu: 'slow', prec: 'fp16', batch: 1 } });
  assert.equal(p.slo, 'Misses the 50 ms target even at 1 user. Try a faster GPU or FP8.');
});
