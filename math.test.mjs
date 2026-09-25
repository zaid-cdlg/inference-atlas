// Hand-derived checks for math.js. Every expected value is worked out from the
// standard formulas and public model configs (Hugging Face config.json) or public
// GPU spec sheets. Params are rounded to keep the arithmetic readable by hand.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  kvBytesPerToken, kvPerGpuPerToken, headsSplitOk, chooseTP, decodeItl,
  splitTokens, prefillTime, tokensPerSec, blendedApiPrice,
  kvPerGpuPerSeq, USE_CASES, maxUsers, operatingBatch, selfHostPerM, selfHostPerDay, breakEven, vllmCommand, fmtCount,
} from './math.js';

const close = (actual, expected, rel = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= Math.abs(expected) * rel,
    `expected ${expected}, got ${actual}`);

// Llama 3 70B config.json: 80 layers, 64 heads, 8 KV heads, head_dim 128.
const llama70b = {
  params: 70e9, active_params: 70e9, layers: 80, heads: 64, kv_heads: 8, head_dim: 128, mla: null,
};
// Llama 3.1 8B config.json: 32 layers, 32 heads, 8 KV heads, head_dim 128.
const llama8b = {
  params: 8e9, active_params: 8e9, layers: 32, heads: 32, kv_heads: 8, head_dim: 128, mla: null,
};
// DeepSeek V3 config.json: 61 layers, kv_lora_rank 512, qk_rope_head_dim 64.
const deepseekV3 = {
  params: 671e9, active_params: 37e9, layers: 61, heads: 128, kv_heads: 128, head_dim: 128,
  mla: { kv_lora_rank: 512, qk_rope_head_dim: 64 },
};
// Gemma 3 27B config.json: 62 layers, 16 KV heads, head_dim 128, sliding window 1024 on
// 5 of every 6 layers, so 10 full-attention layers and 52 sliding ones.
const gemma27b = {
  params: 27e9, active_params: 27e9, layers: 62, heads: 32, kv_heads: 16, head_dim: 128, mla: null,
  attn: { full: 10, sliding: 52, window: 1024, linear: 0, approx: false },
};
// Qwen3.5 27B config.json: 64 layers, 4 KV heads, head_dim 256, full attention on every 4th
// layer (16), linear attention on the other 48.
const qwen35 = {
  params: 27e9, active_params: 27e9, layers: 64, heads: 24, kv_heads: 4, head_dim: 256, mla: null,
  attn: { full: 16, sliding: 0, window: null, linear: 48, approx: false },
};
// H100 SXM spec sheet: 80 GB, 3.35 TB/s, 989 TFLOPS dense BF16, 1979 dense FP8.
const h100 = { vram_gb: 80, bandwidth_gbs: 3350, tflops: { fp16: 989, fp8: 1979 }, fp8: true };

test('KV bytes per token: GQA = 2 x kv_heads x head_dim x layers x bytes', () => {
  // 2 x 8 x 128 x 80 x 2 B = 327,680 B = 320 KiB
  assert.equal(kvBytesPerToken(llama70b, 'fp16'), 327680);
  assert.equal(kvBytesPerToken(llama70b, 'fp8'), 163840);
});

test('KV bytes per token: MLA = (kv_lora_rank + qk_rope_head_dim) x layers x bytes', () => {
  // (512 + 64) x 61 x 2 B = 70,272 B
  assert.equal(kvBytesPerToken(deepseekV3, 'fp16'), 70272);
});

test('per-GPU KV: sharded when TP <= kv_heads, replicated beyond, full latent for MLA', () => {
  assert.equal(kvPerGpuPerToken(llama70b, 'fp16', 1), 327680);
  assert.equal(kvPerGpuPerToken(llama70b, 'fp16', 8), 327680 / 8);
  // kv_heads = 2 at TP 8: each rank holds 8/2 = 4 copies' worth, so kv x 4 / 8 = kv / 2
  const kv2 = { ...llama70b, kv_heads: 2 };
  assert.equal(kvPerGpuPerToken(kv2, 'fp16', 8), (2 * 2 * 128 * 80 * 2) / 2);
  assert.equal(kvPerGpuPerToken(deepseekV3, 'fp16', 8), 70272);
});

test('head split rule follows vLLM', () => {
  assert.equal(headsSplitOk(llama70b, 8), true);
  // 24 heads, 6 KV heads: TP 2 ok; TP 4 fails (6 % 4); TP 8 fails (8 % 6)
  const m = { heads: 24, kv_heads: 6 };
  assert.equal(headsSplitOk(m, 2), true);
  assert.equal(headsSplitOk(m, 4), false);
  assert.equal(headsSplitOk(m, 8), false);
  // TP above kv_heads needs TP to be a multiple of kv_heads
  assert.equal(headsSplitOk({ heads: 32, kv_heads: 2 }, 8), true);
});

test('chooseTP: Llama 70B FP16 on H100 at 8k context fits on 2 GPUs', () => {
  // TP 1: 140e9 B > 72e9 usable. TP 2: 70e9 + 163,840 x 8192 = 71.34e9 <= 72e9.
  assert.deepEqual(chooseTP(llama70b, h100, 'fp16', 'fp16', 8192), { tp: 2 });
});

test('chooseTP: longer context pushes Llama 70B FP16 to 4 GPUs', () => {
  // TP 2: 70e9 + 163,840 x 32,768 = 75.4e9 > 72e9. TP 4: 35e9 + 81,920 x 32,768 = 37.7e9.
  assert.deepEqual(chooseTP(llama70b, h100, 'fp16', 'fp16', 32768), { tp: 4 });
});

test('chooseTP: FP8 weights fit Llama 70B on one H100', () => {
  // 70e9 + 163,840 (fp8 KV) x 8192 = 71.34e9 <= 72e9
  assert.deepEqual(chooseTP(llama70b, h100, 'fp8', 'fp8', 8192), { tp: 1 });
});

test('chooseTP: fits memory only at a TP the head count cannot split', () => {
  // Synthetic: 100e9 FP16 = 200e9 B. TP 2 (100e9) and TP 4 (50e9 + KV) fits first at 4.
  // 24 heads / 6 KV heads cannot split 4 or 8 ways, so this is unsupported.
  const m = { params: 100e9, active_params: 100e9, layers: 40, heads: 24, kv_heads: 6, head_dim: 128, mla: null };
  assert.deepEqual(chooseTP(m, h100, 'fp16', 'fp16', 8192), { error: 'heads', fitTp: 4, maxValidTp: 2 });
});

test('chooseTP: more than 8 GPUs is multi-node', () => {
  // DeepSeek V3 FP16 = 1342e9 B; / 8 = 167.75e9 > 72e9
  assert.deepEqual(chooseTP(deepseekV3, h100, 'fp16', 'fp16', 8192), { error: 'multi_node' });
});

test('decode ITL at batch 1 is memory-bound', () => {
  // memory = (16e9 + 1 x 131,072 x 1024) / 3.35e12 = 16.134e9 / 3.35e12 s
  // compute = 2 x 8e9 x 1 / (989e12 x 0.5) = 0.032 ms, far smaller
  close(decodeItl(llama8b, h100, 'fp16', 'fp16', 1, 1, 1024), (16e9 + 131072 * 1024) / 3.35e12);
});

test('decode ITL at batch 500 is compute-bound (roofline)', () => {
  // memory = (16e9 + 500 x 131,072 x 128) / 3.35e12 = 7.28 ms
  // compute = 2 x 8e9 x 500 / (989e12 x 0.5) = 16.18 ms, wins
  const itl = decodeItl(llama8b, h100, 'fp16', 'fp16', 1, 500, 128);
  close(itl, (2 * 8e9 * 500) / (989e12 * 0.5));
  assert.ok(itl > (16e9 + 500 * 131072 * 128) / 3.35e12);
});

test('decode ITL under TP applies the 0.85 comm factor and per-GPU KV', () => {
  // TP 2: (140e9 / 2 + 4 x 163,840 x 4096) / (3.35e12 x 0.85)
  close(decodeItl(llama70b, h100, 'fp16', 'fp16', 2, 4, 4096),
    (70e9 + 4 * 163840 * 4096) / (3.35e12 * 0.85));
});

test('decode ITL uses FP8 peak FLOPS only for FP8 weights, INT4 computes at FP16', () => {
  close(decodeItl(llama8b, h100, 'fp8', 'fp8', 1, 500, 128), (2 * 8e9 * 500) / (1979e12 * 0.5));
  close(decodeItl(llama8b, h100, 'int4', 'fp16', 1, 500, 128), (2 * 8e9 * 500) / (989e12 * 0.5));
});

test('MoE decode streams active weights only', () => {
  // DeepSeek V3 FP8 at TP 8, batch 1, 4k: (37e9 / 8 + 70,272/2 x 4096) / (3.35e12 x 0.85)
  close(decodeItl(deepseekV3, h100, 'fp8', 'fp8', 8, 1, 4096),
    (37e9 / 8 + 35136 * 4096) / (3.35e12 * 0.85));
});

test('head split failure is monotone in TP, so there is never a larger valid TP to fall back to', () => {
  // Heads divide by TP only if they divide by every smaller power of two. KV heads that
  // fail at TP (not divisible, or TP not a multiple) fail at 2 x TP for the same reason.
  for (let heads = 1; heads <= 128; heads++) {
    for (let kv = 1; kv <= heads; kv++) {
      if (heads % kv) continue;
      for (const tp of [1, 2, 4]) {
        if (!headsSplitOk({ heads, kv_heads: kv }, tp)) {
          assert.equal(headsSplitOk({ heads, kv_heads: kv }, tp * 2), false, `${heads}/${kv} at ${tp * 2}`);
        }
      }
    }
  }
});

test('splitTokens: in = ctx x r/(r+1), out = ctx x 1/(r+1)', () => {
  // r = 3: 4096 x 3/4 = 3072 in, 1024 out
  assert.deepEqual(splitTokens(4096, 3), { inTok: 3072, outTok: 1024 });
});

test('prefill is compute-bound and skips the cached prefix', () => {
  // 2 x 8e9 x 12,000 x (1 - 0.7) / (989e12 x 0.5) = 5.76e13 / 4.945e14 s
  close(prefillTime(llama8b, h100, 'fp16', 1, 12000, 0.7), (2 * 8e9 * 12000 * 0.3) / (989e12 * 0.5));
  // no cache, FP8 peak, TP 2
  close(prefillTime(llama8b, h100, 'fp8', 2, 12000, 0), (2 * 8e9 * 12000) / (2 * 1979e12 * 0.5));
});

test('agents case: B prefills share one replica, so tokens/s = B(in+out) / (B x prefill + out x ITL)', () => {
  // Llama 3.1 8B FP16, H100, TP 1, B = 8, avg ctx 16,384, r = 10, 70% cached prefix
  const { inTok, outTok } = splitTokens(16384, 10);
  const pre = prefillTime(llama8b, h100, 'fp16', 1, inTok, 0.7);
  const itl = decodeItl(llama8b, h100, 'fp16', 'fp16', 1, 8, 16384);
  const tps = tokensPerSec(8, inTok, outTok, pre, itl);
  close(tps, (8 * 16384) / (8 * pre + outTok * itl));
  // Charging each request only its own prefill would overstate throughput
  assert.ok(tps < (8 * 16384) / (pre + outTok * itl));
});

test('blended API price uses the cache-read price for the cached prompt share', () => {
  // $/1M: prompt 0.6, completion 2.4, cache read 0.15. Agents r = 10, 70% cached:
  // 10/11 x (0.7 x 0.15 + 0.3 x 0.6) + 1/11 x 2.4 = (2.85 + 2.4) / 11
  const p = blendedApiPrice({ prompt: 0.6, completion: 2.4, cache_read: 0.15 }, 10, 0.7);
  close(p.perM, 5.25 / 11);
  assert.equal(p.cachePriced, true);
});

test('blended API price falls back to the prompt price when there is no cache price', () => {
  // 10/11 x 0.6 + 1/11 x 2.4 = 8.4 / 11
  const p = blendedApiPrice({ prompt: 0.6, completion: 2.4 }, 10, 0.7);
  close(p.perM, 8.4 / 11);
  assert.equal(p.cachePriced, false);
  // chat, no cache: (3 x 0.6 + 2.4) / 4 = 1.05
  close(blendedApiPrice({ prompt: 0.6, completion: 2.4, cache_read: 0.15 }, 3, 0).perM, 1.05);
});

test('use-case defaults table', () => {
  assert.deepEqual(USE_CASES.chat, { maxCtx: 8192, avgCtx: 4096, r: 3, itlPin: 0.05, cache: 0 });
  assert.deepEqual(USE_CASES.agents, { maxCtx: 32768, avgCtx: 16384, r: 10, itlPin: 0.15, cache: 0.7 });
  assert.deepEqual(USE_CASES.batch, { maxCtx: 8192, avgCtx: 2048, r: 5, itlPin: null, cache: 0 });
});

test('maxUsers = floor((VRAM x 0.9 - weights / TP) / (per-GPU KV x avg ctx))', () => {
  // 8B FP16: (72e9 - 16e9) / (131,072 x 4096) = 56e9 / 536,870,912 = 104.3
  assert.equal(maxUsers(llama8b, h100, 'fp16', 'fp16', 1, 4096), 104);
  // 70B FP8: (72e9 - 70e9) / (163,840 x 4096) = 2.98
  assert.equal(maxUsers(llama70b, h100, 'fp8', 'fp8', 1, 4096), 2);
  // 70B FP16 on one GPU does not fit at all
  assert.equal(maxUsers(llama70b, h100, 'fp16', 'fp16', 1, 4096), 0);
});

test('operating batch: largest batch under the ITL pin, capped by maxUsers', () => {
  const run = (pin, override) => operatingBatch(llama8b, h100, 'fp16', 'fp16', 1, 4096, 104, pin, override);
  // 50 ms pin: ITL at B = 104 is (16e9 + 104 x 536,870,912) / 3.35e12 = 21.4 ms, so all users fit
  assert.deepEqual(run(0.05), { batch: 104, sloMiss: false });
  // 10 ms pin: B = 32 gives 33.18e9 / 3.35e12 = 9.90 ms, B = 33 gives 10.07 ms
  assert.deepEqual(run(0.01), { batch: 32, sloMiss: false });
  // no pin (batch use case): B = maxUsers
  assert.deepEqual(run(null), { batch: 104, sloMiss: false });
  // 4 ms pin: even B = 1 takes 16.54e9 / 3.35e12 = 4.94 ms
  assert.deepEqual(run(0.004), { batch: 1, sloMiss: true });
  // user override wins, but never above maxUsers or below 1
  assert.deepEqual(run(0.01, 16), { batch: 16, sloMiss: false });
  assert.deepEqual(run(0.01, 500), { batch: 104, sloMiss: false });
});

test('self-host $/1M tokens = (TP x $/hr / 3600) / (tokens/s x utilization) x 1e6', () => {
  // TP 2 at $3/hr, 1000 tok/s, 60%: (6 / 3600) / 600 x 1e6 = $2.78
  close(selfHostPerM(2, 3, 1000, 0.6), 6e6 / (3600 * 600));
});

test('self-host $/day is a step function of replicas', () => {
  // capacity 4.32e7 tokens/day per replica; TP 1 at $2/hr = $48/day per replica
  assert.equal(selfHostPerDay(4.32e7, 4.32e7, 1, 2), 48);
  assert.equal(selfHostPerDay(4.32e7 + 1, 4.32e7, 1, 2), 96);
});

test('break-even: first crossing, or which side wins across 1e5..1e10 tokens/day', () => {
  // TP 1 at $2/hr, 1000 tok/s at 50%: capacity 4.32e7/day, $48/day per replica.
  const be = (apiPerM, tps = 1000, util = 0.5) => breakEven({ tps, util, tp: 1, usdHr: 2, apiPerM });
  // API $2/M: $48 buys 2.4e7 tokens, inside one replica's capacity
  assert.deepEqual(be(2), { kind: 'cross', tpd: 2.4e7 });
  // API $1/M: break-even 4.8e7 is above one replica's capacity, so the API always wins
  assert.deepEqual(be(1), { kind: 'api' });
  // API $1000/M: break-even 4.8e4 is below the chart range
  assert.deepEqual(be(1000), { kind: 'self' });
  // Huge capacity, API $0.001/M: break-even 4.8e10 is above the chart range
  assert.deepEqual(be(0.001, 1e6, 1), { kind: 'api' });
});

test('vllm serve command carries TP, context, batch and FP8 flags', () => {
  const base = { hfId: 'meta-llama/Llama-3.3-70B-Instruct', tp: 2, maxCtx: 8192, batch: 38 };
  assert.equal(vllmCommand({ ...base, prec: 'fp16', kvDtype: 'fp16' }),
    'vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max-model-len 8192 '
    + '--gpu-memory-utilization 0.9 --max-num-seqs 38');
  assert.equal(vllmCommand({ ...base, prec: 'fp8', kvDtype: 'fp8' }),
    'vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max-model-len 8192 '
    + '--gpu-memory-utilization 0.9 --max-num-seqs 38 --quantization fp8 --kv-cache-dtype fp8');
});

test('fmtCount rounds to 2 significant figures with a K/M/B/T suffix', () => {
  assert.equal(fmtCount(42.3e6), '42M');
  assert.equal(fmtCount(1.46e9), '1.5B');
  assert.equal(fmtCount(123456), '120K');
  assert.equal(fmtCount(999), '1K');
  assert.equal(fmtCount(950), '950');
});

test('layer-aware KV per sequence: full x ctx, sliding x min(ctx, window), linear x 0', () => {
  // Gemma 3 27B FP16, per layer per token 2 x 16 x 128 x 2 B = 8192 B.
  // 4096 ctx: (10 x 4096 + 52 x 1024) x 8192 = 771,751,936 B
  assert.equal(kvPerGpuPerSeq(gemma27b, 'fp16', 1, 4096), 771751936);
  // Context inside the window: every layer holds all 512 tokens
  assert.equal(kvPerGpuPerSeq(gemma27b, 'fp16', 1, 512), 62 * 512 * 8192);
  // TP 2 splits the 16 KV heads
  assert.equal(kvPerGpuPerSeq(gemma27b, 'fp16', 2, 4096), 771751936 / 2);
  // Qwen3.5 27B, per layer per token 2 x 4 x 256 x 2 B = 4096 B; only 16 layers keep KV
  assert.equal(kvPerGpuPerSeq(qwen35, 'fp16', 1, 10000), 16 * 10000 * 4096);
  // No layout recorded: every layer is full attention
  assert.equal(kvPerGpuPerSeq(llama70b, 'fp16', 1, 4096), 327680 * 4096);
});

test('max users, TP fit and decode ITL use the layer-aware KV', () => {
  // (72e9 - 54e9) / 771,751,936 = 23.3
  assert.equal(maxUsers(gemma27b, h100, 'fp16', 'fp16', 1, 4096), 23);
  // 128k context: (10 x 131,072 + 52 x 1024) x 8192 = 11.17e9 B, 54e9 + 11.17e9 <= 72e9 fits on 1 GPU
  assert.deepEqual(chooseTP(gemma27b, h100, 'fp16', 'fp16', 131072), { tp: 1 });
  // batch 4 at 4096: (54e9 + 4 x 771,751,936) / 3.35e12
  close(decodeItl(gemma27b, h100, 'fp16', 'fp16', 1, 4, 4096), (54e9 + 4 * 771751936) / 3.35e12);
});
