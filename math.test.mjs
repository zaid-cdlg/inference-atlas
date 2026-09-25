// Hand-derived checks for math.js. Every expected value is worked out from the
// standard formulas and public model configs (Hugging Face config.json) or public
// GPU spec sheets. Params are rounded to keep the arithmetic readable by hand.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  kvBytesPerToken, kvPerGpuPerToken, headsSplitOk, chooseTP, decodeItl,
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
