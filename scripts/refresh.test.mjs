// Refresh pipeline checks against saved (trimmed) OpenRouter and Hugging Face responses.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { selectListings, normalize, fetchModel, buildCatalogue, validate } from './refresh.mjs';

const fx = (p) => JSON.parse(readFileSync(new URL(`../test/fixtures/${p}`, import.meta.url)));
const hf = (name) => ({ config: fx(`hf/${name}.config.json`), api: fx(`hf/${name}.api.json`) });

test('listings: keep hf ids, drop :free and :batch, dedup to the cheapest paid slug, prices per 1M', () => {
  const m = selectListings(fx('openrouter-models.json').data);
  assert.deepEqual([...m.keys()], ['meta-llama/Llama-3.3-70B-Instruct', 'Qwen/Qwen3.8-27B', 'zai-org/GLM-5.3']);
  assert.deepEqual(m.get('meta-llama/Llama-3.3-70B-Instruct'), {
    hf_id: 'meta-llama/Llama-3.3-70B-Instruct', slug: 'meta-llama/llama-3.3-70b-instruct',
    name: 'Meta: Llama 3.3 70B Instruct', context_length: 131072,
    pricing: { prompt: 0.1, completion: 0.32, cache_read: null },
  });
  // The :free slug is dropped, so the paid slug's cache-read price is kept
  assert.deepEqual(m.get('Qwen/Qwen3.8-27B').pricing, { prompt: 0.42, completion: 3, cache_read: 0.085 });
  // The :batch slug (0.45 + 2.0) is async batch pricing, so the interactive slug is kept
  assert.equal(m.get('zai-org/GLM-5.3').slug, 'z-ai/glm-5.3');
  assert.deepEqual(m.get('zai-org/GLM-5.3').pricing, { prompt: 1.4, completion: 4.4, cache_read: 0.26 });
  // Third-party descriptions are never stored
  assert.ok(!('description' in m.get('zai-org/GLM-5.3')));
  // A model listed only as :free is not in the catalogue at all
  const free = { id: 'x/y:free', hugging_face_id: 'x/Y', name: 'Y', context_length: 1, pricing: { prompt: '0', completion: '0' } };
  assert.equal(selectListings([free]).size, 0);
  // Same for a model listed only as :batch
  assert.equal(selectListings([{ ...free, id: 'x/y:batch', pricing: { prompt: '0.000001', completion: '0.000001' } }]).size, 0);
});

test('normalize GQA without head_dim: head_dim = hidden_size / heads', () => {
  const { config, api } = hf('qwen2.5-7b');
  assert.deepEqual(normalize(config, api), {
    params: 7615616512, active_params: 7615616512, layers: 28, heads: 28, kv_heads: 4,
    head_dim: 128, mla: null, moe: false, max_ctx: 32768,
    // sliding_window is set but use_sliding_window is false: every layer is full attention
    attn: { full: 28, sliding: 0, window: null, linear: 0, approx: false },
    quant: null, native_bytes: null,
  });
});

test('normalize MHA: missing num_key_value_heads means kv_heads = heads', () => {
  const { config, api } = hf('phi-2');
  assert.equal(normalize(config, api).kv_heads, 32);
});

test('normalize unwraps text_config (multimodal wrappers)', () => {
  const { config, api } = hf('gemma-3-27b');
  const n = normalize(config, api);
  assert.equal(n.layers, 62);
  assert.equal(n.heads, 32);
  assert.equal(n.kv_heads, 16);
  assert.equal(n.head_dim, 128);
});

test('normalize MoE: active = total - all routed experts + the k routed per token', () => {
  // Expert = gate + up + down = 3 x hidden x expert intermediate size.
  // Qwen3-30B-A3B: 3 x 2048 x 768, 128 experts, 8 per token, 48 layers
  const q = 3 * 2048 * 768;
  const qn = normalize(hf('qwen3-30b-a3b').config, hf('qwen3-30b-a3b').api);
  assert.equal(qn.active_params, 30532122624 - 128 * q * 48 + 8 * q * 48);
  assert.equal(qn.moe, true);
  // Mixtral 8x7B has no moe_intermediate_size, so experts use intermediate_size 14,336
  const x = 3 * 4096 * 14336;
  const xn = normalize(hf('mixtral-8x7b').config, hf('mixtral-8x7b').api);
  assert.equal(xn.active_params, 46702792704 - 8 * x * 32 + 2 * x * 32);
});

test('normalize MLA + MoE (DeepSeek V3): latent KV, dense first layers, MTP layer counted', () => {
  // 61 layers - 3 dense + 1 next-token-prediction layer = 59 MoE layers
  const e = 3 * 7168 * 2048;
  const n = normalize(hf('deepseek-v3').config, hf('deepseek-v3').api);
  assert.deepEqual(n.mla, { kv_lora_rank: 512, qk_rope_head_dim: 64 });
  assert.equal(n.params, 684531386000);
  assert.equal(n.active_params, 684531386000 - 256 * e * 59 + 8 * e * 59);
});

test('attention layout: sliding_window_pattern (Gemma 3), 1 full layer in every 6', () => {
  assert.deepEqual(normalize(hf('gemma-3-27b').config, hf('gemma-3-27b').api).attn,
    { full: 10, sliding: 52, window: 1024, linear: 0, approx: false });
});

test('attention layout: layer_types with linear attention (Qwen3.5), linear layers keep no KV', () => {
  assert.deepEqual(normalize(hf('qwen3.5-27b').config, hf('qwen3.5-27b').api).attn,
    { full: 16, sliding: 0, window: null, linear: 48, approx: false });
});

test('attention layout: full_attention_interval without layer_types (Qwen3-Next)', () => {
  // 48 layers, full attention every 4th: 12 full, 36 linear
  assert.deepEqual(normalize(hf('qwen3-coder-next').config, hf('qwen3-coder-next').api).attn,
    { full: 12, sliding: 0, window: null, linear: 36, approx: false });
});

test('attention layout: unknown layouts count every layer as full and are marked approximate', () => {
  // Mistral 7B v0.1 has a sliding_window but no per-layer layout
  assert.deepEqual(normalize(hf('mistral-7b-v0.1').config, hf('mistral-7b-v0.1').api).attn,
    { full: 32, sliding: 0, window: null, linear: 0, approx: true });
  // An unfamiliar layer type counts as full attention; mamba and conv layers keep no KV
  const { config, api } = hf('qwen2.5-7b');
  const types = [...Array(26).fill('deepseek_sparse_attention'), 'mamba', 'conv'];
  assert.deepEqual(normalize({ ...config, layer_types: types }, api).attn,
    { full: 26, sliding: 0, window: null, linear: 2, approx: true });
});

test('native FP8 (DeepSeek V3): bytes per dtype from the HF breakdown', () => {
  const n = normalize(hf('deepseek-v3').config, hf('deepseek-v3').api);
  assert.equal(n.quant, 'fp8');
  // BF16 x 2 + F8_E4M3 x 1 + F32 x 4
  assert.equal(n.native_bytes, 3918786560 * 2 + 680571043840 + 41555600 * 4);
});

test('native MXFP4 (gpt-oss-20b): U8 blocks count as 4.25-bit params, attention stays BF16', () => {
  const n = normalize(hf('gpt-oss-20b').config, hf('gpt-oss-20b').api);
  assert.equal(n.quant, 'mxfp4');
  assert.equal(n.native_bytes, 1804459584 * 2 + 19110297600 * 0.53125);
  // Experts use experts_per_token: 3 x 2880 x 2880 per expert, 32 experts, 4 per token, 24 layers
  const e = 3 * 2880 * 2880;
  assert.equal(n.active_params, 20914757184 - 32 * e * 24 + 4 * e * 24);
  assert.deepEqual(n.attn, { full: 12, sliding: 12, window: 128, linear: 0, approx: false });
});

test('AWQ/GPTQ 4-bit is INT4; other quantization formats are quarantined', () => {
  const { config, api } = hf('qwen2.5-7b');
  assert.equal(normalize({ ...config, quantization_config: { quant_method: 'awq', bits: 4 } }, api).quant, 'int4');
  assert.throws(() => normalize({ ...config, quantization_config: { quant_method: 'compressed-tensors' } }, api),
    /unsupported quantization: compressed-tensors/);
  assert.throws(() => normalize({ ...config, quantization_config: { quant_method: 'gptq', bits: 3 } }, api),
    /unsupported quantization: gptq/);
});

test('MLA head_dim = qk_nope_head_dim + qk_rope_head_dim, not hidden_size / heads', () => {
  // GLM-4.7-Flash: 2048 / 20 heads = 102.4, but its heads are 192 + 64 = 256 wide
  const g = normalize(hf('glm-4.7-flash').config, hf('glm-4.7-flash').api);
  assert.equal(g.head_dim, 256);
  assert.deepEqual(g.mla, { kv_lora_rank: 512, qk_rope_head_dim: 64 });
  // DeepSeek V3: 128 + 64
  assert.equal(normalize(hf('deepseek-v3').config, hf('deepseek-v3').api).head_dim, 192);
  // An explicit head_dim of 0 (GLM-5.3-Flash) is ignored for MLA models too
  const zero = { ...hf('glm-4.7-flash').config, head_dim: 0 };
  assert.equal(normalize(zero, hf('glm-4.7-flash').api).head_dim, 256);
});

test('normalize rejects configs missing core shape keys', () => {
  assert.throws(() => normalize({ hidden_size: 4096 }, hf('qwen2.5-7b').api), /num_hidden_layers/);
});

test('normalize rejects nonsense results so the model is quarantined, not the whole run', () => {
  // MoE expert maths larger than the whole checkpoint gives active params below zero
  assert.throws(() => normalize(hf('qwen3-30b-a3b').config, { safetensors: { total: 1e9 } }), /bad active_params/);
  // hidden_size not divisible by heads and no head_dim: fractional head_dim
  const odd = { ...hf('qwen2.5-7b').config, num_attention_heads: 27 };
  assert.throws(() => normalize(odd, hf('qwen2.5-7b').api), /bad head_dim/);
});

// Fake network: URL -> fixture, anything else -> HTTP error
const net = (routes) => async (url) => {
  if (url in routes) return routes[url];
  const err = new Error(`HTTP ${routes.status ?? 404} ${url}`);
  err.status = routes.status ?? 404;
  throw err;
};
const HF = 'https://huggingface.co';

test('params fall back to the safetensors index size / dtype bytes', async () => {
  const id = 'Qwen/Qwen2.5-7B-Instruct';
  const get = net({
    [`${HF}/${id}/resolve/main/config.json`]: hf('qwen2.5-7b').config,
    [`${HF}/api/models/${id}?expand[]=safetensors`]: { id },
    [`${HF}/${id}/resolve/main/model.safetensors.index.json`]: fx('hf/qwen2.5-7b.index.json'),
  });
  // 15,231,233,024 bytes / 2 (bfloat16)
  assert.equal((await fetchModel(id, get)).params, 15231233024 / 2);
});

test('catalogue: gated 403 keeps the previous entry, new failures are quarantined, delisted models drop', async () => {
  const listings = selectListings(fx('openrouter-models.json').data);
  const prevLlama = {
    hf_id: 'meta-llama/Llama-3.3-70B-Instruct', slug: 'old', name: 'old', context_length: 1,
    pricing: { prompt: 9, completion: 9, cache_read: null },
    arch: { ...normalize(hf('qwen2.5-7b').config, hf('qwen2.5-7b').api) },
  };
  const prevDelisted = { ...prevLlama, hf_id: 'gone/Model' };
  const glm = normalize(hf('qwen2.5-7b').config, hf('qwen2.5-7b').api);
  const fetchArch = async (id) => {
    if (id === 'zai-org/GLM-5.3') return glm;
    const err = new Error('HTTP 403');
    err.status = 403;
    throw err;
  };
  const r = await buildCatalogue(listings, [prevLlama, prevDelisted], fetchArch);
  assert.deepEqual(r.models.map((m) => m.hf_id), ['meta-llama/Llama-3.3-70B-Instruct', 'zai-org/GLM-5.3']);
  // Previous architecture, fresh listing and price
  const llama = r.models[0];
  assert.equal(llama.arch, prevLlama.arch);
  assert.equal(llama.pricing.prompt, 0.1);
  assert.deepEqual(r.kept, ['meta-llama/Llama-3.3-70B-Instruct']);
  assert.deepEqual(r.quarantined, [{ hf_id: 'Qwen/Qwen3.8-27B', reason: 'HTTP 403' }]);
});

test('validation gate: schema, >= 90% of the previous count; a missing featured model only warns', () => {
  const good = { hf_id: 'a/b', slug: 'a/b', name: 'B', context_length: 8192, pricing: null,
    arch: normalize(hf('qwen2.5-7b').config, hf('qwen2.5-7b').api) };
  const models = Array.from({ length: 9 }, (_, i) => ({ ...good, hf_id: `a/m${i}` }));
  assert.deepEqual(validate(models, 10, ['a/m0']), { errors: [], warnings: [] });
  assert.deepEqual(validate(models, 11, []).errors, ['model count 9 is below 90% of the previous 11']);
  assert.deepEqual(validate(models, 10, ['x/y']).warnings, ['featured model x/y is missing']);
  assert.deepEqual(validate([], 0, []).errors, ['no models']);
  const bad = [{ ...good, arch: { ...good.arch, kv_heads: 0 } }, { ...good, hf_id: 'a/c', pricing: { prompt: -1, completion: 1, cache_read: null } }];
  assert.deepEqual(validate(bad, 0, []).errors, ['a/b: bad arch.kv_heads', 'a/c: bad pricing']);
});
