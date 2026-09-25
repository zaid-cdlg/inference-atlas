// Refresh pipeline checks against saved (trimmed) OpenRouter and Hugging Face responses.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { selectListings, normalize, fetchModel, buildCatalogue, validate } from './refresh.mjs';

const fx = (p) => JSON.parse(readFileSync(new URL(`../test/fixtures/${p}`, import.meta.url)));
const hf = (name) => ({ config: fx(`hf/${name}.config.json`), api: fx(`hf/${name}.api.json`) });

test('listings: keep hf ids, drop :free, dedup to the cheapest paid slug, prices per 1M', () => {
  const m = selectListings(fx('openrouter-models.json').data);
  assert.deepEqual([...m.keys()], ['meta-llama/Llama-3.3-70B-Instruct', 'Qwen/Qwen3.8-27B', 'zai-org/GLM-5.3']);
  assert.deepEqual(m.get('meta-llama/Llama-3.3-70B-Instruct'), {
    hf_id: 'meta-llama/Llama-3.3-70B-Instruct', slug: 'meta-llama/llama-3.3-70b-instruct',
    name: 'Meta: Llama 3.3 70B Instruct', context_length: 131072,
    pricing: { prompt: 0.1, completion: 0.32, cache_read: null },
  });
  // The :free slug is dropped, so the paid slug's cache-read price is kept
  assert.deepEqual(m.get('Qwen/Qwen3.8-27B').pricing, { prompt: 0.42, completion: 3, cache_read: 0.085 });
  // 0.45 + 2.0 beats 1.4 + 4.4
  assert.equal(m.get('zai-org/GLM-5.3').slug, 'z-ai/glm-5.3:batch');
  // Third-party descriptions are never stored
  assert.ok(!('description' in m.get('zai-org/GLM-5.3')));
  // A model listed only as :free is not in the catalogue at all
  const free = { id: 'x/y:free', hugging_face_id: 'x/Y', name: 'Y', context_length: 1, pricing: { prompt: '0', completion: '0' } };
  assert.equal(selectListings([free]).size, 0);
});

test('normalize GQA without head_dim: head_dim = hidden_size / heads', () => {
  const { config, api } = hf('qwen2.5-7b');
  assert.deepEqual(normalize(config, api), {
    params: 7615616512, active_params: 7615616512, layers: 28, heads: 28, kv_heads: 4,
    head_dim: 128, mla: null, moe: false, max_ctx: 32768,
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
