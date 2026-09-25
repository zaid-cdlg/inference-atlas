// Weekly catalogue refresh: OpenRouter listing -> Hugging Face config.json -> data/models.json.
// Fails closed: if the validation gate fails, nothing is written and the job exits non-zero.
//
//   HF_TOKEN=... node scripts/refresh.mjs
import { readFileSync, writeFileSync, existsSync, appendFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const OPENROUTER = 'https://openrouter.ai/api/v1/models';
const HF = 'https://huggingface.co';
const MODELS_FILE = new URL('../data/models.json', import.meta.url);
const FEATURED_FILE = new URL('../data/featured.json', import.meta.url);

// OpenRouter prices are USD per token as strings. Store USD per 1M tokens.
function perM(raw) {
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? Number((n * 1e6).toPrecision(12)) : null;
}

// One entry per hf_id: drop entries without one, `:free` slugs and `:batch` slugs (async
// batch pricing, not comparable to live serving), then keep the cheapest paid slug
// (prompt + completion). Descriptions are never stored.
export function selectListings(data) {
  const out = new Map();
  for (const m of data) {
    if (!m.hugging_face_id || /:(free|batch)$/.test(m.id)) continue;
    const prompt = perM(m.pricing?.prompt);
    const completion = perM(m.pricing?.completion);
    const paid = prompt !== null && completion !== null && prompt + completion > 0;
    const entry = {
      hf_id: m.hugging_face_id, slug: m.id, name: m.name, context_length: m.context_length,
      pricing: paid ? { prompt, completion, cache_read: perM(m.pricing.input_cache_read) } : null,
    };
    const prev = out.get(entry.hf_id);
    const cost = (e) => (e.pricing ? e.pricing.prompt + e.pricing.completion : Infinity);
    if (!prev || cost(entry) < cost(prev)) out.set(entry.hf_id, entry);
  }
  return out;
}

const ARCH_KEYS = ['params', 'active_params', 'layers', 'heads', 'kv_heads', 'head_dim'];

// First bad architecture field, or undefined. Counts must be positive integers.
export function archError(a) {
  const bad = ARCH_KEYS.find((k) => {
    const v = a?.[k];
    return !(Number.isFinite(v) && v > 0 && (k.endsWith('params') || Number.isInteger(v)));
  });
  return bad ?? (a.active_params > a.params ? 'active_params' : undefined);
}

// Layer types that keep no per-token KV cache (a small constant state instead).
const NO_KV_LAYERS = ['linear_attention', 'mamba', 'conv'];

// How many layers keep full KV, a sliding window of KV, or none. A layout we cannot read
// counts every layer as full attention (never fewer users than reality) and is approximate.
export function attnLayout(c, layers) {
  if (Array.isArray(c.layer_types)) {
    let full = 0;
    let sliding = 0;
    let linear = 0;
    let approx = false;
    for (const t of c.layer_types) {
      if (t === 'sliding_attention') sliding++;
      else if (NO_KV_LAYERS.includes(t)) linear++;
      else {
        full++;
        if (t !== 'full_attention' && t !== 'attention') approx = true;
      }
    }
    if (sliding && !(c.sliding_window > 0)) {
      full += sliding;
      sliding = 0;
      approx = true;
    }
    return { full, sliding, window: sliding ? c.sliding_window : null, linear, approx };
  }
  if (c.full_attention_interval > 0) {
    const full = Math.floor(layers / c.full_attention_interval);
    return { full, sliding: 0, window: null, linear: layers - full, approx: false };
  }
  if (c.sliding_window_pattern > 0 && c.sliding_window > 0) {
    const full = Math.floor(layers / c.sliding_window_pattern);
    return { full, sliding: layers - full, window: c.sliding_window, linear: 0, approx: false };
  }
  const unknown = (c.sliding_window > 0 && c.use_sliding_window !== false)
    || c.attention_chunk_size > 0 || c.hybrid_override_pattern != null;
  return { full: layers, sliding: 0, window: null, linear: 0, approx: unknown };
}

// Native checkpoint format from quantization_config. Anything we cannot size is rejected,
// which quarantines the model.
function nativeFormat(config) {
  const q = config.quantization_config ?? config.text_config?.quantization_config;
  if (!q) return null;
  if (q.quant_method === 'fp8' || q.quant_method === 'mxfp4') return q.quant_method;
  if ((q.quant_method === 'awq' || q.quant_method === 'gptq') && q.bits === 4) return 'int4';
  throw new Error(`unsupported quantization: ${q.quant_method}`);
}

// Bytes per param for each safetensors dtype in the HF API breakdown. For MXFP4 checkpoints
// the U8 entries are the 4-bit expert weights plus their shared scales (4.25 bits).
const DTYPE_PARAM_BYTES = { BF16: 2, F16: 2, F32: 4, F8_E4M3: 1, F8_E5M2: 1, I8: 1 };

function nativeBytes(parameters, quant) {
  let sum = 0;
  for (const [dtype, n] of Object.entries(parameters ?? {})) {
    const b = dtype === 'U8' && quant === 'mxfp4' ? 0.53125 : DTYPE_PARAM_BYTES[dtype];
    if (b === undefined) return null;
    sum += n * b;
  }
  return sum || null;
}

const first = (c, keys) => keys.map((k) => c[k]).find((v) => v != null);

// Architecture numbers the math needs, from config.json plus the HF API param count.
// MoE active params are approximate: routed expert = gate + up + down projections.
export function normalize(config, api) {
  const c = config.text_config ?? config;
  for (const k of ['num_hidden_layers', 'num_attention_heads', 'hidden_size']) {
    if (!(c[k] > 0)) throw new Error(`config missing ${k}`);
  }
  const params = api.safetensors?.total;
  if (!(params > 0)) throw new Error('no parameter count');
  const layers = c.num_hidden_layers;
  const heads = c.num_attention_heads;
  const quant = nativeFormat(config);

  let active = params;
  const experts = first(c, ['n_routed_experts', 'num_local_experts', 'num_experts']);
  const k = first(c, ['num_experts_per_tok', 'experts_per_token']);
  const expertSize = first(c, ['moe_intermediate_size', 'intermediate_size']);
  if (experts > 1 && k > 0 && expertSize > 0) {
    // DeepSeek-style: the first k layers are dense, and next-token-prediction layers are MoE too.
    const moeLayers = layers - (c.first_k_dense_replace ?? 0) + (c.num_nextn_predict_layers ?? 0);
    const perLayer = 3 * c.hidden_size * expertSize * moeLayers;
    active = params - experts * perLayer + k * perLayer;
  }

  const arch = {
    params,
    active_params: active,
    layers,
    heads,
    kv_heads: c.num_key_value_heads ?? heads,
    // MLA query/key heads are nope + rope wide; head_dim there is often absent or 0.
    head_dim: c.kv_lora_rank ? c.qk_nope_head_dim + c.qk_rope_head_dim : c.head_dim ?? c.hidden_size / heads,
    mla: c.kv_lora_rank ? { kv_lora_rank: c.kv_lora_rank, qk_rope_head_dim: c.qk_rope_head_dim } : null,
    moe: experts > 1,
    max_ctx: c.max_position_embeddings ?? null,
    attn: attnLayout(c, layers),
    quant,
    native_bytes: quant ? nativeBytes(api.safetensors.parameters, quant) : null,
  };
  // Hybrid or unusual layouts (linear attention, partial MoE) produce nonsense here.
  // Throwing quarantines just this model instead of failing the whole run.
  const bad = archError(arch);
  if (bad) throw new Error(`unsupported architecture: bad ${bad}`);
  return arch;
}

const DTYPE_BYTES = { float32: 4, bfloat16: 2, float16: 2, float8_e4m3fn: 1 };

// get(url) resolves to parsed JSON or throws an Error with .status.
export async function fetchModel(id, get) {
  const config = await get(`${HF}/${id}/resolve/main/config.json`);
  let api = await get(`${HF}/api/models/${id}?expand[]=safetensors`);
  if (!api.safetensors?.total) {
    const index = await get(`${HF}/${id}/resolve/main/model.safetensors.index.json`);
    const dtype = (config.text_config ?? config).torch_dtype ?? config.torch_dtype;
    api = { safetensors: { total: index.metadata.total_size / (DTYPE_BYTES[dtype] ?? 2) } };
  }
  return normalize(config, api);
}

// A failing existing model keeps its previous architecture with the fresh listing; a
// failing new model is quarantined. Models gone from the listing are dropped.
export async function buildCatalogue(listings, previous, fetchArch) {
  const prevById = new Map(previous.map((m) => [m.hf_id, m]));
  const models = [];
  const kept = [];
  const quarantined = [];
  for (const listing of listings.values()) {
    try {
      models.push({ ...listing, arch: await fetchArch(listing.hf_id) });
    } catch (err) {
      const prev = prevById.get(listing.hf_id);
      if (prev) {
        models.push({ ...listing, arch: prev.arch });
        kept.push(listing.hf_id);
      } else {
        quarantined.push({ hf_id: listing.hf_id, reason: err.message });
      }
    }
  }
  return { models, kept, quarantined };
}

const price = (v) => typeof v === 'number' && v >= 0;

// Global signals only. A missing featured model is a warning, never a failure.
export function validate(models, previousCount, featured) {
  const errors = [];
  if (!models.length) errors.push('no models');
  if (models.length < 0.9 * previousCount) {
    errors.push(`model count ${models.length} is below 90% of the previous ${previousCount}`);
  }
  for (const m of models) {
    const badKey = archError(m.arch);
    if (typeof m.hf_id !== 'string' || badKey) errors.push(`${m.hf_id}: bad arch.${badKey}`);
    const p = m.pricing;
    if (p !== null && !(price(p.prompt) && price(p.completion) && (p.cache_read === null || price(p.cache_read)))) {
      errors.push(`${m.hf_id}: bad pricing`);
    }
  }
  const ids = new Set(models.map((m) => m.hf_id));
  const warnings = featured.filter((id) => !ids.has(id)).map((id) => `featured model ${id} is missing`);
  return { errors, warnings };
}

async function main() {
  const token = process.env.HF_TOKEN;
  const get = async (url) => {
    const auth = token && url.startsWith(HF) ? { Authorization: `Bearer ${token}` } : {};
    const res = await fetch(url, { headers: auth });
    if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status });
    return res.json();
  };
  const read = (f, fallback) => (existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : fallback);
  const previous = read(MODELS_FILE, { models: [] }).models;
  const featured = read(FEATURED_FILE, []);

  const listings = selectListings((await get(OPENROUTER)).data);
  const { models, kept, quarantined } = await buildCatalogue(listings, previous, (id) => fetchModel(id, get));
  const { errors, warnings } = validate(models, previous.length, featured);

  const summary = [
    `## Catalogue refresh`, `${models.length} models (previous ${previous.length})`,
    ...kept.map((id) => `- kept previous entry: ${id}`),
    ...quarantined.map((q) => `- quarantined: ${q.hf_id} (${q.reason})`),
    ...warnings.map((w) => `- warning: ${w}`),
    ...errors.map((e) => `- **error: ${e}**`),
  ].join('\n');
  console.log(summary);
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${summary}\n`);
  if (errors.length) process.exit(1);

  const ids = new Set(models.map((m) => m.hf_id));
  const out = {
    as_of: new Date().toISOString().slice(0, 10),
    featured: featured.filter((id) => ids.has(id)),
    models,
  };
  writeFileSync(MODELS_FILE, `${JSON.stringify(out, null, 1)}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
