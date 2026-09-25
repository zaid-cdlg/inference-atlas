// Every featured model must survive the refresh (never quarantined) and its attention
// layout must match what its config.json says, read by hand from each model's page.
// Fixtures are trimmed config.json + HF API responses; gated repos were read from the
// ungated unsloth/ mirror of the same config (see _read_from in each fixture).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalize } from '../scripts/refresh.mjs';

// hf_id: [full layers, sliding layers, linear layers, window, native format]
export const FEATURED_LAYOUT = {
  'meta-llama/Meta-Llama-3.1-8B-Instruct': [32, 0, 0, null, null],
  'Qwen/Qwen3-8B': [36, 0, 0, null, null],
  'Qwen/Qwen3.5-9B': [8, 0, 24, null, null],
  'google/gemma-3-4b-it': [5, 29, 0, 1024, null],
  'mistralai/Ministral-3-8B-Instruct-2512': [34, 0, 0, null, 'fp8'],
  'google/gemma-4-31B-it': [10, 50, 0, 1024, null],
  'Qwen/Qwen3.6-27B': [16, 0, 48, null, null],
  'Qwen/Qwen3-32B': [64, 0, 0, null, null],
  'mistralai/Mistral-Small-3.2-24B-Instruct-2506': [40, 0, 0, null, null],
  'google/gemma-3-27b-it': [10, 52, 0, 1024, null],
  'microsoft/phi-4': [40, 0, 0, null, null],
  'meta-llama/Llama-3.3-70B-Instruct': [80, 0, 0, null, null],
  'mistralai/Devstral-2-123B-Instruct-2512': [88, 0, 0, null, 'fp8'],
  'openai/gpt-oss-20b': [12, 12, 0, 128, 'mxfp4'],
  'openai/gpt-oss-120b': [18, 18, 0, 128, 'mxfp4'],
  'Qwen/Qwen3.6-35B-A3B': [10, 0, 30, null, null],
  'google/gemma-4-26B-A4B-it': [5, 25, 0, 1024, null],
  'zai-org/GLM-4.5-Air': [46, 0, 0, null, null],
  'Qwen/Qwen3-235B-A22B-Instruct-2507': [94, 0, 0, null, null],
  'deepseek-ai/DeepSeek-V3.2': [61, 0, 0, null, 'fp8'],
};

const fixture = (id, kind) =>
  JSON.parse(readFileSync(new URL(`fixtures/featured/${id.replace('/', '__')}.${kind}.json`, import.meta.url)));

for (const [id, [full, sliding, linear, window, quant]] of Object.entries(FEATURED_LAYOUT)) {
  test(`featured ${id}: normalizes (not quarantined) with the expected layout`, () => {
    const arch = normalize(fixture(id, 'config'), fixture(id, 'api'));
    assert.deepEqual(arch.attn, { full, sliding, window, linear, approx: false });
    assert.equal(arch.quant, quant);
  });
}

test('Gemma 4 names its per-token expert count top_k_experts', () => {
  // 26B-A4B: expert = 3 x 2816 x 704; 128 experts, 8 per token, 30 layers
  const id = 'google/gemma-4-26B-A4B-it';
  const arch = normalize(fixture(id, 'config'), fixture(id, 'api'));
  const e = 3 * 2816 * 704;
  assert.equal(arch.moe, true);
  assert.equal(arch.active_params, arch.params - 128 * e * 30 + 8 * e * 30);
});
