// Every featured model must survive the refresh (never quarantined) and its attention
// layout must match what its config.json says, read by hand from each model's page.
// Fixtures are trimmed config.json + HF API responses; gated repos were read from the
// ungated unsloth/ mirror of the same config (see _read_from in each fixture).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalize } from '../scripts/refresh.mjs';
import { FEATURED_LAYOUT } from './featured-layout.mjs';


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
