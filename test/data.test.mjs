// Schema checks for the hand-kept data. Every number carries a source and a date.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { FEATURED_LAYOUT } from './featured-layout.mjs';
import { DEFAULT_MODEL, FALLBACK_GPU } from '../state.js';
import { validate } from '../scripts/refresh.mjs';

const read = (p) => JSON.parse(readFileSync(new URL(`../data/${p}`, import.meta.url)));
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const cited = (s, what) => {
  assert.ok(s, `${what}: no source`);
  assert.match(s.source_url ?? s.url, /^(https:\/\/|gpus\.sources\.md#)/, `${what}: bad source url`);
  assert.match(s.checked_date, DATE, `${what}: bad checked_date`);
};

test('gpus.json: 9 GPUs, every value positive and cited with a date', () => {
  const { gpus } = read('gpus.json');
  assert.equal(gpus.length, 9);
  assert.equal(new Set(gpus.map((g) => g.id)).size, 9);
  assert.ok(gpus.some((g) => g.id === FALLBACK_GPU));
  for (const g of gpus) {
    for (const k of ['vram_gb', 'bandwidth_gbs', 'usd_per_hr']) {
      assert.ok(g[k] > 0, `${g.id}.${k}`);
      cited(g.sources[k], `${g.id}.${k}`);
    }
    assert.ok(g.peak_tflops.fp16 > 0, `${g.id} fp16`);
    cited(g.sources['peak_tflops.fp16'], `${g.id}.peak_tflops.fp16`);
    assert.equal(typeof g.fp8, 'boolean');
    cited(g.sources.fp8, `${g.id}.fp8`);
    if (g.fp8) {
      assert.ok(g.peak_tflops.fp8 > 0, `${g.id} fp8 peak`);
      cited(g.sources['peak_tflops.fp8'], `${g.id}.peak_tflops.fp8`);
    } else {
      assert.equal(g.peak_tflops.fp8, null);
    }
    if (g.verify) assert.ok(g.verify_reason?.length > 10, `${g.id} verify_reason`);
  }
});

test('quality.json: every quant penalty row quotes a dated source', () => {
  const { quant_penalty: rows } = read('quality.json');
  assert.deepEqual(rows.map((r) => r.id), ['fp16', 'fp8', 'int8', 'int4-gptq', 'int4-awq']);
  for (const r of rows) {
    assert.ok(r.label && r.quote, r.id);
    assert.ok(r.sources.length > 0, r.id);
    for (const s of r.sources) cited(s, r.id);
  }
});

test('quality.json: the INT4 pick for each model is a real repo with a publisher', () => {
  const { quantized_repos: repos } = read('quality.json');
  for (const [id, entry] of Object.entries(repos)) {
    assert.ok(id in FEATURED_LAYOUT, `${id} is not featured`);
    const pick = entry.int4;
    if (!pick) continue;
    assert.equal(pick.url, `https://huggingface.co/${pick.repo}`, id);
    assert.ok(['awq', 'gptq', 'qat'].includes(pick.method), `${id} method`);
    assert.ok(['official', 'trusted', 'community'].includes(pick.publisher), `${id} publisher`);
    // A pre-quantized model only runs in its own format, so it never gets an INT4 pick
    assert.equal(FEATURED_LAYOUT[id][4], null, `${id} is natively quantized`);
  }
  assert.deepEqual(repos['google/gemma-4-31B-it'].int4,
    { repo: 'google/gemma-4-31B-it-qat-w4a16-ct', url: 'https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct', method: 'qat', publisher: 'official' });
  assert.equal(repos['Qwen/Qwen3.6-27B'].int4.publisher, 'community');
  assert.equal(repos['Qwen/Qwen3.6-35B-A3B'].int4.publisher, 'community');
});

test('featured.json is the cross-checked list and contains the page default', () => {
  const featured = read('featured.json');
  assert.deepEqual([...featured].sort(), Object.keys(FEATURED_LAYOUT).sort());
  assert.ok(featured.includes(DEFAULT_MODEL));
});

test('models.json passes the refresh validation gate', () => {
  const m = read('models.json');
  assert.match(m.as_of, DATE);
  assert.deepEqual(validate(m.models, 0, []).errors, []);
});
