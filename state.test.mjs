// Permalink parsing: round-trip, fallbacks and hostile input.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseState, serializeState, urlSyncer, DEFAULT_MODEL } from './state.js';

const ctx = {
  models: [DEFAULT_MODEL, 'Qwen/Qwen2.5-7B-Instruct'],
  gpus: { h100: { fp8: true }, a100_80: { fp8: false } },
  int4: [DEFAULT_MODEL],
};
const parse = (qs) => parseState(new URLSearchParams(qs), ctx);

test('no params: example model, chat defaults, auto GPU/precision/batch, no notices', () => {
  assert.deepEqual(parse(''), {
    state: {
      model: DEFAULT_MODEL, gpu: null, prec: null, kv: 'fp16', use: 'chat',
      max_ctx: 8192, avg_ctx: 4096, r: 3, cache: 0, batch: null, util: 60, tpd: 1e7,
    },
    notices: [],
  });
});

test('use case sets the defaults for omitted numbers', () => {
  const { state } = parse('use=agents');
  assert.equal(state.max_ctx, 32768);
  assert.equal(state.avg_ctx, 16384);
  assert.equal(state.r, 10);
  assert.equal(state.cache, 70);
});

test('round-trip is identical, and default values are left out of the link', () => {
  const qs = 'model=Qwen%2FQwen2.5-7B-Instruct&gpu=h100&prec=fp8&kv=fp8&use=agents&avg_ctx=8000&batch=12&util=80&tpd=50000000';
  const { state } = parse(qs);
  assert.equal(serializeState(state), qs);
  assert.deepEqual(parse(serializeState(state)).state, state);
  assert.equal(serializeState(parse('').state), '');
});

test('numbers are clamped to their ranges, and avg context never exceeds max context', () => {
  const { state } = parse('max_ctx=100&avg_ctx=999999&r=1000&cache=100&batch=0&util=1&tpd=1e20');
  assert.equal(state.max_ctx, 512);
  assert.equal(state.avg_ctx, 512);
  assert.equal(state.r, 100);
  assert.equal(state.cache, 95);
  assert.equal(state.batch, 1);
  assert.equal(state.util, 5);
  assert.equal(state.tpd, 1e12);
  assert.equal(parse('max_ctx=9999999').state.max_ctx, 1048576);
});

test('garbage numbers fall back silently to defaults', () => {
  const { state, notices } = parse('max_ctx=abc&r=&cache=1e400&batch=-Infinity&util=NaN');
  assert.equal(state.max_ctx, 8192);
  assert.equal(state.r, 3);
  assert.equal(state.cache, 0);
  assert.equal(state.batch, null);
  assert.equal(state.util, 60);
  assert.deepEqual(notices, []);
});

test('removed model: example model and a notice, other params kept', () => {
  const { state, notices } = parse('model=gone%2Fmodel&gpu=h100&r=5');
  assert.equal(state.model, DEFAULT_MODEL);
  assert.equal(state.gpu, 'h100');
  assert.equal(state.r, 5);
  assert.deepEqual(notices, ['model_removed']);
});

test('unknown GPU falls back to H100 with a notice', () => {
  const { state, notices } = parse('gpu=rtx9090');
  assert.equal(state.gpu, 'h100');
  assert.deepEqual(notices, ['gpu_unknown']);
});

test('precision not available: FP8 on a non-FP8 GPU, or INT4 without a repo, falls back to FP16', () => {
  assert.deepEqual(parse('gpu=a100_80&prec=fp8'), {
    ...parse('gpu=a100_80&prec=fp16'), notices: ['prec_unavailable'],
  });
  assert.equal(parse('model=Qwen%2FQwen2.5-7B-Instruct&prec=int4').state.prec, 'fp16');
  assert.equal(parse('prec=int4').state.prec, 'int4');
});

test('hostile params are inert: never echoed, never prototype keys', () => {
  const { state, notices } = parse(
    'model=%3Cscript%3Ealert(1)%3C%2Fscript%3E&gpu=__proto__&prec=constructor&kv=%22%3E&use=toString',
  );
  assert.equal(state.model, DEFAULT_MODEL);
  assert.equal(state.gpu, 'h100');
  assert.equal(state.prec, null);
  assert.equal(state.kv, 'fp16');
  assert.equal(state.use, 'chat');
  assert.deepEqual(notices, ['model_removed', 'gpu_unknown', 'link_invalid']);
  assert.ok(!JSON.stringify(state).includes('<'));
});

test('URL sync debounces to one replaceState call per 300 ms burst', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const calls = [];
  const win = { location: { pathname: '/atlas/' }, history: { replaceState: (...a) => calls.push(a) } };
  const sync = urlSyncer(win);
  for (const r of [4, 5, 6]) sync({ ...parse('').state, r });
  t.mock.timers.tick(299);
  assert.equal(calls.length, 0);
  t.mock.timers.tick(1);
  assert.deepEqual(calls, [[null, '', '/atlas/?r=6']]);
});
