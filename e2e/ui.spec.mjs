// Full UI walk-through: every control, every page state, desktop width.
import { test, expect } from '@playwright/test';

const FEATURED = 20;
let errors;

test.beforeEach(async ({ page }) => {
  errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.setViewportSize({ width: 1280, height: 900 });
});
test.afterEach(() => expect(errors).toEqual([]));

const stats = (page) => page.locator('#stats');
const ready = async (page, qs = '') => {
  await page.goto(`/${qs}`);
  await expect(page.locator('#model')).toBeEnabled();
};

test('first paint: verdict, stats, chart with crossing, command, footer', async ({ page }) => {
  await ready(page);
  await expect(page.locator('#verdict')).toHaveText(/^Self-hosting likely wins above ~\d+(\.\d)?[KMB] tokens\/day$/);
  await expect(stats(page)).toHaveText(/^1× B200 180GB · \d+ users at once · \$[\d.]+ vs \$[\d.]+ per 1M tokens$/);
  await expect(page.locator('#chart svg path.api')).toHaveCount(1);
  await expect(page.locator('#chart svg path.self')).toHaveCount(1);
  await expect(page.locator('#chart svg circle.cross')).toHaveCount(1);
  expect(await page.locator('#chart-table tr').count()).toBe(12);
  await expect(page.locator('#command')).toHaveText(/^vllm serve openai\/gpt-oss-120b --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs \d+$/);
  await expect(page.locator('#cmd-note')).toContainText('MXFP4');
  await expect(page.locator('#gpu-hint')).toContainText('An AMD MI300X may cost less here');
  await expect(page.locator('#asof')).toHaveText(/^\d{4}-\d{2}-\d{2}$/);
  await expect(page.locator('#gpudate')).toHaveText(/^\d{4}-\d{2}-\d{2}$/);
  await expect(page.locator('#quality tbody tr')).toHaveCount(5);
});

test('"What these numbers mean" explains every number', async ({ page }) => {
  await ready(page);
  await page.locator('#explain-box summary').click();
  const terms = await page.locator('#explain dt').allTextContents();
  for (const t of ['GPUs', 'Users at once', 'Speed', 'Throughput', 'Self-host price', 'API price', 'Approximate']) {
    expect(terms).toContain(t);
  }
});

test('use case buttons switch defaults, pressed state and the link', async ({ page }) => {
  await ready(page);
  const before = await stats(page).textContent();
  for (const [use, hint] of [['agents', '16K-token'], ['batch', 'Offline'], ['chat', '50 ms']]) {
    await page.locator(`[data-use=${use}]`).click();
    await expect(page.locator(`[data-use=${use}]`)).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#use-hint')).toContainText(hint);
    if (use === 'agents') {
      await expect(page).toHaveURL(/use=agents/);
      await expect(stats(page)).not.toHaveText(before);
    }
  }
  await expect(page).not.toHaveURL(/use=/);
});

test('every GPU can be picked by hand and gives a result or a clear error', async ({ page }) => {
  await ready(page, '?model=meta-llama%2FLlama-3.3-70B-Instruct');
  const ids = await page.locator('#gpu option').evaluateAll((os) => os.map((o) => o.value).filter(Boolean));
  expect(ids).toHaveLength(9);
  for (const id of ids) {
    await page.locator('#gpu').selectOption(id);
    await expect(page).toHaveURL(new RegExp(`gpu=${id}`));
    const ok = await stats(page).textContent();
    if (ok) {
      await expect(page.locator('#command')).toContainText('--tensor-parallel-size');
      await expect(page.locator('#cmd-note')).toHaveText(id === 'mi300x' ? /ROCm build/ : /^(?!.*ROCm)/);
    } else {
      await expect(page.locator('#cmd-error')).toBeVisible();
      await expect(page.locator('#verdict')).toHaveText(/doesn't fit|can't be split/);
    }
  }
  await page.locator('#gpu').selectOption('');
  await expect(page.locator('#gpu-hint')).toContainText('lowest cost per token');
});

test('precision: options follow the model and GPU, INT4 swaps in the quantized repo', async ({ page }) => {
  await ready(page);
  // gpt-oss is published in MXFP4: only that runs
  const enabled = () => page.locator('#prec option:not([disabled])').evaluateAll((os) => os.map((o) => o.value));
  expect(await enabled()).toEqual(['', 'mxfp4']);
  await expect(page.locator('#prec-hint')).toContainText('published in MXFP4');

  await ready(page, '?model=meta-llama%2FLlama-3.3-70B-Instruct&gpu=h100');
  expect(await enabled()).toEqual(['', 'fp16', 'fp8', 'int4']);
  await page.locator('#prec').selectOption('int4');
  await expect(page.locator('#command')).toContainText('vllm serve casperhansen/llama-3.3-70b-instruct-awq');
  await page.locator('#prec').selectOption('fp8');
  await expect(page.locator('#command')).toContainText('--quantization fp8');

  await page.locator('#gpu').selectOption('a100_80');
  expect(await enabled()).toEqual(['', 'fp16', 'int4']);
  await expect(page.locator('#prec-hint')).toContainText('FP8: This GPU has no FP8 support.');

  await ready(page, '?model=Qwen%2FQwen3.6-27B&gpu=h100');
  await expect(page.locator('#prec option[value=int4]')).toHaveText('INT4 (community quant)');
  await page.locator('#prec').selectOption('int4');
  await expect(page.locator('#cmd-note')).toContainText('(community quant)');
});

test('every featured model loads, in every use case, without errors', async ({ page }) => {
  await ready(page);
  const names = await page.locator('#model-list option').evaluateAll((os) => os.map((o) => o.label));
  expect(names).toHaveLength(FEATURED);
  for (const id of names) {
    for (const use of ['chat', 'agents', 'batch']) {
      await ready(page, `?model=${encodeURIComponent(id)}&use=${use}`);
      await expect(page.locator('#notices')).toBeEmpty();
      await expect(stats(page)).toContainText('users at once');
      await expect(page.locator('#command')).toContainText(`--max-num-seqs`);
    }
  }
});

test('model picker: full list on focus, search, no-match message', async ({ page }) => {
  await ready(page);
  const input = page.locator('#model');
  await input.focus();
  await expect(input).toHaveValue('');
  await input.fill('zzz');
  await input.press('Enter');
  await expect(page.locator('#model-hint')).toHaveText("No model matches 'zzz'. Try a family name like 'qwen'.");
  await input.fill('Qwen3 8B');
  await input.press('Enter');
  await expect(page).toHaveURL(/model=Qwen%2FQwen3-8B/);
  await expect(page.locator('#model-hint')).toHaveText('');
});

test('error states: does not fit, head split, SLO miss', async ({ page }) => {
  await ready(page, '?model=meta-llama%2FLlama-3.3-70B-Instruct&gpu=t4');
  await expect(page.locator('#verdict')).toHaveText(/doesn't fit on T4 16GB, even 8 of them/);
  await expect(page.locator('#command-card')).toBeHidden();
  await expect(page.locator('#chart-fig')).toBeHidden();
  await expect(page.locator('#kv')).toBeHidden();

  await ready(page, '?model=microsoft%2Fphi-4&gpu=t4');
  await expect(page.locator('#verdict')).toHaveText(/attention heads can't be split/);

  await ready(page, '?model=meta-llama%2FMeta-Llama-3.1-8B-Instruct&gpu=l4&prec=fp16');
  await expect(page.locator('#slo')).toHaveText('Misses the 50 ms target even at 1 user. Try a faster GPU or FP8.');
  await expect(stats(page)).toContainText('per 1M tokens');
});

test('"API wins everywhere" state shows the chart note and no crossing dot', async ({ page }) => {
  await ready(page, '?model=meta-llama%2FLlama-3.3-70B-Instruct');
  await expect(page.locator('#verdict')).toHaveText('The API is likely cheaper at any volume up to 10B tokens/day');
  await expect(page.locator('#chart-note')).toHaveText('API is cheaper across 100K to 10B tokens/day.');
  await expect(page.locator('#chart svg circle.cross')).toHaveCount(0);
});

test('chart: hover reads the costs, "you" marker follows tokens per day', async ({ page }) => {
  await ready(page);
  await page.locator('#chart').scrollIntoViewIfNeeded();
  const box = await page.locator('#chart svg').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.locator('#chart-readout')).toHaveText(/^At [\d.]+[KMB] tokens\/day: self-hosting \$[\d.,]+\/day, API \$[\d.,<]+\/day\.$/);
  const youX = () => page.locator('#chart line.you').getAttribute('x1');
  const x1 = Number(await youX());
  await page.locator('#tune summary').click();
  await page.locator('#tpd').fill('1000000000');
  await page.locator('#tpd').press('Tab');
  await expect(page).toHaveURL(/tpd=1000000000/);
  expect(Number(await youX())).toBeGreaterThan(x1);
});

test('tune assumptions: every input changes the result and the link', async ({ page }) => {
  await ready(page, '?model=meta-llama%2FLlama-3.3-70B-Instruct&gpu=h100&prec=fp16');
  await page.locator('#tune summary').click();
  const users = async () => Number((await stats(page).textContent()).match(/(\d+) users/)[1]);

  const u0 = await users();
  await page.locator('#kvdtype').selectOption('fp8');
  await expect(page).toHaveURL(/kv=fp8/);
  expect(await users()).toBeGreaterThan(u0);

  const edits = [['#maxctx', '16384', /max_ctx=16384/], ['#ratio', '10', /r=10/], ['#cache', '50', /cache=50/], ['#util', '90', /util=90/], ['#batch', '2', /batch=2/]];
  // Every number on the page: stats, command, explanations and the chart's data table
  const numbers = () => page.evaluate(() => ['stats', 'command', 'explain', 'chart-table'].map((id) => document.getElementById(id).textContent).join('|'));
  for (const [sel, v, url] of edits) {
    const before = await numbers();
    await page.locator(sel).fill(v);
    await page.locator(sel).press('Tab');
    await expect(page).toHaveURL(url);
    expect(await numbers(), sel).not.toBe(before);
  }
  await expect(page.locator('#command')).toContainText('--max-model-len 16384');
  await expect(page.locator('#command')).toContainText('--max-num-seqs 2');
  // Out-of-range input is clamped, not echoed
  await page.locator('#cache').fill('500');
  await page.locator('#cache').press('Tab');
  await expect(page.locator('#cache')).toHaveValue('95');
  // Clearing the batch goes back to auto
  await page.locator('#batch').fill('');
  await page.locator('#batch').press('Tab');
  await expect(page).not.toHaveURL(/batch=/);
});

test('KV sim: fills on view, slider changes the user count and heading', async ({ page }) => {
  await ready(page, '?model=meta-llama%2FLlama-3.3-70B-Instruct&gpu=h100');
  await page.locator('#kv').scrollIntoViewIfNeeded();
  await expect(page.locator('#kv-legend')).toContainText('users fit.', { timeout: 5000 });
  const n = Number((await page.locator('#kv-h').textContent()).match(/\d+/)[0]);
  expect(await page.locator('#kv-users .u').count()).toBe(n);
  await page.locator('#avg').fill('8192');
  await expect(page.locator('#avg')).toHaveAttribute('aria-valuetext', '8k tokens');
  await expect(page.locator('#kv-h')).not.toHaveText(`Why ${n} users fit`);
  await expect(page).toHaveURL(/avg_ctx=8192/);
});

test('copy command and copy link put the right text on the clipboard', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await ready(page, '?use=agents');
  await page.locator('#copy-cmd').click();
  await expect(page.locator('#copy-cmd')).toHaveText('Copied');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(await page.locator('#command').textContent());
  await page.locator('#copy-link').click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('http://127.0.0.1:8002/?use=agents');
  await expect(page.locator('#copy-cmd')).toHaveText('Copy command', { timeout: 3000 });
});

test('permalinks: a shared link restores every control; bad links show notices', async ({ page }) => {
  await ready(page, '?model=Qwen%2FQwen3-32B&gpu=h100&prec=fp8&kv=fp8&use=agents&avg_ctx=8192&r=5&cache=40&util=80&tpd=50000000');
  await expect(page.locator('#model')).toHaveValue('Qwen3 32B');
  await expect(page.locator('#gpu')).toHaveValue('h100');
  await expect(page.locator('#prec')).toHaveValue('fp8');
  await expect(page.locator('[data-use=agents]')).toHaveAttribute('aria-pressed', 'true');
  await page.locator('#tune summary').click();
  await expect(page.locator('#kvdtype')).toHaveValue('fp8');
  await expect(page.locator('#ratio')).toHaveValue('5');
  await expect(page.locator('#cache')).toHaveValue('40');

  await ready(page, '?model=%3Cscript%3Ealert(1)%3C%2Fscript%3E&gpu=rtx9090&use=x');
  await expect(page.locator('#notices p')).toHaveText([
    'That model is no longer listed. Showing the example instead.',
    "That GPU isn't in our list. Showing the H100 instead.",
    "Part of that link wasn't valid, so defaults are used for it.",
  ]);
  expect(await page.content()).not.toContain('<script>alert');
});

test('history does not grow while adjusting controls', async ({ page }) => {
  await ready(page);
  const h = await page.evaluate(() => history.length);
  for (const use of ['agents', 'batch', 'chat', 'agents']) await page.locator(`[data-use=${use}]`).click();
  await page.waitForTimeout(400);
  expect(await page.evaluate(() => history.length)).toBe(h);
});

test('"How we estimate" link opens the explanation', async ({ page }) => {
  await ready(page);
  await page.locator('#how-link').click();
  await expect(page.locator('#how')).toHaveAttribute('open', '');
  await expect(page).toHaveURL(/#how$/);
});

test('keyboard: every control is reachable and shows a focus ring', async ({ page }) => {
  await ready(page);
  const seen = new Set();
  for (let i = 0; i < 16; i++) {
    await page.keyboard.press('Tab');
    const f = await page.evaluate(() => {
      const e = document.activeElement;
      return { id: e.id || e.dataset.use || e.textContent.trim().slice(0, 20), ring: getComputedStyle(e).outlineStyle };
    });
    seen.add(f.id);
    expect(f.ring, f.id).toBe('solid');
  }
  for (const id of ['model', 'chat', 'agents', 'batch', 'gpu', 'prec', 'copy-cmd', 'copy-link']) expect([...seen]).toContain(id);
});

test('reduced motion: the KV sim shows its final state at once', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await ready(page);
  await expect(page.locator('#kv-legend')).toContainText('users fit.');
  await expect(page.locator('.kv-cover')).toHaveCount(0);
});

test('data load failure shows an error with Retry, not a blank page', async ({ page }) => {
  await page.route('**/data/models.json', (r) => r.abort());
  await page.goto('/');
  await expect(page.locator('#notices')).toContainText("Couldn't load the model list.");
  await expect(page.locator('#notices button')).toHaveText('Retry');
  errors = errors.filter((e) => !e.includes('ERR_FAILED'));
});
