// The Hugging Face Space rejects the whole upload when its README metadata breaks a rule,
// so the rules the deploy depends on are checked here, before CI.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('space/README.md front matter passes the Hugging Face Space metadata rules', () => {
  const text = readFileSync(new URL('../space/README.md', import.meta.url), 'utf8');
  const fm = text.match(/^---\n([\s\S]*?)\n---\n/);
  assert.ok(fm, 'front matter block');
  const meta = Object.fromEntries(fm[1].split('\n').map((l) => l.match(/^(\w+):\s*(.*)$/)).filter(Boolean).map((m) => [m[1], m[2]]));
  assert.equal(meta.sdk, 'static');
  assert.ok(meta.title, 'title');
  // HF: "short_description length must be less than or equal to 60 characters long"
  assert.ok(meta.short_description.length <= 60, `short_description is ${meta.short_description.length} chars`);
});
