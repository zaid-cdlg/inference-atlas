import { test, expect } from '@playwright/test';

test('phone 375x812: the verdict and its cost line are above the fold, no errors', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/');
  await expect(page.locator('#stats')).toContainText('per 1M tokens');
  for (const id of ['#verdict', '#stats']) {
    const box = await page.locator(id).boundingBox();
    expect(box.y, `${id} top`).toBeGreaterThanOrEqual(0);
    expect(box.y + box.height, `${id} bottom`).toBeLessThanOrEqual(812);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(375);
  expect(errors).toEqual([]);
});

test('model picker offers every featured model, not just the current one', async ({ page }) => {
  await page.goto('/');
  const input = page.locator('#model');
  await expect(input).toHaveValue('gpt-oss-120b');
  // A datalist only suggests options that match the box, so focus must empty it
  await input.focus();
  await expect(input).toHaveValue('');
  await expect(input).toHaveAttribute('placeholder', 'gpt-oss-120b');
  expect(await page.locator('#model-list option').count()).toBe(20);
  // Leaving without a choice puts the current model back
  await input.blur();
  await expect(input).toHaveValue('gpt-oss-120b');
  // Picking another model works
  await input.focus();
  await input.fill('Llama 3.3 70B');
  await input.press('Enter');
  await expect(page.locator('#stats')).toContainText('per 1M tokens');
  await expect(page).toHaveURL(/model=meta-llama%2FLlama-3\.3-70B-Instruct/);
  await input.blur();
  await expect(input).toHaveValue('Llama 3.3 70B');
});
