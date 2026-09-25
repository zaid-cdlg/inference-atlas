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
