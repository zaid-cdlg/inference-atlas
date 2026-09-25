// Regression tests from /qa on 2026-09-25. Report: .gstack/qa-reports/qa-report-127-0-0-1-2026-09-25.md
import { test, expect } from '@playwright/test';

// Regression: ISSUE-001 — the footer "sources" link opened raw Markdown (text/markdown),
// which GitHub Pages serves unrendered or as a download.
test('footer sources link opens the rendered sources page on GitHub', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('footer a', { hasText: 'sources' }))
    .toHaveAttribute('href', 'https://github.com/zaid-cdlg/inference-atlas/blob/main/data/gpus.sources.md');
});

// Regression: ISSUE-002 — on a Turing GPU, a model published only in MXFP4 hid its MXFP4
// option and never said why it can't run there.
test('precision picker shows the native MXFP4 option as disabled, with the Turing reason', async ({ page }) => {
  await page.goto('/?model=openai%2Fgpt-oss-20b&gpu=t4');
  await expect(page.locator('#prec option[value=mxfp4]')).toBeDisabled();
  await expect(page.locator('#prec-hint')).toContainText('vLLM has no MXFP4 kernel for Turing GPUs such as the T4.');
  // On a GPU that can run it, MXFP4 is offered and enabled
  await page.goto('/?model=openai%2Fgpt-oss-20b&gpu=l4');
  await expect(page.locator('#prec option[value=mxfp4]')).toBeEnabled();
});
