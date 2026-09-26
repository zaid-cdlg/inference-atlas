// Browser checks for the static page, served by e2e/serve.mjs on its own port so they never
// clash with a local preview server.
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  use: { baseURL: 'http://127.0.0.1:8002', browserName: 'chromium' },
  webServer: {
    command: 'node e2e/serve.mjs 8002',
    url: 'http://127.0.0.1:8002/',
    reuseExistingServer: false,
  },
});
