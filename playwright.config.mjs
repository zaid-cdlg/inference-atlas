// Browser smoke checks for the static page. Serves the repo root on port 8001.
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  use: { baseURL: 'http://127.0.0.1:8001', browserName: 'chromium' },
  webServer: {
    command: 'python3 -m http.server 8001 --bind 127.0.0.1',
    url: 'http://127.0.0.1:8001/',
    reuseExistingServer: true,
  },
});
