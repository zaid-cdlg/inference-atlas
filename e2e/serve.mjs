// Static server for the browser tests (python3 http.server drops connections when a page
// fires a dozen requests at once). Serves the repo root: node e2e/serve.mjs [port]
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.md': 'text/markdown', '.svg': 'image/svg+xml', '.webp': 'image/webp', '.woff2': 'font/woff2',
};

createServer(async (req, res) => {
  try {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
    const body = await readFile(join(root, path.endsWith('/') ? `${path}index.html` : path));
    res.writeHead(200, { 'content-type': TYPES[extname(path) || '.html'] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
}).listen(Number(process.argv[2] ?? 8002), '127.0.0.1');
