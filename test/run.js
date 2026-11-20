// End-to-end test: real extension in Chromium + fake Discord call + local server.
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path'), os = require('os');
const { spawn } = require('child_process');
const SCRATCH = process.argv[2];
const EXT_SRC = path.join(__dirname, '..', 'extension');
const EXT = path.join(SCRATCH, 'ext-test'), OUT = path.join(SCRATCH, process.env.OUT_DIR || 'tx-e2e');
fs.rmSync(EXT, { recursive: true, force: true }); if (!process.env.KEEP_OUT) fs.rmSync(OUT, { recursive: true, force: true });
fs.cpSync(EXT_SRC, EXT, { recursive: true });
const man = JSON.parse(fs.readFileSync(path.join(EXT, 'manifest.json')));
for (const cs of man.content_scripts) cs.matches.push('http://localhost/*');   // test page stands in for discord.com
man.host_permissions.push('http://localhost/*');
fs.writeFileSync(path.join(EXT, 'manifest.json'), JSON.stringify(man));
const bg = path.join(EXT, 'background.js');
fs.writeFileSync(bg, fs.readFileSync(bg, 'utf8').replace('ws://127.0.0.1:8765', 'ws://127.0.0.1:8766'));

const site = http.createServer((req, res) => {
  const u = req.url.split('?')[0];
  const f = path.join(__dirname, 'site', u === '/' ? 'index.html' : u.slice(1));
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': f.endsWith('.html') ? 'text/html' : 'audio/wav' }); fs.createReadStream(f).pipe(res);
}).listen(8899);

(async () => {
  const srv = spawn(process.env.SERVER_PY || path.join(__dirname, '..', 'server', '.venv', 'bin', 'python'),
    [path.join(__dirname, '..', 'server', 'server.py'), '--port', '8766', '--out', OUT, ...(process.env.SERVER_ARGS || '').split(' ').filter(Boolean)], { stdio: ['ignore', 'pipe', 'pipe'] });
  global.__srv = srv;
  let log = ''; srv.stdout.on('data', (d) => (log += d)); srv.stderr.on('data', (d) => (log += d));
  while (!log.includes('Model ready')) await new Promise((r) => setTimeout(r, 500));
  console.log('STEP server ready');

  const ctx = await chromium.launchPersistentContext(fs.mkdtempSync(path.join(os.tmpdir(), 'dt-')), {
    channel: 'chromium', headless: true,
    args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`],
  });
  const page = await ctx.newPage();
  page.on('console', (m) => console.log('  [page]', m.text()));
  await page.goto('http://localhost:8899/' + (process.env.BUSY ? '?busy=1' : '') + (process.env.NOWL ? (process.env.BUSY ? '&' : '?') + 'nowl=1' : ''));
  await page.waitForTimeout(1500);
  console.log('STEP page loaded');
  await page.click('#join');
  console.log('STEP joined');            // real user gesture, like clicking "Join Voice"
  let dash = null;
  if (process.env.DASH) {
    dash = await ctx.newPage();
    dash.on('console', (m) => { if (m.type() === 'error') console.log('  [dash error]', m.text()); });
    dash.on('pageerror', (e) => console.log('  [dash exception]', e.message));
    await dash.setViewportSize({ width: 1320, height: 860 });
    await dash.emulateMedia({ colorScheme: 'dark' });
    await dash.goto('http://127.0.0.1:8766/');
  }
  await page.waitForTimeout(13000);
  if (dash) {
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-live.png') });
    await dash.click('#p-ext'); await dash.waitForTimeout(300);
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-pop.png') });
    await dash.keyboard.press('Escape');
    const scroll = await dash.evaluate(() => { window.scrollTo(0, 800); document.documentElement.scrollTop = 800; return { scrollY: window.scrollY, docScroll: document.documentElement.scrollTop, overflow: document.documentElement.scrollHeight - innerHeight }; });
    console.log('PAGE SCROLL CHECK', JSON.stringify(scroll));
  }
  await page.waitForTimeout(15000);
  if (dash) {
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-dark.png') });
    await dash.fill('#hq', 'pricing'); await dash.waitForTimeout(800);
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-search.png') });
    await dash.click('.hit'); await dash.waitForTimeout(800);
    console.log('AFTER HIT', await dash.evaluate(() => document.querySelector('#where').innerText + ' | flashed line: ' + (document.querySelector('.line.new')?.innerText || 'none')));
    await dash.fill('#hq', ''); await dash.waitForTimeout(400);
    await dash.click('.call >> nth=2'); await dash.waitForTimeout(800);
    console.log('OPEN OLD CALL', await dash.evaluate(() => document.querySelector('#where').innerText + ' | lines: ' + document.querySelectorAll('.line').length));
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-groups.png') });
    const before = await dash.evaluate(() => document.querySelectorAll('.call').length);
    await dash.click('.srv >> nth=1'); await dash.waitForTimeout(300);
    const after = await dash.evaluate(() => document.querySelectorAll('.call').length);
    console.log('COLLAPSE second server: calls visible', before, '->', after, '| servers:', await dash.evaluate(() => [...document.querySelectorAll('.srv .nm')].map((e) => e.textContent).join(' / ')));
    await dash.click('.srv >> nth=1'); await dash.waitForTimeout(300);
    await dash.emulateMedia({ colorScheme: 'light' });
    await dash.waitForTimeout(500);
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-light.png') });
    await dash.emulateMedia({ colorScheme: 'dark' });
    await dash.setViewportSize({ width: 1320, height: 1500 });
    await dash.waitForTimeout(500);
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-tall.png') });
    await dash.emulateMedia({ colorScheme: 'light' });
    await dash.setViewportSize({ width: 420, height: 900 });
    await dash.waitForTimeout(400);
    await dash.screenshot({ path: path.join(SCRATCH, 'dash-narrow.png'), fullPage: true });
  }
  const state = await page.evaluate(() => ({ harness: window.__harness, hook: window.__dtHook && { capture: window.__dtHook.capture, tracks: window.__dtHook.tracks, chunks: window.__dtHook.chunks, frames: window.__dtHook.frames, errors: window.__dtHook.errors, live: window.__dtHook.live, names: window.__dtHook.names, nameVia: window.__dtHook.nameVia, loadedAt: window.__dtHook.loadedAt } }));
  console.log('PAGE STATE', JSON.stringify(state));
  await ctx.close();
  await new Promise((r) => setTimeout(r, 6000));   // let Whisper finish
  srv.kill(); site.close();
  const statusLog = path.join(SCRATCH, 'status.log');
  if (fs.existsSync(statusLog)) console.log('STATUS.LOG last:', fs.readFileSync(statusLog, 'utf8').trim().split('\n').pop());
  console.log('SERVER LOG\n' + log.split('\n').filter((l) => !/Fetching|Download|Reconstruction/.test(l)).join('\n'));
})().catch((e) => { console.error(e); process.exit(1); });
process.on('exit', () => { try { global.__srv && global.__srv.kill(); } catch {} });
