// Per-account proxy test: the real Electron app, two accounts, each with its
// own session. The listener starts with a Webshare list-line proxy; the main
// account has none. We check Chromium resolves each session's proxy separately,
// then change and remove the proxy through the dashboard bridge.
//   node test/proxy-app-test.js <scratch-dir>
const { _electron: electron } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const SCRATCH = path.resolve(process.argv[2] || fs.mkdtempSync(path.join(require('os').tmpdir(), 'ct-proxy-')));
fs.mkdirSync(SCRATCH, { recursive: true });
const DESKTOP = path.join(__dirname, '..', 'app', 'desktop');
const PORT = 8773;
const exe = process.env.EXE || (process.platform === 'darwin'
  ? path.join(DESKTOP, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(DESKTOP, 'node_modules', 'electron', 'dist', 'electron.exe'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const fail = (m) => { console.log('FAIL', m); failed++; };

const WH = 'p.webshare.io:80:alice:s3cret';

(async () => {
  const ud = path.join(SCRATCH, 'userdata');
  fs.rmSync(ud, { recursive: true, force: true }); fs.mkdirSync(ud, { recursive: true });
  fs.writeFileSync(path.join(ud, 'accounts.json'), JSON.stringify([
    { id: 'main', role: 'main', partition: 'persist:test-main' },
    { id: 'l1', role: 'listener', partition: 'persist:test-l1', muted: true, proxy: WH },
  ]));
  const app = await electron.launch({ executablePath: exe, args: [DESKTOP], timeout: 120000,
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_USER_DATA: ud,
      CT_DISCORD_URL: 'http://127.0.0.1:59999/', CT_SERVER_ARGS: `--out ${path.join(SCRATCH, 'tx')}` } });
  const logs = []; app.process().stdout.on('data', (d) => logs.push(String(d))); app.process().stderr.on('data', (d) => logs.push(String(d)));
  let dash = null;
  const dashUrl = new RegExp(`^(http://127\\.0\\.0\\.1:${PORT}|data:)`);
  for (let i = 0; i < 150 && !dash; i++) {
    dash = app.windows().find((w) => dashUrl.test(w.url())) || null;
    if (!dash) await sleep(200);
  }
  if (!dash) { fail('dashboard window missing'); await app.close(); return; }
  for (let i = 0; i < 60 && !(await dash.evaluate(() => typeof window.ctApp === 'object').catch(() => false)); i++) await sleep(250);

  const resolve = (partition) => app.evaluate(async ({ session }, p) => session.fromPartition(p).resolveProxy('https://discord.com'), partition);

  // Startup: the listener's stored proxy is applied to its session only.
  const l1 = await resolve('persist:test-l1');
  const main = await resolve('persist:test-main');
  console.log('START listener', l1, '| main', main);
  if (!/PROXY p\.webshare\.io:80\b/i.test(l1)) fail('startup proxy not applied to the listener session');
  if (/p\.webshare\.io/i.test(main)) fail('the main account picked up the listener\'s proxy');

  // The accounts list shows the proxy without its password.
  const info = await dash.evaluate(() => window.ctApp.accounts());
  const shown = (info.accounts || []).find((a) => a.id === 'l1') || {};
  console.log('SHOWN', JSON.stringify({ proxy: shown.proxy, mode: shown.proxyMode }));
  if (shown.proxy !== 'http://alice:***@p.webshare.io:80') fail('account list does not show a redacted proxy: ' + shown.proxy);
  if (JSON.stringify(info).includes('s3cret')) fail('the proxy password leaked to the dashboard');

  // Change it live, through the same bridge the Proxy… dialog uses.
  const set = await dash.evaluate(() => window.ctApp.setProxy('l1', 'socks5://bob:pw@127.0.0.1:1080'));
  console.log('SET socks5', JSON.stringify(set));
  if (set?.error) fail('setting a socks5 proxy failed: ' + set.error);
  const sock = await resolve('persist:test-l1');
  console.log('AFTER socks5', sock);
  if (!/SOCKS5 127\.0\.0\.1:1080\b/i.test(sock)) fail('the changed proxy was not applied');
  // It is saved, encrypted when the OS supports it, and decrypts back to the same value.
  const kept = JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8')).find((a) => a.id === 'l1') || {};
  console.log('SAVED AS', kept.proxyEnc ? 'encrypted' : kept.proxy ? 'plain text (no OS encryption here)' : 'nothing');
  if (!kept.proxy && !kept.proxyEnc) fail('the proxy was not saved');
  if (kept.proxyEnc) {
    const dec = await app.evaluate(({ safeStorage }, b64) => {
      try { return safeStorage.decryptString(Buffer.from(b64, 'base64')); } catch { return ''; }
    }, kept.proxyEnc);
    if (dec !== 'socks5://bob:pw@127.0.0.1:1080') fail('the saved proxy does not decrypt back: ' + dec);
  }

  // Bad input is explained and changes nothing.
  const bad = await dash.evaluate(() => window.ctApp.setProxy('l1', 'socks5://'));
  console.log('BAD', JSON.stringify(bad));
  if (!bad?.error) fail('invalid proxy was accepted');
  if (!/SOCKS5 127\.0\.0\.1:1080\b/i.test(await resolve('persist:test-l1'))) fail('a rejected proxy changed the session');

  // Removing it goes back to the system setting.
  const cleared = await dash.evaluate(() => window.ctApp.setProxy('l1', ''));
  const after = await resolve('persist:test-l1');
  console.log('CLEARED', JSON.stringify(cleared), '->', after);
  if (cleared?.error) fail('removing the proxy failed: ' + cleared.error);
  if (/127\.0\.0\.1:1080|p\.webshare\.io/i.test(after)) fail('the proxy stayed after removal');
  const saved = JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8'));
  const stored = saved.find((a) => a.id === 'l1') || {};
  if (stored.proxy || stored.proxyEnc) fail('accounts.json still has the removed proxy');
  const encavail = await app.evaluate(({ safeStorage }) => { try { return safeStorage.isEncryptionAvailable(); } catch { return false; } });
  if (encavail && fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8').includes('s3cret')) fail('the proxy password is stored in clear text');

  await app.evaluate(({ app }) => app.quit()); await sleep(2500);
  if (failed) { console.log(`\n${failed} FAILED`); console.log(logs.join('').split('\n').slice(-25).join('\n')); process.exitCode = 1; }
  else console.log('\nPASS');
})().catch((e) => { console.error(e); process.exit(1); });
