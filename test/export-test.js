// Export: transcript button (Markdown) and right-click a server (CSV), check files.
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path'), fs = require('fs');
const SCRATCH = process.argv[2], OUT = path.join(SCRATCH, 'tx-exp');
(async () => {
  const srv = spawn(path.join(__dirname, '..', 'server', '.venv', 'bin', 'python'), [path.join(__dirname, '..', 'server', 'server.py'), '--port', '8766', '--out', OUT]);
  process.on('exit', () => srv.kill());
  let log = ''; srv.stdout.on('data', (d) => (log += d)); srv.stderr.on('data', (d) => (log += d));
  while (!log.includes('Dashboard:')) await new Promise((r) => setTimeout(r, 300));
  const b = await chromium.launch(); const page = await b.newPage({ acceptDownloads: true, viewport: { width: 1320, height: 860 } });
  page.on('pageerror', (e) => console.log('PAGE ERROR', e.message));
  await page.goto('http://127.0.0.1:8766/'); await page.waitForTimeout(1500);
  const grab = async (fn) => { const [dl] = await Promise.all([page.waitForEvent('download'), fn()]); const p = path.join(SCRATCH, dl.suggestedFilename()); await dl.saveAs(p); return p; };
  await page.click('#export'); await page.waitForTimeout(200);
  await page.screenshot({ path: path.join(SCRATCH, 'export-dialog.png') });
  const md = await grab(() => page.click('.modal button[data-f="md"]'));
  const mdText = fs.readFileSync(md, 'utf8');
  console.log('MD file:', path.basename(md), '|', mdText.split('\n').length, 'lines | head:', JSON.stringify(mdText.split('\n').slice(0, 5).join(' / ')).slice(0, 220));
  await page.click('.srv >> nth=0', { button: 'right' }); await page.click('#ctx button[data-a="export"]'); await page.waitForTimeout(200);
  const csv = await grab(() => page.click('.modal button[data-f="csv"]'));
  const rows = fs.readFileSync(csv, 'utf8').trim().split('\n');
  console.log('CSV file:', path.basename(csv), '|', rows.length - 1, 'rows | header:', rows[0], '| row1:', rows[1].slice(0, 120));
  for (const f of ['srt', 'json', 'txt']) {
    await page.click('#export'); await page.waitForTimeout(150);
    const p = await grab(() => page.click(`.modal button[data-f="${f}"]`));
    const t = fs.readFileSync(p, 'utf8');
    if (f === 'json') { const j = JSON.parse(t); console.log('JSON ok:', j.length, 'call(s),', j[0].lines.length, 'lines, first:', JSON.stringify(j[0].lines[0]).slice(0, 120)); }
    else console.log(f.toUpperCase() + ':', JSON.stringify(t.split('\n').slice(0, f === 'srt' ? 4 : 3).join(' / ')).slice(0, 180));
  }
  await b.close(); process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
