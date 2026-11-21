// Proxy parsing: the formats one account's proxy field accepts, what Chromium
// is given, and what the accounts list is allowed to show.
//   node test/proxy-test.js
const { parseProxy, redactProxy, proxyLogin } = require('../app/desktop/proxy.js');

let failed = 0;
const ok = (name, cond, extra = '') => {
  if (!cond) { failed++; console.log('FAIL', name, extra); }
  else console.log('ok  ', name);
};
const eq = (name, got, want) => ok(name, JSON.stringify(got) === JSON.stringify(want), `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);

// No proxy: follow the system, or explicitly direct.
eq('empty -> system', parseProxy('').config, { mode: 'system' });
eq('spaces -> system', parseProxy('   ').config, { mode: 'system' });
eq('direct -> direct', parseProxy('direct').config, { mode: 'direct' });

// Webshare's list line: address:port:username:password.
{
  const p = parseProxy('p.webshare.io:80:alice:s3cret');
  ok('webshare list parses', p.ok);
  eq('webshare list rules', p.config, { mode: 'fixed_servers', proxyRules: 'p.webshare.io:80' });
  eq('webshare list user', p.proxy.username, 'alice');
  eq('webshare list pass', p.proxy.password, 's3cret');
}
// Webshare's proxy URL, including a percent-encoded password.
{
  const p = parseProxy('http://alice-rotate:p%40ss@p.webshare.io:80');
  ok('webshare URL parses', p.ok);
  eq('webshare URL pass', p.proxy.password, 'p@ss');
  eq('webshare URL rules', p.config.proxyRules, 'p.webshare.io:80');
}
// Other common shapes.
eq('host:port', parseProxy('proxy.example.com:3128').config, { mode: 'fixed_servers', proxyRules: 'proxy.example.com:3128' });
eq('socks5', parseProxy('socks5://u:p@127.0.0.1:1080').config, { mode: 'fixed_servers', proxyRules: 'socks5://127.0.0.1:1080' });
eq('socks5h -> socks5', parseProxy('socks5h://127.0.0.1:1080').config.proxyRules, 'socks5://127.0.0.1:1080');
eq('socks4', parseProxy('socks4://127.0.0.1:1080').config.proxyRules, 'socks4://127.0.0.1:1080');
eq('https proxy', parseProxy('https://u:p@proxy.example.com:443').config.proxyRules, 'https://proxy.example.com:443');
eq('user:pass@host:port', parseProxy('u:p@proxy.example.com:3128').proxy.host, 'proxy.example.com');
eq('user:pass:host:port', parseProxy('u:p:proxy.example.com:3128').proxy.host, 'proxy.example.com');
eq('user:pass:host:port port', parseProxy('u:p:proxy.example.com:3128').proxy.port, 3128);
eq('space separated', parseProxy('proxy.example.com 3128 u p').proxy.username, 'u');
eq('IPv6', parseProxy('[::1]:8080:u:p').config.proxyRules, '[::1]:8080');
eq('password with colon', parseProxy('proxy.example.com:3128:u:a:b').proxy.password, 'a:b');
eq('PAC', parseProxy('pac:https://example.com/proxy.pac').config, { mode: 'pac_script', pacScript: 'https://example.com/proxy.pac' });

// Bad input must explain itself, not guess.
for (const bad of ['socks5://', 'proxy.example.com:0', 'proxy.example.com:99999', 'ftp://proxy.example.com:21', 'justahost', 'http://']) {
  const p = parseProxy(bad);
  ok(`rejects ${bad || 'empty-ish'}`, !p.ok && !!p.error, JSON.stringify(p));
}

// The accounts list never gets the password.
ok('redacts password', !redactProxy('http://alice:s3cret@p.webshare.io:80').includes('s3cret'));
ok('redacts list password', !redactProxy('p.webshare.io:80:alice:s3cret').includes('s3cret'));
eq('redacted label', redactProxy('p.webshare.io:80:alice:s3cret'), 'http://alice:***@p.webshare.io:80');
eq('redacted without credentials', redactProxy('proxy.example.com:3128'), 'http://proxy.example.com:3128');

// Chromium asks for credentials by proxy host and port.
eq('login matches', proxyLogin('p.webshare.io:80:alice:s3cret', 'p.webshare.io', 80), { username: 'alice', password: 's3cret' });
eq('login host mismatch', proxyLogin('p.webshare.io:80:alice:s3cret', 'other.example.com', 80), null);
eq('login port mismatch', proxyLogin('p.webshare.io:80:alice:s3cret', 'p.webshare.io', 81), null);
eq('login without credentials', proxyLogin('proxy.example.com:3128', 'proxy.example.com', 3128), null);

if (failed) { console.log(`\n${failed} FAILED`); process.exitCode = 1; }
else console.log('\nPASS');
