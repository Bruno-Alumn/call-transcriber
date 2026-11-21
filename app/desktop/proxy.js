// Proxy settings for one account's Discord window.
//
// Each account has its own Chromium session, and Chromium can use a different
// proxy per session. Chromium's proxy rules never carry a username or password,
// so we keep those here and hand them to Chromium from the app's 'login' event
// when the proxy asks for them.
//
// Accepts the formats people copy from a provider. Webshare, for one, lists
// `address:port:username:password` and builds `http://user:pass@host:port` URLs:
//
//   host:port
//   host:port:user:password
//   user:password@host:port
//   user:password:host:port
//   http://user:password@host:port
//   https://user:password@host:port
//   socks4://host:port
//   socks5://user:password@host:port   (socks, socks5h and socks4a also work)
//   pac:https://example.com/proxy.pac
//
// An empty value means "no proxy of your own": the account follows the
// system's network settings, like the rest of the app.

const SCHEMES = {
  http:    { rule: 'http',   port: 80 },
  https:   { rule: 'https',  port: 443 },
  socks:   { rule: 'socks5', port: 1080 },
  socks4:  { rule: 'socks4', port: 1080 },
  socks4a: { rule: 'socks4', port: 1080 },
  socks5:  { rule: 'socks5', port: 1080 },
  socks5h: { rule: 'socks5', port: 1080 },
};

const FORMAT = 'Enter a proxy as host:port, host:port:user:password, or a URL such as http://user:password@host:port.';

const isPort = (v) => /^\d{1,5}$/.test(String(v)) && Number(v) >= 1 && Number(v) <= 65535;
const decode = (v) => { try { return decodeURIComponent(v); } catch { return String(v); } };
const bareHost = (v) => String(v || '').trim().replace(/^\[|\]$/g, '');

function result(config, extra = {}) {
  return { ok: true, config, ...extra };
}

function badHost(host) {
  return !host || /[\s/?#@]/.test(host) || (host.includes(':') && !/^[0-9a-f:.]+$/i.test(host));
}

// `scheme` is the one the user wrote (kept for display); `rule` is what
// Chromium's proxy rules understand.
function build(scheme, rule, host, port, username, password, defaultPort) {
  host = bareHost(host).toLowerCase();
  const p = port === '' || port == null ? defaultPort : Number(port);
  if (badHost(host)) return { ok: false, error: FORMAT };
  if (!Number.isInteger(p) || p < 1 || p > 65535) {
    return { ok: false, error: `The proxy port must be a number from 1 to 65535${port ? ` (got "${port}")` : ''}.` };
  }
  const h = host.includes(':') ? `[${host}]` : host;       // IPv6
  const proxyRules = rule === 'http' ? `${h}:${p}` : `${rule}://${h}:${p}`;
  return result({ mode: 'fixed_servers', proxyRules }, {
    proxy: { scheme, host, port: p, username: username || '', password: password || '' },
  });
}

function fromUrl(raw, scheme, spec) {
  let u;
  try { u = new URL(raw); } catch { return null; }
  if (!u.hostname) return { ok: false, error: FORMAT };
  return build(scheme, spec.rule, u.hostname, u.port || spec.port, decode(u.username), decode(u.password), spec.port);
}

function fromList(raw, scheme, spec) {
  let s = String(raw || '').trim();
  if (!s) return { ok: false, error: FORMAT };

  let username = '', password = '';

  // user:password@host:port
  const at = s.lastIndexOf('@');
  if (at > 0) {
    const cred = s.slice(0, at);
    const sep = cred.indexOf(':');
    if (sep < 0) return { ok: false, error: FORMAT };
    username = decode(cred.slice(0, sep));
    password = decode(cred.slice(sep + 1));
    s = s.slice(at + 1).trim();
  }

  // host port [user [password]] (space- or tab-separated lines)
  if (/\s/.test(s)) {
    const parts = s.split(/\s+/);
    if (parts.length < 2) return { ok: false, error: FORMAT };
    if (!username) { username = parts[2] || ''; password = parts[3] || ''; }
    return build(scheme, spec.rule, parts[0], parts[1], username, password, spec.port);
  }

  const parts = s.split(':');

  // [::1]:port[:user:password]
  if (/^\[/.test(s)) {
    const m = /^\[([^\]]+)\]:(\d+)(?::([^:]*)(?::(.*))?)?$/.exec(s);
    if (!m) return { ok: false, error: FORMAT };
    if (!username) { username = m[3] || ''; password = m[4] || ''; }
    return build(scheme, spec.rule, m[1], m[2], username, password, spec.port);
  }

  let host, port;
  if (parts.length === 2) {
    [host, port] = parts;
  } else if (parts.length >= 3 && isPort(parts[1])) {
    // address:port:username:password (Webshare's list format)
    host = parts[0]; port = parts[1];
    if (!username) { username = parts[2] || ''; password = parts.slice(3).join(':'); }
  } else if (parts.length >= 4 && isPort(parts[parts.length - 1])) {
    // user:password:host:port
    host = parts[parts.length - 2]; port = parts[parts.length - 1];
    if (!username) { username = parts[0] || ''; password = parts.slice(1, -2).join(':'); }
  } else {
    return { ok: false, error: FORMAT };
  }
  return build(scheme, spec.rule, host, port, username, password, spec.port);
}

// Returns one of:
//   { ok: true, config, proxy? }   a proxy to use (config goes to session.setProxy)
//   { ok: true, config }           no proxy: system (empty) or direct ("direct")
//   { ok: false, error }           a message to show the person
function parseProxy(input) {
  const raw = String(input == null ? '' : input).trim().replace(/^["']+|["']+$/g, '');
  if (!raw) return result({ mode: 'system' });
  if (/^(none|off|direct|direct:\/\/|no proxy)$/i.test(raw)) return result({ mode: 'direct' }, { direct: true });

  if (/^pac[:+]/i.test(raw)) {
    const url = raw.replace(/^pac[:+]/i, '');
    try {
      const u = new URL(url);
      if (u.protocol !== 'http:' && u.protocol !== 'https:') throw 0;
      return result({ mode: 'pac_script', pacScript: u.href }, { pac: u.href });
    } catch {
      return { ok: false, error: 'Enter a PAC address as pac:https://example.com/proxy.pac.' };
    }
  }

  const m = /^([a-z][a-z0-9+.-]*):\/\//i.exec(raw);
  if (m) {
    const scheme = m[1].toLowerCase();
    const spec = SCHEMES[scheme];
    if (!spec) return { ok: false, error: `This proxy type isn't supported: ${scheme}. Use http, https, socks4 or socks5.` };
    return fromUrl(raw, scheme, spec) || fromList(raw.slice(m[0].length), scheme, spec);
  }

  return fromList(raw, 'http', SCHEMES.http);
}

// A short, safe description for the accounts list: never the password.
function redactProxy(input) {
  const p = parseProxy(input);
  if (!p.ok) return 'invalid proxy';
  if (p.pac) return `PAC ${p.pac}`;
  if (!p.proxy) return '';
  const { scheme, host, port, username } = p.proxy;
  const h = host.includes(':') ? `[${host}]` : host;
  return `${scheme}://${username ? `${username}:***@` : ''}${h}:${port}`;
}

// Credentials for the proxy host:port Chromium is asking about. Returns null
// when there is nothing to send (or this isn't one of our proxies).
function proxyLogin(input, host, port) {
  const p = parseProxy(input);
  if (!p.ok || !p.proxy || !p.proxy.username) return null;
  if (String(p.proxy.host).toLowerCase() !== String(host || '').toLowerCase()) return null;
  if (Number(p.proxy.port) !== Number(port)) return null;
  return { username: p.proxy.username, password: p.proxy.password };
}

module.exports = { parseProxy, redactProxy, proxyLogin };
