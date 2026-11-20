// Screen-share picker (runs in picker.html).
const list = document.getElementById('list');
const go = document.getElementById('go');
let chosen = null;

function card(src) {
  const b = document.createElement('button');
  b.className = 'src';
  const th = document.createElement('div'); th.className = 'th';
  if (src.thumb) { const img = document.createElement('img'); img.src = src.thumb; img.alt = ''; th.appendChild(img); }
  else th.textContent = src.placeholder || '';
  const nm = document.createElement('div'); nm.className = 'nm'; nm.textContent = src.name;
  if (src.note) { const sm = document.createElement('small'); sm.textContent = src.note; nm.appendChild(sm); }
  b.append(th, nm);
  b.onclick = () => { document.querySelectorAll('.src').forEach((x) => x.classList.remove('on')); b.classList.add('on'); chosen = src.id; go.disabled = false; };
  b.ondblclick = () => window.picker.choose(src.id);
  return b;
}

function section(title, items) {
  if (!items.length) return;
  const h = document.createElement('h2'); h.textContent = title;
  const g = document.createElement('div'); g.className = 'grid';
  items.forEach((s) => g.appendChild(card(s)));
  list.append(h, g);
}

window.picker.sources().then((s) => {
  list.textContent = '';
  section('Earshot', s.app);
  section('Screens', s.screens);
  section('Windows', s.windows);
  if (s.note) { const n = document.createElement('div'); n.className = 'sub'; n.style.marginTop = '12px'; n.textContent = s.note; list.appendChild(n); }
  list.querySelector('.src')?.click();
});
go.onclick = () => chosen && window.picker.choose(chosen);
document.getElementById('cancel').onclick = () => window.picker.choose(null);
addEventListener('keydown', (e) => { if (e.key === 'Escape') window.picker.choose(null); if (e.key === 'Enter' && chosen) window.picker.choose(chosen); });
