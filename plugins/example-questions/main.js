// Example Earshot plugin: a "Questions" panel plus a mark on every question in
// the transcript. Copy this folder to start your own plugin (see PLUGINS.md).
export default function (earshot) {
  const esc = earshot.escape;
  const isQuestion = (r) => /\?\s*$/.test(r.text);

  // 1. Mark questions in the transcript.
  earshot.transcript.decorate((r) => (isQuestion(r) ? '<span class="q-mark">question</span>' : ''));

  // 2. A panel you can place anywhere with Customize.
  let body = null;
  earshot.panels.add({ id: 'questions', title: 'Questions', render: (el) => { body = el; draw(); } });

  function draw() {
    if (!body) return;
    const lines = earshot.data.lines();
    const items = [];
    lines.forEach((r, i) => {
      if (!isQuestion(r)) return;
      const who = earshot.data.speakerOf(r);
      // the first other person to speak within 20 s counts as the answer
      const ans = lines.slice(i + 1).find((x) => x.t - r.end < 20000 && earshot.data.speakerOf(x) !== who);
      items.push(`<div class="q-item"><span class="q-who" style="color:${earshot.data.colorFor(who)}">${esc(earshot.data.nameFor(who))}</span> ${esc(r.text)}
        <div class="q-ans">${ans ? `${esc(earshot.data.nameFor(earshot.data.speakerOf(ans)))}: ${esc(ans.text)}` : 'no answer yet'}</div></div>`);
    });
    body.innerHTML = items.length ? `<div class="q-list">${items.slice(-30).reverse().join('')}</div>` : '<div style="color:var(--text-muted)">No questions yet.</div>';
  }

  // 3. Keep it current.
  earshot.on('line', () => setTimeout(draw, 50));
  earshot.on('call', draw);
}
