# Earshot plugins

A plugin is a folder in `plugins/` with a `manifest.json` and a JavaScript
module. It runs inside the dashboard and can add panels, mark up transcript
lines, add header buttons, restyle anything and react to new lines as they
arrive. Plugins are off until you turn them on in **Settings › Plugins**, and
changes apply when the dashboard reloads (<kbd>⌘</kbd><kbd>R</kbd>).

[`plugins/example-questions`](plugins/example-questions) is a complete
example: a Questions panel and a mark on every question in the transcript.
Copy it to start your own.

> Plugins run with the same access as the dashboard: they can read every
> transcript and use the app's controls. Only turn on plugins you trust.

## The folder

```
plugins/
  my-plugin/
    manifest.json
    main.js
    style.css        (optional)
```

```json
{
  "name": "My plugin",
  "version": "1.0.0",
  "description": "One line shown in Settings.",
  "main": "main.js",
  "style": "style.css"
}
```

The folder name is the plugin's id (letters, numbers, `.`, `_` and `-`).
Files are served from `/plugins/<id>/`, so a plugin can load its own images
and modules with relative paths. Your own plugins stay out of git; only the
`example-*` folders ship with Earshot.

## main.js

`main.js` is an ES module. Its default export is called once with the API:

```js
export default function (earshot) {
  earshot.panels.add({
    id: 'hello',
    title: 'Hello',
    render: (body) => { body.textContent = 'Lines so far: ' + earshot.data.lines().length; },
  });
  earshot.on('line', (line) => earshot.ui.toast(`${line.speaker}: ${line.text}`));
}
```

## The API

| Call | What it does |
|:--|:--|
| `earshot.panels.add({ id, title, render(body, earshot, card), show, grow })` | Adds a panel. It appears in the last column the first time (`show: false` keeps it in **Customize › Add a panel**). `grow` 1–3 makes it share the column's height. Returns the card element. |
| `earshot.panels.remove(id)` | Removes a panel this plugin added. |
| `earshot.panels.list()`, `earshot.panels.element(id)` | Every panel id, built-in ones included, and a panel's element, for changing built-in panels. |
| `earshot.transcript.decorate(fn)` | `fn(line)` returns HTML added after that line's text, or nothing. Escape anything you take from the transcript with `earshot.escape`. |
| `earshot.on(event, fn)` | `line` (a new line, live), `state` (connection, accounts, model; every 2 s), `call` (another call opened), `layout` (panels moved). Returns a function that stops listening. |
| `earshot.data.lines()`, `.allLines()`, `.calls()`, `.call()` | The open call's lines, the whole day's lines, every call, and the open call. |
| `earshot.data.state()`, `.stats()` | The live state and the open call's stats (talk time and words per person). |
| `earshot.data.speakerOf(line)`, `.nameFor(key)`, `.colorFor(key)` | A line's speaker key, that person's display name and their colour. |
| `earshot.ui.addHeaderButton({ id, title, html, onClick })` | Adds a button to the dashboard header. |
| `earshot.ui.addStyle(css)` | Adds CSS. The dashboard's colours are CSS variables such as `--accent`, `--text-muted` and `--grid`. |
| `earshot.ui.toast(text)`, `earshot.ui.modal(title, html)` | A short message, or a dialog; `modal` returns its body element. |
| `earshot.ui.rerender()` | Redraws every built-in panel. |
| `earshot.get(path)` | Reads the local server's JSON, for example `/api/calls` or `/api/day?d=2026-11-20`. |
| `earshot.escape(text)` | HTML-escapes text. |

A line looks like `{ t, end, user, speaker, text, ch, chName, guild, guildName }`,
with `t` and `end` in milliseconds since 1970.

## Changing the layout without a plugin

**Customize** (the layout button in the header) lets anyone drag panels
between columns, size them, hide them, change the number of columns and
bring hidden panels back, including panels from plugins. Normal and compact
mode each keep their own layout, and **Reset** restores the default.
