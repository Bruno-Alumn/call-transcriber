"""Call Transcriber for Windows.

Owns the local transcription server (starts it on launch, stops it on quit),
shows the live dashboard in a native window (WebView2 via pywebview) and keeps
running in the system tray when the window is closed. If this launcher dies,
the server notices (--parent-pid) and shuts itself down.

Run with the server's venv:  server\\.venv\\Scripts\\pythonw.exe app\\windows\\call_transcriber.pyw
(The installer creates Start-menu and desktop shortcuts that do exactly this.)
"""
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil
import webview

ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT / "server"
PORT = int(os.environ.get("CT_PORT", "8765"))
BASE = f"http://127.0.0.1:{PORT}/"
IS_WIN = sys.platform == "win32"
ICON = Path(__file__).with_name("icon.ico")
ICON_PNG = Path(__file__).with_name("icon.png")
LOG = ROOT / "server.log"


def server_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":                 # console-less launcher -> python.exe for the server
        exe = exe.with_name("python.exe")
    return str(exe)


def http(path: str, timeout=1.0, app_header=False):
    req = urllib.request.Request(BASE + path.lstrip("/"), headers={"X-CT": "1"} if app_header else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def server_up() -> bool:
    try:
        status, body = http("api/state")
        return status == 200 and body.startswith(b"{")
    except Exception:
        return False


SPLASH = """<html><head><meta name="color-scheme" content="light dark"><style>
html,body{{margin:0;height:100%;overflow:hidden}}
body{{display:grid;place-items:center;font:15px "Segoe UI",system-ui,sans-serif;background:#f4f4f2;color:#52514e}}
@media (prefers-color-scheme:dark){{body{{background:#111110;color:#c3c2b7}}}}
.s{{width:28px;height:28px;border-radius:50%;border:3px solid #3987e544;border-top-color:#3987e5;animation:r 1s linear infinite;margin:0 auto 16px}}
@keyframes r{{to{{transform:rotate(360deg)}}}} .e{{color:#e66767}}
button{{margin-top:16px;font:inherit;font-size:14px;padding:8px 18px;border-radius:10px;border:0;background:#3987e5;color:#fff;cursor:pointer}}
</style></head><body><div style="text-align:center">{spin}<div class="{cls}">{text}</div>{button}</div></body></html>"""


class App:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.stopped_by_user = False
        self.window = None
        self.tray = None
        self.quitting = False
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- views
    def splash(self, text, error=False, offer_start=False):
        button = "<button onclick=\"pywebview.api.server('start')\">Start server</button>" if offer_start else ""
        spin = "" if (error or offer_start) else '<div class="s"></div>'
        html = SPLASH.format(spin=spin, cls="e" if error else "", text=text, button=button)
        if self.window:
            self.window.load_html(html)

    def show_dashboard(self):
        if self.window:
            self.window.load_url(BASE)

    # ---------------------------------------------------------------- server
    def strays(self):
        """Our own server(s) already on the port: started from a terminal, or left over."""
        out = []
        try:
            conns = psutil.net_connections(kind="tcp")
        except psutil.Error:                                # (macOS needs admin for this; Windows doesn't)
            return out
        for c in conns:
            if c.laddr and c.laddr.port == PORT and c.status == psutil.CONN_LISTEN and c.pid:
                if self.proc and c.pid == self.proc.pid:
                    continue
                try:
                    if any("server.py" in a for a in psutil.Process(c.pid).cmdline()):
                        out.append(c.pid)
                except psutil.Error:
                    pass
        return out

    def stop_pids(self, pids):
        try:
            http("api/shutdown", app_header=True)         # clean stop: flushes pending lines
        except Exception:
            pass
        procs = []
        for pid in pids:
            try:
                p = psutil.Process(pid)
                procs += [p] + p.children(recursive=True)   # venv stub + real interpreter
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.Error:
                pass

    def start(self):
        with self.lock:
            self.stopped_by_user = False
            if self.proc and self.proc.poll() is None:
                self.show_dashboard(); return
            self.splash("Starting the transcription server…")
            strays = self.strays()
            if strays:
                self.stop_pids(strays)
            py = server_python()
            if not Path(py).exists() or not (SERVER_DIR / "server.py").exists():
                self.splash("Couldn't find the server. Run install.cmd again.", error=True); return
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
            log = open(LOG, "w", encoding="utf-8")
            self.proc = subprocess.Popen(
                [py, "-u", "server.py", "--port", str(PORT), "--parent-pid", str(os.getpid())]
                + (["--engine", os.environ["CT_ENGINE"]] if os.environ.get("CT_ENGINE") else []),
                cwd=SERVER_DIR, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags)
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
        for _ in range(480):                               # model loading can take a while the first time
            if self.proc is None or self.proc.poll() is not None:
                return
            if server_up():
                self.show_dashboard(); return
            time.sleep(0.5)
        self.splash("The server is taking too long to start. See server.log.", error=True)

    def _watch(self, proc):
        proc.wait()
        if self.quitting or proc is not self.proc:
            return
        self.proc = None
        if self.stopped_by_user:
            self.splash("The server is stopped. Nothing is being transcribed.", offer_start=True)
        else:
            self.splash(f"The server stopped unexpectedly (exit {proc.returncode}). See server.log.", error=True, offer_start=True)

    def stop(self):
        with self.lock:
            p = self.proc
            if not p or p.poll() is not None:
                self.splash("The server is stopped. Nothing is being transcribed.", offer_start=True); return
            self.stopped_by_user = True
            self.splash("Stopping the server…")
        self.stop_pids([p.pid])

    def restart(self):
        p = self.proc
        if p and p.poll() is None:
            self.stopped_by_user = True
            self.splash("Restarting the server…")
            self.stop_pids([p.pid])
            self.proc = None
        self.start()

    def relaunch_after_update(self, deps: bool):
        self.splash("Installing the update…")
        self.stopped_by_user = True
        p = self.proc
        if p and p.poll() is None:
            self.stop_pids([p.pid])
        if deps:
            import shutil
            uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv.exe")
            subprocess.run([uv, "sync"], cwd=SERVER_DIR, creationflags=subprocess.CREATE_NO_WINDOW if IS_WIN else 0,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # free the single-instance lock, start the new launcher, then quit this one
        global _instance_lock
        if _instance_lock:
            _instance_lock.close(); _instance_lock = None
        exe = Path(sys.executable)
        exe = exe.with_name("pythonw.exe") if IS_WIN else exe
        subprocess.Popen([str(exe), str(Path(__file__).resolve())], cwd=ROOT,
                         creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if IS_WIN else 0)
        self.quit()

    # ---------------------------------------------------------------- window / tray
    def show(self, *_):
        if self.window:
            self.window.show()
            self.window.restore()

    def quit(self, *_):
        self.quitting = True
        p = self.proc
        if p and p.poll() is None:
            self.stop_pids([p.pid])
        if self.tray:
            self.tray.stop()
        if self.window:
            self.window.destroy()

    def on_closing(self):
        if self.quitting or not self.tray:
            self.quit(); return True
        self.window.hide()                                   # keep transcribing in the tray
        return False

    def make_tray(self):
        try:
            import pystray
            from PIL import Image
        except ImportError:
            return None
        def openfolder(*_):
            os.startfile(ROOT / "transcripts") if IS_WIN else subprocess.run(["open", ROOT / "transcripts"])
        menu = pystray.Menu(
            pystray.MenuItem("Open Call Transcriber", self.show, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Restart server", lambda *_: threading.Thread(target=self.restart, daemon=True).start()),
            pystray.MenuItem("Stop server", lambda *_: threading.Thread(target=self.stop, daemon=True).start()),
            pystray.MenuItem("Open transcripts folder", openfolder),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit (stops transcribing)", self.quit),
        )
        image = Image.open(ICON_PNG if ICON_PNG.exists() else ICON)
        return pystray.Icon("CallTranscriber", image, "Call Transcriber", menu)


class Api:
    """Exposed to the dashboard as window.pywebview.api"""
    def __init__(self, app: App):
        self._app = app

    def server(self, action):
        fn = {"start": self._app.start, "stop": self._app.stop, "restart": self._app.restart}.get(action)
        if fn:
            threading.Thread(target=fn, daemon=True).start()
        return True

    def save(self, name, content):
        """Save dialog for transcript exports."""
        downloads = Path.home() / "Downloads"
        res = self._app.window.create_file_dialog(webview.SAVE_DIALOG, directory=str(downloads if downloads.exists() else Path.home()),
                                                  save_filename=name)
        path = res[0] if isinstance(res, (list, tuple)) and res else res
        if not path:
            return False
        Path(path).write_text(content, encoding="utf-8", newline="")
        return True

    def relaunch(self, rebuild=False, deps=False):
        """After the server pulled a new release: stop it, refresh dependencies, reopen."""
        threading.Thread(target=self._app.relaunch_after_update, args=(bool(deps),), daemon=True).start()
        return True


_instance_lock = None


def already_running() -> bool:
    """Single instance via a reserved localhost port (only one process can hold it).
    (A process scan doesn't work on Windows: a venv's python.exe is a stub that
    starts the real interpreter as a second process with the same command line.)"""
    import socket
    global _instance_lock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", PORT + 1000))      # 9765 for the default port
        s.listen(1)
    except OSError:
        s.close()
        return True
    _instance_lock = s                           # held until the app exits
    return False


def main():
    if already_running():
        # Another copy owns the server; just show the dashboard in the browser.
        import webbrowser
        webbrowser.open(BASE)
        return
    app = App()
    app.window = webview.create_window(
        "Call Transcriber", html=SPLASH.format(spin='<div class="s"></div>', cls="", text="Starting…", button=""),
        js_api=Api(app), width=1320, height=860, min_size=(980, 620), background_color="#111110")
    app.window.events.closing += app.on_closing
    if IS_WIN:
        app.tray = app.make_tray()
        if app.tray:
            app.tray.run_detached()
    # func runs in its own thread once the window exists
    webview.start(func=app.start, private_mode=False, storage_path=str(ROOT / ".webview"))
    app.quit()


if __name__ == "__main__":
    main()
