"""Call Transcriber for Windows: starts the desktop app (app/desktop, Electron).

Kept so that shortcuts and updates from older versions (which ran this file
with pythonw) still open the app. Installs the app's Electron runtime with npm
the first time if it's missing.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "app" / "desktop"
ELECTRON = DESKTOP / "node_modules" / "electron" / "dist" / "electron.exe"
NO_WINDOW = 0x08000000


def main():
    if not ELECTRON.exists():
        npm = shutil.which("npm.cmd") or shutil.which("npm")
        if not npm:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, "Call Transcriber needs Node.js to finish updating.\n\n"
                                             "Install it from https://nodejs.org, then run install.cmd again.",
                                             "Call Transcriber", 0x10)
            return
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=DESKTOP, creationflags=NO_WINDOW)
        subprocess.run(["node", "node_modules/electron/install.js"], cwd=DESKTOP, creationflags=NO_WINDOW)
    subprocess.Popen([str(ELECTRON), str(DESKTOP)], cwd=ROOT, creationflags=0x00000008 | 0x00000200)  # detached
    if os.environ.get("CT_FIX_SHORTCUTS", "1") == "1":
        fix_shortcuts()


def fix_shortcuts():
    """Point the Start-menu/desktop shortcuts straight at the app (no Python in between)."""
    ps = ("$s = New-Object -ComObject WScript.Shell; "
          "foreach ($d in @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('Desktop'))) { "
          "$p = Join-Path $d 'Call Transcriber.lnk'; if (Test-Path $p) { $l = $s.CreateShortcut($p); "
          f"$l.TargetPath = '{ELECTRON}'; $l.Arguments = '\"{DESKTOP}\"'; $l.WorkingDirectory = '{ROOT}'; "
          f"$l.IconLocation = '{ROOT / 'app' / 'windows' / 'icon.ico'}'; $l.Save() }} }}")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], creationflags=NO_WINDOW)


if __name__ == "__main__":
    main()
