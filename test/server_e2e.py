"""End-to-end server test (same on macOS, Windows, Linux).

Starts server.py on a spare port with a temp transcripts folder, feeds three
overlapping speakers exactly as the Chrome extension does (per-person audio,
names, voice channel), and checks the transcript.

  python test/server_e2e.py [--engine onnx|mlx|torch|auto]
"""
import argparse, asyncio, base64, json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import numpy as np, soundfile as sf, websockets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PORT = 8799
SPEAKERS = [  # id, name, clip, start (s), words that must be heard
    ("111111111111111111", "Alice", "a.wav", 0.5, ["model", "fast", "documents"]),
    ("222222222222222222", "Bob", "b.wav", 1.5, ["pricing", "week", "switch"]),
    ("333333333333333333", "Carol", "c.wav", 2.0, ["best", "demo"]),
]
CHANNEL = {"id": "700000000000000001", "name": "Test Channel", "guild": "900000000000000001", "guildName": "Test Server"}


def get(path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5))


async def feed():
    tracks = []
    for uid, name, clip, at, _ in SPEAKERS:
        a, sr = sf.read(HERE / "site" / clip, dtype="float32")
        assert sr == 16000, clip
        x = np.concatenate([np.zeros(int(16000 * at), np.float32), a, np.zeros(16000 * 3, np.float32)])
        tracks.append((uid, name, (np.clip(x, -1, 1) * 32767).astype("<i2")))
    n = max(len(t[2]) for t in tracks)
    async with websockets.connect(f"ws://127.0.0.1:{PORT}", max_size=2**22) as ws:
        t0 = time.time() * 1000
        for i in range(0, n, 3200):                      # 200 ms chunks, speech only
            for uid, name, pcm in tracks:
                c = pcm[i:i + 3200]
                if len(c) and np.abs(c).max() > 60:
                    await ws.send(json.dumps({"type": "track-audio", "user": uid, "name": name, "ch": CHANNEL,
                                              "t": t0 + i / 16, "pcm": base64.b64encode(c.tobytes()).decode()}))
            await asyncio.sleep(0.05)                    # faster than real time is fine
        await asyncio.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="onnx")
    ap.add_argument("--switch-to", default=None, help="model id to switch to (live) before feeding audio")
    args = ap.parse_args()
    out = Path(tempfile.mkdtemp()) / "transcripts"
    env = dict(os.environ, PYTHONUTF8="1")
    srv = subprocess.Popen([sys.executable, "-u", str(ROOT / "server" / "server.py"), "--port", str(PORT),
                            "--engine", args.engine, "--out", str(out)], env=env)
    try:
        for _ in range(900):                             # first run downloads the model
            try:
                if get("/api/state")["modelReady"]:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            sys.exit("server never became ready")
        if args.switch_to:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/models/select?id={args.switch_to}", headers={"X-CT": "1"})
            print("switch:", json.load(urllib.request.urlopen(req, timeout=10)))
            for _ in range(900):
                st = get("/api/state")
                if st["modelReady"] and st["modelId"] == args.switch_to:
                    break
                time.sleep(1)
            else:
                sys.exit("model switch never finished")
            print("now running:", st["modelId"], st["engine"], st["model"])
        asyncio.run(feed())
        deadline = time.time() + 120
        lines = []
        while time.time() < deadline and len(lines) < len(SPEAKERS):
            lines = get(f"/api/day?d={time.strftime('%Y-%m-%d')}")
            time.sleep(1)
        failures = []
        for uid, name, _, _, words in SPEAKERS:
            mine = [l for l in lines if l["speaker"] == name]
            text = " ".join(l["text"] for l in mine).lower()
            missing = [w for w in words if not re.search(rf"\b{w}", text)]
            print(f"{name:6s} -> {text!r}")
            if not mine:
                failures.append(f"{name}: no lines")
            elif missing:
                failures.append(f"{name}: missing {missing}")
            if any(l.get("chName") != CHANNEL["name"] or l.get("guildName") != CHANNEL["guildName"] for l in mine):
                failures.append(f"{name}: channel not tagged")
        calls = get("/api/calls")
        if not calls or calls[0].get("chName") != CHANNEL["name"]:
            failures.append("calls list missing the test channel")
        print("engine:", get("/api/state")["engine"])
        if failures:
            sys.exit("FAILED: " + "; ".join(failures))
        print("PASS")
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
