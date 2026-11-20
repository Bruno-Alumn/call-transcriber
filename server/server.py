"""Local transcription server for the Discord Transcriber extension.

The extension sends each call participant's audio as its own stream
(16 kHz mono int16, tagged with their Discord user id and name). Per person,
we cut the stream into utterances on silence, transcribe each utterance
locally with an MLX speech model, and append "[time] Name: text" to
transcripts/YYYY-MM-DD.md.

Run:  uv run server.py            (see --help for model choices)
"""

import argparse
import sys
import asyncio
import base64
import datetime as dt
import gc
import json
import os
import queue
import signal
import subprocess
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

SR = 16000
ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
DASHBOARD = Path(__file__).resolve().parent / "dashboard"

HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "you", "bye.", "bye", ".", "thank you very much.", "subtitles by the amara.org community",
}


FILLER = re.compile(r"(?i)(^|[\s,])(?:um+|uh+|uhm|erm|hmm+|mm+)(?=[\s,.!?]|$)[,]?")


def tidy(text: str) -> str:
    """Readability pass: drop um/uh fillers, capitalise 'i' and sentence starts."""
    t = FILLER.sub(r"\1", text)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,")
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), t)
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t


# ---------------------------------------------------------------- segmenter
class Segmenter:
    """Energy-gate VAD for one person's stream: 256 ms frames -> utterances."""

    def __init__(self, out: queue.Queue, threshold: float, label: str, uid: str = "",
                 silence_ms=700, max_ms=25000, preroll_ms=300):
        self.out, self.threshold, self.label, self.uid = out, threshold, label, uid
        self.channel: dict | None = None
        self.silence_ms, self.max_ms, self.preroll_ms = silence_ms, max_ms, preroll_ms
        self.pre: deque[tuple[float, np.ndarray]] = deque()
        self.cur: list[tuple[float, np.ndarray]] = []
        self.silent_ms = 0.0
        self.noise = threshold
        self.last_seen = time.time()
        self.last_end = 0.0          # audio-clock end of the previous chunk (ms)

    def push(self, t: float, pcm: np.ndarray):
        self.last_seen = time.time()
        dur = len(pcm) / SR * 1000
        # The browser only sends speech, so a jump in the audio timeline is
        # silence that was skipped: count it, and end the utterance if it's long.
        gap = t - self.last_end if self.last_end else 0
        self.last_end = t + dur
        if self.cur and gap > 300:
            self.silent_ms += gap
            if self.silent_ms >= self.silence_ms:
                self.flush()
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        speech = rms > max(self.threshold, self.noise * 3)
        if not speech:
            self.noise = 0.98 * self.noise + 0.02 * rms
        if self.cur:
            self.cur.append((t, pcm))
            self.silent_ms = 0 if speech else self.silent_ms + dur
            if self.silent_ms >= self.silence_ms or self.cur[-1][0] + dur - self.cur[0][0] >= self.max_ms:
                self.flush()
        elif speech:
            self.cur = list(self.pre) + [(t, pcm)]
            self.pre.clear()
            self.silent_ms = 0
        else:
            self.pre.append((t, pcm))
            while self.pre and t - self.pre[0][0] > self.preroll_ms:
                self.pre.popleft()

    def flush(self):
        if not self.cur:
            return
        t0 = self.cur[0][0]
        audio = np.concatenate([p for _, p in self.cur])
        self.cur, self.silent_ms = [], 0
        if len(audio) >= SR * 0.4:
            self.out.put((t0, audio, self.label, self.uid, self.channel))


# ---------------------------------------------------------------- output
class Writer:
    """Appends each line to the readable .md transcript and to a .jsonl data
    file (used by the dashboard's analytics), then pushes it to the dashboard."""

    def __init__(self, hub: "Hub"):
        self.hub = hub

    def _paths(self, when):
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        md = TRANSCRIPTS / f"{when:%Y-%m-%d}.md"
        if not md.exists():
            md.write_text(f"# Discord voice transcript, {when:%A %B} {when.day}, {when.year}\n\n", encoding="utf-8")
        return md, TRANSCRIPTS / f"{when:%Y-%m-%d}.jsonl"

    def write(self, t_ms: float, dur_s: float, uid: str, speaker: str, text: str, ch: dict | None = None):
        when = dt.datetime.fromtimestamp(t_ms / 1000)
        md, jl = self._paths(when)
        with md.open("a", encoding="utf-8") as f:
            f.write(f"**[{when:%H:%M:%S}] {speaker}:** {text}\n\n")
        rec = {"t": int(t_ms), "end": int(t_ms + dur_s * 1000), "user": uid, "speaker": speaker, "text": text}
        if ch and ch.get("id"):
            rec.update({"ch": ch["id"], "chName": ch.get("name"), "guild": ch.get("guild"), "guildName": ch.get("guildName")})
        with jl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{when:%H:%M:%S}] {speaker}: {text}", flush=True)
        self.hub.publish({"type": "line", **rec})


MD_LINE = re.compile(r"^\*\*\[(\d\d):(\d\d):(\d\d)\] (.+?):\*\* (.*)$")


def load_day(day: str) -> list[dict]:
    """All lines for a day. Older lines that predate the .jsonl data file are
    recovered from the .md transcript (duration estimated from word count)."""
    recs: list[dict] = []
    jl, md = TRANSCRIPTS / f"{day}.jsonl", TRANSCRIPTS / f"{day}.md"
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    first = min((r["t"] for r in recs), default=float("inf"))
    if md.exists():
        base = dt.datetime.strptime(day, "%Y-%m-%d")
        for line in md.read_text(encoding="utf-8").splitlines():
            m = MD_LINE.match(line)
            if not m:
                continue
            h, mi, se, who, text = m.groups()
            t = int((base.replace(hour=int(h), minute=int(mi), second=int(se))).timestamp() * 1000)
            if t >= first - 1000:
                continue
            words = max(1, len(text.split()))
            recs.append({"t": t, "end": t + int(words / 2.6 * 1000), "user": "", "speaker": who, "text": text})
    recs.sort(key=lambda r: r["t"])
    return recs


GAP_MS = 15 * 60 * 1000
_day_cache: dict[str, tuple[float, list, list]] = {}


def day_data(day: str) -> tuple[list[dict], list[dict]]:
    """(lines, calls) for a day, cached until its files change."""
    files = [TRANSCRIPTS / f"{day}.jsonl", TRANSCRIPTS / f"{day}.md"]
    stamp = sum(f.stat().st_mtime for f in files if f.exists())
    hit = _day_cache.get(day)
    if hit and hit[0] == stamp:
        return hit[1], hit[2]
    recs = load_day(day)
    calls = build_calls(day, recs)
    _day_cache[day] = (stamp, recs, calls)
    return recs, calls


def build_calls(day: str, recs: list[dict]) -> list[dict]:
    """Split a day into calls: a new call starts when the voice channel
    changes or after 15 quiet minutes."""
    calls: list[dict] = []
    for r in recs:
        cur = calls[-1] if calls else None
        ch = r.get("ch")
        if cur and r["t"] - cur["end"] < GAP_MS and (not ch or not cur["ch"] or ch == cur["ch"]):
            cur["lines"].append(r); cur["end"] = max(cur["end"], r["end"])
            if ch and not cur["ch"]:
                cur.update(ch=ch, chName=r.get("chName"), guild=r.get("guild"), guildName=r.get("guildName"))
        else:
            calls.append({"day": day, "start": r["t"], "end": r["end"], "ch": ch, "chName": r.get("chName"),
                          "guild": r.get("guild"), "guildName": r.get("guildName"), "lines": [r]})
    out = []
    for c in calls:
        talk: dict[str, float] = {}
        words = 0
        for r in c["lines"]:
            talk[r["speaker"]] = talk.get(r["speaker"], 0) + (r["end"] - r["t"]) / 1000
            words += len(r["text"].split())
        people = sorted(talk.items(), key=lambda kv: -kv[1])
        out.append({"id": f"{day}@{c['start']}", "day": day, "start": c["start"], "end": c["end"],
                    "ch": c["ch"], "chName": c["chName"], "guild": c["guild"], "guildName": c["guildName"],
                    "people": [p for p, _ in people], "lines": len(c["lines"]), "words": words,
                    "preview": next((r["text"] for r in reversed(c["lines"]) if len(r["text"]) > 20), c["lines"][-1]["text"])[:120]})
    return out


def srv_key(c: dict) -> str:
    if c.get("guild"):
        return "g:" + c["guild"]
    if c.get("guildName"):
        return "gn:" + c["guildName"]
    return "dm" if c.get("chName") == "Private call" else "unknown"


def ch_key(c: dict) -> str:
    return srv_key(c) + "|" + (c.get("ch") or c.get("chName") or "?")


def group_match(c: dict, kind: str, key: str) -> bool:
    return (kind == "call" and c["id"] == key) or (kind == "channel" and ch_key(c) == key) or (kind == "server" and srv_key(c) == key)


def archive_path() -> Path:
    return TRANSCRIPTS.parent / "archive.json"


def archived_keys() -> set[str]:
    try:
        return set(json.loads(archive_path().read_text(encoding="utf-8")).get("keys", []))
    except Exception:
        return set()


def set_archived(key: str, on: bool):
    keys = archived_keys()
    (keys.add if on else keys.discard)(key)
    archive_path().write_text(json.dumps({"keys": sorted(keys)}, indent=1), encoding="utf-8")


def delete_group(kind: str, key: str) -> dict:
    """Remove every line of a call / channel / server from the transcripts.
    Removed lines go to transcripts/.trash/ (recoverable), not into the void."""
    trash = TRANSCRIPTS / ".trash"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    n_lines = n_calls = 0
    for d in all_days():
        _, calls = day_data(d)
        targets = [c for c in calls if group_match(c, kind, key)]
        if not targets:
            continue
        n_calls += len(targets)
        ranges = [(c["start"], c["end"], c["ch"]) for c in targets]
        def hit(t, ch, slack=0):
            return any(s0 - slack <= t <= e0 + slack and (not ch or not cc or ch == cc) for s0, e0, cc in ranges)
        trash.mkdir(parents=True, exist_ok=True)
        jl, md = TRANSCRIPTS / f"{d}.jsonl", TRANSCRIPTS / f"{d}.md"
        if jl.exists():
            keep, gone = [], []
            for line in jl.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    keep.append(line); continue
                (gone if hit(r["t"], r.get("ch")) else keep).append(line)
            if gone:
                with (trash / f"{d}-deleted-{stamp}.jsonl").open("a", encoding="utf-8") as f:
                    f.write("\n".join(gone) + "\n")
                tmp = jl.with_suffix(".jsonl.tmp"); tmp.write_text("".join(l + "\n" for l in keep), encoding="utf-8"); tmp.replace(jl)
                n_lines += len(gone)
        if md.exists():
            base = dt.datetime.strptime(d, "%Y-%m-%d")
            out, gone, skip_blank = [], [], False
            for line in md.read_text(encoding="utf-8").splitlines():
                m = MD_LINE.match(line)
                if m:
                    h, mi, se = (int(x) for x in m.groups()[:3])
                    t = int(base.replace(hour=h, minute=mi, second=se).timestamp() * 1000)
                    if hit(t, None, slack=1000):         # .md has 1-second timestamps
                        gone.append(line); skip_blank = True; continue
                if skip_blank and line == "":
                    skip_blank = False; continue
                skip_blank = False
                out.append(line)
            if gone:
                with (trash / f"{d}-deleted-{stamp}.md").open("a", encoding="utf-8") as f:
                    f.write("\n\n".join(gone) + "\n")
                tmp = md.with_suffix(".md.tmp"); tmp.write_text("\n".join(out) + "\n", encoding="utf-8"); tmp.replace(md)
                if not jl.exists():
                    n_lines += len(gone)
        _day_cache.pop(d, None)
    set_archived(f"{kind}:{key}", False)
    return {"calls": n_calls, "lines": n_lines}


def all_days() -> list[str]:
    return sorted({f.stem for f in TRANSCRIPTS.glob("*.md")} | {f.stem for f in TRANSCRIPTS.glob("*.jsonl")}, reverse=True)


# ---------------------------------------------------------------- updates
# Updates come from GitHub release tags (vX.Y.Z) via git, which is already
# signed in on each machine (the repo is private: no tokens in the app).
def git(*args, timeout=60):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3]) or (0,)


class Updater:
    def __init__(self, hub: "Hub"):
        self.hub = hub
        self.lock = threading.Lock()
        self.state = {"current": self.current(), "latest": None, "available": False, "notes": "",
                      "checked": None, "error": None, "busy": False, "applied": None}

    @staticmethod
    def current() -> str:
        try:
            return (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        except OSError:
            return "0.0.0"

    def check(self) -> dict:
        st = self.state
        st["current"] = self.current()
        if not (ROOT / ".git").exists():
            st.update(error="This copy isn't a git checkout, so it can't update itself. Install with git clone to get updates.")
            return st
        rc, out = git("fetch", "--tags", "--force", "--quiet", "origin", timeout=90)
        if rc != 0:
            st.update(error="Couldn't reach GitHub: " + out[-200:], checked=time.time())
            return st
        rc, out = git("tag", "-l", "v*", "--sort=-v:refname")
        latest = next((t for t in out.splitlines() if re.fullmatch(r"v\d+\.\d+\.\d+", t.strip())), None)
        notes = git("tag", "-l", "--format=%(contents)", latest)[1] if latest else ""
        st.update(latest=latest, notes=notes.strip()[:4000], checked=time.time(), error=None,
                  available=bool(latest) and vtuple(latest) > vtuple(st["current"]))
        return st

    def apply(self) -> dict:
        with self.lock:
            st = self.check()
            if st.get("error"):
                return {"ok": False, "error": st["error"]}
            if not st["available"]:
                return {"ok": False, "error": "Already up to date."}
            st["busy"] = True
            try:
                tag = st["latest"]
                rc, dirty = git("status", "--porcelain", "--untracked-files=no")
                if dirty:
                    return {"ok": False, "error": "This copy has local code changes; not touching them. (git status shows them.)"}
                rc, branch = git("rev-parse", "--abbrev-ref", "HEAD")
                if branch.strip() != "main":
                    return {"ok": False, "error": f"On branch {branch.strip()!r}, expected main."}
                rc, changed = git("diff", "--name-only", "HEAD", tag)
                files = [f for f in changed.splitlines() if f.strip()]
                rc, out = git("merge", "--ff-only", "--quiet", tag)
                if rc != 0:
                    return {"ok": False, "error": "Couldn't fast-forward to " + tag + ": " + out[-200:]}
                result = {"ok": True, "from": st["current"], "version": self.current(), "files": len(files),
                          "rebuildMac": any(f.startswith("app/mac/") or f.startswith("app/desktop/package") for f in files),
                          "app": any(f.startswith("app/desktop/package") for f in files),   # new Electron / npm packages
                          "deps": any(f in ("server/pyproject.toml", "server/uv.lock") for f in files),
                          "extension": any(f.startswith("extension/") for f in files),
                          "server": any(f.startswith("server/") for f in files)}
                st.update(applied=result, available=False, current=self.current())
                if result["extension"]:
                    self.hub.tell_extensions({"type": "reload"})     # it reloads itself from disk
                return result
            finally:
                st["busy"] = False

    def loop(self):
        time.sleep(20)
        while True:
            try:
                self.check()
            except Exception as e:
                self.state["error"] = str(e)[:200]
            time.sleep(6 * 3600)


class Hub:
    """Live state for the dashboard: who's connected, who's talking, new lines."""

    def __init__(self, model: str):
        self.model, self.model_ready, self.started = model, False, time.time()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set = set()
        self.extensions = 0
        self.ext_sockets: set = set()
        self.ext_reports: dict = {}       # connection -> (status, time): Chrome and the app can both be connected
        self.res: dict = {}
        self.audio_s = self.proc_s = 0.0             # transcription throughput
        self.latency: deque = deque(maxlen=50)        # s from end of speech to text
        self.people: dict[str, "Segmenter"] = {}
        self.queue: queue.Queue | None = None
        self.names_file = TRANSCRIPTS.parent / "names.json"
        try:
            self.names: dict[str, str] = json.loads(self.names_file.read_text(encoding="utf-8"))
        except Exception:
            self.names = {}

    def learn_name(self, uid: str, name: str):
        """Remember the real name behind a Discord id (relabels old user-1234 lines)."""
        if not name or re.fullmatch(r"user-\d{4}", name) or self.names.get(uid) == name:
            return
        self.names[uid] = name
        try:
            self.names_file.write_text(json.dumps(self.names, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass

    def state(self) -> dict:
        now = time.time()
        ppl = [{"user": uid, "name": p.label, "talking": bool(p.cur) and now - p.last_seen < 1.5,
                "lastSeen": int(p.last_seen * 1000)} for uid, p in self.people.items() if now - p.last_seen < 120]
        # Report from the connection that matters: in a call first, then the app, then the latest.
        reports = [r for r in self.ext_reports.values() if now - r[1] < 15] or \
                  sorted(self.ext_reports.values(), key=lambda r: r[1])[-1:]
        best = max(reports, key=lambda r: (bool(r[0].get("pcs")) and bool(r[0].get("tracks")),
                                           r[0].get("client") == "app", r[1]), default=({}, 0.0))
        st, at = best
        age = now - at
        ext = {"connected": self.extensions > 0,
               "discordTab": self.extensions > 0 and age < 15,          # a Discord tab reports every 5 s
               "paused": bool(st.get("paused")) if age < 15 else False,
               "inCall": bool(st.get("pcs")) and bool(st.get("tracks")),
               "streams": len(st.get("live") or []),
               "named": sum(1 for n in (st.get("live") or []) if not re.fullmatch(r"\d{15,22}", str(n))),
               "audio": st.get("ctx"), "hook": st.get("v"), "errors": st.get("errors") or [],
               "channel": st.get("channel") if age < 15 else None,
               "client": st.get("client") or "chrome",           # "app": Discord inside the desktop app
               "lastReport": int(at * 1000) if at else None}
        return {"type": "state", "model": self.model, "engine": getattr(self, "engine", None), "modelReady": self.model_ready,
                "modelId": getattr(self, "model_id", None), "modelLoading": getattr(self, "loading", None),
                "modelError": getattr(self, "model_error", None),
                "extensions": self.extensions, "ext": ext, "people": ppl, "queue": self.queue.qsize() if self.queue else 0,
                "uptime": int(now - self.started), "now": int(now * 1000), "pid": os.getpid(),
                "res": {**self.res, "speed": round(self.audio_s / self.proc_s, 1) if self.proc_s else None,
                        "latency": round(sum(self.latency) / len(self.latency), 2) if self.latency else None,
                        "audioMin": round(self.audio_s / 60, 1)}}

    def sample_resources(self):
        """Every 2 s: this server's CPU % (of one core), memory, and model memory."""
        last_cpu, last_t = time.process_time(), time.time()
        while True:
            time.sleep(2)
            cpu, t = time.process_time(), time.time()
            pct = 100 * (cpu - last_cpu) / max(1e-6, t - last_t)
            last_cpu, last_t = cpu, t
            try:
                import psutil
                rss = psutil.Process().memory_info().rss / 2**20
            except Exception:
                rss = None
            mlx = None
            try:
                if "mlx.core" in sys.modules:
                    mlx = sys.modules["mlx.core"].get_active_memory() / 2**20
                elif "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
                    mlx = sys.modules["torch"].cuda.memory_allocated() / 2**20
            except Exception:
                pass
            self.res = {"cpu": round(pct, 1), "rssMB": round(rss) if rss else None, "modelMB": round(mlx) if mlx else None,
                        "cores": os.cpu_count()}

    def tell_extensions(self, msg: dict):   # safe to call from any thread
        data = json.dumps(msg)
        async def send():
            for w in list(self.ext_sockets):
                try:
                    await w.send(data)
                except Exception:
                    pass
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(send()))

    def publish(self, msg: dict):           # safe to call from any thread
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._send(msg)))

    async def _send(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False)
        for c in list(self.clients):
            try:
                await c.send(data)
            except Exception:
                self.clients.discard(c)


# ---------------------------------------------------------------- model
# ---------------------------------------------------------------- engines
# mlx  : Apple Silicon (Metal) via mlx-audio — the Mac default.
# torch: Hugging Face transformers on an NVIDIA GPU (CUDA) or the CPU.
# onnx : ONNX Runtime (onnx-asr) — light, fast on any CPU; the Windows default
#        when there is no NVIDIA GPU.
ENGINE_DEFAULTS = {
    "mlx": "ibm-granite/granite-speech-5.0-470m-turboctc",
    "torch": "ibm-granite/granite-speech-5.0-470m-turboctc",
    "onnx": "nemo-parakeet-tdt-0.6b-v2",
}


# Models the picker offers. "repos" maps engine -> model id for that runtime;
# "bench" holds our measured results on real meeting speech (AMI, 120 clips):
# word error rate % and speed (x real time) on the machines we tested.
MODELS = [
    {"id": "granite-470m", "name": "Granite Speech 5.0 · 470M", "default": True,
     "about": "Best all-rounder: near-top accuracy, very fast, light. English + 6 languages.", "langs": "English + 6 more",
     "repos": {"mlx": "ibm-granite/granite-speech-5.0-470m-turboctc", "torch": "ibm-granite/granite-speech-5.0-470m-turboctc"},
     "bench": {"mlx": {"wer": 5.5, "speed": 249, "mem": 1.6}, "torch": {"wer": 5.5, "speed": 82, "mem": 0.9}}},
    {"id": "granite-2b", "name": "Granite Speech 5.0 · 2B",
     "about": "Most accurate we measured; ~5x the memory and much slower. English + 6 languages.", "langs": "English + 6 more",
     "repos": {"mlx": "ibm-granite/granite-speech-5.0-2b", "torch": "ibm-granite/granite-speech-5.0-2b"},
     "bench": {"mlx": {"wer": 5.4, "speed": 13, "mem": 9.1}, "torch": {"wer": 5.4, "speed": 11}}},
    {"id": "parakeet-v3", "name": "Parakeet TDT 0.6B v3",
     "about": "Very fast, 25 European languages, auto-detects language.", "langs": "25 European",
     "repos": {"mlx": "mlx-community/parakeet-tdt-0.6b-v3", "onnx": "nemo-parakeet-tdt-0.6b-v3"},
     "bench": {"mlx": {"wer": 7.0, "speed": 116, "mem": 3.5}, "onnx": {"wer": 7.4, "speed": 30, "mem": 2.0}}},
    {"id": "qwen3-1.7b", "name": "Qwen3-ASR · 1.7B",
     "about": "52 languages incl. Chinese, Japanese, Korean, Arabic.", "langs": "52 languages",
     "repos": {"mlx": "mlx-community/Qwen3-ASR-1.7B-8bit", "torch": "Qwen/Qwen3-ASR-1.7B-hf"},
     "bench": {"mlx": {"wer": 8.5, "speed": 18, "mem": 5.5}, "torch": {"wer": 8.5, "speed": 10}}},
    {"id": "whisper-turbo", "name": "Whisper large-v3-turbo",
     "about": "OpenAI's model: 99 languages; least accurate on fast chatter.", "langs": "99 languages",
     "repos": {"mlx": "mlx-community/whisper-large-v3-turbo", "torch": "openai/whisper-large-v3-turbo"},
     "bench": {"mlx": {"wer": 13.0, "speed": 15, "mem": 2.5}, "torch": {"wer": 13.0, "speed": 30}}},
]


def engines_here() -> list[str]:
    """Speech runtimes this machine can use, best first."""
    out = []
    import platform
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_audio  # noqa: F401
            out.append("mlx")
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out.append("torch")
    except Exception:
        pass
    try:
        import onnx_asr  # noqa: F401
        out.append("onnx")
    except Exception:
        pass
    return out


def models_here() -> list[dict]:
    have = engines_here()
    out = []
    for m in MODELS:
        eng = next((e for e in have if e in m["repos"]), None)
        out.append({**{k: v for k, v in m.items() if k != "repos"}, "engine": eng, "repo": m["repos"].get(eng) if eng else None,
                    "available": bool(eng), "stats": m.get("bench", {}).get(eng) if eng else None})
    return out


def settings_path() -> Path:
    return TRANSCRIPTS.parent / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(**kw):
    st = load_settings(); st.update(kw)
    settings_path().write_text(json.dumps(st, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- personality read
# A quick, for-fun personality sketch of one person from what they said in a
# call, written by a small local language model (Liquid AI LFM3-1.7B). Loaded
# on first use, unloaded after 10 idle minutes.
LLM_REPOS = {"mlx": "LiquidAI/LFM3-1.7B-MLX-8bit", "torch": "LiquidAI/LFM3-1.7B"}
LLM_NAME = "LFM3 1.7B"
PERSONA_SYSTEM = """You write a quick, light-hearted personality read of one person, based only on things they said in a voice call. It is a playful guess, not an assessment.

Rules:
- Base every point on what they actually said. Quote short bits as evidence.
- Never guess or mention health, mental health, sexuality, religion, politics, ethnicity, age, or other sensitive traits.
- Be fair and kind. No insults, no diagnoses.
- If the lines are few or generic, say the read is shaky.

Answer in exactly this format and nothing else:
**In a sentence:** <one sentence>

**Big Five (rough guess)**
- Openness: <low|medium|high> - <short reason>
- Conscientiousness: <low|medium|high> - <short reason>
- Extraversion: <low|medium|high> - <short reason>
- Agreeableness: <low|medium|high> - <short reason>
- Emotional stability: <low|medium|high> - <short reason>

**Comes across as:** <three short traits, comma-separated>

**Evidence:**
- "<short quote>"
- \"<short quote>\"

Use two or three quotes, then stop."""


def tidy_persona(text: str) -> str:
    """Small models sometimes keep going after the format ends: keep one
    Evidence section with at most three quotes, drop anything after it."""
    out, in_ev, quotes = [], False, 0
    for line in text.strip().splitlines():
        if re.match(r"^\W*evidence", line, re.I):
            if in_ev:
                break
            in_ev = True
            out.append("**Evidence:**")
            continue
        if in_ev:
            if line.strip().startswith(("-", "•", "*")):
                quotes += 1
                if quotes > 3:
                    break
            elif line.strip():
                break
        out.append(line)
    return "\n".join(out).strip()


class PersonalityReader:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = self.proc = None
        self.engine = None
        self.last_used = 0.0
        self.jobs: dict[str, dict] = {}
        threading.Thread(target=self._idle_unload, daemon=True).start()

    @staticmethod
    def engine_here() -> tuple[str | None, str]:
        import platform
        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                import mlx_vlm  # noqa: F401
                return "mlx", ""
            except ImportError:
                return None, "Install the Mac add-on: cd server && uv sync"
        try:
            import torch
            if torch.cuda.is_available():
                return "torch", ""
        except Exception:
            pass
        return None, "Needs an Apple Silicon Mac or an NVIDIA graphics card."

    def info(self) -> dict:
        eng, why = self.engine_here()
        return {"available": bool(eng), "engine": eng, "model": LLM_NAME, "reason": why, "loaded": self.model is not None}

    def _load(self, eng: str):
        if self.model is not None:
            return
        repo = LLM_REPOS[eng]
        print(f"Personality read: loading {repo} …", flush=True)
        if eng == "mlx":
            from mlx_vlm import load
            self.model, self.proc = load(repo)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.proc = AutoTokenizer.from_pretrained(repo)
            self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
        self.engine = eng

    def _generate(self, messages: list[dict]) -> str:
        if self.engine == "mlx":
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(self.proc, self.model.config, messages, num_images=0)
            out = generate(self.model, self.proc, prompt, max_tokens=420, temperature=0.3, repetition_penalty=1.1, verbose=False)
            return (out.text if hasattr(out, "text") else str(out)).strip()
        import torch
        ids = self.proc.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, enable_thinking=False).to("cuda")
        with torch.inference_mode():
            gen = self.model.generate(**ids, max_new_tokens=420, do_sample=True, temperature=0.3, top_p=0.9,
                                      repetition_penalty=1.05)
        return self.proc.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _idle_unload(self):
        while True:
            time.sleep(30)
            with self.lock:
                if self.model is not None and time.time() - self.last_used > 600:
                    self.model = self.proc = None
                    gc.collect()
                    print("Personality read: model unloaded (idle).", flush=True)

    def start(self, day: str, t0: int, t1: int, ch: str, user: str, name: str) -> dict:
        eng, why = self.engine_here()
        if not eng:
            return {"error": why}
        recs, _ = day_data(day)
        mine = [r for r in recs if t0 <= r["t"] <= t1 and (not ch or not r.get("ch") or r.get("ch") == ch)
                and ((user and r.get("user") == user) or (not user and r.get("speaker") == name))]
        said = [r["text"].strip() for r in mine if r.get("text", "").strip()]
        words = sum(len(x.split()) for x in said)
        if len(said) < 4 or words < 30:
            return {"error": f"Not enough to go on yet: {name} has said {words} words in this call. Try again once they've talked a bit more."}
        picked, total = [], 0                      # most recent lines, up to ~2500 words
        for x in reversed(said):
            total += len(x.split())
            if total > 2500:
                break
            picked.append(x)
        picked.reverse()
        jid = os.urandom(6).hex()
        job = {"id": jid, "status": "loading" if self.model is None else "thinking", "name": name,
               "lines": len(picked), "words": sum(len(x.split()) for x in picked), "model": LLM_NAME, "started": time.time()}
        self.jobs[jid] = job
        messages = [{"role": "system", "content": PERSONA_SYSTEM},
                    {"role": "user", "content": f"Name: {name}\nThings {name} said in this call, oldest first:\n"
                                                + "\n".join("- " + x for x in picked)}]

        def run():
            try:
                with self.lock:
                    self._load(eng)
                    job["status"] = "thinking"
                    text = self._generate(messages)
                    self.last_used = time.time()
                job.update(status="done", text=tidy_persona(text), seconds=round(time.time() - job["started"], 1))
            except Exception as e:
                job.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        threading.Thread(target=run, daemon=True).start()
        for old in [k for k, v in self.jobs.items() if time.time() - v["started"] > 3600]:
            self.jobs.pop(old, None)
        return {"id": jid, **{k: job[k] for k in ("status", "lines", "words")}}


def pick_engine(requested: str) -> str:
    if requested != "auto":
        return requested
    import platform
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    try:
        import torch
        if torch.cuda.is_available():
            return "torch"
    except Exception:
        pass
    return "onnx"


def load_engine(engine: str, repo: str, language: str | None):
    """Returns run(audio float32 16 kHz) -> text."""
    if engine == "mlx" and "whisper" in repo.lower():      # Whisper: its own MLX package
        import mlx_whisper
        kw = {"language": language} if language else {}
        return lambda audio: mlx_whisper.transcribe(audio, path_or_hf_repo=repo, condition_on_previous_text=False, **kw)["text"]
    if engine == "mlx":
        import mlx.core as mx
        from mlx_audio.stt import load
        model = load(repo)
        kw = {"language": language} if language and ("whisper" in repo.lower() or "qwen" in repo.lower()) else {}
        def run(audio):
            out = model.generate(mx.array(audio), **kw)
            return out.text if hasattr(out, "text") else out if isinstance(out, str) else ""
        return run
    if engine == "torch":
        import torch
        from transformers import AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        half = torch.float16 if dev == "cuda" else torch.float32
        low = repo.lower()
        proc = AutoProcessor.from_pretrained(repo)
        def to_dev(inp, dtype):
            return {k: (v.to(dev, dtype=dtype) if v.is_floating_point() else v.to(dev)) if hasattr(v, "to") else v for k, v in inp.items()}
        if "turboctc" in low:                                # Granite 470M (CTC)
            from transformers import AutoModelForCTC
            model = AutoModelForCTC.from_pretrained(repo, dtype=half).to(dev).eval()
            def run(audio):
                with torch.inference_mode():
                    inp = to_dev(proc([audio], sampling_rate=SR), model.dtype)
                    return proc.batch_decode(model.generate(**inp), skip_special_tokens=True)[0]
            return run
        if "granite-speech" in low:                          # Granite 2B (speech LLM)
            from transformers import AutoModelForSpeechSeq2Seq
            model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
            tok = proc.tokenizer
            prompt = tok.apply_chat_template([{"role": "user", "content": "<|audio|>transcribe the speech with proper punctuation and capitalization."}],
                                             tokenize=False, add_generation_prompt=True)
            def run(audio):
                with torch.inference_mode():
                    inp = proc(prompt, torch.from_numpy(audio).unsqueeze(0), device=dev, return_tensors="pt").to(dev)
                    out = model.generate(**inp, max_new_tokens=200, do_sample=False, num_beams=1)
                    return tok.batch_decode(out[:, inp["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
            return run
        if "qwen3-asr" in low:                               # Qwen3-ASR (52 languages)
            from transformers import AutoModelForMultimodalLM
            model = AutoModelForMultimodalLM.from_pretrained(repo, dtype=half).to(dev).eval()
            def run(audio):
                with torch.inference_mode():
                    inp = proc.apply_transcription_request(audio=audio, sampling_rate=SR).to(model.device, model.dtype)
                    out = model.generate(**inp, max_new_tokens=256)
                    return proc.decode(out[:, inp["input_ids"].shape[1]:], return_format="transcription_only")[0]
            return run
        if "whisper" in low:                                 # Whisper
            from transformers import pipeline
            pipe = pipeline("automatic-speech-recognition", model=repo, dtype=half, device=dev)
            kw = {"language": language} if language else {}
            return lambda audio: pipe({"raw": audio, "sampling_rate": SR}, generate_kwargs=kw)["text"]
        raise SystemExit(f"No torch loader for {repo}")
    if engine == "onnx":
        import onnx_asr
        model = onnx_asr.load_model(repo, quantization="int8")
        return lambda audio: model.recognize(audio, sample_rate=SR)
    raise SystemExit(f"Unknown engine {engine!r} (use mlx, torch, onnx or auto)")


def transcriber(q: queue.Queue, writer: Writer, hub: Hub, language: str | None):
    """Transcribes queued utterances. Switches models live when hub.want changes
    (utterances wait in the queue while the new model loads)."""
    import gc
    current, model_run = None, None
    while True:
        want = hub.want                                   # (id, engine, repo)
        if want != current:
            mid, engine, repo = want
            hub.model_ready, hub.loading = False, mid
            print(f"Loading {repo} on {engine} (first run downloads it)…", flush=True)
            try:
                new_run = load_engine(engine, repo, language)
                new_run(np.zeros(SR, dtype=np.float32))  # warm up
            except Exception as e:
                print("model load failed:", e, flush=True)
                hub.model_error = f"{mid}: {str(e)[:200]}"
                if current is None:
                    raise
                hub.want = current                        # fall back to the previous model
                hub.model_ready, hub.loading = True, None
                continue
            model_run = new_run
            current = want
            hub.model_id, hub.engine, hub.model = mid, engine, repo
            hub.model_ready, hub.loading, hub.model_error = True, None, None
            hub.audio_s = hub.proc_s = 0.0; hub.latency.clear()
            gc.collect()
            try:
                if engine == "mlx":
                    sys.modules["mlx.core"].clear_cache()
                elif engine == "torch":
                    sys.modules["torch"].cuda.empty_cache()
            except Exception:
                pass
            print("Model ready.", flush=True)
        try:
            t0, audio, who, uid, ch = q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            started = time.time()
            text = (model_run(audio) or "").strip()
            hub.proc_s += time.time() - started
            hub.audio_s += len(audio) / SR
            hub.latency.append(max(0.0, time.time() - (t0 / 1000 + len(audio) / SR)))
        except Exception as e:  # keep running on one bad chunk
            print("transcribe error:", e, flush=True)
            continue
        text = tidy(text)
        if text and text.lower() not in HALLUCINATIONS:
            writer.write(t0, len(audio) / SR, uid, who, text, ch)


# ---------------------------------------------------------------- websocket
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--engine", default="auto", choices=["auto", "mlx", "torch", "onnx"],
                    help="speech runtime (auto: mlx on Apple Silicon, torch with an NVIDIA GPU, else onnx)")
    ap.add_argument("--model", default=None, help="model id for the engine (default depends on the engine)")
    ap.add_argument("--language", default="en", help="for Whisper/Qwen models; 'auto' to detect")
    ap.add_argument("--threshold", type=float, default=0.008, help="speech RMS threshold (0-1)")
    ap.add_argument("--out", default=None, help="transcript folder (default: ../transcripts)")
    ap.add_argument("--parent-pid", type=int, default=0, help="exit when this process (the app) goes away")
    args = ap.parse_args()
    if args.out:
        globals()["TRANSCRIPTS"] = Path(args.out)
    # Which model: --model (any repo, for experiments) > saved choice > default.
    if args.model:
        engine = pick_engine(args.engine)
        want = ("custom", engine, args.model)
    else:
        here = [m for m in models_here() if m["available"]]
        saved = load_settings().get("model")
        forced = None if args.engine == "auto" else args.engine
        choice = next((m for m in here if m["id"] == saved and (not forced or m["engine"] == forced)), None) \
            or next((m for m in here if m.get("default") and (not forced or m["engine"] == forced)), None) \
            or (here[0] if here and not forced else None)
        if choice:
            want = (choice["id"], choice["engine"], choice["repo"])
        else:
            engine = pick_engine(args.engine)
            want = ("default", engine, ENGINE_DEFAULTS[engine])
    engine = want[1]
    language = None if args.language == "auto" else args.language

    # One way to stop, on every OS: set this event (from any thread). Signals
    # aren't reliable here (a background process may have SIGINT ignored, and
    # Windows has no Ctrl-C for windowless processes).
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    request_stop = lambda why: (print(why, flush=True), loop.call_soon_threadsafe(stop.set))

    if args.parent_pid:
        def watch_parent():
            import psutil
            while True:
                time.sleep(2)
                if not psutil.pid_exists(args.parent_pid):
                    request_stop("App is gone; shutting down.")
                    return
        threading.Thread(target=watch_parent, daemon=True).start()

    q: queue.Queue = queue.Queue()
    hub = Hub(want[2])
    hub.engine, hub.want, hub.model_id, hub.loading, hub.model_error = engine, want, want[0], want[0], None
    hub.queue = q
    hub.loop = asyncio.get_running_loop()
    threading.Thread(target=hub.sample_resources, daemon=True).start()
    writer = Writer(hub)
    updater = Updater(hub)
    persona = PersonalityReader()
    threading.Thread(target=updater.loop, daemon=True).start()
    threading.Thread(target=transcriber, args=(q, writer, hub, language), daemon=True).start()

    async def http(connection, request):
        # Plain HTTP (the dashboard); WebSocket upgrades fall through to handle().
        # Only answer requests addressed to this machine (blocks DNS-rebinding
        # tricks where another website pretends to be 127.0.0.1).
        host = request.headers.get("Host", "")
        if host.split(":")[0] not in ("127.0.0.1", "localhost"):
            return Response(403, "Forbidden", Headers([("Content-Length", "0")]), b"")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            origin = request.headers.get("Origin")
            if origin and not re.match(r"^(https://discord\.com|chrome-extension://|http://(127\.0\.0\.1|localhost)(:\d+)?$)", origin):
                return Response(403, "Forbidden", Headers([("Content-Length", "0")]), b"")
            return None
        path, _, query = request.path.partition("?")
        params = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
        def respond(body, ctype="application/json", status=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False)
            if isinstance(body, str):
                body = body.encode("utf-8")
            return Response(status, {200: "OK", 400: "Bad Request", 403: "Forbidden"}.get(status, "Not Found"),
                            Headers([("Content-Type", ctype), ("Cache-Control", "no-store"),
                                     ("Content-Length", str(len(body)))]), body)
        if path in ("/", "/index.html"):
            return respond((DASHBOARD / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            return respond(hub.state())
        if path == "/api/names":
            return respond(hub.names)
        if path == "/api/calls":
            arch = archived_keys()
            calls = []
            for d in all_days():
                for c in reversed(day_data(d)[1]):
                    hit = next((k for k in (f"call:{c['id']}", f"channel:{ch_key(c)}", f"server:{srv_key(c)}") if k in arch), None)
                    calls.append({**c, "srvKey": srv_key(c), "chKey": ch_key(c), "archived": hit})
            return respond(calls)
        if path == "/api/models":
            return respond({"current": hub.model_id, "loading": hub.loading, "error": hub.model_error, "models": models_here()})
        if path == "/api/models/select":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            mid = params.get("id", "")
            m = next((m for m in models_here() if m["id"] == mid and m["available"]), None)
            if not m:
                return respond({"error": "model not available on this machine"}, status=400)
            hub.want = (m["id"], m["engine"], m["repo"])
            save_settings(model=m["id"])
            return respond({"ok": True, "loading": m["id"]})
        if path == "/api/persona/info":
            return respond(persona.info())
        if path == "/api/persona/start":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            try:
                from urllib.parse import unquote
                res = persona.start(params.get("day", ""), int(params.get("from", 0)), int(params.get("to", 0)),
                                    unquote(params.get("ch", "")), unquote(params.get("user", "")), unquote(params.get("name", "")))
            except (ValueError, OSError) as e:
                res = {"error": str(e)}
            return respond(res, status=400 if "error" in res else 200)
        if path == "/api/persona":
            job = persona.jobs.get(params.get("id", ""))
            return respond(job or {"error": "unknown job"}, status=200 if job else 404)
        if path == "/api/update":
            return respond(updater.state)
        if path in ("/api/update/check", "/api/update/apply"):
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            # git can take seconds: never block the audio event loop on it
            if path == "/api/update/check":
                return respond(await asyncio.to_thread(updater.check))
            return respond(await asyncio.to_thread(updater.apply))
        if path == "/api/shutdown":
            # The app asks the server to stop cleanly (Windows has no Ctrl-C for
            # background processes). Same guard as other changes.
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            threading.Timer(0.2, request_stop, args=("Shutdown requested by the app.",)).start()
            return respond({"ok": True})
        if path in ("/api/archive", "/api/delete"):
            # Changes data: only the dashboard itself may ask (custom header =>
            # other websites can't send it without a CORS preflight, which fails).
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            from urllib.parse import unquote_plus
            kind, key = params.get("kind", ""), unquote_plus(params.get("key", ""))
            if kind not in ("call", "channel", "server") or not key:
                return respond({"error": "bad request"}, status=400)
            if path == "/api/archive":
                set_archived(f"{kind}:{key}", params.get("on", "1") == "1")
                return respond({"ok": True})
            if params.get("dry") == "1":
                n_calls = sum(1 for d in all_days() for c in day_data(d)[1] if group_match(c, kind, key))
                n_lines = sum(c["lines"] for d in all_days() for c in day_data(d)[1] if group_match(c, kind, key))
                return respond({"calls": n_calls, "lines": n_lines})
            return respond(delete_group(kind, key))
        if path == "/api/search":
            from urllib.parse import unquote_plus
            q = unquote_plus(params.get("q", "")).strip().lower()
            if len(q) < 2:
                return respond([])
            hits = []
            for d in all_days():
                recs, calls = day_data(d)
                for r in reversed(recs):
                    if q in r["text"].lower() or q in r["speaker"].lower():
                        call = next((c for c in calls if c["start"] <= r["t"] <= c["end"]), None)
                        hits.append({"day": d, "t": r["t"], "speaker": r["speaker"], "text": r["text"],
                                     "chName": r.get("chName") or (call or {}).get("chName"), "call": call and call["id"]})
                        if len(hits) >= 150:
                            return respond(hits)
            return respond(hits)
        if path == "/api/days":
            days = sorted({f.stem for f in TRANSCRIPTS.glob("*.md")} | {f.stem for f in TRANSCRIPTS.glob("*.jsonl")}, reverse=True)
            return respond(days)
        if path == "/api/day":
            day = params.get("d", dt.date.today().isoformat())
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                return respond({"error": "bad day"}, status=404)
            return respond(load_day(day))
        return respond({"error": "not found"}, status=404)

    async def dashboard(ws):
        hub.clients.add(ws)
        try:
            await ws.send(json.dumps(hub.state()))
            async for _ in ws:
                pass
        finally:
            hub.clients.discard(ws)

    async def tick():
        while True:
            await asyncio.sleep(1)
            if hub.clients:
                await hub._send(hub.state())
    asyncio.create_task(tick())

    async def handle(ws):
        if ws.request.path.startswith("/live"):
            return await dashboard(ws)
        people = hub.people
        hub.extensions += 1
        hub.ext_sockets.add(ws)
        last_status = time.time()
        print("Extension connected.", flush=True)

        async def flush_quiet():
            # Someone left or went silent without trailing quiet audio: flush them.
            while True:
                await asyncio.sleep(1)
                for p in list(people.values()):
                    if p.cur and time.time() - p.last_seen > 2.0:
                        p.flush()
        flusher = asyncio.create_task(flush_quiet())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                ev = json.loads(msg)
                if ev["type"] == "track-audio":
                    uid = ev["user"]
                    if uid not in people:
                        people[uid] = Segmenter(q, args.threshold, label=f"user-{uid[-4:]}", uid=uid)
                    if ev.get("name"):
                        people[uid].label = ev["name"]
                        hub.learn_name(uid, ev["name"])
                    if ev.get("ch"):
                        people[uid].channel = ev["ch"]
                    pcm = np.frombuffer(base64.b64decode(ev["pcm"]), dtype="<i2").astype(np.float32) / 32768.0
                    people[uid].push(ev["t"], pcm)
                elif ev["type"] == "hook-status":
                    hub.ext_reports[ws] = ({**ev["s"], "paused": ev.get("paused")}, time.time())
                    with (TRANSCRIPTS.parent / "status.log").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"),
                                            **ev["s"], "paused": ev.get("paused")}) + "\n")
                if time.time() - last_status > 30:
                    last_status = time.time()
                    live = [p.label for p in people.values() if time.time() - p.last_seen < 30]
                    print(f"  STATUS {len(live)} people talking recently: {', '.join(live) or '-'}  "
                          f"(queue {q.qsize()})", flush=True)
        finally:
            hub.extensions -= 1
            hub.ext_sockets.discard(ws)
            hub.ext_reports.pop(ws, None)
            flusher.cancel()
            for p in people.values():
                p.flush()
            print("Extension disconnected.", flush=True)

    async with websockets.serve(handle, "127.0.0.1", args.port, max_size=2**22, process_request=http):
        print(f"Dashboard: http://127.0.0.1:{args.port}/  ·  transcripts → {TRANSCRIPTS}", flush=True)
        await stop.wait()
    print("Server stopped.", flush=True)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
