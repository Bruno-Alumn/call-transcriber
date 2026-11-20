"""Local transcription server for Earshot.

The desktop app sends each call participant's audio as its own stream
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
import shutil
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


# ---------------------------------------------------------------- voice detection
# Loudness alone can't tell a voice from keyboard clicks, a fan, breathing or
# music, and every noise burst that starts an utterance comes back as a junk
# line ("I.", "Oh."). Silero VAD (v5: a 2 MB ONNX model, CPU, ONNX Runtime)
# scores each 32 ms of audio with a speech probability. The model is
# recurrent, so each person has their own VadStream (state + leftover
# samples); the model itself is loaded once, in the background at startup,
# and shared. Until it is ready, or if it can't be loaded, Segmenter falls
# back to the loudness gate alone.
VAD_REPO, VAD_FILE = "onnx-community/silero-vad", "onnx/model.onnx"
VAD_WINDOW, VAD_CONTEXT = 512, 64   # samples per score (32 ms at 16 kHz); samples of the previous window in front
# Loudness bars, as multiples of the speech threshold (which sensitivity sets).
# The VAD tells speech from noise, not this person from a quiet voice in their
# room (TV, a roommate, other people leaking back through their speakers), so
# an utterance still has to start as loud as the threshold; once it is going,
# quieter audio may continue it (soft words, trailing off), but only down to
# VAD_REL of the utterance's own level.
VAD_START = 1.0                     # x threshold: an utterance starts only above this (and with speech)
VAD_FLOOR = 0.25                    # x threshold: below this nothing continues an utterance
VAD_WARM = 0.5                      # x threshold: between utterances, quieter than this isn't scored
VAD_REL = 0.15                      # x the utterance's level (-16 dB): quieter audio doesn't continue it
VAD_MIN_VOICED_MS = 160             # an utterance needs this much speech-scored audio to be transcribed (see flush)


class Vad:
    model = None                    # the shared scoring function once loaded
    state = "off"                   # off -> loading -> ready | failed

    @classmethod
    def load(cls):
        cls.state = "loading"
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            try:
                path = hf_hub_download(VAD_REPO, VAD_FILE, local_files_only=True)   # cached: no network
            except Exception:
                path = hf_hub_download(VAD_REPO, VAD_FILE)
            opts = ort.SessionOptions()
            # One thread: a 32 ms window is tiny, and threads would cost more than the math.
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            sr = np.array(SR, dtype=np.int64)

            def score(x, state):
                out, state = sess.run(None, {"input": x[None, :], "state": state, "sr": sr})
                return float(out[0, 0]), state
            score(np.zeros(VAD_CONTEXT + VAD_WINDOW, np.float32), np.zeros((2, 1, 128), np.float32))
            cls.model, cls.state = score, "ready"
            print("  Speech detection: Silero VAD ready", flush=True)
        except Exception as e:
            cls.state = "failed"
            print(f"  Speech detection: Silero VAD unavailable ({type(e).__name__}: {e}); using loudness only", flush=True)

    @classmethod
    def load_async(cls):
        if cls.state == "off":
            cls.state = "loading"
            threading.Thread(target=cls.load, daemon=True, name="vad-load").start()


class VadStream:
    """One person's VAD: feed chunks of any length, get a probability per 32 ms."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros(VAD_CONTEXT, np.float32)
        self.rest = np.zeros(0, np.float32)      # samples short of a full window, kept for next time

    def feed(self, score, pcm: np.ndarray) -> list[float]:
        x = np.concatenate([self.rest, pcm]) if len(self.rest) else pcm
        n = len(x) // VAD_WINDOW
        probs = []
        for i in range(n):
            w = x[i * VAD_WINDOW:(i + 1) * VAD_WINDOW]
            p, self.state = score(np.concatenate([self.ctx, w]), self.state)
            self.ctx = w[-VAD_CONTEXT:]
            probs.append(p)
        self.rest = x[n * VAD_WINDOW:]
        return probs


# ---------------------------------------------------------------- segmenter
class Segmenter:
    """One person's stream -> utterances. Loudness is a cheap first gate; what
    passes it is scored by the VAD (once loaded), which decides speech."""

    def __init__(self, out: queue.Queue, threshold: float, label: str, uid: str = "",
                 silence_ms=700, max_ms=25000, preroll_ms=300, vad_threshold: float | None = 0.5):
        self.out, self.threshold, self.label, self.uid = out, threshold, label, uid
        self.vad_threshold = vad_threshold        # None: loudness only
        self.channel: dict | None = None
        self.silence_ms, self.max_ms, self.preroll_ms = silence_ms, max_ms, preroll_ms
        self.pre: deque[tuple[float, np.ndarray]] = deque()
        self.cur: list[tuple[float, np.ndarray]] = []
        self.silent_ms = 0.0
        self.noise = threshold
        self.last_seen = time.time()
        self.last_end = 0.0          # audio-clock end of the previous chunk (ms)
        self.vad = VadStream()
        self.voiced_ms = 0.0         # VAD-scored speech in the current utterance
        self.cur_vad = False         # the current utterance was started by the VAD
        self.level = 0.0             # typical loudness of the current utterance's speech
        self.carry = False           # the last utterance was cut at max_ms: what follows continues it
        self.cur_carry = False       # the current utterance is such a continuation
        self.dropped = 0             # utterances the VAD found too little speech in

    def push(self, t: float, pcm: np.ndarray):
        self.last_seen = time.time()
        dur = len(pcm) / SR * 1000
        # The browser only sends speech, so a jump in the audio timeline is
        # silence that was skipped: count it, and end the utterance if it's long.
        gap = t - self.last_end if self.last_end else 0
        self.last_end = t + dur
        if gap > 300:
            self.vad.reset()                     # unrelated audio follows: start the VAD fresh
            if self.cur or self.carry:
                self.silent_ms += gap
                if self.silent_ms >= self.silence_ms:
                    self.flush()
                    self.carry = False           # (a pause this long ends a carry too)
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        score = Vad.model if self.vad_threshold is not None else None
        voiced = 0
        going = bool(self.cur) or self.carry
        if score is None:
            speech = rms > max(self.threshold, self.noise * 3)
        elif rms > self.threshold * VAD_WARM or going:
            # Near a voice's loudness (or mid-utterance, where the VAD's
            # memory of the voice matters): ask the model. Silero is slow to
            # rise at onsets, so once speaking, a lower bar keeps it going.
            probs = self.vad.feed(score, pcm)
            voiced = sum(p > self.vad_threshold for p in probs)
            stay = self.vad_threshold - 0.15 if going else self.vad_threshold
            bar = max(self.threshold * VAD_FLOOR, self.level * VAD_REL) if going else self.threshold * VAD_START
            speech = rms > bar and any(p > stay for p in probs)
            if speech:
                self.level = 0.8 * self.level + 0.2 * rms if going else rms
        else:
            self.vad.reset()                     # quiet, between utterances: nothing worth remembering
            speech = False
        if not speech:
            self.noise = 0.98 * self.noise + 0.02 * rms   # (only the loudness-only gate uses it)
        if self.cur:
            self.cur.append((t, pcm))
            self.voiced_ms += voiced * VAD_WINDOW / SR * 1000
            self.silent_ms = 0 if speech else self.silent_ms + dur
            if self.silent_ms >= self.silence_ms:
                self.flush()
            elif self.cur[-1][0] + dur - self.cur[0][0] >= self.max_ms:
                self.flush()
                self.carry = True                # mid-speech cut: the rest is the same person still talking
        elif speech:
            self.cur = list(self.pre) + [(t, pcm)]
            self.pre.clear()
            self.silent_ms = 0
            self.voiced_ms = voiced * VAD_WINDOW / SR * 1000
            self.cur_vad = score is not None
            self.cur_carry, self.carry = self.carry, False
        else:
            if self.carry:                       # a pause after the cut: still the same talk until it's a long one
                self.silent_ms += dur
                self.carry = self.silent_ms < self.silence_ms
            self.pre.append((t, pcm))
            while self.pre and t - self.pre[0][0] > self.preroll_ms:
                self.pre.popleft()

    def flush(self):
        if not self.cur:
            return
        t0 = self.cur[0][0]
        audio = np.concatenate([p for _, p in self.cur])
        # (sample offset in `audio`, clock ms) per chunk: skipped silence means
        # audio time != clock time, and lines are split back onto the clock.
        marks, off = [], 0
        for t, p in self.cur:
            marks.append((off, t)); off += len(p)
        self.cur, self.silent_ms = [], 0
        self.carry = False
        # A quick "no" or "yes" scores about 250 ms of speech; a cough, a click
        # or a breath that fooled the VAD for a window or two doesn't. (The
        # tail of a talk cut at max_ms is exempt: it can be one short word.)
        if self.cur_vad and not self.cur_carry and self.voiced_ms < VAD_MIN_VOICED_MS:
            self.dropped += 1
            print(f"  Speech detection: dropped a {len(audio) / SR:.1f} s sound with only "
                  f"{self.voiced_ms:.0f} ms of speech ({self.dropped} so far for this person)", flush=True)
            return
        if self.cur_vad and self.cur_carry and len(audio) < SR * 0.4:
            # The last word of a talk cut at max_ms can be a single chunk:
            # pad it with silence rather than lose it to the length minimum.
            audio = np.pad(audio, (0, int(SR * 0.4) - len(audio)))
        if len(audio) >= SR * 0.4:
            self.out.put((t0, audio, self.label, self.uid, self.channel, marks))


# ---------------------------------------------------------------- output
class Writer:
    """Appends each line to the readable .md transcript and to a .jsonl data
    file (used by the dashboard's analytics), then pushes it to the dashboard."""

    def __init__(self, hub: "Hub"):
        self.hub = hub
        self.q = None                    # the utterance queue (set by main)
        self.pending: list = []          # .md lines waiting for anything said earlier
        self.lock = threading.Lock()

    def _paths(self, when):
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        md = TRANSCRIPTS / f"{when:%Y-%m-%d}.md"
        if not md.exists():
            md.write_text(f"# Voice call transcript, {when:%A %B} {when.day}, {when.year}\n\n", encoding="utf-8")
        return md, TRANSCRIPTS / f"{when:%Y-%m-%d}.jsonl"

    def write(self, t_ms: float, dur_s: float, uid: str, speaker: str, text: str, ch: dict | None = None, bg: bool = False):
        when = dt.datetime.fromtimestamp(t_ms / 1000)
        md, jl = self._paths(when)
        rec = {"id": os.urandom(5).hex(), "t": int(t_ms), "end": int(t_ms + dur_s * 1000), "user": uid, "speaker": speaker, "text": text}
        if bg:
            rec["bg"] = True             # not the person's own voice (a video, someone in their room)
        with self.lock:
            self.pending.append((t_ms, when, speaker, text, (ch or {}).get("name"), time.time(), rec["id"], bg))
        self.flush_md()
        if ch and ch.get("id"):
            rec.update({"ch": ch["id"], "chName": ch.get("name"), "guild": ch.get("guild"), "guildName": ch.get("guildName")})
        with jl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{when:%H:%M:%S}] {speaker}: {text}", flush=True)
        self.hub.publish({"type": "line", **rec})
        return rec

    def replace(self, old: list[dict], parts: list[tuple[float, float, str]]):
        """The 2B pass rewrote an utterance: new lines replace the live ones, in
        the data file, the dashboard and (if not written yet) the .md file."""
        if not old or not parts:
            return
        base = old[0]
        ids = [r["id"] for r in old]
        # Deleted (or archived away) since? Then there is nothing to rewrite.
        days = {dt.datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d") for r in old}
        alive = {r.get("id") for d in days for r in load_day(d)}
        if not all(i in alive for i in ids):
            return
        new = []
        for i, (t_ms, end_ms, text) in enumerate(parts):
            rec = {"id": os.urandom(5).hex(), "t": int(t_ms), "end": int(max(end_ms, t_ms + 300)), "user": base["user"],
                   "speaker": base["speaker"], "text": text, "refined": True}
            for k in ("ch", "chName", "guild", "guildName", "bg"):   # the 2B rewrite keeps the background flag
                if k in base:
                    rec[k] = base[k]
            new.append(rec)
        # A note in every day file that held an old line (a call can cross
        # midnight), then each new line in its own day's file.
        for d in sorted(days):
            _, jl = self._paths(dt.datetime.strptime(d, "%Y-%m-%d"))
            with jl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"tomb": True, "t": int(min(r["t"] for r in old)), "replaces": ids}) + "\n")
        for rec in new:
            _, jl = self._paths(dt.datetime.fromtimestamp(rec["t"] / 1000))
            with jl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with self.lock:
            if all(any(x[6] == i for x in self.pending) for i in ids):      # not in the .md yet: swap them
                self.pending = [x for x in self.pending if x[6] not in ids]
                for rec in new:
                    w = dt.datetime.fromtimestamp(rec["t"] / 1000)
                    self.pending.append((rec["t"], w, rec["speaker"], rec["text"], rec.get("chName"), time.time(), rec["id"], bool(rec.get("bg"))))
        self.hub.publish({"type": "replace", "ids": ids, "lines": new})
        self.flush_md()

    def watermark(self) -> float:
        """Clock time (ms) of the earliest speech that could still produce a line:
        someone mid-sentence, an utterance waiting in the queue, or the one being
        transcribed now. Everything said before it is final."""
        marks = []
        try:
            marks += [p.cur[0][0] for p in list(self.hub.people.values()) if p.cur]
        except Exception:
            pass
        try:
            marks += [item[0] for item in list(self.q.queue)] if self.q else []
        except Exception:
            pass
        if getattr(self.hub, "busy_t0", None) is not None:
            marks.append(self.hub.busy_t0)
        try:      # lines waiting for the 2B rewrite (gives up on ones older than a minute)
            marks += [it["t0"] for it in list(getattr(self.hub, "refine_q", ())) if time.time() - it["queued"] < 60]
        except Exception:
            pass
        return min(marks) if marks else float("inf")

    def flush_md(self, force: bool = False):
        """Write .md lines in the order they were said, not the order they were
        transcribed (a long sentence finishes after a short reply that came later)."""
        with self.lock:
            if not self.pending:
                return
            wm, now = self.watermark(), time.time()
            self.pending.sort(key=lambda x: x[0])
            ready = [x for x in self.pending if force or x[0] < wm or now - x[5] > 90]
            if not ready:
                return
            cut = max(i for i, x in enumerate(self.pending) if x in ready) + 1
            ready, self.pending = self.pending[:cut], self.pending[cut:]
        for t_ms, when, speaker, text, chname, _, _, bg in ready:
            md, _ = self._paths(when)
            with md.open("a", encoding="utf-8") as f:
                # Background voices are marked, so the readable transcript doesn't put a video's words in their mouth.
                f.write(f"**[{when:%H:%M:%S}] {speaker}{f' · {chname}' if chname else ''}:** {BG_MARK if bg else ''}{text}\n\n")


BG_MARK = "(background voice) "      # .md prefix for lines that didn't sound like the speaker
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
    gone = {x for r in recs for x in (r.get("replaces") or [])}   # lines the 2B pass rewrote
    if gone or any(r.get("tomb") for r in recs):
        recs = [r for r in recs if not r.get("tomb") and r.get("id") not in gone]
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
    open_by_ch: dict = {}              # several calls can run at once, one per voice channel
    for r in recs:
        ch = r.get("ch")
        cur = open_by_ch.get(ch) if ch else (calls[-1] if calls else None)
        if ch and cur is None and calls and not calls[-1]["ch"] and r["t"] - calls[-1]["end"] < GAP_MS:
            cur = calls[-1]                # older lines without a channel: the call they belong to
        if cur and r["t"] - cur["end"] < GAP_MS and (not ch or not cur["ch"] or ch == cur["ch"]):
            cur["lines"].append(r); cur["end"] = max(cur["end"], r["end"])
            if ch and not cur["ch"]:
                cur.update(ch=ch, chName=r.get("chName"), guild=r.get("guild"), guildName=r.get("guildName"))
                open_by_ch[ch] = cur
        else:
            calls.append({"day": day, "start": r["t"], "end": r["end"], "ch": ch, "chName": r.get("chName"),
                          "guild": r.get("guild"), "guildName": r.get("guildName"), "lines": [r]})
            if ch:
                open_by_ch[ch] = calls[-1]
    out = []
    for c in calls:
        talk: dict[str, float] = {}
        words = 0
        own = [r for r in c["lines"] if not r.get("bg")]     # background voices aren't the person talking
        for r in own:
            talk[r["speaker"]] = talk.get(r["speaker"], 0) + (r["end"] - r["t"]) / 1000
            words += len(r["text"].split())
        people = sorted(talk.items(), key=lambda kv: -kv[1])
        out.append({"id": f"{day}@{c['start']}", "day": day, "start": c["start"], "end": c["end"],
                    "ch": c["ch"], "chName": c["chName"], "guild": c["guild"], "guildName": c["guildName"],
                    "people": [p for p, _ in people], "lines": len(own), "words": words,
                    "total": len(c["lines"]),          # every line, background included (what a delete removes)
                    "preview": next((r["text"] for r in reversed(own) if len(r["text"]) > 20), own[-1]["text"] if own else "")[:120]})
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
                          "server": any(f.startswith("server/") for f in files)}
                st.update(applied=result, available=False, current=self.current())
                return result
            finally:
                st["busy"] = False

    def loop(self):
        time.sleep(20)
        while True:
            try:
                if current_settings()["autoUpdate"]:
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
        self.ext_reports: dict = {}       # connection -> (status, time)
        # One entry per Discord window (account): who it is and whether it is the
        # one transcribing its channel (two of your accounts in one channel must
        # not transcribe everything twice).
        self.conns: dict = {}             # ws -> {account, role, userId, label, since, capturing}
        self.uid_owner: dict = {}         # person -> (connection, last chunk time): one source per person
        self.res: dict = {}
        self.parent_pid = 0
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

    def own_ids(self) -> set:
        ids = {c.get("userId") for c in self.conns.values() if c.get("userId")}
        for st, _ in self.ext_reports.values():
            if (st.get("me") or {}).get("id"):
                ids.add(st["me"]["id"])
        return ids

    def elect(self) -> list:
        """Per voice channel, pick the one connection that transcribes it:
        a listener account before your main one, then whoever joined first.
        Returns [(ws, on)] for connections whose role changed."""
        now, groups = time.time(), {}
        for ws, c in self.conns.items():
            st, at = self.ext_reports.get(ws, ({}, 0.0))
            ch = (st.get("channel") or {}).get("id") if now - at < 15 and st.get("pcs") and st.get("tracks") else None
            if ch:
                groups.setdefault(ch, []).append((ws, c))
        chosen, grouped = set(), set()
        for members in groups.values():
            members.sort(key=lambda m: (m[1].get("role") != "listener", m[1]["since"]))
            chosen.add(members[0][0])
            grouped.update(m[0] for m in members)
        changes = []
        for ws, c in self.conns.items():
            want = ws in chosen or ws not in grouped
            if c["capturing"] != want:
                c["capturing"] = want
                changes.append((ws, want))
        return changes

    def accounts_state(self, now: float) -> list:
        own, out = self.own_ids(), []
        for ws, c in self.conns.items():
            st, at = self.ext_reports.get(ws, ({}, 0.0))
            fresh = now - at < 15
            me = st.get("me") or {}
            self_st = (me.get("self") or {}) if fresh else {}
            out.append({"account": c.get("account"), "role": c.get("role"), "label": c.get("label") or me.get("name"),
                        "userId": c.get("userId") or me.get("id"), "capturing": c["capturing"],
                        "inCall": fresh and bool(st.get("pcs")) and bool(st.get("tracks")),
                        "channel": st.get("channel") if fresh else None,
                        "streams": len([x for x in (st.get("liveIds") or []) if x not in own]) if fresh else 0,
                        "present": [x for x in (st.get("liveIds") or []) if x not in own] if fresh else [],
                        "deafened": bool(self_st.get("deaf") or self_st.get("selfDeaf")),
                        "lastReport": int(at * 1000) if at else None})
        return out

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
               "present": [x for x in (st.get("liveIds") or []) if x not in self.own_ids()] if age < 15 else [],   # in the call now (not your accounts)
               "client": st.get("client") or "chrome",           # "app": Discord inside the desktop app
               "lastReport": int(at * 1000) if at else None}
        return {"type": "state", "model": self.model, "engine": getattr(self, "engine", None), "modelReady": self.model_ready,
                "modelId": getattr(self, "model_id", None), "modelLoading": getattr(self, "loading", None),
                "modelError": getattr(self, "model_error", None),
                "refine": getattr(self, "refine_id", None), "refineLoading": getattr(self, "refine_loading", None),
                "refineQueue": len(getattr(self, "refine_q", ()) or ()), "refined": getattr(self, "refined", 0),
                "extensions": self.extensions, "ext": ext, "people": ppl,
                "accounts": self.accounts_state(now), "own": sorted(self.own_ids()), "queue": self.queue.qsize() if self.queue else 0,
                "uptime": int(now - self.started), "now": int(now * 1000), "pid": os.getpid(),
                "res": {**self.res, "speed": round(self.audio_s / self.proc_s, 1) if self.proc_s else None,
                        "latency": round(sum(self.latency) / len(self.latency), 2) if self.latency else None,
                        "audioMin": round(self.audio_s / 60, 1)}}

    def sample_resources(self):
        """Every 2 s, what Earshot uses, measured the way the OS reports it:
        the speech server, the app and its call windows (the app is this
        server's parent process and its helpers), model / GPU memory, and the
        machine as a whole."""
        import psutil
        procs: dict[int, psutil.Process] = {}
        cores = os.cpu_count() or 1
        last_gpu, gpu = 0.0, {}

        def mem(p: psutil.Process) -> float:
            """Bytes of memory a process really uses (Activity Monitor's / Task Manager's number)."""
            if sys.platform == "darwin":
                fp = phys_footprint(p.pid)
                if fp:
                    return fp
            mi = p.memory_info()
            return getattr(mi, "private", 0) or mi.rss          # Windows: private bytes

        def tracked(pid: int):
            if pid not in procs:
                p = psutil.Process(pid)
                p.cpu_percent(None)                                 # first call primes the counter
                procs[pid] = p
            return procs[pid]

        while True:
            time.sleep(2)
            try:
                me = tracked(os.getpid())
                app_pids: list[int] = []
                if self.parent_pid and psutil.pid_exists(self.parent_pid):
                    parent = psutil.Process(self.parent_pid)
                    app_pids = [parent.pid] + [c.pid for c in parent.children(recursive=True) if c.pid != me.pid
                                               and c.pid not in {x.pid for x in me.children(recursive=True)}]
                server_pids = [me.pid] + [c.pid for c in me.children(recursive=True)]
                live = set(server_pids + app_pids)
                for pid in list(procs):
                    if pid not in live:
                        procs.pop(pid, None)
                cpu_s = mem_s = cpu_a = mem_a = 0.0
                for pid in server_pids:
                    try:
                        p = tracked(pid); cpu_s += p.cpu_percent(None); mem_s += mem(p)
                    except psutil.Error:
                        pass
                for pid in app_pids:
                    try:
                        p = tracked(pid); cpu_a += p.cpu_percent(None); mem_a += mem(p)
                    except psutil.Error:
                        pass
                vm = psutil.virtual_memory()
                model = vram_used = vram_total = None
                if "mlx.core" in sys.modules:                         # Apple Silicon: GPU memory is shared memory
                    mx = sys.modules["mlx.core"]
                    model = mx.get_active_memory() + mx.get_cache_memory()
                elif "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
                    torch = sys.modules["torch"]
                    model = torch.cuda.memory_reserved()
                    free, total = torch.cuda.mem_get_info()
                    vram_used, vram_total = total - free, total       # every program's use of the card
                if time.time() - last_gpu > 5 and shutil.which("nvidia-smi"):
                    last_gpu = time.time()
                    try:
                        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.splitlines()[0]
                        u, used, tot, name = [x.strip() for x in out.split(",", 3)]
                        gpu = {"gpuLoad": float(u), "gpuName": name}
                        vram_used, vram_total = float(used) * 2**20, float(tot) * 2**20
                    except Exception:
                        gpu = {}
                elif gpu and vram_used is None:
                    pass
                MB = lambda b: round(b / 2**20) if b else None
                self.res = {
                    "cores": cores,
                    "cpu": round(cpu_s, 1),                                         # speech server, % of one core
                    "cpuServer": round(cpu_s / cores, 1), "cpuApp": round(cpu_a / cores, 1),
                    "cpuTotal": round((cpu_s + cpu_a) / cores, 1),                  # % of the whole processor
                    "memServerMB": MB(mem_s), "memAppMB": MB(mem_a), "memTotalMB": MB(mem_s + mem_a),
                    "rssMB": MB(mem_s + mem_a),
                    "modelMB": MB(model), "unified": "mlx.core" in sys.modules,
                    "vramUsedMB": MB(vram_used), "vramTotalMB": MB(vram_total), **gpu,
                    "sysMemUsedMB": MB(vm.total - vm.available), "sysMemTotalMB": MB(vm.total),
                    "appProcs": len(app_pids),
                }
            except Exception as e:
                self.res = {**self.res, "error": str(e)[:120]}

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
     "about": "Fast and light: transcribes live. English + 6 languages.", "langs": "English + 6 more",
     "repos": {"mlx": "ibm-granite/granite-speech-5.0-470m-turboctc", "torch": "ibm-granite/granite-speech-5.0-470m-turboctc"},
     "bench": {"mlx": {"wer": 5.5, "speed": 249, "mem": 1.6}, "torch": {"wer": 5.5, "speed": 82, "mem": 0.9}}},
    {"id": "granite-2b", "name": "Granite Speech 5.0 · 2B",
     "about": "The most accurate; about 5x the memory and much slower. English + 6 languages.", "langs": "English + 6 more",
     "repos": {"mlx": "ibm-granite/granite-speech-5.0-2b", "torch": "ibm-granite/granite-speech-5.0-2b"},
     "bench": {"mlx": {"wer": 5.4, "speed": 13, "mem": 9.1}, "torch": {"wer": 5.4, "speed": 11}}},
    # Not offered: used automatically on PCs without an NVIDIA GPU, where Granite can't keep up.
    {"id": "parakeet-v3", "name": "Parakeet TDT 0.6B v3", "fallback": True,
     "about": "Used on this PC because it has no NVIDIA graphics card: Granite needs one to keep up.", "langs": "25 European",
     "repos": {"onnx": "nemo-parakeet-tdt-0.6b-v3"},
     "bench": {"onnx": {"wer": 7.4, "speed": 30, "mem": 2.0}}},
]
GRANITE = ("granite-470m", "granite-2b")


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


def phys_footprint(pid: int) -> int | None:
    """macOS: a process's physical footprint, the memory figure Activity Monitor
    shows (RSS misses GPU buffers that MLX models live in)."""
    try:
        import ctypes, ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        class RUsageV2(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
                "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
                "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
                "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
                "diskio_bytesread", "diskio_byteswritten")]
        buf = RUsageV2()
        if lib.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(2), ctypes.byref(buf)) == 0:
            return int(buf.phys_footprint)
    except Exception:
        pass
    return None


def model_plan() -> dict:
    """Which models run: the Granite ones you switched on (470M live, 2B live, or
    470M live + 2B rewriting each line), or Parakeet when Granite can't run here."""
    here = {m["id"]: m for m in models_here() if m["available"]}
    granite = [g for g in GRANITE if g in here]
    if not granite:
        fb = here.get("parakeet-v3") or next(iter(here.values()), None)
        return {"live": fb, "refine": None, "selected": [], "fallback": True}
    st = load_settings()
    sel = [g for g in (st.get("models") or []) if g in granite]
    if not sel and st.get("model") in granite:          # the old single choice
        sel = [st["model"]]
    if not sel:
        sel = ["granite-470m"] if "granite-470m" in granite else granite[:1]
    live = here["granite-470m"] if "granite-470m" in sel else here[sel[0]]
    refine = here["granite-2b"] if len(sel) == 2 else None
    return {"live": live, "refine": refine, "selected": sel, "fallback": False}


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


# Settings the dashboard can change (stored with the model choice in settings.json).
SETTING_DEFAULTS = {"splitLines": True, "personality": True, "autoUpdate": True, "sensitivity": "normal", "language": "en",
                    "background": "dim"}
SETTING_CHOICES = {"sensitivity": ["high", "normal", "low"], "background": ["show", "dim", "hide"],   # how the dashboard shows "bg" lines
                   "language": ["en", "auto", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk", "ja", "zh", "ko"]}
SENSITIVITY = {"high": 0.5, "normal": 1.0, "low": 2.0}     # x the speech threshold: high hears quieter voices
VAD_SENSITIVITY = {"high": 0.35, "normal": 0.5, "low": 0.65}  # speech probability the VAD must reach (Silero's default is 0.5)


PLUGINS_DIR = ROOT / "plugins"
PLUGIN_TYPES = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml", ".webp": "image/webp", ".html": "text/html; charset=utf-8"}


def list_plugins() -> list[dict]:
    """Plugins: one folder each in plugins/, described by manifest.json. Off until turned on in Settings."""
    on = set(load_settings().get("plugins") or [])
    out = []
    if PLUGINS_DIR.is_dir():
        for d in sorted(PLUGINS_DIR.iterdir()):
            mf = d / "manifest.json"
            if not d.is_dir() or not mf.is_file() or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", d.name):
                continue
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({"id": d.name, "name": str(m.get("name") or d.name)[:80], "version": str(m.get("version") or "")[:20],
                        "description": str(m.get("description") or "")[:300], "main": str(m.get("main") or "main.js"),
                        "style": m.get("style"), "enabled": d.name in on})
    return out


def plugin_file(rel: str):
    """A file inside plugins/<id>/, or None (never outside that folder)."""
    try:
        p = (PLUGINS_DIR / rel).resolve()
        p.relative_to(PLUGINS_DIR.resolve())
    except (ValueError, OSError):
        return None
    if not p.is_file() or p.suffix.lower() not in PLUGIN_TYPES:
        return None
    return p


def current_settings() -> dict:
    st = load_settings()
    return {k: st.get(k, v) for k, v in SETTING_DEFAULTS.items()}


def set_setting(key: str, raw: str) -> dict:
    if key not in SETTING_DEFAULTS:
        return {"error": "unknown setting"}
    if isinstance(SETTING_DEFAULTS[key], bool):
        val = raw in ("1", "true", "on")
    else:
        val = raw if raw in SETTING_CHOICES.get(key, []) else None
        if val is None:
            return {"error": "not one of " + ", ".join(SETTING_CHOICES.get(key, []))}
    save_settings(**{key: val})
    return {"ok": True, key: val}


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
        self.cache: dict[tuple, tuple[int, str]] = {}    # (day, call start, ch, who) -> (words, job id)
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

    def start(self, day: str, t0: int, t1: int, ch: str, user: str, name: str, fresh: bool = False) -> dict:
        if not current_settings()["personality"]:
            return {"error": "Personality reads are turned off in Settings."}
        eng, why = self.engine_here()
        if not eng:
            return {"error": why}
        recs, _ = day_data(day)
        mine = [r for r in recs if t0 <= r["t"] <= t1 and (not ch or not r.get("ch") or r.get("ch") == ch) and not r.get("bg")
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
        # Reuse the last read of this person in this call; redo it every 10 minutes while they keep talking.
        ck = (day, t0, ch, user or name)
        hit = self.cache.get(ck)
        stale = hit and time.time() - self.jobs.get(hit[1], {}).get("started", 0) >= 600 and words - hit[0] >= 20
        if hit and not fresh and not stale:
            old = self.jobs.get(hit[1])
            if old and old["status"] in ("loading", "thinking", "done"):
                return {"id": old["id"], **{k: old[k] for k in ("status", "lines", "words")}, "cached": True}
        jid = os.urandom(6).hex()
        self.cache[ck] = (words, jid)
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


# ---------------------------------------------------------------- timing
# One utterance can hold several sentences while someone else replies in
# between. Split it at sentence ends and pauses, and put each part at the time
# it was actually said, so the transcript interleaves who said what when.
ENGLISH_SYMBOLS = set("£€°’‘“”—–…")              # non-ASCII that English text still uses


def english_mask(tokenizer, size: int, symbols: bool = False):
    """Tokens a model may emit when the call is in English: only ones made of
    plain ASCII, so it can't drift into another language's words or script.
    symbols=True also allows tokens that decode to ASCII plus ENGLISH_SYMBOLS
    (currency, degrees, curly quotes, dashes), for the 2B that writes them."""
    ok = np.zeros(size, dtype=bool)
    markers = str.maketrans("", "", "\u2581\u0120\u010a")      # SentencePiece / byte-BPE space and newline marks
    name = getattr(tokenizer, "convert_ids_to_tokens", None) or getattr(tokenizer, "id_to_token")
    for i in range(size):
        try:
            t = name(i)
        except Exception:
            t = None
        if t is None:
            continue
        t = str(t).translate(markers)
        ok[i] = all(ord(c) < 128 for c in t)
        if symbols and not ok[i]:
            # byte-level BPE names these by their bytes ("Â£"): look at the decoded text
            # (a piece of a multi-byte character decodes to U+FFFD and stays forbidden)
            try:
                d = tokenizer.decode([i])
            except Exception:
                continue
            # (only tokens with a symbol in them: not tabs and control bytes, which the byte names forbid)
            ok[i] = any(c in ENGLISH_SYMBOLS for c in d) and all(32 <= ord(c) < 127 or c in ENGLISH_SYMBOLS for c in d)
    return ok


def en_token_cap(audio_len: int) -> int:
    """Most tokens the 2B may write, English only, for this much audio. English
    speech is under 5 tokens a second (AMI meetings: at most 4.9); with the mask
    on, speech in another script can make the model repeat itself until it runs
    out of tokens (a minute on a 10 s clip), so stop well before that."""
    return int(10 * audio_len / SR) + 20


def looping(ids) -> bool:
    """True when the output ends in the same 1-4 tokens 8 times over ("3 3 3 ...")."""
    for p in range(1, 5):
        if len(ids) >= 8 * p and all(ids[-k] == ids[-k - p] for k in range(1, 7 * p + 1)):
            return True
    return False


def ctc_pieces(ids, audio_len: int, decode, blank: int) -> list[tuple[str, float, float]]:
    """Greedy CTC frames -> [(text added by this token, start s, end s)]."""
    n = len(ids)
    spf = audio_len / SR / max(1, n)                  # seconds per output frame
    toks, prev = [], None
    for i, t in enumerate(ids):
        t = int(t)
        if t != blank and t != prev:
            toks.append([t, i, i])
        elif t != blank and t == prev and toks:
            toks[-1][2] = i
        prev = t
    out, before = [], ""
    for k in range(len(toks)):
        now = decode([x[0] for x in toks[:k + 1]])
        out.append((now[len(before):] if now.startswith(before) else now[len(before):], toks[k][1] * spf, (toks[k][2] + 1) * spf))
        before = now
    return out


def split_timed(pieces: list[tuple[str, float, float]], pause_s: float = 0.6) -> list[tuple[str, float, float]]:
    """Group timed pieces into parts at sentence ends and pauses."""
    parts, buf, start, end = [], "", None, 0.0
    for i, (txt, a, b) in enumerate(pieces):
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        if start is None and txt.strip():
            start = a
        buf += txt; end = b
        sentence_end = buf.rstrip().endswith((".", "?", "!")) and (nxt is None or nxt[0][:1].isspace())
        pause = nxt is not None and nxt[1] - b > pause_s and nxt[0][:1].isspace()
        if (sentence_end or pause or nxt is None) and buf.strip():
            parts.append([buf.strip(), start if start is not None else a, end]); buf, start = "", None
    merged = []                                       # don't leave one-word fragments on their own
    for p in parts:
        if merged and len(p[0].split()) < 2 and not p[0].endswith(("?", "!")):
            merged[-1][0] += " " + p[0]; merged[-1][2] = p[2]
        else:
            merged.append(p)
    return [tuple(p) for p in merged]


def split_by_text(text: str, audio_s: float) -> list[tuple[str, float, float]]:
    """No timing from the model: split at sentence ends, time by share of characters."""
    sents = [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x]
    if len(sents) <= 1:
        return [(text.strip(), 0.0, audio_s)]
    total, pos, out = sum(len(x) for x in sents), 0, []
    for x in sents:
        a = audio_s * pos / total; pos += len(x)
        out.append((x, a, audio_s * pos / total))
    merged = []
    for p in out:
        if merged and len(p[0].split()) < 2:
            merged[-1] = (merged[-1][0] + " " + p[0], merged[-1][1], p[2])
        else:
            merged.append(p)
    return merged


def to_clock(marks: list[tuple[int, float]], sec: float) -> float:
    """Audio seconds within an utterance -> clock ms (skipped silence restored)."""
    if not marks:
        return 0.0
    pos = sec * SR
    best = marks[0]
    for m in marks:
        if m[0] <= pos:
            best = m
        else:
            break
    return best[1] + (pos - best[0]) / SR * 1000


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
        if "granite-speech" in repo.lower() and "turboctc" not in repo.lower():      # Granite 2B
            kw = {"punctuate": True}      # (a language hint made 2B less accurate on English: 6.7% vs 5.3% words wrong)
        if "turboctc" in repo.lower():                   # Granite 470M (CTC): word timing from its frames
            feat = sys.modules[type(model).__module__].compute_features
            blank = model.config.pad_token_id
            tok = model._tokenizer
            mask = None
            def run_ctc(audio):
                nonlocal mask
                x = feat(model._load_audio(mx.array(audio)), num_mel_bins=model.config.encoder_config.num_mel_bins)[None]
                logits = model(x)[0]
                if language == "en":                     # English only: never pick another language's tokens
                    if mask is None:
                        m = english_mask(tok, logits.shape[-1]); m[blank] = True
                        mask = mx.array(np.where(m, 0.0, -1e9).astype(np.float32))
                    logits = logits + mask
                ids = np.array(mx.argmax(logits, axis=-1))
                pieces = ctc_pieces(ids, len(audio), lambda t: tok.decode(t, skip_special_tokens=True), blank)
                return {"text": "".join(p[0] for p in pieces).strip(), "pieces": pieces}
            return run_ctc
        if "granite-speech" in repo.lower() and language == "en":      # Granite 2B, English only
            # model.generate takes no logits processors, so this runs its own
            # (non-streaming, greedy) decoding steps with a mask that forbids
            # tokens which aren't plain ASCII: no other scripts or accented
            # letters (Latin-script words of another language can still slip
            # through, just without accents). Same prompt as generate().
            # When the speech isn't English the mask can make it repeat itself
            # instead of ending: such output is dropped (returns ""), see en_token_cap.
            from mlx_audio.lm.generate import generate_step
            gs = sys.modules[type(model).__module__]
            tok = model._tokenizer
            ask = gs._resolve_prompt("asr", None, None, punctuate=True)
            system = gs.PLUS_SYSTEM_PROMPT if model.is_plus else None
            mask = None
            def only_english(tokens, logits):
                nonlocal mask
                if mask is None:
                    m = english_mask(tok, logits.shape[-1], symbols=True)
                    for i in [tok.eos_token_id, *getattr(tok, "all_special_ids", [])]:
                        if i is not None and i < len(m):
                            m[i] = True
                    mask = mx.array(np.where(m, 0.0, -np.inf).astype(np.float32))
                return logits.astype(mx.float32) + mask
            def run_en(audio):
                feats, n_audio = model._extract_features(model._load_audio(mx.array(audio)))
                audio_feats = model.get_audio_features(feats)
                ids = model._build_prompt(n_audio, ask, system_prompt=system)
                embeds = model._build_inputs_embeds(ids, audio_feats)
                out, ended = [], False
                for t, _ in generate_step(prompt=ids, input_embeddings=embeds.squeeze(0), model=model,
                                          max_tokens=en_token_cap(len(audio)), logits_processors=[only_english]):
                    if t == tok.eos_token_id:
                        ended = True
                        break
                    out.append(int(t))
                    if looping(out):
                        break
                return tok.decode(out, skip_special_tokens=True) if ended else ""
            return run_en
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
            blank = model.config.pad_token_id
            tk = getattr(proc, "tokenizer", proc)
            mask = None
            def run(audio):
                nonlocal mask
                with torch.inference_mode():
                    inp = to_dev(proc([audio], sampling_rate=SR), model.dtype)
                    logits = model(**inp).logits[0]
                    if language == "en":                 # English only: never pick another language's tokens
                        if mask is None:
                            m = english_mask(tk, logits.shape[-1]); m[blank] = True
                            mask = torch.tensor(np.where(m, 0.0, -1e4), dtype=logits.dtype, device=logits.device)
                        logits = logits + mask
                ids = logits.argmax(-1).cpu().numpy()
                pieces = ctc_pieces(ids, len(audio), lambda t: tk.decode(t, skip_special_tokens=True), blank)
                return {"text": "".join(p[0] for p in pieces).strip(), "pieces": pieces}
            return run
        if "granite-speech" in low:                          # Granite 2B (speech LLM)
            from transformers import AutoModelForSpeechSeq2Seq
            model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
            tok = proc.tokenizer
            ask = "<|audio|>transcribe the speech with proper punctuation and capitalization."
            prompt = tok.apply_chat_template([{"role": "user", "content": ask}],
                                             tokenize=False, add_generation_prompt=True)
            gen = {}
            if language == "en":                         # English only: forbid tokens that aren't plain ASCII
                from transformers import LogitsProcessor, LogitsProcessorList
                class OnlyEnglish(LogitsProcessor):
                    bad = None
                    def __call__(self, input_ids, scores):
                        if self.bad is None:
                            m = english_mask(tok, scores.shape[-1], symbols=True)
                            for i in [tok.eos_token_id, *tok.all_special_ids]:
                                if i is not None and i < len(m):
                                    m[i] = True
                            self.bad = torch.from_numpy(~m).to(scores.device)
                        return scores.masked_fill(self.bad, -float("inf"))
                gen["logits_processor"] = LogitsProcessorList([OnlyEnglish()])
            def run(audio):
                with torch.inference_mode():
                    inp = proc(prompt, torch.from_numpy(audio).unsqueeze(0), device=dev, return_tensors="pt").to(dev)
                    cap = min(200, en_token_cap(len(audio))) if gen else 200
                    out = model.generate(**inp, max_new_tokens=cap, do_sample=False, num_beams=1, **gen)
                    new = out[0, inp["input_ids"].shape[-1]:].tolist()
                    if gen and (looping([t for t in new if t != tok.pad_token_id]) or
                                (tok.eos_token_id not in new and len(new) >= cap)):
                        return ""                    # English only, but it looped: not English speech
                    return tok.decode(new, skip_special_tokens=True)
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


def transcriber(q: queue.Queue, writer: Writer, hub: Hub, language: str | None, language_forced: bool = False):
    """Transcribes queued utterances. Switches models live when hub.want changes
    (utterances wait in the queue while the new model loads)."""
    import gc
    from concurrent.futures import ThreadPoolExecutor
    # Own-voice detection runs on the CPU next to transcription (on the GPU
    # with MLX/torch), so it adds no wait; one worker keeps it in order.
    try:
        from voiceprint import Voiceprints
        voices = Voiceprints()
    except Exception as e:               # missing file or onnxruntime: transcribe without it
        print("Own-voice detection off:", e, flush=True)
        voices = None
    voice_pool = ThreadPoolExecutor(1)
    current, model_run = None, None
    refine_cur, refine_run = None, None
    refine_lang = None                   # the language the rewrite model was loaded for (2B is English-only on "en", else the same)
    loader: dict = {}                    # background load of the rewrite model
    def free_memory(engine):
        gc.collect()
        try:
            if engine == "mlx":
                sys.modules["mlx.core"].clear_cache()
            elif engine == "torch":
                sys.modules["torch"].cuda.empty_cache()
        except Exception:
            pass
    while True:
        lang = current_settings()["language"] if not language_forced else language
        want = (*hub.want, lang)                          # (id, engine, repo, language)
        if want != current and refine_cur and want[:3] == refine_cur and refine_run \
                and (refine_lang == "en") == (lang == "en"):
            # Both -> 2B only: the rewrite model is already loaded; make it the live one.
            model_run, refine_run, refine_cur, hub.refine_id = refine_run, None, None, None
            hub.refine_q.clear()
            current = want
            hub.model_id, hub.engine, hub.model = want[0], want[1], want[2]
            hub.model_ready, hub.loading, hub.model_error = True, None, None
            free_memory(hub.engine)
            print("Model ready (the 2B rewrite model is now live).", flush=True)
        if want != current:
            mid, engine, repo, lang = want
            language = None if lang == "auto" else lang
            if refine_run is not None and (hub.want_refine is None or (refine_lang == "en") != (language == "en")):
                # free the rewrite model first (also when English only was turned on or off:
                # it reloads for that; 2B is the same for every other language)
                refine_run, refine_cur, hub.refine_id = None, None, None
                hub.refine_q.clear()
                free_memory(engine)
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
                hub.want = current[:3]                    # fall back to the previous model
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
        # The 2B rewrite pass: loaded when both models are on, dropped when not.
        target = hub.want_refine
        if target != refine_cur:
            if target is None:
                if refine_run is not None or refine_cur is not None:
                    refine_run, refine_cur, hub.refine_id = None, None, None
                    hub.refine_q.clear()
                    free_memory(hub.engine)
            elif loader.get("target") != target:
                # Load (and on first use download) 2B in the background: live lines keep coming.
                loader.clear(); loader["target"] = target
                hub.refine_loading = target[0]
                def load_refine(tg=target, lg=language):
                    try:
                        print(f"Loading {tg[2]} for the rewrite pass…", flush=True)
                        if tg[1] == "mlx":
                            # MLX ties a model to the thread that loads it: only download here
                            # (the slow part); the transcriber thread loads it, in seconds.
                            from huggingface_hub import snapshot_download
                            snapshot_download(tg[2])
                            loader.update(done=True, run=None, target=tg)
                        else:
                            loader.update(done=True, run=load_engine(tg[1], tg[2], lg), target=tg, lang=lg)
                    except Exception as e:
                        loader.update(done=True, error=f"{tg[0]}: {str(e)[:200]}", target=tg)
                threading.Thread(target=load_refine, daemon=True).start()
            elif loader.get("done"):
                if loader.get("error"):
                    print("rewrite model failed:", loader["error"], flush=True)
                    hub.model_error = loader["error"]
                    hub.want_refine = None
                elif loader.get("run") is not None and (loader.get("lang") == "en") != (language == "en"):
                    loader.clear()                                         # English only changed while loading: load again
                else:
                    try:
                        run = loader["run"] or load_engine(target[1], target[2], language)
                        run(np.zeros(SR, dtype=np.float32))                # warm up here, on this thread
                        refine_run, refine_cur, hub.refine_id, refine_lang = run, target, target[0], language
                        print("Rewrite pass ready.", flush=True)
                    except Exception as e:
                        hub.model_error = f"{target[0]}: {str(e)[:200]}"; hub.want_refine = None
                loader.clear(); hub.refine_loading = None
        elif loader and loader.get("target") != target:
            loader.clear(); hub.refine_loading = None      # turned off again while it was loading
        writer.flush_md()
        # Live lines first; the rewrite pass uses the time in between.
        if refine_run and hub.refine_q and q.empty():
            it = hub.refine_q.popleft()
            if time.time() - it["queued"] < 60:
                try:
                    res2 = refine_run(it["audio"]) or ""
                    text2 = tidy((res2["text"] if isinstance(res2, dict) else res2).strip())
                    old_text = " ".join(r["text"] for r in it["lines"])
                    if text2 and text2.lower() not in HALLUCINATIONS and text2 != old_text:
                        writer.replace(it["lines"], refined_parts(text2, it))
                        hub.refined += 1
                except Exception as e:
                    print("rewrite error:", e, flush=True)
            writer.flush_md()
            continue
        try:
            item = q.get(timeout=0.5)
            t0, audio, who, uid, ch = item[:5]
            marks = item[5] if len(item) > 5 else [(0, t0)]
            hub.busy_t0 = t0
        except queue.Empty:
            continue
        voice = voice_pool.submit(voices.embed, audio) if voices else None
        try:
            started = time.time()
            res = model_run(audio) or ""
            text = (res["text"] if isinstance(res, dict) else res).strip()
            hub.proc_s += time.time() - started
            hub.audio_s += len(audio) / SR
            hub.latency.append(max(0.0, time.time() - (t0 / 1000 + len(audio) / SR)))
        except Exception as e:  # keep running on one bad chunk
            print("transcribe error:", e, flush=True)
            hub.busy_t0 = None
            continue
        if not text or tidy(text).lower() in HALLUCINATIONS:
            hub.busy_t0 = None
            continue
        if not current_settings()["splitLines"]:
            parts = [(text, 0.0, len(audio) / SR)]
        else:
            parts = split_timed(res["pieces"]) if isinstance(res, dict) and res.get("pieces") else split_by_text(text, len(audio) / SR)
        keep = [(p, a, b) for p, a, b in ((tidy(p), a, b) for p, a, b in parts) if p and p.lower() not in HALLUCINATIONS]
        bg = False
        if voice and keep:              # only speech that became a line shapes the voices (not music or noise)
            try:
                bg = voices.judge(uid or who, voice.result(timeout=10), len(audio) / SR)
            except Exception as e:      # never lose a line over this
                print("voiceprint error:", e, flush=True)
        written = []
        for part, a, b in keep:
            start = to_clock(marks, a)
            written.append(writer.write(start, max(0.3, (to_clock(marks, b) - start) / 1000), uid, who, part, ch, bg))
        if refine_run and written:
            hub.refine_q.append({"t0": t0, "audio": audio, "marks": marks, "lines": written, "queued": time.time()})
        hub.busy_t0 = None


def refined_parts(text: str, it: dict) -> list[tuple[float, float, str]]:
    """Times for the rewritten text: if it splits into as many sentences as the
    live lines, keep each live line's timing; otherwise share the utterance's
    span by text length."""
    lines, audio_s = it["lines"], len(it["audio"]) / SR
    sents = split_by_text(text, audio_s) if current_settings()["splitLines"] else [(text, 0.0, audio_s)]
    if len(sents) == len(lines):
        return [(r["t"], r["end"], s[0]) for r, s in zip(lines, sents)]
    lo, hi = min(r["t"] for r in lines), max(r["end"] for r in lines)
    clamp = lambda x: min(max(x, lo), hi)
    return [(clamp(to_clock(it["marks"], a)), clamp(to_clock(it["marks"], b)), s) for s, a, b in sents]


# ---------------------------------------------------------------- websocket
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--engine", default="auto", choices=["auto", "mlx", "torch", "onnx"],
                    help="speech runtime (auto: mlx on Apple Silicon, torch with an NVIDIA GPU, else onnx)")
    ap.add_argument("--model", default=None, help="model id for the engine (default depends on the engine)")
    ap.add_argument("--language", default="en", help="for Whisper/Qwen models; 'auto' to detect")
    ap.add_argument("--threshold", type=float, default=0.008, help="speech RMS threshold (0-1)")
    ap.add_argument("--no-vad", action="store_true", help="decide speech by loudness only (no Silero VAD)")
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
        forced = None if args.engine == "auto" else args.engine
        plan = model_plan()
        choice = plan["live"] if plan["live"] and (not forced or plan["live"]["engine"] == forced) else None
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
    rp = None if args.model else model_plan()["refine"]
    hub.want_refine = (rp["id"], rp["engine"], rp["repo"]) if rp and rp["engine"] == engine else None
    hub.refine_q = deque()               # utterances waiting for the 2B rewrite
    hub.refine_id, hub.refine_loading, hub.refined = None, None, 0
    hub.queue = q
    hub.loop = asyncio.get_running_loop()
    hub.parent_pid = args.parent_pid
    threading.Thread(target=hub.sample_resources, daemon=True).start()
    if not args.no_vad:
        Vad.load_async()                 # speech detection; loudness only until it's ready
    writer = Writer(hub)
    writer.q = q
    updater = Updater(hub)
    persona = PersonalityReader()
    threading.Thread(target=updater.loop, daemon=True).start()
    lang_cli = "--language" in sys.argv                  # an explicit --language wins over the setting
    threading.Thread(target=transcriber, args=(q, writer, hub, language, lang_cli), daemon=True).start()

    async def http(connection, request):
        # Plain HTTP (the dashboard); WebSocket upgrades fall through to handle().
        # Only answer requests addressed to this machine (blocks DNS-rebinding
        # tricks where another website pretends to be 127.0.0.1).
        host = request.headers.get("Host", "")
        if host.split(":")[0] not in ("127.0.0.1", "localhost"):
            return Response(403, "Forbidden", Headers([("Content-Length", "0")]), b"")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            origin = request.headers.get("Origin")
            if origin and not re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", origin):   # the app (no Origin) or the dashboard
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
        if path == "/logo.png":
            return respond((DASHBOARD / "logo.png").read_bytes(), "image/png")
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
            plan = model_plan()
            return respond({"current": hub.model_id, "loading": hub.loading or hub.refine_loading, "error": hub.model_error,
                            "selected": plan["selected"], "fallback": plan["fallback"], "refine": hub.refine_id,
                            "models": [m for m in models_here() if not m.get("fallback") or plan["fallback"]]})
        if path == "/api/models/select":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            ids = [x for x in params.get("ids", params.get("id", "")).split(",") if x]
            avail = {m["id"]: m for m in models_here() if m["available"] and m["id"] in GRANITE}
            if not ids or any(x not in avail for x in ids):
                return respond({"error": "Choose Granite 470M, 2B or both (at least one)."}, status=400)
            save_settings(models=[g for g in GRANITE if g in ids])
            plan = model_plan()
            hub.want = (plan["live"]["id"], plan["live"]["engine"], plan["live"]["repo"])
            r = plan["refine"]
            hub.want_refine = (r["id"], r["engine"], r["repo"]) if r else None
            return respond({"ok": True, "selected": plan["selected"]})
        if path == "/api/plugins":
            return respond(list_plugins())
        if path == "/api/plugins/enable":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            pid, on = params.get("id", ""), params.get("on") == "1"
            if pid not in {p["id"] for p in list_plugins()}:
                return respond({"error": "no such plugin"}, status=404)
            cur = [x for x in (load_settings().get("plugins") or []) if x != pid] + ([pid] if on else [])
            save_settings(plugins=cur)
            return respond({"ok": True, "enabled": cur})
        if path.startswith("/plugins/"):
            from urllib.parse import unquote
            f = plugin_file(unquote(path[len("/plugins/"):]))
            if not f:
                return respond({"error": "not found"}, status=404)
            return respond(f.read_bytes(), PLUGIN_TYPES[f.suffix.lower()])
        if path == "/api/settings":
            if request.headers.get("X-CT") == "1" and params.get("key"):
                from urllib.parse import unquote
                res = set_setting(params["key"], unquote(params.get("value", "")))
                if res.get("ok") and params["key"] == "sensitivity":
                    for p in hub.people.values():
                        p.threshold = args.threshold * SENSITIVITY[res["sensitivity"]]
                        if p.vad_threshold is not None:
                            p.vad_threshold = VAD_SENSITIVITY[res["sensitivity"]]
                return respond(res, status=400 if "error" in res else 200)
            return respond({**current_settings(), "choices": SETTING_CHOICES})
        if path == "/api/persona/info":
            return respond(persona.info())
        if path == "/api/persona/start":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            try:
                from urllib.parse import unquote
                res = persona.start(params.get("day", ""), int(params.get("from", 0)), int(params.get("to", 0)),
                                    unquote(params.get("ch", "")), unquote(params.get("user", "")), unquote(params.get("name", "")),
                                    fresh=params.get("fresh") == "1")
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
                n_lines = sum(c["total"] for d in all_days() for c in day_data(d)[1] if group_match(c, kind, key))
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
                        hits.append({"day": d, "t": r["t"], "speaker": r["speaker"], "text": r["text"], "bg": bool(r.get("bg")),
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
        conn = hub.conns[ws] = {"account": None, "role": "main", "userId": None, "label": None,
                                "since": time.time(), "capturing": True}

        async def reelect():
            for w, on in hub.elect():
                try:
                    await w.send(json.dumps({"type": "capture", "on": on}))
                except Exception:
                    pass
        last_status = time.time()
        print("Discord window connected.", flush=True)

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
                    if not conn["capturing"]:          # another of your accounts is transcribing this channel
                        continue
                    uid = ev["user"]
                    # Two of your accounts in one call both hear this person: take their
                    # audio from one connection only (until it goes quiet for 2 s).
                    own = hub.uid_owner.get(uid)
                    now_t = time.time()
                    if (own and own[0] is not ws and own[0] in hub.conns and hub.conns[own[0]]["capturing"]
                            and now_t - own[1] < 2.0):
                        continue                       # (hands over at once when that one stands by)
                    hub.uid_owner[uid] = (ws, now_t)
                    if uid not in people:
                        sens = current_settings()["sensitivity"]
                        people[uid] = Segmenter(q, args.threshold * SENSITIVITY.get(sens, 1.0),
                                                label=f"user-{uid[-4:]}", uid=uid,
                                                vad_threshold=None if args.no_vad else VAD_SENSITIVITY.get(sens, 0.5))
                    if ev.get("name"):
                        people[uid].label = ev["name"]
                        hub.learn_name(uid, ev["name"])
                    if ev.get("ch"):
                        people[uid].channel = ev["ch"]
                    pcm = np.frombuffer(base64.b64decode(ev["pcm"]), dtype="<i2").astype(np.float32) / 32768.0
                    people[uid].push(ev["t"], pcm)
                elif ev["type"] == "hello":
                    conn.update(account=ev.get("account"), role="listener" if ev.get("role") == "listener" else "main",
                                userId=ev.get("userId") or conn["userId"], label=ev.get("label") or conn["label"])
                    await reelect()
                elif ev["type"] == "hook-status":
                    hub.ext_reports[ws] = ({**ev["s"], "paused": ev.get("paused")}, time.time())
                    if ev["s"].get("role") in ("main", "listener"):
                        conn["role"] = ev["s"]["role"]
                    await reelect()
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
            hub.conns.pop(ws, None)
            for w, on in hub.elect():          # someone else in that channel takes over
                try:
                    await w.send(json.dumps({"type": "capture", "on": on}))
                except Exception:
                    pass
            flusher.cancel()
            for p in people.values():
                p.flush()
            print("Discord window disconnected.", flush=True)

    async with websockets.serve(handle, "127.0.0.1", args.port, max_size=2**22, process_request=http):
        print(f"Dashboard: http://127.0.0.1:{args.port}/  ·  transcripts → {TRANSCRIPTS}", flush=True)
        await stop.wait()
    writer.flush_md(force=True)                   # write any lines still waiting for earlier speech
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
