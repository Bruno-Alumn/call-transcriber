"""Voice detection test: noise must not start utterances, speech must.

Feeds per-person streams to the server's Segmenter in 200 ms chunks, as the
desktop app does (chunks of digital silence are not sent):
  (a) noise only (white, pink, fan hum, keyboard clicks, a music-like tone
      mix, breathing), all loud enough to pass the old loudness gate: with
      Silero VAD no utterance may be queued (and the loudness-only gate is
      checked to queue them, so the noise really is a test);
  (b) speech (test/site/*.wav, plain and over noise; plus AMI meeting clips
      when the parquet file and pyarrow are present): the speech must still be
      queued, and the queued audio must cover it (recall is printed).
  (b2) speech cut to 100 ms is dropped, short answers ("No.", "Yes.", spoken
      fast by macOS `say`, skipped elsewhere) are kept;
  (f) quiet voices that aren't the participant (a TV, a roommate, other people
      leaking back through their speakers): a voice at -52..-44 dBFS must not
      start lines at 'low', and at 'normal' only above -46 dBFS; a quiet voice
      right after the participant's own must not be glued onto their line;
  (g) a talk longer than the 25 s cut (3 s here): the last word after the
      cut is kept, even when it is a single chunk.
Also: the loudness-only fallback when the model is missing, and the cost per
200 ms chunk.

  cd server && uv run python ../test/vad_test.py [--ami path/to/0.parquet] [--ami-clips 120]
"""
import argparse, io, os, queue, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, soundfile as sf

HERE = Path(__file__).resolve().parent
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, str(HERE.parent / "server"))
import server as S

ap = argparse.ArgumentParser()
ap.add_argument("--ami", default=os.environ.get("AMI_PARQUET", ""))
ap.add_argument("--ami-clips", type=int, default=120)
args = ap.parse_args(ARGS)

SR, CHUNK, THRESHOLD = 16000, 3200, 0.008          # the server's --threshold default
rng = np.random.default_rng(7)
failures = []


def segment(x: np.ndarray, vad=0.5, threshold=THRESHOLD, **kw):
    """Run one person's stream through a Segmenter; returns (utterances, segmenter)."""
    q = queue.Queue()
    seg = S.Segmenter(q, threshold, "test", vad_threshold=vad, **kw)
    pcm16 = (np.clip(x, -1, 1) * 32767).astype("<i2")
    for i in range(0, len(x), CHUNK):
        c = pcm16[i:i + CHUNK]
        if len(c) and np.abs(c).max() > 60:          # the app only sends non-silent chunks
            seg.push(i / 16, c.astype(np.float32) / 32768.0)
    seg.flush()
    return [q.get() for _ in range(q.qsize())], seg


def covered(utts) -> list[tuple[int, int]]:
    """Sample ranges of the stream that ended up in queued utterances."""
    out = []
    for t0, audio, *_, marks in utts:
        for k, (off, t) in enumerate(marks):
            end = marks[k + 1][0] if k + 1 < len(marks) else len(audio)
            s = int(round(t * 16))
            out.append((s, s + end - off))
    return out


def covered_mask(n, utts):
    mask = np.zeros(n, bool)
    for a, b in covered(utts):
        mask[a:b] = True
    return mask


def speech_recall(x, utts, loud_at=THRESHOLD):
    """Share of the loud 32 ms frames of x (the speech) inside queued audio."""
    mask = covered_mask(len(x), utts)
    n = len(x) // 512
    fr = x[:n * 512].reshape(n, 512)
    loud = np.sqrt((fr ** 2).mean(1)) > loud_at
    inside = mask[:n * 512].reshape(n, 512).all(1)
    return float((loud & inside).sum() / max(1, loud.sum()))


def rms_to(x, level):
    return (x * level / np.sqrt(np.mean(x ** 2))).astype(np.float32)


# ------------------------------------------------------------- noise makers
def white(sec):
    return rms_to(rng.standard_normal(SR * sec), 0.03)


def pink(sec):
    f = np.fft.rfft(rng.standard_normal(SR * sec))
    f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
    return rms_to(np.fft.irfft(f, SR * sec), 0.03)


def fan(sec):
    t = np.arange(SR * sec) / SR
    brown = np.cumsum(rng.standard_normal(SR * sec)); brown -= np.convolve(brown, np.ones(800) / 800, "same")
    return rms_to(rms_to(brown, 1) + 0.6 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t), 0.03)


def keyboard(sec):
    x = np.zeros(SR * sec, np.float32)
    i = 0
    while i < len(x) - 2000:                         # typing: a click every 80-300 ms, a pause now and then
        n = 400
        click = rng.standard_normal(n) * np.exp(-np.arange(n) / 60) * rng.uniform(0.2, 0.6)
        click = np.convolve(click, [1, -0.9], "same")      # bright, like a key
        x[i:i + n] += click
        if rng.random() < 0.5:                       # key release
            x[i + 1200:i + 1200 + n] += 0.5 * click
        i += int(SR * (rng.uniform(0.08, 0.3) if rng.random() > 0.08 else rng.uniform(0.6, 1.2)))
    return x + rms_to(rng.standard_normal(len(x)), 0.002)


def music(sec):
    """Tone mix: chords of harmonic notes, a new chord every 0.4 s, plucked envelope."""
    t = np.arange(int(SR * 0.4)) / SR
    out = []
    for _ in range(int(sec / 0.4)):
        root = 110 * 2 ** (rng.integers(0, 24) / 12)
        chord = np.zeros_like(t)
        for step in (0, 4, 7, 12):
            f0 = root * 2 ** (step / 12)
            for h in range(1, 6):
                chord += np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h
        out.append(chord * np.exp(-t * 3))
    return rms_to(np.concatenate(out), 0.05)


def breathing(sec):
    """Breaths: band-limited noise, 1.2 s inhale / 1.8 s exhale, smooth envelope."""
    n = SR * sec
    f = np.fft.rfft(rng.standard_normal(n))
    hz = np.fft.rfftfreq(n, 1 / SR)
    f *= np.exp(-((hz - 900) / 600) ** 2)            # breathy: centred near 1 kHz
    x = np.fft.irfft(f, n)
    t = np.arange(n) / SR
    env = np.clip(np.sin(2 * np.pi * t / 3.0), 0, None) ** 1.5 + 0.6 * np.clip(-np.sin(2 * np.pi * t / 3.0), 0, None) ** 1.5
    return rms_to(x * env, 0.03)


NOISES = {"white noise": white, "pink noise": pink, "fan + hum": fan, "keyboard clicks": keyboard,
          "music-like tones": music, "breathing": breathing}

# ------------------------------------------------------------- the model
t = time.time()
S.Vad.load()
if S.Vad.model is None:
    sys.exit("FAIL: Silero VAD did not load")
print(f"Silero VAD loaded in {time.time() - t:.1f}s")

print("\n(a) noise only, 20 s each: utterances queued (loudness-only gate -> with VAD)")
for name, make in NOISES.items():
    x = make(20)
    old, _ = segment(x, vad=None)
    for sens, vt in S.VAD_SENSITIVITY.items():
        new, seg = segment(x, vad=vt)
        if sens == "normal":
            print(f"  {name:18s} rms {np.sqrt(np.mean(x ** 2)):.3f}   {len(old):2d} -> {len(new)}   "
                  f"(dropped as too little speech: {seg.dropped})")
        if new:
            failures.append(f"{name}: {len(new)} utterance(s) queued at sensitivity {sens}")
    if not old:
        failures.append(f"{name}: not loud enough to fool the loudness gate, so this proves nothing")

print("\n(b) speech: test/site clips (1 s silence around each)")
site = sorted((HERE / "site").glob("*.wav"))
noisy_bed = pink(30) * 0.4                          # rms 0.012: a noisy room under the voice
for p in site:
    a, sr = sf.read(p, dtype="float32")
    a = a.mean(1) if a.ndim > 1 else a
    assert sr == SR, p
    x = np.concatenate([np.zeros(SR, np.float32), a, np.zeros(SR, np.float32)])
    row = []
    for label, stream, vt in (("clean", x, 0.5), ("over noise", x + noisy_bed[:len(x)], 0.5), ("loudness only", x, None)):
        utts, _ = segment(stream, vad=vt)
        r = speech_recall(x, utts)
        row.append(f"{label} {len(utts)} utt, {100 * r:5.1f}%")
        if not utts or r < 0.95:
            failures.append(f"{p.name} ({label}): recall {100 * r:.1f}% in {len(utts)} utterance(s)")
    print(f"  {p.name:7s} " + " | ".join(row))

print(f"\n(b2) blips: speech cut to 100 ms is dropped (< {S.VAD_MIN_VOICED_MS} ms voiced), a 400 ms word is kept")
a, _ = sf.read(site[0], dtype="float32")
for ms, keep in ((100, False), (400, True)):
    for start in (8000, 24000, 40000):
        x = np.concatenate([np.zeros(SR, np.float32), a[start:start + 16 * ms], np.zeros(SR, np.float32)])
        utts, seg = segment(x)
        print(f"  {ms} ms from {start / SR:.1f}s: {len(utts)} queued, {seg.dropped} dropped")
        if bool(utts) != keep:
            failures.append(f"{ms} ms blip at {start}: {'not ' if keep else ''}queued")

if shutil.which("say"):
    print("\n(b3) short answers, spoken fast (macOS say, 340 words/min): queued at every sensitivity")
    with tempfile.TemporaryDirectory() as d:
        for voice in ("Samantha", "Daniel"):
            for word in ("No", "Yes", "Sure", "Yeah"):
                f = Path(d) / f"{word}{voice}.wav"
                r = subprocess.run(["say", "-v", voice, "-r", "340", "-o", str(f), "--data-format=LEI16@16000", word + "."],
                                   capture_output=True)
                if r.returncode or not f.exists():
                    continue                             # (that voice isn't installed)
                w, _ = sf.read(f, dtype="float32")
                w = w * 0.06 / np.sqrt(np.mean(w[np.abs(w) > 0.01 * np.abs(w).max()] ** 2))   # the word at rms 0.06
                x = np.concatenate([np.zeros(SR, np.float32), w, np.zeros(SR, np.float32)])
                kept = []
                for sens, vt in S.VAD_SENSITIVITY.items():
                    utts, seg = segment(x, vad=vt, threshold=THRESHOLD * S.SENSITIVITY[sens])
                    kept.append(f"{sens} {'kept' if utts else 'DROPPED'}")
                    if not utts:
                        failures.append(f"short answer '{word}.' ({voice}) not queued at sensitivity {sens}")
                print(f"  {word + '.':6s} {voice:9s} " + ", ".join(kept))
else:
    print("\n(b3) short answers: skipped (no macOS say)")

print("\n(c) fallback: no model -> loudness gate still queues speech")
model, S.Vad.model = S.Vad.model, None
a, _ = sf.read(site[0], dtype="float32")
utts, _ = segment(np.concatenate([np.zeros(SR, np.float32), a, np.zeros(SR, np.float32)]))
print(f"  {site[0].name}: {len(utts)} utterance(s)")
if not utts:
    failures.append("fallback: speech not queued without the model")
S.Vad.model = model

if args.ami and Path(args.ami).exists():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None
        print("\n(d) AMI: skipped (no pyarrow)")
    if pq:
        tab = pq.read_table(args.ami)
        rows = tab.slice(0, tab.num_rows).to_pylist()[::max(1, tab.num_rows // args.ami_clips)][:args.ami_clips]
        # (sensitivity, VAD threshold, loudness threshold), as the server sets them
        sets = [("loudness only", None, THRESHOLD)] + [(s, vt, THRESHOLD * S.SENSITIVITY[s]) for s, vt in S.VAD_SENSITIVITY.items()]
        got = {name: [] for name, *_ in sets}
        for r in rows:
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            a = a.mean(1) if a.ndim > 1 else a
            x = np.concatenate([np.zeros(SR, np.float32), a, np.zeros(SR, np.float32)])
            for name, vt, thr in sets:
                utts, _ = segment(x, vad=vt, threshold=thr)
                got[name].append((bool(utts), speech_recall(x, utts)))
        print(f"\n(d) AMI meeting clips ({len(rows)}, real headset audio): clips with speech queued, loud frames covered")
        for name, res in got.items():
            hit = sum(h for h, _ in res) / len(res)
            cov = sum(c for _, c in res) / len(res)
            print(f"  {name:14s} {100 * hit:5.1f}% of clips   {100 * cov:5.1f}% of loud frames")
        # Some AMI headsets are quieter than the normal threshold (as they were
        # for the loudness gate before): 'high' is the setting for those.
        for name, need in (("high", 0.98), ("normal", 0.95)):
            hit = sum(h for h, _ in got[name]) / len(got[name])
            if hit < need:
                failures.append(f"AMI: only {100 * hit:.1f}% of clips queued at sensitivity {name}")
else:
    print("\n(d) AMI: skipped (pass --ami path/to/0.parquet)")

print("\n(f) quiet voices that aren't the participant: b.wav scaled down, seconds queued")
b, _ = sf.read(HERE / "site" / "b.wav", dtype="float32")
for level in (0.0025, 0.004, 0.006):
    x = np.concatenate([np.zeros(SR, np.float32), rms_to(b, level), np.zeros(SR, np.float32)])
    row = []
    for sens, vt in S.VAD_SENSITIVITY.items():
        utts, _ = segment(x, vad=vt, threshold=THRESHOLD * S.SENSITIVITY[sens])
        sec = sum(len(u[1]) for u in utts) / SR
        row.append(f"{sens} {sec:.1f}s")
        if sec and (sens == "low" or (sens == "normal" and level < 0.005)):
            failures.append(f"quiet voice at rms {level}: {sec:.1f}s queued at sensitivity {sens}")
    print(f"  rms {level:.4f} ({20 * np.log10(level):.0f} dBFS): " + " | ".join(row))
own, other = rms_to(sf.read(site[0], dtype="float32")[0], 0.03), rms_to(sf.read(HERE / "site" / "c.wav", dtype="float32")[0], 0.003)
x = np.concatenate([np.zeros(SR, np.float32), own, other, np.zeros(SR, np.float32)])
utts, _ = segment(x)
mask = covered_mask(len(x), utts)
a0 = SR + len(own)
glued = mask[a0:a0 + len(other)].sum() / SR
print(f"  own voice (rms 0.03) then straight into a quiet one (0.003): own {100 * mask[SR:a0].mean():.0f}% kept, "
      f"{glued:.1f}s of {len(other) / SR:.1f}s of the quiet voice glued on")
if mask[SR:a0].mean() < 0.95 or glued > 0.9:          # 0.7 s of trailing silence always ends a line, plus a chunk
    failures.append(f"quiet voice glued onto the participant's line: {glued:.1f}s")

print("\n(g) a talk longer than the cut (max 3 s here, 25 s in the server): the words after the cut are kept")
loud = np.flatnonzero(np.sqrt(np.convolve(own ** 2, np.ones(512) / 512, "same")) > 0.01)
start = loud[0]
for after in (0.1, 0.15, 0.2, 0.25, 0.3):                # this much speech after the cut, then silence
    end = start + int((3.0 + after) * SR)                # (the cut lands 3 s into the line)
    x = np.concatenate([np.zeros(SR, np.float32), own[start:end], np.zeros(SR, np.float32)])
    utts, seg = segment(x, max_ms=3000)
    m = covered_mask(len(x), utts)[SR:SR + end - start]
    frames = loud[(loud >= start + int(2.9 * SR)) & (loud < end)] - start   # the loud audio around and after the cut
    tail = m[frames].mean() if len(frames) else 1.0
    print(f"  {int(after * 1000):3d} ms past the cut: {len(utts)} utterance(s), {100 * tail:.0f}% of the speech "
          f"around and after the cut kept, {seg.dropped} dropped")
    if tail < 0.95:
        failures.append(f"long talk: {100 * tail:.0f}% of the speech after the cut kept ({int(after * 1000)} ms past the cut)")

print("\n(e) cost: Silero per 200 ms chunk (CPU, one thread)")
# In the server this runs on the asyncio thread that also serves the
# websockets and the dashboard, so it is time that thread is busy, not a
# share of spare cores.
for name, x in (("talking (white noise, rms 0.03)", white(20)), ("open mic, quiet fan (rms 0.003)", rms_to(fan(20), 0.003))):
    seg = S.Segmenter(queue.Queue(), THRESHOLD, "cost")
    t = time.perf_counter()
    for i in range(0, len(x), CHUNK):
        seg.push(i / 16, x[i:i + CHUNK])
    ms = (time.perf_counter() - t) / (len(x) / CHUNK) * 1000
    print(f"  {name:32s} {ms:.2f} ms per chunk per person = {ms / 2:.2f}% of the server's event-loop thread")

# A scrap of quiet sound long before someone speaks must not become the start of
# their line (a line once "lasted" 39 minutes this way).
import queue as _q
_out = _q.Queue(); _seg = S.Segmenter(_out, 0.01, "late", vad_threshold=None)
_quiet = np.zeros(1600, np.float32) + 0.001
_loud = (np.sin(np.arange(1600) / 5) * 0.3).astype(np.float32)
_seg.push(1_000_000, _quiet)
_t1 = 1_000_000 + 39 * 60_000
for i in range(20): _seg.push(_t1 + i * 100, _loud)
for i in range(12): _seg.push(_t1 + 2000 + i * 100, _quiet)
_seg.flush()
_starts = []
while not _out.empty(): _starts.append(_out.get()[0])
if not _starts or any(abs(x - _t1) > 1000 for x in _starts):
    failures.append(f"old lead-in glued to a line: starts {[round((x - _t1) / 1000) for x in _starts]} s from the speech")
else:
    print("  a line starts where its speech does, not at old sound before a long pause: ok")

print("\nFAIL:\n  " + "\n  ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
