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
Also: the loudness-only fallback when the model is missing, and the cost per
200 ms chunk.

  cd server && uv run python ../test/vad_test.py [--ami path/to/0.parquet] [--ami-clips 120]
"""
import argparse, io, os, queue, sys, time
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


def segment(x: np.ndarray, vad=0.5, threshold=THRESHOLD):
    """Run one person's stream through a Segmenter; returns (utterances, segmenter)."""
    q = queue.Queue()
    seg = S.Segmenter(q, threshold, "test", vad_threshold=vad)
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


def speech_recall(x, utts):
    """Share of the loud 32 ms frames of x (the speech) inside queued audio."""
    mask = np.zeros(len(x), bool)
    for a, b in covered(utts):
        mask[a:b] = True
    n = len(x) // 512
    fr = x[:n * 512].reshape(n, 512)
    loud = np.sqrt((fr ** 2).mean(1)) > THRESHOLD
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

print(f"\n(b2) blips: speech cut to 150 ms is dropped (< {S.VAD_MIN_VOICED_MS} ms voiced), a 400 ms word is kept")
a, _ = sf.read(site[0], dtype="float32")
for ms, keep in ((150, False), (400, True)):
    for start in (8000, 24000, 40000):
        x = np.concatenate([np.zeros(SR, np.float32), a[start:start + 16 * ms], np.zeros(SR, np.float32)])
        utts, seg = segment(x)
        print(f"  {ms} ms from {start / SR:.1f}s: {len(utts)} queued, {seg.dropped} dropped")
        if bool(utts) != keep:
            failures.append(f"{ms} ms blip at {start}: {'not ' if keep else ''}queued")

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
        got = {None: [], 0.35: [], 0.5: [], 0.65: []}
        for r in rows:
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            a = a.mean(1) if a.ndim > 1 else a
            x = np.concatenate([np.zeros(SR, np.float32), a, np.zeros(SR, np.float32)])
            for vt in got:
                utts, _ = segment(x, vad=vt)
                got[vt].append((bool(utts), speech_recall(x, utts)))
        print(f"\n(d) AMI meeting clips ({len(rows)}, real headset audio): clips with speech queued, loud frames covered")
        for vt, res in got.items():
            name = "loudness only" if vt is None else f"VAD {vt:.2f}"
            hit = sum(h for h, _ in res) / len(res)
            cov = sum(c for _, c in res) / len(res)
            print(f"  {name:14s} {100 * hit:5.1f}% of clips   {100 * cov:5.1f}% of loud frames")
        hit = sum(h for h, _ in got[0.5]) / len(got[0.5])
        if hit < 0.98:
            failures.append(f"AMI: only {100 * hit:.1f}% of clips queued at normal sensitivity")
else:
    print("\n(d) AMI: skipped (pass --ami path/to/0.parquet)")

print("\n(e) cost: Silero per 200 ms chunk (one person, CPU, one thread)")
seg = S.Segmenter(queue.Queue(), THRESHOLD, "cost")
x = (white(20) + 0.0).astype(np.float32)
t = time.perf_counter()
for i in range(0, len(x), CHUNK):
    seg.push(i / 16, x[i:i + CHUNK])
ms = (time.perf_counter() - t) / (len(x) / CHUNK) * 1000
print(f"  {ms:.2f} ms per chunk -> one core keeps up with ~{int(200 / ms)} people talking at once")

print("\nFAIL:\n  " + "\n  ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
