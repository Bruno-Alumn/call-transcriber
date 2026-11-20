"""Timing test: three sentences with short pauses in one utterance must come
back as three lines at the right times (each within 0.4 s of the truth).
  cd server && uv run python ../test/timing_test.py [--engine mlx|torch]"""
import argparse, sys, numpy as np
sys.argv, args = sys.argv[:1], argparse.ArgumentParser().parse_known_args()[1]
sys.path.insert(0, ".")
import soundfile as sf
import server as S
eng = args[args.index("--engine") + 1] if "--engine" in args else S.pick_engine("auto")
def wav(n):
    a, sr = sf.read(f"../test/site/{n}.wav", dtype="float32"); a = a.mean(1) if a.ndim > 1 else a
    if sr != 16000:
        from math import gcd
        import scipy.signal as ss
        a = ss.resample_poly(a, 16000 // gcd(16000, sr), sr // gcd(16000, sr)).astype(np.float32)
    return a
a, b, c = wav("a"), wav("b"), wav("c")
audio = np.concatenate([a, np.zeros(8000, np.float32), b, np.zeros(4000, np.float32), c])
truth = [0.0, (len(a) + 8000) / 16000, (len(a) + 8000 + len(b) + 4000) / 16000]
run = S.load_engine(eng, S.ENGINE_DEFAULTS[eng] if eng != "mlx" else "ibm-granite/granite-speech-5.0-470m-turboctc", "en")
r = run(audio)
parts = S.split_timed(r["pieces"]) if isinstance(r, dict) else S.split_by_text(r, len(audio) / 16000)
for p in parts:
    print("  %5.2f-%5.2f  %s" % (p[1], p[2], p[0]))
ok = len(parts) == 3 and all(abs(p[1] - t) < 0.4 for p, t in zip(parts, truth))
print("PASS" if ok else "FAIL", eng)
sys.exit(0 if ok else 1)
