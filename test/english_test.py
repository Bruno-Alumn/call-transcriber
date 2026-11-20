"""English-only test for Granite 2B: with the Language setting on English, the
speech LLM may only write plain-ASCII tokens (no other scripts or accented
letters; it does not translate, so French speech still comes out as French
words without accents), while end-of-text and other special tokens stay allowed.
2B also may write the few symbols English uses (£ € ° curly quotes, dashes), and
when speech in another script makes it repeat itself it stops early and writes
nothing instead of thousands of characters of junk.
  cd server && uv run python ../test/english_test.py            (tokenizer only, quick)
  cd server && uv run python ../test/english_test.py --engine mlx|torch
     also runs the 2B model on IBM's multilingual sample (non-English speech):
     without a language it writes accented French, with English only ASCII;
     on a Mac also Japanese speech made with `say` (skipped elsewhere)."""
import argparse, glob, os, sys, numpy as np
sys.argv, args = sys.argv[:1], argparse.ArgumentParser().parse_known_args()[1]
sys.path.insert(0, ".")
import server as S
from transformers import AutoTokenizer
REPO = "ibm-granite/granite-speech-5.0-2b"
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        fails.append(what)

tok = AutoTokenizer.from_pretrained(REPO)
m = S.english_mask(tok, len(tok))
check(0.5 * len(m) < m.sum() < len(m), f"{m.sum()} of {len(m)} tokens allowed")
check(bool(m[tok.eos_token_id]), "end-of-text allowed")
for w in [" hello", " Hello,", " don't", " 42.", "?"]:
    check(all(m[i] for i in tok.encode(w, add_special_tokens=False)), f"{w!r} allowed")
for w in ["é", "日本", "привет", "ß"]:
    check(not any(m[i] for i in tok.encode(w, add_special_tokens=False)), f"{w!r} forbidden")
m2 = S.english_mask(tok, len(tok), symbols=True)       # what 2B uses
for w in [" £20", " €5", " 20°", " it’s", " “yes”", " so —", " well…"]:
    check(all(m2[i] for i in tok.encode(w, add_special_tokens=False)), f"2B: {w!r} allowed")
for w in ["é", "日本", "привет", "ß", "\t", "\r\n"]:
    check(not any(m2[i] for i in tok.encode(w, add_special_tokens=False)), f"2B: {w!r} forbidden")
check(S.looping([5, 9] * 8) and S.looping([3] * 8) and not S.looping([1, 2, 3, 4, 5, 6, 7, 8] * 2), "repeat guard")

eng = args[args.index("--engine") + 1] if "--engine" in args else None
if eng:
    import soundfile as sf
    snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{REPO.replace('/', '--')}/snapshots/*/multilingual_sample.wav"))
    if not snap:
        from huggingface_hub import hf_hub_download
        snap = [hf_hub_download(REPO, "multilingual_sample.wav")]
    a, sr = sf.read(snap[0], dtype="float32"); a = a.mean(1) if a.ndim > 1 else a
    if sr != 16000:
        from math import gcd
        import scipy.signal as ss
        a = ss.resample_poly(a, 16000 // gcd(16000, sr), sr // gcd(16000, sr)).astype(np.float32)
    a = a[16000 * 15:]                       # from 15 s on the sample is French (before that, English)
    free = S.load_engine(eng, REPO, None)(a)
    print("  any language:", ascii(free[:120]))
    en = S.load_engine(eng, REPO, "en")(a)
    print("  English only:", ascii(en[:120]))
    check(any(ord(c) > 127 for c in free), "without a language the model writes non-ASCII here")
    check(en.strip() != "" and all(ord(c) < 128 for c in en), "with English it writes only ASCII")
    import shutil, subprocess, tempfile, time
    if shutil.which("say") and shutil.which("afconvert") and "Kyoko" in subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout:
        # Japanese speech with English only: this clip made 2B repeat "3 3 3 ..." for a minute
        d = tempfile.mkdtemp()
        subprocess.run(["say", "-v", "Kyoko", "-o", f"{d}/ja.aiff", "皆さん、こんにちは。今日の会議では来月の予算について話します。"
                        "それから新しいプロジェクトの計画も確認しましょう。"], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", f"{d}/ja.aiff", f"{d}/ja.wav"], check=True)
        ja = sf.read(f"{d}/ja.wav", dtype="float32")[0]
        run = S.load_engine(eng, REPO, "en")
        t = time.time(); out = run(ja); took = time.time() - t
        print(f"  Japanese, English only: {len(out)} chars in {took:.1f}s:", ascii(out[:80]))
        check(len(out) < 400, "Japanese speech with English only: no pages of repeated junk")
        if eng == "mlx":                     # (torch here may be on CPU, much slower either way)
            check(took < 6, "and it ends quickly")
    else:
        print("  (no `say` with a Japanese voice here: skipped the Japanese check)")
print("PASS" if not fails else "FAIL", eng or "tokenizer")
sys.exit(0 if not fails else 1)
