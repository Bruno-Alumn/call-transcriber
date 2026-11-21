"""Earshot is English only: both Granite models must write English text (plain
ASCII plus the few symbols English uses) whatever language is spoken, even when
no language is passed in, and the Language setting can't be changed.
  cd server && uv run python ../test/force_english_test.py --engine mlx   (Mac: makes speech with `say`)"""
import argparse, json, os, subprocess, sys, tempfile, numpy as np
sys.argv, args = sys.argv[:1], argparse.ArgumentParser().parse_known_args()[1]
sys.path.insert(0, ".")
import server as S
eng = args[args.index("--engine") + 1] if "--engine" in args else "mlx"
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok: fails.append(what)

# the setting is pinned to English
check(S.SETTING_CHOICES["language"] == ["en"], "Language setting offers English only")
old = S.load_settings
S.load_settings = lambda: {"language": "fr"}
check(S.current_settings()["language"] == "en", "an old settings.json saying French still reads as English")
S.load_settings = old

SAY = [("Eddy (Spanish (Spain))", "Hola a todos, ¿qué tal estáis? Mañana vamos a la reunión con el equipo de diseño."),
       ("Eddy (French (France))", "Bonjour à tous, je pense que la réunion de demain est très importante pour l'équipe."),
       ("Anna", "Guten Morgen, wir müssen über die Größe der neuen Datei sprechen, das ist wichtig."),
       ("Eddy (Japanese (Japan))", "こんにちは、明日の会議について話しましょう。新しいバージョンはとても速いです。"),
       ("Eddy (Chinese (China mainland))", "大家好，我们明天开会讨论新的版本，这个很重要。"),
       ("Samantha", "Okay everyone, the build passed, so let's ship the release on Thursday.")]
def clip(voice, text):
    d = tempfile.mkdtemp()
    aiff, raw = os.path.join(d, "a.aiff"), os.path.join(d, "a.pcm")
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, raw], check=True)
    a = np.frombuffer(open(raw, "rb").read()[44:], dtype=np.int16).astype(np.float32) / 32768
    return a
english_ok = lambda t: all(ord(c) < 128 or c in S.ENGLISH_SYMBOLS for c in t)
# lines in another language written with plain letters are dropped; English with a few foreign words stays
for t, want in [("Hola todos, que tal estis? Maana vamos a la reunion con el equipo.", True),
                ("Bonjour a tous, je pense que la reunion de demain est importante.", True),
                ("Guten Morgen, wir mussen uber die Grosse der neuen Datei sprechen.", True),
                ("I went to La Paz with Jose and we ate tacos.", False), ("Yep, Marooney spawn de de.", False),
                ("The meeting is at noon, de facto.", False), ("Okay everyone, the build passed.", False),
                ("Guten Morgen, we mussten, um die Gruppe der neuen Datei sprechen, das is wichtig.", True),
                ("I went to La Paz with Jose de la Cruz.", False),
                ("Ja.", True), ("Es ist…", True), ("Oui", True), ("Bien.", True),
                ("Yo.", False), ("Okay.", False), ("No.", False), ("So", False), ("Ha ha ha!", False), ("Um…", False),
                ("Herr Prager, Herr Kommissar, liebe Kolleginnen, liebe Kollegen!", True), ("Es un buen momento.", True),
                ("Parece que.", True), ("Nicht genug.", True), ("He's from Uganda.", False), ("That van is OP, man.", False)]:
    check(S.not_english(t) == want, f"{'dropped' if want else 'kept'}: {t!r}")
clips = [(v, clip(v, t)) for v, t in SAY]
for mid in ["granite-470m", "granite-2b"]:
    repo = next(m for m in S.MODELS if m["id"] == mid)["repos"][eng]
    run = S.load_engine(eng, repo, None)              # no language given: English must still be forced
    print(f"{mid} ({eng})")
    for v, a in clips:
        out = run(a)
        text = out["text"] if isinstance(out, dict) else out
        print(f"    {v.split(' ')[0] if '(' not in v else v.split('(')[1].split(' ')[0]:9} -> {text!r}")
        check(english_ok(text), f"{mid}: {v} written in English characters only")
    eng_text = (run(clips[-1][1]) or {})
    eng_text = eng_text["text"] if isinstance(eng_text, dict) else eng_text
    check("thursday" in eng_text.lower() and "release" in eng_text.lower(), f"{mid}: English speech still transcribed: {eng_text!r}")
    del run
    import gc; gc.collect()
    if eng == "mlx":
        import mlx.core as mx; mx.clear_cache()
print("FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
