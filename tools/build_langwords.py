"""Builds server/langwords.txt.gz: how common each word is in English and in six
other languages (Zipf scale, from the wordfreq project's data), for telling
English lines from lines in another language. Run with wordfreq installed:
  python tools/build_langwords.py"""
import gzip
from pathlib import Path
from wordfreq import top_n_list, zipf_frequency
LANGS = ["en", "de", "es", "fr", "it", "pt", "nl"]
words = set()
for l in LANGS:
    words |= {w for w in top_n_list(l, 15000) if w.isalpha() or "'" in w}
rows = sorted(words)
out = Path(__file__).resolve().parent.parent / "server" / "langwords.txt.gz"
with gzip.open(out, "wt", encoding="utf-8") as f:
    f.write("# word " + " ".join(LANGS) + "  (Zipf frequency; data: wordfreq, https://github.com/rspeer/wordfreq)\n")
    for w in rows:
        f.write(w + " " + " ".join(f"{zipf_frequency(w, l):.1f}" for l in LANGS) + "\n")
print(len(rows), "words ->", out, out.stat().st_size // 1024, "KB")
