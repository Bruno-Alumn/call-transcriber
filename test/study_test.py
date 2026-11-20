"""Study mode test: runs with no model (FakeLabeller) on a fictional server.

Builds synthetic transcripts in a temporary directory (one fictional server with
ten fictional people who talk in deliberately different styles, one excluded
person, plus noise lines that must be skipped), then checks the mechanisms
listed in study/SPEC.md "Tests". Prints PASS/FAIL per check and exits non-zero
if any check fails.

Run from the repo root:
  server/.venv/bin/python test/study_test.py            # Mac
  server\\.venv\\Scripts\\python.exe test\\study_test.py  # Windows
"""
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fictional server
# ---------------------------------------------------------------------------

GUILD = "700000000000000001"
GUILD_NAME = "Hollowmere Tavern"
CH = "800000000000000001"
CH_NAME = "hearthside"
OTHER_GUILD = "700000000000000002"

# Each person: user id, display name, relative turn weight, sentence pool.
# Display names are unusual on purpose so a substring search can't hit by chance.
PEOPLE = {
    "swearer": ("900000000000000101", "Brannoc Thistlewhite", 1.0, [
        "what the fuck is this shit, damn it",
        "this is so fucking stupid, i hate this crap",
        "oh hell no, that was bullshit and you know it",
        "damn that guy is an idiot, screw him",
        "shit shit shit, i am so pissed off right now",
        "god damn it, this stupid game is garbage",
    ]),
    "warm": ("900000000000000102", "Odalys Fernwick", 1.0, [
        "thank you so much, that was really lovely of you",
        "i love playing with you all, you are wonderful",
        "great job everyone, that was awesome and so fun",
        "aw that is so sweet, i really appreciate it",
        "you are all amazing friends, i am so happy tonight",
        "nice work, glad you are here with us, love it",
    ]),
    "questioner": ("900000000000000103", "Pelleas Quorvane", 1.0, [
        "what do you think we should do next?",
        "why did that happen, does anyone know?",
        "how does this mechanic actually work?",
        "where are we going after this round?",
        "can someone explain what the plan is?",
        "who is playing tomorrow night then?",
    ]),
    "interrupter": ("900000000000000104", "Saffi Grendlemoor", 1.0, [
        "no wait wait listen to me for a second",
        "hold on hold on that is not what happened",
        "actually let me just say something here",
        "no no the point is we go left not right",
        "sorry but i have to jump in on this one",
    ]),
    "anxious": ("900000000000000105", "Tamsin Vellacourt", 1.0, [
        "i am really worried about this, i feel nervous",
        "i'm scared i will mess it up again, i always do",
        "i feel sad and lonely today honestly",
        "i'm afraid my connection will drop, i am anxious",
        "i don't know, i feel kind of upset and tired",
    ]),
    "talker": ("900000000000000106", "Jorund Ashbellow", 2.0, [
        "so we all went out together last weekend and everyone had a great time, "
        "the whole group was there and we talked for hours about everything",
        "okay everyone listen, we should all meet up together on saturday, "
        "our team is the best and we always have fun when we hang out",
        "so the story is that we went to the lake and our friends brought food "
        "and we stayed until midnight just talking and laughing",
    ]),
    "tentative": ("900000000000000107", "Ilsabet Corrowan", 1.0, [
        "maybe we could try that, i guess, not sure",
        "perhaps it is probably fine, i think, kinda",
        "i guess it might work, sorta, idk really",
        "maybe, i am unsure, probably not though i think",
    ]),
    "gamer": ("900000000000000108", "Wendel Haskrigg", 1.0, [
        "push the left lane and reload before the boss",
        "heal me now, i am low, the tank is down",
        "grab the loot and respawn at the second checkpoint",
        "use your ultimate on the boss when it is staggered",
    ]),
    "joker": ("900000000000000109", "Merrit Oakhollow", 1.0, [
        "haha that is hilarious, lol you guys are clowns",
        "lmao imagine being that bad at a video game",
        "ha ha okay that one was actually funny",
        "lol did you just walk into the wall again",
    ]),
    "quiet": ("900000000000000110", "Caddoc Ferrisdale", 0.6, [
        "yeah sounds good",
        "okay sure thing",
        "right, makes sense",
        "yep i am here",
    ]),
}
# The excluded person. Every line carries a marker word so a leak is easy to find.
EXCLUDED = ("900000000000000999", "Ysolde Quillonmere", 1.0, [
    "zarquonblue is my favourite colour honestly",
    "i said zarquonblue three times already today",
    "the zarquonblue team should go first this round",
])
EXCLUDED_MARKER = "zarquonblue"


def make_transcripts(tdir: Path) -> list[dict]:
    """Write three days of transcripts, two calls a day, plus lines that must be
    skipped (other server, missing guild, empty user, hallucinations, one-word
    lines). Returns all raw rows of the target server (including the excluded
    person) for leak checks."""
    rng = random.Random(1234)
    pool = list(PEOPLE.values()) + [EXCLUDED]
    weights = [p[2] for p in pool]
    interrupter_id = PEOPLE["interrupter"][0]
    target_rows = []
    for day in range(3):
        date = f"2026-03-{10 + day:02d}"
        rows = []
        for start_hour in (19, 22):
            # Local time so hour-of-day features see the intended hours.
            t = int(time.mktime(time.strptime(f"{date} {start_hour}:00:00", "%Y-%m-%d %H:%M:%S")) * 1000)
            prev_end, prev_user, prev_t = t, None, t
            for _ in range(160):
                uid, name, _, lines = rng.choices(pool, weights)[0]
                if uid == prev_user:          # avoid long monologues by one speaker
                    continue
                text = rng.choice(lines)
                words = len(text.split())
                if uid == interrupter_id and prev_user is not None and rng.random() < 0.8:
                    # Barge in well before the previous speaker finishes.
                    start = max(prev_t + 200, prev_end - rng.randint(700, 1500))
                else:
                    start = prev_end + rng.randint(200, 1500)
                end = start + words * 320 + 300
                row = {"t": start, "end": end, "user": uid, "speaker": name, "text": text,
                       "ch": CH, "chName": CH_NAME, "guild": GUILD, "guildName": GUILD_NAME}
                rows.append(row)
                target_rows.append(row)
                prev_end, prev_user, prev_t = max(prev_end, end), uid, start
            # Noise lines of the target server that must be dropped by load_lines.
            base = prev_end + 5000
            rows.append({"t": base, "end": base + 500, "user": PEOPLE["quiet"][0], "speaker": "Caddoc Ferrisdale",
                         "text": "Thank you.", "ch": CH, "chName": CH_NAME, "guild": GUILD, "guildName": GUILD_NAME})
            rows.append({"t": base + 600, "end": base + 900, "user": PEOPLE["quiet"][0], "speaker": "Caddoc Ferrisdale",
                         "text": "yes", "ch": CH, "chName": CH_NAME, "guild": GUILD, "guildName": GUILD_NAME})
        # Lines that can't be attributed or belong elsewhere.
        t0 = rows[0]["t"]
        rows.append({"t": t0 + 1, "end": t0 + 900, "user": "900000000000000555", "speaker": "Outsider Nobody",
                     "text": "this line belongs to another server entirely", "ch": "800000000000000009",
                     "chName": "elsewhere", "guild": OTHER_GUILD, "guildName": "Other Place"})
        rows.append({"t": t0 + 2, "end": t0 + 900, "user": "900000000000000556", "speaker": "Old Format",
                     "text": "an old line with no guild or channel at all"})
        rows.append({"t": t0 + 3, "end": t0 + 900, "user": "", "speaker": "Nameless",
                     "text": "a very old line with an empty user id", "ch": CH, "chName": CH_NAME,
                     "guild": GUILD, "guildName": GUILD_NAME})
        # Files are append-order, not necessarily sorted: shuffle a little.
        rng.shuffle(rows)
        with open(tdir / f"{date}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return target_rows


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    """Decorator: run the check, record PASS/FAIL; an exception is a FAIL."""
    def wrap(fn):
        try:
            ok, detail = fn()
        except Exception as e:  # a module missing or raising is a failure, not a crash
            ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        RESULTS.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return fn
    return wrap


def key_of(row) -> str:
    """Line.key as defined in the spec."""
    return hashlib.sha1(f"{row['t']}|{row['user']}|{row['text']}".encode()).hexdigest()[:16]


def venv_python() -> str:
    """The server venv's python (Mac or Windows layout), else the current one."""
    for rel in ("server/.venv/bin/python", "server/.venv/Scripts/python.exe"):
        if (ROOT / rel).exists():
            return str(ROOT / rel)
    return sys.executable


def run_cli(*args, timeout=900):
    return subprocess.run([venv_python(), "-m", "study", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="study_test_"))
    tdir = tmp / "transcripts"
    tdir.mkdir()
    out = tmp / "out"
    raw = make_transcripts(tdir)
    excl_file = tmp / "exclude.txt"
    excl_file.write_text("# people who asked not to be studied\n" + EXCLUDED[0] + "\n", encoding="utf-8")
    excluded_keys = {key_of(r) for r in raw if r["user"] == EXCLUDED[0]}
    included_keys = {key_of(r) for r in raw if r["user"] != EXCLUDED[0]}
    print(f"synthetic data in {tdir} ({len(raw)} server lines, {len(excluded_keys)} from the excluded person)")

    # Modules are imported inside each check, so a missing or broken module
    # fails only the checks that need it.
    import importlib
    def mod(name):
        return importlib.import_module(f"study.{name}")

    # -- replies / overlaps on a hand-built example ---------------------------
    @check("replies and overlaps on a hand-built example")
    def _():
        data = mod("data")
        def L(t, end, user, text="some words here"):
            return data.Line(t=t, end=end, user=user, speaker=user, text=text, ch="c1", ch_name="c",
                             guild="g", guild_name="G")
        lines = [
            L(0, 2000, "A"),          # A speaks
            L(3000, 5000, "B"),       # B replies 1000 ms after A ends
            L(4000, 6000, "A"),       # A barges in 1000 ms before B ends: overlap + reply with latency -1000
            L(20000, 22000, "C"),     # 14 s after the last line: not a reply
            L(21900, 23000, "D"),     # D starts 100 ms before C ends: reply (-100) but too short for an overlap
        ]
        for i, ln in enumerate(lines):
            ln.idx = i
        data.group_calls(lines)
        reps = [tuple(r) for r in data.replies(lines)]
        ovs = [tuple(o) for o in data.overlaps(lines)]
        want_reps = {("B", "A", 1000), ("A", "B", -1000), ("D", "C", -100)}
        problems = []
        if set(reps) != want_reps:
            problems.append(f"replies {reps} != {sorted(want_reps)}")
        if set(ovs) != {("A", "B")}:
            problems.append(f"overlaps {ovs} != [('A', 'B')]")
        return not problems, "; ".join(problems)

    # -- kappa ----------------------------------------------------------------
    @check("kappa: identical -> 1, textbook pair -> 0.4, chance -> ~0, empty -> nan")
    def _():
        agreement = mod("agreement")
        a = ["x", "y", "z", "x", "y"] * 10
        k1 = agreement.kappa(a, list(a))
        # 50 items: both yes 20, A yes/B no 5, A no/B yes 10, both no 15 -> po .7, pe .5, kappa .4
        ka = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
        kb = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
        k2 = agreement.kappa(ka, kb)
        rng = random.Random(7)
        cats = ["positive", "neutral", "negative"]
        ra = [rng.choice(cats) for _ in range(5000)]
        rb = [rng.choice(cats) for _ in range(5000)]
        k3 = agreement.kappa(ra, rb)
        k4 = agreement.kappa([], [])
        ok = abs(k1 - 1) < 1e-9 and abs(k2 - 0.4) < 1e-6 and abs(k3) < 0.05 and math.isnan(k4)
        return ok, f"identical={k1:.3f} textbook={k2:.3f} chance={k3:.3f} empty={k4}"

    # -- ridge calibration ----------------------------------------------------
    @check("ridge LOO r > 0.9 when truth is linear in the features")
    def _():
        traits = mod("traits")
        rng = np.random.default_rng(3)
        feats = ["f_a", "f_b", "f_c", "f_d"]
        n = 40
        X = rng.normal(size=(n, len(feats)))
        people = {f"p{i:02d}": {f: float(X[i, j]) for j, f in enumerate(feats)} for i in range(n)}
        coefs = rng.normal(size=(len(traits.TRAITS), len(feats)))
        truth = {}
        for i in range(n):
            truth[f"p{i:02d}"] = {tr: float(3 + X[i] @ coefs[k] + rng.normal(scale=0.05))
                                  for k, tr in enumerate(traits.TRAITS)}
        cal = traits.fit_calibration(people, truth, feats, alpha=1.0)
        loo = cal["loo_r"]
        ok = all(loo[tr] > 0.9 for tr in traits.TRAITS) and cal["n"] == n
        # And it must refuse with fewer than 8 users.
        few = {u: people[u] for u in list(people)[:5]}
        try:
            traits.fit_calibration(few, truth, feats)
            ok, refused = False, "did not refuse 5 users"
        except ValueError:
            refused = "refuses 5 users"
        return ok, ", ".join(f"{tr[:4]}={loo[tr]:.3f}" for tr in traits.TRAITS) + f", n={cal['n']}, {refused}"

    # -- in-process pipeline: load, label (fake), features, prior scores --------
    pipe = {}

    @check("load_lines drops the excluded person, other servers and noise")
    def _():
        data = mod("data")
        excl = data.load_exclude(excl_file)
        lines = data.load_lines(tdir, GUILD, exclude=excl)
        data.group_calls(lines)
        pipe["lines"] = lines
        users = {ln.user for ln in lines}
        problems = []
        if excl != {EXCLUDED[0]}:
            problems.append(f"load_exclude -> {excl}")
        if EXCLUDED[0] in users:
            problems.append("excluded user loaded")
        if any(EXCLUDED_MARKER in ln.text for ln in lines):
            problems.append("excluded text loaded")
        if any(ln.guild != GUILD for ln in lines) or "" in users:
            problems.append("foreign or unattributed lines loaded")
        if any(ln.text.strip().lower() in ("thank you.", "yes") for ln in lines):
            problems.append("hallucination / one-word lines loaded")
        if [ln.t for ln in lines] != sorted(ln.t for ln in lines):
            problems.append("not sorted by t")
        if users != {p[0] for p in PEOPLE.values()}:
            problems.append(f"users {len(users)} != {len(PEOPLE)}")
        if data.load_exclude(tmp / "missing.txt") != set():
            problems.append("missing exclude file should give an empty set")
        return not problems, "; ".join(problems) or f"{len(lines)} lines, {len(users)} people"

    def labels_for(lines):
        """FakeLabeller labels keyed by line key (no cache needed in-process)."""
        from study.labels import FakeLabeller
        lab = FakeLabeller()
        items, prev = [], ""
        for ln in lines:
            items.append((ln.key, prev, ln.text))
            prev = ln.text
        return dict(zip((i[0] for i in items), lab.label(items)))

    def score_lines(lines, labels):
        """Prior trait scores for a subset of lines."""
        data, features, traits = mod("data"), mod("features"), mod("traits")
        people = features.person_features(lines, labels, data.replies(lines), data.overlaps(lines))
        return traits.prior_scores(people)

    @check("prior directions: heavy swearer < warm person on agreeableness")
    def _():
        lines = pipe["lines"]
        labels = labels_for(lines)
        pipe["labels"] = labels
        scores = score_lines(lines, labels)
        sw, wa = PEOPLE["swearer"][0], PEOPLE["warm"][0]
        a_sw, a_wa = scores[sw]["agreeableness"], scores[wa]["agreeableness"]
        return a_sw < a_wa, f"swearer={a_sw:+.2f} warm={a_wa:+.2f}"

    @check("split-half returns finite values for every trait")
    def _():
        lines, labels = pipe["lines"], pipe["labels"]
        traits = mod("traits")
        sh = traits.split_half(lambda ls: score_lines(ls, labels), lines)
        missing = [tr for tr in traits.TRAITS if tr not in sh]
        bad = [tr for tr in traits.TRAITS if tr in sh and not math.isfinite(sh[tr])]
        return not missing and not bad, ", ".join(f"{tr[:4]}={sh.get(tr, float('nan')):+.2f}"
                                                  for tr in traits.TRAITS)

    # -- the CLI end to end ---------------------------------------------------
    @check("`guilds` command lists the fictional server")
    def _():
        r = run_cli("guilds", "--transcripts", str(tdir), timeout=120)
        return r.returncode == 0 and GUILD in r.stdout, f"exit {r.returncode}" + (
            f": {r.stderr.strip()[-400:]}" if r.returncode else "")

    cli = {}

    @check("`run --engine fake` completes end to end")
    def _():
        t0 = time.time()
        r = run_cli("run", "--guild", GUILD, "--transcripts", str(tdir), "--exclude", str(excl_file),
                    "--out", str(out), "--engine", "fake", "--min-words", "100")
        cli["ok"] = r.returncode == 0
        missing = [f for f in ("report.md", "people.csv", "server.json", "labels.jsonl") if not (out / f).exists()]
        detail = f"exit {r.returncode} in {time.time() - t0:.0f}s"
        if missing:
            detail += f", missing {missing}"
        if r.returncode:
            detail += f": {r.stderr.strip()[-800:]}"
        return r.returncode == 0 and not missing, detail

    @check("excluded person absent from every output file and the label cache")
    def _():
        files = [p for p in out.rglob("*") if p.is_file()]
        if not files:
            return False, "no output files"
        needles = [EXCLUDED[0], EXCLUDED[1], EXCLUDED_MARKER] + sorted(excluded_keys)
        leaks = []
        for p in files:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for n in needles:
                if n in txt:
                    leaks.append(f"{p.name}: {n[:20]}")
        cache = out / "labels.jsonl"
        if not cache.exists():
            leaks.append("labels.jsonl missing, can't verify the cache")
        else:
            keys = {json.loads(l)["key"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}
            if keys & excluded_keys:
                leaks.append(f"{len(keys & excluded_keys)} excluded keys in cache")
            # Guard against a vacuous pass: the cache must hold the included lines' labels.
            if not keys:
                leaks.append("label cache is empty")
            elif not keys <= included_keys:
                leaks.append(f"{len(keys - included_keys)} cache keys match no included line")
        return not leaks, "; ".join(leaks[:5]) or f"{len(files)} files clean"

    @check("report.md contains no per-person names or user ids")
    def _():
        rep = out / "report.md"
        if not rep.exists():
            return False, "report.md missing"
        txt = rep.read_text(encoding="utf-8")
        found = []
        for uid, name, _, _ in PEOPLE.values():
            for n in (uid, name, name.split()[0], name.split()[-1]):
                if n in txt:
                    found.append(n)
        return not found, ", ".join(found)

    @check("people.csv has one row per included person")
    def _():
        pc = out / "people.csv"
        if not pc.exists():
            return False, "people.csv missing"
        with open(pc, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        expected = {p[0] for p in PEOPLE.values()}
        # Column should be "user"; if named differently, find the id in any cell.
        users = set()
        for r in rows:
            if r.get("user"):
                users.add(r["user"])
            else:
                users |= {v for v in r.values() if v in expected or v == EXCLUDED[0]}
        return users <= expected and len(users) >= 5, f"{len(users)} people"

    # -- checks added with the review fixes --------------------------------------
    @check("split-half keeps every call whole (all its speakers in the same half)")
    def _():
        traits = mod("traits")
        # Everyone attends every synthetic call, so drop one person from the first
        # call: with per-person parity their later calls would then flip halves.
        first_call = pipe["lines"][0].call
        quiet = PEOPLE["quiet"][0]
        lines = [ln for ln in pipe["lines"] if not (ln.call == first_call and ln.user == quiet)]
        halves = []

        def spy(subset):
            halves.append({ln.call for ln in subset})
            return score_lines(subset, pipe["labels"])

        traits.split_half(spy, lines)
        if len(halves) != 2:
            return False, f"score_fn called {len(halves)} times"
        both = halves[0] & halves[1]
        return not both, f"{len(both)} calls split across halves" if both else f"{len(halves[0])}+{len(halves[1])} calls"

    @check("social-only: turn-taking rates use one line set for numerator and denominator")
    def _():
        features = mod("features")
        lines, labels = pipe["lines"], pipe["labels"]
        full = features.features_for(lines, labels)
        soc = features.features_for(lines, labels, social=True)
        bad = []
        for u, f in soc.items():
            for k in ("replies_given_per_100_lines", "interrupts_per_100_lines"):
                a, b = f[k], full[u][k]
                if not (math.isclose(a, b, rel_tol=1e-9) or (math.isnan(a) and math.isnan(b))):
                    bad.append(f"{u[-3:]}:{k[:7]} {a:.0f} vs {b:.0f}")
        return not bad, "; ".join(bad[:4]) or f"{len(soc)} people consistent"

    @check("exclusion file with BOM / UTF-16 still excludes; malformed ids are refused")
    def _():
        data = mod("data")
        p1 = tmp / "ex_bom.txt"; p1.write_bytes(b"\xef\xbb\xbf" + EXCLUDED[0].encode() + b"\r\n")
        p2 = tmp / "ex_u16.txt"; p2.write_bytes((EXCLUDED[0] + "\r\n").encode("utf-16"))
        p3 = tmp / "ex_mention.txt"; p3.write_text(f"<@!{EXCLUDED[0]}>  # a pasted mention\n", encoding="utf-8")
        bad = [p.name for p in (p1, p2, p3) if data.load_exclude(p) != {EXCLUDED[0]}]
        p4 = tmp / "ex_bad.txt"; p4.write_text(f"{EXCLUDED[0]},\nnot-an-id\n", encoding="utf-8")
        try:
            data.load_exclude(p4)
            bad.append("malformed entry accepted")
        except ValueError:
            pass
        return not bad, ", ".join(bad)

    @check("late exclusion also scrubs an earlier sample.csv (rows and 'previous' text)")
    def _():
        o = tmp / "out_late"
        common = ["--guild", GUILD, "--transcripts", str(tdir), "--out", str(o)]
        r1 = run_cli("run", *common, "--engine", "fake", "--min-words", "10", "--boot", "5")
        r2 = run_cli("sample", *common, "--n", "500")
        before = (o / "sample.csv").read_text(encoding="utf-8-sig") if (o / "sample.csv").exists() else ""
        r3 = run_cli("run", *common, "--exclude", str(excl_file), "--engine", "fake", "--min-words", "10",
                     "--boot", "5")
        txt = (o / "sample.csv").read_text(encoding="utf-8-sig") if (o / "sample.csv").exists() else ""
        if any(r.returncode for r in (r1, r2, r3)):
            return False, f"exits {r1.returncode}/{r2.returncode}/{r3.returncode}: {r3.stderr.strip()[-300:]}"
        if EXCLUDED_MARKER not in before:
            return False, "test setup: marker not in the first sample"
        leaked = [k for k in excluded_keys if k in txt]
        problems = []
        if EXCLUDED_MARKER in txt:
            problems.append("excluded text still in sample.csv")
        if leaked:
            problems.append(f"{len(leaked)} excluded keys still in sample.csv")
        if not txt:
            problems.append("sample.csv was removed instead of rewritten")
        return not problems, "; ".join(problems)

    @check("label parsing maps near-synonyms; mlx schema constrains every key to its list")
    def _():
        labels = mod("labels")
        problems = []
        for raw, key, want in (("command", "act", "request"), ("instruction", "act", "request"),
                               ("disappointment", "emotion", "sadness"), ("frustration", "emotion", "anger"),
                               ("nervous", "emotion", "fear")):
            got = labels._clean_value(raw, labels.LABELS[key])
            if got != want:
                problems.append(f"{raw}->{got}")
        schema = labels.label_schema()
        props = schema.get("properties", {})
        if set(schema.get("required", [])) != set(labels.LABELS) or schema.get("additionalProperties") is not False:
            problems.append("schema keys")
        if any(props.get(k, {}).get("enum") != v for k, v in labels.LABELS.items()):
            problems.append("schema enums")
        # The derailed answer from the review must now parse.
        ans = '{"sentiment": "neutral", "act": "command", "emotion": "none", "tone": "neutral", "context": "gameplay"}'
        if labels.parse_answer(ans) is None:
            problems.append("synonym answer not parsed")
        return not problems, "; ".join(problems)

    @check("bootstrap keeps calls whole; few-call people get no interval; segments are one reply")
    def _():
        data, traits = mod("data"), mod("traits")
        lines, labels = pipe["lines"], pipe["labels"]
        problems = []
        # Cluster bootstrap: every replicate call holds all of its original speakers.
        speakers = {}
        for ln in lines:
            speakers.setdefault(ln.call, set()).add(ln.user)
        by_user_call = {}
        for ln in lines:
            by_user_call.setdefault(ln.user, {}).setdefault(ln.call, []).append(ln)
        broken = []

        def spy(subset):
            seen = {}
            for ln in subset:
                seen.setdefault(ln.call, set()).add(ln.user)
            broken.extend(c for c, us in seen.items() if us != speakers[c.split("#b")[0]])
            return score_lines(subset, labels)

        cis = traits.bootstrap_ci(by_user_call, spy, n=5, seed=1)
        if broken:
            problems.append(f"{len(broken)} replicate calls lost speakers")
        if not all(math.isfinite(cis[u]["agreeableness"][0]) for u in cis):
            problems.append("6-call people should get finite intervals")
        few = traits.bootstrap_ci(by_user_call, spy, n=3, seed=1, min_calls=7)
        if any(math.isfinite(v[0]) for d in few.values() for v in d.values()):
            problems.append("people below min_calls got an interval")
        # Two Whisper segments of one turn by B are one reply, not two.
        def L(t, end, user):
            return data.Line(t=t, end=end, user=user, speaker=user, text="a b c", ch="c", ch_name="c",
                             guild="g", guild_name="G")
        seg = [L(0, 2000, "A"), L(2500, 3500, "B"), L(3600, 4500, "B")]
        data.group_calls(seg)
        if data.replies(seg) != [("B", "A", 500)]:
            problems.append(f"segmented turn -> {data.replies(seg)}")
        return not problems, "; ".join(problems)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed; output kept in {tmp}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
