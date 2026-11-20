"""Command line for study mode. See study/SPEC.md.

    python -m study guilds
    python -m study run --guild ID [--since D] [--until D] [--exclude FILE] [--out DIR]
                        [--engine auto|mlx|torch|fake] [--model REPO] [--min-words 1000] [--limit N]
                        [--social-only] [--questionnaire CSV] [--boot 200]
    python -m study sample --guild ID [--n 200] [--out DIR] [--exclude FILE] [--since D] [--until D]
    python -m study agree --sheet FILE

Every command also takes --transcripts DIR (default: transcripts/ in the repo root).

The sibling modules are imported inside each command, so `import study.__main__`
works even if one of them is broken; the error then shows up only for the
command that needs it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPTS = REPO_ROOT / "transcripts"
DEFAULT_OUT_BASE = Path(__file__).resolve().parent / "out"

PROGRESS_EVERY = 100   # print a progress line every this many newly labelled lines


def _out_dir(args) -> Path:
    return Path(args.out) if args.out else DEFAULT_OUT_BASE / str(args.guild)


def _read_exclude(path: str | None) -> set[str]:
    """Exclusion list. A path that was given but does not exist is an error:
    silently running without the list would leak the people it names."""
    from study.data import load_exclude
    if path is None:
        return set()
    p = Path(path)
    if not p.is_file():
        sys.exit(f"study: exclusion file not found: {p}")
    return load_exclude(p)


# ------------------------------------------------------------------- guilds
def cmd_guilds(args) -> int:
    from study.data import list_guilds
    rows = list_guilds(Path(args.transcripts))
    if not rows:
        print(f"No server-tagged lines found in {args.transcripts}")
        return 0
    print(f"{'server id':<22} {'lines':>8}  {'first':<10}  {'last':<10}  name")
    for r in rows:
        print(f"{str(r.get('guild')):<22} {r.get('lines', 0):>8}  {str(r.get('first', '')):<10}  "
              f"{str(r.get('last', '')):<10}  {r.get('guildName', '')}")
    return 0


# ---------------------------------------------------------------------- run
def _purge_cache(cache_path: Path, transcripts: Path, guild: str, since, until, exclude: set[str]) -> int:
    """Remove label-cache entries for lines by people now on the exclusion list.

    The cache stores only line hashes and labels, but a person added to the list
    after an earlier run must still disappear from it. Their line keys are
    recomputed from the raw transcripts (same formula as data.line_key) and the
    matching cache rows are dropped. Returns the number of rows removed.
    """
    if not exclude or not cache_path.exists():
        return 0
    from study.data import line_key, _transcript_files, _iter_records
    bad: set[str] = set()
    for _date, path in _transcript_files(transcripts, None, None):   # all dates: be thorough
        for rec in _iter_records(path):
            user = str(rec.get("user") or "")
            if user not in exclude:
                continue
            text = str(rec.get("text") or "").strip()
            try:
                t = int(rec.get("t"))
            except (TypeError, ValueError):
                continue
            bad.add(line_key(t, user, text))
    if not bad:
        return 0
    kept, removed = [], 0
    for row in cache_path.read_text(encoding="utf-8").splitlines():
        try:
            if json.loads(row).get("key") in bad:
                removed += 1
                continue
        except (json.JSONDecodeError, AttributeError):
            pass
        kept.append(row)
    if removed:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text("".join(r + "\n" for r in kept), encoding="utf-8")
        tmp.replace(cache_path)
    return removed


def _label_stats(lines, labels: dict, model: str) -> dict:
    """Aggregate label quality: how many lines have labels, how many answers parsed,
    and how often each value was given."""
    from study.labels import LABELS
    keys = {ln.key for ln in lines}
    attempted = [k for k in keys if k in labels]
    parsed = [labels[k] for k in attempted if labels[k] is not None]
    dist = {name: dict(Counter(lab.get(name) for lab in parsed if lab.get(name)))
            for name in LABELS}
    return {
        "model": model,
        "lines": len(keys),
        "attempted": len(attempted),
        "labelled": len(parsed),
        "unlabelled": len(keys) - len(attempted),
        "parsed_share": (len(parsed) / len(attempted)) if attempted else float("nan"),
        "distribution": dist,
    }


def cmd_run(args) -> int:
    from study.data import load_lines, group_calls, replies, overlaps
    from study.labels import make_labeller, LabelCache, label_lines
    from study.features import person_features, social_only
    from study.traits import (TRAITS, PRIOR_WEIGHTS, prior_scores, bootstrap_ci, split_half,
                              fit_calibration, apply_calibration, load_questionnaire)
    from study.habits import server_habits
    from study.report import write_report

    transcripts = Path(args.transcripts)
    out = _out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    exclude = _read_exclude(args.exclude)
    t0 = time.time()

    # 1. Load. Excluded people are dropped inside load_lines before anything else.
    lines = load_lines(transcripts, args.guild, args.since, args.until, exclude)
    if not lines:
        print(f"study: no usable lines for server {args.guild} in {transcripts}", file=sys.stderr)
        return 1
    calls = group_calls(lines)           # sets line.call once; nothing below regroups
    guild_name = next((ln.guild_name for ln in reversed(lines) if ln.guild_name), str(args.guild))
    print(f"Loaded {len(lines)} lines in {len(calls)} calls from {guild_name} "
          f"({len(exclude)} people excluded).", flush=True)

    # 2. Label. The cache makes this resumable: an interrupted run picks up where it stopped.
    cache_path = out / "labels.jsonl"
    removed = _purge_cache(cache_path, transcripts, args.guild, args.since, args.until, exclude)
    if removed:
        print(f"Removed {removed} cached labels of excluded people.", flush=True)
    labeller = make_labeller(args.engine, args.model)
    cache = LabelCache(cache_path)
    state = {"next": PROGRESS_EVERY, "t": time.time()}

    def progress(*a):
        # label_lines calls progress(done, todo) after each batch; print every 100 lines.
        done = int(a[0]) if a else 0
        todo = int(a[1]) if len(a) > 1 else 0
        if done >= state["next"] or (todo and done >= todo):
            rate = done / max(1e-6, time.time() - state["t"])
            left = (todo - done) / rate if rate > 0 and todo else 0
            print(f"  labelled {done}/{todo} new lines ({rate:.1f}/s, ~{left / 60:.0f} min left)", flush=True)
            while state["next"] <= done:
                state["next"] += PROGRESS_EVERY

    print(f"Labelling with {labeller.name} (cached labels are reused)...", flush=True)
    labels = label_lines(lines, labeller, cache, progress=progress, limit=args.limit)
    lstats = _label_stats(lines, labels, labeller.name)
    agreement_path = out / "agreement.json"
    if agreement_path.exists():
        try:
            lstats["agreement"] = json.loads(agreement_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    # 3. Turn-taking is measured on ALL lines (who answers whom does not depend
    #    on topic); --social-only restricts only the text used for features.
    reps = replies(lines)
    ovs = overlaps(lines)
    feat_lines = social_only(lines, labels) if args.social_only else lines
    if args.social_only:
        print(f"Social-only: {len(feat_lines)} of {len(lines)} lines kept for features.", flush=True)

    # 4. Features for everyone, then the word threshold applied here, so the
    #    number of people below it can be reported.
    all_people = person_features(feat_lines, labels, reps, ovs, min_words=0)
    included = {u for u, f in all_people.items() if f.get("words", 0) >= args.min_words}
    people = {u: all_people[u] for u in sorted(included)}
    print(f"{len(people)} people with at least {args.min_words} words "
          f"({len(all_people) - len(people)} below).", flush=True)

    # Score function used for the main scores, the bootstrap and split-half.
    # The included set is fixed here: subsets are NOT re-thresholded (half the
    # data would otherwise drop most people). Lines keep their call ids from
    # step 1, and duplicated lines from resampling are left as they are.
    feats_for_cal = sorted({f for w in PRIOR_WEIGHTS.values() for f in w})
    settings_extra: dict = {}
    calibration = None

    def features_of(subset):
        subset = sorted(subset, key=lambda ln: ln.t)
        feats = person_features(subset, labels, replies(subset), overlaps(subset), min_words=0)
        return {u: f for u, f in feats.items() if u in included}

    def score_people(pf):
        if len(pf) < 2:
            return {u: {t: float("nan") for t in TRAITS} for u in pf}
        if calibration is not None:
            # relative=True: re-z-scored across people, same scale as prior scores
            return apply_calibration(pf, calibration, relative=True)
        return prior_scores(pf)

    def compute(subset):
        return score_people(features_of(subset))

    # 5. Calibration, if a questionnaire was supplied.
    if args.questionnaire:
        truth = load_questionnaire(Path(args.questionnaire))
        truth = {u: v for u, v in truth.items() if u not in exclude and u in people}
        try:
            calibration = fit_calibration(people, truth, feats_for_cal)
            print(f"Calibrated on {calibration.get('n')} questionnaires.", flush=True)
        except ValueError as e:
            settings_extra["calibration_error"] = str(e)
            print(f"study: calibration skipped ({e}); using literature weights.", file=sys.stderr, flush=True)

    scores = score_people(people)

    # 6. Uncertainty: bootstrap over each person's calls, and split-half reliability.
    cis: dict = {}
    split: dict = {t: float("nan") for t in TRAITS}
    if len(people) >= 2 and args.boot > 0:
        by_user_call: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for ln in feat_lines:
            if ln.user in included:
                by_user_call[ln.user][ln.call].append(ln)
        by_user_call = {u: dict(c) for u, c in by_user_call.items()}
        print(f"Bootstrap ({args.boot} resamples)...", flush=True)
        cis = bootstrap_ci(by_user_call, compute, n=args.boot, seed=0)
    if len(people) >= 2:
        incl_lines = [ln for ln in feat_lines if ln.user in included]
        split = split_half(compute, incl_lines)

    # 7. Server habits on all lines.
    habits = server_habits(lines, calls, reps)

    # 8. Report.
    keys_by_user: dict[str, list[str]] = defaultdict(list)
    names: dict[str, str] = {}
    for ln in lines:
        keys_by_user[ln.user].append(ln.key)
        if ln.speaker:
            names[ln.user] = ln.speaker        # latest display name wins
    labelled_share = {u: sum(1 for k in ks if labels.get(k) is not None) / len(ks)
                      for u, ks in keys_by_user.items() if ks}
    reliability = {"split_half": split,
                   "calibration": ({k: calibration.get(k) for k in ("loo_r", "n", "n_by_trait", "weights", "intercept")}
                                   if calibration else None)}
    settings = {
        "guild": str(args.guild),
        "since": args.since, "until": args.until,
        "first_date": time.strftime("%Y-%m-%d", time.localtime(lines[0].t / 1000)),
        "last_date": time.strftime("%Y-%m-%d", time.localtime(lines[-1].t / 1000)),
        "engine": args.engine, "model": labeller.name,
        "min_words": args.min_words, "limit": args.limit, "social_only": bool(args.social_only),
        "questionnaire": bool(args.questionnaire), "calibrated": calibration is not None,
        "boot": args.boot,
        "excluded_count": len(exclude),           # a count only, never the ids
        "lines": len(feat_lines),
        "people_total": len(all_people), "people_included": len(people),
        "people_below_min": len(all_people) - len(people),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        # per-person, read by write_report for people.csv only (never written to server.json)
        "names": {u: names.get(u, "") for u in people},
        "labelled_share_by_user": {u: labelled_share.get(u, float("nan")) for u in people},
        **settings_extra,
    }
    write_report(out, guild_name, people, scores, cis, reliability, habits, lstats, settings)
    print(f"Done in {time.time() - t0:.0f} s. Wrote {out / 'report.md'}, {out / 'server.json'} and "
          f"{out / 'people.csv'} (people.csv is per-person: researcher only).", flush=True)
    if lstats["unlabelled"]:
        print(f"{lstats['unlabelled']} lines still unlabelled; run the same command again to continue.")
    return 0


# ------------------------------------------------------------------- sample
def _cached_labels(cache_path: Path) -> dict[str, dict]:
    """key -> labels from labels.jsonl, for whichever model wrote them (the last
    entry for a key wins). Read directly so no model has to be loaded."""
    out: dict[str, dict] = {}
    if not cache_path.exists():
        return out
    for row in cache_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(row)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("key") and rec.get("labels"):
            out[rec["key"]] = rec["labels"]
    return out


def cmd_sample(args) -> int:
    from study.data import load_lines, group_calls
    from study.agreement import export_sample
    out = _out_dir(args)
    exclude = _read_exclude(args.exclude)
    labels = _cached_labels(out / "labels.jsonl")
    if not labels:
        sys.exit(f"study: no labels in {out / 'labels.jsonl'}; run `python -m study run` first.")
    lines = load_lines(Path(args.transcripts), args.guild, args.since, args.until, exclude)
    group_calls(lines)   # so the sheet's "previous line" stays within one call
    lines = [ln for ln in lines if ln.key in labels]
    sheet = out / "sample.csv"
    n = export_sample(lines, {ln.key: labels[ln.key] for ln in lines}, sheet, n=args.n, seed=0)
    print(f"Wrote {n} rows to {sheet}. Fill in the human_* columns, then run:\n"
          f"  python -m study agree --sheet {sheet}")
    return 0


# -------------------------------------------------------------------- agree
def cmd_agree(args) -> int:
    from study.agreement import score_sample
    from study.report import jsonable
    sheet = Path(args.sheet)
    if not sheet.is_file():
        sys.exit(f"study: sheet not found: {sheet}")
    res = score_sample(sheet)
    print(f"{'label':<10} {'rows':>5} {'kappa':>7} {'accuracy':>9}")
    for name, v in res.items():
        k, a = v.get("kappa"), v.get("accuracy")
        ks = f"{k:.2f}" if isinstance(k, (int, float)) and not math.isnan(k) else "-"
        as_ = f"{100 * a:.0f}%" if isinstance(a, (int, float)) and not math.isnan(a) else "-"
        print(f"{name:<10} {v.get('n', 0):>5} {ks:>7} {as_:>9}")
    # Saved next to the sheet so the next `run` in that folder puts it in the report.
    dest = sheet.parent / "agreement.json"
    dest.write_text(json.dumps(jsonable(res), indent=2) + "\n", encoding="utf-8")
    print(f"Saved {dest}; the next `run` with this output folder includes it in the report.")
    return 0


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m study",
                                description="Offline study of one server's saved transcripts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS),
                        help="folder with YYYY-MM-DD.jsonl files (default: transcripts/ in the repo)")

    g = sub.add_parser("guilds", help="list servers found in the transcripts")
    common(g)

    r = sub.add_parser("run", help="label, score and write the report for one server")
    common(r)
    r.add_argument("--guild", required=True, help="server id (see `guilds`)")
    r.add_argument("--since", help="first date, YYYY-MM-DD (inclusive)")
    r.add_argument("--until", help="last date, YYYY-MM-DD (inclusive)")
    r.add_argument("--exclude", help="file with one user id per line to leave out entirely")
    r.add_argument("--out", help="output folder (default: study/out/<guild>/)")
    r.add_argument("--engine", default="auto", choices=["auto", "mlx", "torch", "fake"],
                   help="labelling backend; 'fake' needs no model (dry runs and tests)")
    r.add_argument("--model", default=None, help="override the model repo for mlx/torch")
    r.add_argument("--min-words", type=int, default=1000,
                   help="people with fewer words are left out of the scores (default 1000)")
    r.add_argument("--limit", type=int, default=None, help="label at most N new lines this run")
    r.add_argument("--social-only", action="store_true",
                   help="compute features only from lines labelled as social chat")
    r.add_argument("--questionnaire", help="CSV of questionnaire scores to calibrate against")
    r.add_argument("--boot", type=int, default=200, help="bootstrap resamples for intervals (0 to skip)")

    s = sub.add_parser("sample", help="export a hand-labelling sheet from existing labels")
    common(s)
    s.add_argument("--guild", required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--out", help="output folder used by `run` (default: study/out/<guild>/)")
    s.add_argument("--exclude", help="exclusion list (same file as for `run`)")
    s.add_argument("--since")
    s.add_argument("--until")

    a = sub.add_parser("agree", help="score a filled-in hand-labelling sheet")
    common(a)
    a.add_argument("--sheet", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"guilds": cmd_guilds, "run": cmd_run, "sample": cmd_sample, "agree": cmd_agree}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
