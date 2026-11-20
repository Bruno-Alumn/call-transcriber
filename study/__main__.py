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
import csv
import datetime as _dt
import json
import math
import os
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
    """Exclusion list. A path that was given but does not exist, or an entry
    that is not a user id, is an error: silently running without (part of)
    the list would leak the people it names."""
    from study.data import load_exclude
    if path is None:
        return set()
    p = Path(path)
    if not p.is_file():
        sys.exit(f"study: exclusion file not found: {p}")
    try:
        return load_exclude(p)
    except ValueError as e:
        sys.exit(f"study: exclusion list rejected, nothing was run: {e}")


def _check_transcripts(path: str) -> Path:
    p = Path(path)
    if not p.is_dir():
        sys.exit(f"study: transcripts folder not found: {p}")
    return p


def _iso_date(value: str) -> str:
    """argparse type: a real calendar date, normalised to YYYY-MM-DD (file names compare as strings)."""
    try:
        return _dt.date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date in the form YYYY-MM-DD")


def _nonneg_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number")
    if n < 0:
        raise argparse.ArgumentTypeError(f"{value} must be 0 or more")
    return n


def _check_dates(args) -> None:
    if args.since and args.until and args.since > args.until:
        sys.exit(f"study: --since {args.since} is after --until {args.until}")


def _no_lines_message(transcripts: Path, guild: str, since, until) -> str:
    """Why load_lines came back empty: unknown server, or nothing in the date range."""
    from study.data import list_guilds
    row = next((r for r in list_guilds(transcripts) if str(r.get("guild")) == str(guild)), None)
    if row is None:
        return (f"study: server {guild} not found in {transcripts}; "
                f"run `python -m study guilds --transcripts {transcripts}` to list the servers")
    if since or until:
        return (f"study: server {guild} has no usable lines between {since or 'the start'} and "
                f"{until or 'the end'}; its transcripts run from {row.get('first')} to {row.get('last')}")
    return f"study: server {guild} has no usable lines (all too short, unattributed or excluded)"


def _write_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write via a temp file and os.replace, so a failed write never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
    os.replace(tmp, path)


# ------------------------------------------------------------------- guilds
def cmd_guilds(args) -> int:
    from study.data import list_guilds
    rows = list_guilds(_check_transcripts(args.transcripts))
    if not rows:
        print(f"No server-tagged lines found in {args.transcripts}")
        return 0
    print(f"{'server id':<22} {'lines':>8}  {'first':<10}  {'last':<10}  name")
    for r in rows:
        print(f"{str(r.get('guild')):<22} {r.get('lines', 0):>8}  {str(r.get('first', '')):<10}  "
              f"{str(r.get('last', '')):<10}  {r.get('guildName', '')}")
    return 0


# ---------------------------------------------------------------------- run
def _scrub_for_exclusion(out: Path, transcripts: Path, guild: str, exclude: set[str]) -> dict:
    """Bring files left by earlier runs in `out` in line with the exclusion list.

    Run whenever an exclusion list is in force. The set of VALID line keys is
    every loaded line of this server over ALL dates (not just --since/--until,
    so labels of other dates are kept), with the exclusion applied. Then:

      labels.jsonl  rows whose key is not valid are dropped: lines of people
                    now on the list, and orphans whose raw lines were deleted
                    from the transcripts (e.g. after a removal request).
      sample.csv    rows whose key is not valid are dropped, and the
                    'previous' column is recomputed from the valid lines (so
                    an excluded person's words no longer appear as context of
                    someone else's row). Filled-in human_* cells are kept.
      agreement.json  recomputed from the scrubbed sheet if rows were dropped;
                    removed if no filled-in row is left.

    Note: `out` is assumed to hold one server (the default study/out/<guild>/).
    Returns counts of what was removed / changed.
    """
    from study.data import load_lines, group_calls
    from study.agreement import _previous_texts, score_sample
    res = {"cache_rows": 0, "sample_rows": 0, "sample_previous": 0, "agreement": ""}
    cache_path, sheet = out / "labels.jsonl", out / "sample.csv"
    if not cache_path.exists() and not sheet.exists():
        return res
    valid = load_lines(transcripts, guild, None, None, exclude)
    group_calls(valid)
    valid_keys = {ln.key for ln in valid}

    if cache_path.exists():
        kept = []
        for row in cache_path.read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            try:
                key = json.loads(row).get("key")
            except (json.JSONDecodeError, AttributeError):
                key = None                  # a torn row: drop it too
            if key in valid_keys:
                kept.append(row)
            else:
                res["cache_rows"] += 1
        if res["cache_rows"]:
            _write_atomic(cache_path, "".join(r + "\n" for r in kept))

    if sheet.exists():
        prev = _previous_texts(valid)
        with open(sheet, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            rows = list(reader)
        if "key" in header:
            new_rows = []
            for r in rows:
                key = r.get("key", "")
                if key not in valid_keys:
                    res["sample_rows"] += 1
                    continue
                if "previous" in r and r["previous"] != prev.get(key, ""):
                    r["previous"] = prev.get(key, "")
                    res["sample_previous"] += 1
                new_rows.append(r)
            if res["sample_rows"] or res["sample_previous"]:
                import io
                buf = io.StringIO()
                w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore")
                w.writeheader()
                w.writerows(new_rows)
                _write_atomic(sheet, buf.getvalue(), encoding="utf-8-sig")
                agreement = out / "agreement.json"
                if res["sample_rows"] and agreement.exists():
                    scored = score_sample(sheet)
                    if any((v or {}).get("n") for v in scored.values()):
                        from study.report import jsonable
                        _write_atomic(agreement, json.dumps(jsonable(scored), indent=2) + "\n")
                        res["agreement"] = "recomputed"
                    else:
                        agreement.unlink()
                        res["agreement"] = "removed"
    return res


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
    from study.data import load_lines, group_calls, replies
    from study.labels import make_labeller, LabelCache, label_lines
    from study.features import features_for, social_only
    from study.traits import (TRAITS, PRIOR_WEIGHTS, CLI_ALPHA, prior_scores, bootstrap_ci, split_half,
                              fit_calibration, apply_calibration, load_questionnaire)
    from study.habits import server_habits
    from study.report import write_report

    # 0. Check every input before anything slow (loading a model, labelling).
    transcripts = _check_transcripts(args.transcripts)
    _check_dates(args)
    exclude = _read_exclude(args.exclude)
    truth_raw = None
    if args.questionnaire:
        qp = Path(args.questionnaire)
        if not qp.is_file():
            sys.exit(f"study: questionnaire not found: {qp}")
        try:
            truth_raw = load_questionnaire(qp)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            sys.exit(f"study: questionnaire rejected: {e}")
    t0 = time.time()

    # 1. Load. Excluded people are dropped inside load_lines before anything else.
    load_stats: dict = {}
    lines = load_lines(transcripts, args.guild, args.since, args.until, exclude, stats=load_stats)
    if not lines:
        print(_no_lines_message(transcripts, args.guild, args.since, args.until), file=sys.stderr)
        return 1
    out = _out_dir(args)
    out.mkdir(parents=True, exist_ok=True)     # only once there is something to write
    calls = group_calls(lines)           # sets line.call once; nothing below regroups
    guild_name = next((ln.guild_name for ln in reversed(lines) if ln.guild_name), str(args.guild))
    dropped = load_stats.get("excluded_lines", {})
    matched = sum(1 for n in dropped.values() if n > 0)
    print(f"Loaded {len(lines)} lines in {len(calls)} calls from {guild_name} "
          f"({len(exclude)} ids on the exclusion list, {matched} of them had lines here: "
          f"{sum(dropped.values())} lines removed).", flush=True)
    unmatched = [u for u, n in dropped.items() if n == 0]
    if unmatched:
        print(f"study: WARNING: {len(unmatched)} id(s) on the exclusion list matched no line of this server "
              f"in this date range: {', '.join(sorted(unmatched)[:10])}. Check they are user ids "
              "(right-click a user > Copy User ID), not server, channel or message ids.",
              file=sys.stderr, flush=True)

    # 2. Bring earlier outputs in line with the exclusion list (cache, hand-label sheet).
    if exclude:
        sc = _scrub_for_exclusion(out, transcripts, args.guild, exclude)
        if sc["cache_rows"]:
            print(f"Removed {sc['cache_rows']} cached labels of excluded people or deleted lines.", flush=True)
        if sc["sample_rows"] or sc["sample_previous"]:
            print(f"Scrubbed sample.csv: {sc['sample_rows']} rows removed, {sc['sample_previous']} "
                  "'previous' cells rewritten." + (f" agreement.json {sc['agreement']}." if sc["agreement"] else ""),
                  flush=True)

    # 3. Label. The cache makes this resumable: an interrupted run picks up where it stopped.
    try:
        labeller = make_labeller(args.engine, args.model)
    except (RuntimeError, ImportError, OSError) as e:
        print(f"study: cannot start the labelling model: {e}", file=sys.stderr)
        return 1
    cache = LabelCache(out / "labels.jsonl")
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
            agr = json.loads(agreement_path.read_text(encoding="utf-8"))
            if any((v or {}).get("n") for v in agr.values()):     # an all-empty result is no result
                lstats["agreement"] = agr
        except (json.JSONDecodeError, AttributeError):
            pass

    # 4. Features. One definition (features.features_for) serves the point
    #    estimate, the bootstrap and split-half: turn-taking is measured on ALL
    #    lines of the calls (who answers whom does not depend on topic), and
    #    --social-only restricts only the text/label features and talk share.
    reps = replies(lines)
    if args.social_only:
        n_soc = len(social_only(lines, labels))
        print(f"Social-only: {n_soc} of {len(lines)} lines kept for text features.", flush=True)
        if n_soc == 0:
            print("study: WARNING: no line is labelled as social chat, so no one can be scored. "
                  "Run without --social-only (or label more lines first).", file=sys.stderr, flush=True)
    all_people = features_for(lines, labels, social=args.social_only)
    included = {u for u, f in all_people.items() if f.get("words", 0) >= args.min_words}
    people = {u: all_people[u] for u in sorted(included)}
    print(f"{len(people)} people with at least {args.min_words} words "
          f"({len(all_people) - len(people)} below).", flush=True)
    if len(people) < 2:
        print(f"study: WARNING: only {len(people)} person(s) reached --min-words {args.min_words}; scores are "
              "relative to the server, so at least 2 are needed (5 for reliability). Try a lower --min-words.",
              file=sys.stderr, flush=True)

    # Score function used for the main scores, the bootstrap and split-half.
    # The included set is fixed here: subsets are NOT re-thresholded (half the
    # data would otherwise drop most people), but everyone's lines stay in the
    # subsets so talk share and replies keep their real context.
    feats_for_cal = sorted({f for w in PRIOR_WEIGHTS.values() for f in w})
    settings_extra: dict = {}
    calibration = None

    def score_people(pf):
        if len(pf) < 2:
            return {u: {t: float("nan") for t in TRAITS} for u in pf}
        if calibration is not None:
            # relative=True: re-z-scored across people, same scale as prior scores
            return apply_calibration(pf, calibration, relative=True)
        return prior_scores(pf)

    def compute(subset):
        return score_people(features_for(subset, labels, social=args.social_only, include=included))

    # 5. Calibration, if a questionnaire was supplied (already read and checked in step 0).
    if truth_raw is not None:
        truth = {u: v for u, v in truth_raw.items() if u not in exclude and u in people}
        try:
            calibration = fit_calibration(people, truth, feats_for_cal, alpha=CLI_ALPHA)
            print(f"Calibrated on {calibration.get('n')} questionnaires.", flush=True)
        except ValueError as e:
            settings_extra["calibration_error"] = str(e)
            print(f"study: WARNING: calibration skipped ({e}); using literature weights.",
                  file=sys.stderr, flush=True)

    scores = score_people(people)

    # 6. Uncertainty: cluster bootstrap over calls, and split-half over whole calls.
    #    Both get every speaker's lines (not only the included people).
    cis: dict = {}
    split: dict = {t: float("nan") for t in TRAITS}
    split_info: dict = {}
    if len(people) >= 2 and args.boot > 0:
        by_user_call: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for ln in lines:
            by_user_call[ln.user][ln.call].append(ln)
        by_user_call = {u: dict(c) for u, c in by_user_call.items()}
        print(f"Bootstrap ({args.boot} resamples)...", flush=True)
        cis = bootstrap_ci(by_user_call, compute, n=args.boot, seed=0)
        cis = {u: v for u, v in cis.items() if u in included}
    if len(people) >= 2:
        split = split_half(compute, lines, info=split_info)

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
                   "split_half_users": split_info.get("n_users"),
                   "calibration": ({k: calibration.get(k) for k in
                                    ("loo_r", "loo_null95", "loo_p", "loo_ci", "n", "n_by_trait", "alpha",
                                     "weights", "intercept")}
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
        # counts only, never the ids
        "excluded_count": matched,                 # ids that actually had lines removed
        "excluded_listed": len(exclude),
        "excluded_unmatched": len(unmatched),
        "lines": len(lines),
        "people_spoke": len(keys_by_user),         # everyone who spoke, from ALL lines
        "people_total": len(all_people),           # people with feature lines (social lines under --social-only)
        "people_included": len(people),
        "people_below_min": len(all_people) - len(people),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        # per-person, read by write_report for people.csv only (never written to server.json)
        "names": {u: names.get(u, "") for u in people},
        "labelled_share_by_user": {u: labelled_share.get(u, float("nan")) for u in people},
        **settings_extra,
    }
    try:
        write_report(out, guild_name, people, scores, cis, reliability, habits, lstats, settings)
    except OSError as e:
        name = getattr(e, "filename", None) or out
        print(f"study: cannot write {name} ({e.strerror or e}); is it open in another program? "
              "Files that were already written are complete; rerun once it is closed.", file=sys.stderr)
        return 1
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
    _check_transcripts(args.transcripts)
    _check_dates(args)
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
    try:
        with open(sheet, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
    except (OSError, UnicodeDecodeError) as e:
        sys.exit(f"study: cannot read {sheet} as a CSV sheet: {e}")
    header = [h.strip() for h in header]
    if "key" not in header or not any(h.startswith("human_") for h in header):
        sys.exit(f"study: {sheet} is not a hand-labelling sheet (needs a 'key' column and human_* columns; "
                 "make one with `python -m study sample`)")
    res = score_sample(sheet)
    print(f"{'label':<10} {'rows':>5} {'kappa':>7} {'accuracy':>9}")
    for name, v in res.items():
        k, a = v.get("kappa"), v.get("accuracy")
        ks = f"{k:.2f}" if isinstance(k, (int, float)) and not math.isnan(k) else "-"
        as_ = f"{100 * a:.0f}%" if isinstance(a, (int, float)) and not math.isnan(a) else "-"
        print(f"{name:<10} {v.get('n', 0):>5} {ks:>7} {as_:>9}")
    if not any((v or {}).get("n") for v in res.values()):
        print("No human_* cells are filled in yet; nothing saved.")
        return 0
    # Saved next to the sheet so the next `run` in that folder puts it in the report.
    dest = sheet.parent / "agreement.json"
    try:
        _write_atomic(dest, json.dumps(jsonable(res), indent=2) + "\n")
    except OSError as e:
        sys.exit(f"study: cannot write {dest} ({e})")
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
    r.add_argument("--since", type=_iso_date, help="first date, YYYY-MM-DD (inclusive)")
    r.add_argument("--until", type=_iso_date, help="last date, YYYY-MM-DD (inclusive)")
    r.add_argument("--exclude", help="file with one user id per line to leave out entirely")
    r.add_argument("--out", help="output folder (default: study/out/<guild>/)")
    r.add_argument("--engine", default="auto", choices=["auto", "mlx", "torch", "fake"],
                   help="labelling backend; 'fake' needs no model (dry runs and tests)")
    r.add_argument("--model", default=None, help="override the model repo for mlx/torch")
    r.add_argument("--min-words", type=_nonneg_int, default=1000,
                   help="people with fewer words are left out of the scores (default 1000)")
    r.add_argument("--limit", type=_nonneg_int, default=None, help="label at most N new lines this run")
    r.add_argument("--social-only", action="store_true",
                   help="compute features only from lines labelled as social chat")
    r.add_argument("--questionnaire", help="CSV of questionnaire scores to calibrate against")
    r.add_argument("--boot", type=_nonneg_int, default=200, help="bootstrap resamples for intervals (0 to skip)")

    s = sub.add_parser("sample", help="export a hand-labelling sheet from existing labels")
    common(s)
    s.add_argument("--guild", required=True)
    s.add_argument("--n", type=_nonneg_int, default=200)
    s.add_argument("--out", help="output folder used by `run` (default: study/out/<guild>/)")
    s.add_argument("--exclude", help="exclusion list (same file as for `run`)")
    s.add_argument("--since", type=_iso_date)
    s.add_argument("--until", type=_iso_date)

    a = sub.add_parser("agree", help="score a filled-in hand-labelling sheet")
    common(a)
    a.add_argument("--sheet", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"guilds": cmd_guilds, "run": cmd_run, "sample": cmd_sample, "agree": cmd_agree}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
