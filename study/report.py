"""Write the study outputs for one server.

Three files land in the output folder:

  people.csv   one row per included person: id, display name, size of their
               data, each trait score with its 95% interval, and flags. This is
               the only file with per-person results; it is for the researcher
               and must not be shared with the server.
  server.json  every aggregate number (settings, label quality, reliability,
               habits summary, trait distribution). No names and no per-person
               scores. Per-person habit details (role, reply partners) go into
               people.csv instead.
  report.md    a plain-language report for the server as a whole. It never
               lists names, user ids or per-person scores.

Inputs are plain dicts so this module does not depend on how the others are
built; every dict is read defensively with .get().
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

# Kept in sync with study.traits.TRAITS (copied so this module imports alone).
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Cut-offs for "low / middle / high" thirds of a standard normal distribution.
# Scores are z-scores relative to the server, so +-0.43 splits a normal
# distribution into three equal parts. Using fixed cut-offs (rather than rank
# terciles, which are 1/3 each by construction) shows whether the server's
# scores are bunched up or spread out.
THIRD_CUT = 0.43

# Flags written to people.csv.
FEW_CALLS = 5          # fewer calls than this: estimates hang on very few sessions, and the
                       # bootstrap gives no interval (kept equal to traits.MIN_CI_CALLS)
WIDE_CI = 1.5          # a 95% interval wider than 1.5 SD is too vague to read much into
FEW_LABELS_SHARE = 0.5  # less than half of the person's lines labelled

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Keys of `settings` that are per-person and must never reach server.json.
_PRIVATE_SETTINGS = {"names", "display_names", "labelled_share_by_user", "exclude_ids"}


# --------------------------------------------------------------------- helpers
def _num(v) -> float:
    """Best-effort float; NaN for anything missing or non-numeric."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _finite(v) -> bool:
    f = _num(v)
    return not (math.isnan(f) or math.isinf(f))


def _fmt(v, digits: int = 2) -> str:
    """Number for display: '-' when missing."""
    return f"{_num(v):.{digits}f}" if _finite(v) else "-"


def _pct(v) -> str:
    return f"{100 * _num(v):.0f}%" if _finite(v) else "-"


def jsonable(obj):
    """Recursively turn obj into something json.dump accepts and that is valid JSON.

    NaN / inf -> null, numpy scalars -> float/int, tuples and sets -> lists,
    dict keys -> str.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    # numpy scalars and arrays without importing numpy here.
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    if hasattr(obj, "item"):
        return jsonable(obj.item())
    try:
        return jsonable(float(obj))
    except (TypeError, ValueError):
        return str(obj)


def _write_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write via <name>.tmp + os.replace: a failed write (file open in Excel,
    read-only) leaves the previous file intact instead of half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _agreement_present(agree) -> bool:
    """An agreement block where no label has any checked row is no result."""
    return bool(agree) and any(_num((v or {}).get("n")) > 0 for v in agree.values() if isinstance(v, dict))


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if _finite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if _finite(x))
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# --------------------------------------------------------------- aggregation
def trait_distribution(scores: dict, cis: dict) -> dict:
    """Per trait: count of people, share in the low / middle / high third, and
    the median width of the 95% intervals (a direct measure of how vague
    individual estimates are)."""
    out = {}
    for trait in TRAITS:
        vals = [_num((s or {}).get(trait)) for s in scores.values()]
        vals = [v for v in vals if _finite(v)]
        widths = []
        for user in scores:
            ci = ((cis or {}).get(user) or {}).get(trait)
            if ci and len(ci) == 2 and _finite(ci[0]) and _finite(ci[1]):
                widths.append(_num(ci[1]) - _num(ci[0]))
        n = len(vals)
        out[trait] = {
            "n": n,
            "low": sum(v < -THIRD_CUT for v in vals) / n if n else float("nan"),
            "middle": sum(-THIRD_CUT <= v <= THIRD_CUT for v in vals) / n if n else float("nan"),
            "high": sum(v > THIRD_CUT for v in vals) / n if n else float("nan"),
            "median_ci_width": _median(widths),
        }
    return out


def habits_summary(habits: dict) -> dict:
    """Aggregate-only view of server_habits(): drops the per-user parts (roles
    by user, reply edges, degrees, top connectors) and keeps counts."""
    habits = habits or {}
    roles = habits.get("roles") or {}
    role_counts: dict[str, int] = {}
    for r in roles.values():
        role_counts[r] = role_counts.get(r, 0) + 1
    net = habits.get("reply_network") or {}
    degrees = [_num(d) for d in (net.get("degree") or {}).values()]
    edges = net.get("edges") or []
    return {
        "heatmap": habits.get("heatmap"),
        "call_sizes": habits.get("call_sizes"),
        "call_lengths_min": habits.get("call_lengths_min"),
        "calls": habits.get("calls"),
        "people": habits.get("people"),
        "hours_spoken": habits.get("hours_spoken"),
        "talk_gini": habits.get("talk_gini"),
        "role_counts": role_counts,
        "reply_network": {
            "pairs": len(edges),
            "replies_in_top_pairs": sum(int(_num(e[2])) for e in edges if len(e) > 2 and _finite(e[2])),
            "median_partners": _median(degrees),
            "max_partners": max(degrees) if degrees else float("nan"),
        },
    }


# -------------------------------------------------------------------- writers
def _people_csv(people_features: dict, scores: dict, cis: dict,
                habits: dict, names: dict, labelled_share: dict) -> str:
    roles = (habits or {}).get("roles") or {}
    degree = ((habits or {}).get("reply_network") or {}).get("degree") or {}
    header = ["user", "name", "words", "lines", "calls", "talk_min", "role", "reply_partners"]
    for t in TRAITS:
        header += [t, f"{t}_lo", f"{t}_hi"]
    header += ["flags"]

    # Order: most words first, so the best-grounded estimates are on top.
    users = sorted(scores.keys() | people_features.keys(),
                   key=lambda u: -_num((people_features.get(u) or {}).get("words")) if _finite(
                       (people_features.get(u) or {}).get("words")) else 0)
    f = io.StringIO()
    w = csv.writer(f)
    w.writerow(header)
    for u in users:
        feats = people_features.get(u) or {}
        sc = scores.get(u) or {}
        ci = (cis or {}).get(u) or {}
        flags = []
        if _finite(feats.get("calls")) and _num(feats.get("calls")) < FEW_CALLS:
            flags.append("few_calls")
        share = labelled_share.get(u)
        if share is not None and _finite(share) and _num(share) < FEW_LABELS_SHARE:
            flags.append("few_labels")
        for t in TRAITS:
            lo_hi = ci.get(t)
            if lo_hi and len(lo_hi) == 2 and _finite(lo_hi[0]) and _finite(lo_hi[1]) \
                    and _num(lo_hi[1]) - _num(lo_hi[0]) > WIDE_CI:
                flags.append("wide_ci")
                break
        row = [u, names.get(u, ""),
               _fmt(feats.get("words"), 0), _fmt(feats.get("lines"), 0), _fmt(feats.get("calls"), 0),
               _fmt(_num(feats.get("talk_s")) / 60, 1), roles.get(u, ""), degree.get(u, "")]
        for t in TRAITS:
            lo_hi = ci.get(t) or (None, None)
            row += [_fmt(sc.get(t)), _fmt(lo_hi[0]), _fmt(lo_hi[1])]
        row.append(";".join(flags))
        w.writerow(row)
    return f.getvalue()


def _heatmap_table(heat) -> list[str]:
    """7 x 24 spoken minutes as a fixed-width text table (in a code block)."""
    if not heat or len(heat) != 7:
        return ["(no heatmap data)"]
    out = ["```", "minutes  " + " ".join(f"{h:>3}" for h in range(24))]
    for d, row in enumerate(heat):
        cells = " ".join(f"{round(_num(v)):>3}" if _finite(v) and _num(v) > 0 else "  ." for v in row)
        out.append(f"{WEEKDAYS[d]:<8} {cells}")
    out.append("```")
    return out


def _report_md(guild_name: str, settings: dict, label_stats: dict, reliability: dict,
               dist: dict, hab: dict) -> str:
    s = settings or {}
    ls = label_stats or {}
    rel = reliability or {}
    L: list[str] = []
    add = L.append

    add(f"# Study report: {guild_name or 'server'}")
    add("")
    add("This report describes the server as a whole. It does not list any person, "
        "and it is not a judgement of anyone.")
    add("")
    n_inc = s.get("people_included")
    if isinstance(n_inc, int) and n_inc < 2:
        add(f"> **No trait scores in this run:** only {n_inc} person(s) reached the "
            f"{s.get('min_words', '?')}-word minimum, and scores are relative to the server, so at least 2 "
            "are needed (5 for the reliability check). Rerun with a lower --min-words"
            + (" or without --social-only" if s.get("social_only") else "") + ".")
        add("")

    # ---- How to read this (first, so nobody skips it)
    calibrated = bool(s.get("calibrated"))
    add("## How to read this")
    add("")
    add("- **Scores are relative to this server.** 0 means \"about average for the people in this "
        "server's calls\", +1 / -1 means one standard deviation above / below that average. "
        "They are not comparable with scores from another server or from a personality test.")
    add("- **They come from published research on language and personality**, not from a test anyone "
        "took. Studies of speech and writing (Mairesse et al. 2007; Yarkoni 2010; Schwartz et al. 2013; "
        "Pennebaker and King 1999) found small links between how people talk and their Big Five "
        "traits, for example people who score high on extraversion tend to talk more and use more "
        "social words. This study counts those same things in the transcripts and adds them up.")
    add("- **Expect modest accuracy.** Methods like this typically agree with questionnaire results "
        "at a correlation of about r = 0.2 to 0.4. That is useful for patterns across a group and "
        "weak for any single person. Treat an individual score as a rough hint, never a fact.")
    if calibrated:
        add("- **Calibrated.** A questionnaire was supplied, so the weights were fitted to those answers "
            "(see Reliability for how well the fit held up on people it had not seen).")
    elif s.get("questionnaire"):
        err = str(s.get("calibration_error") or "reason not recorded").rstrip(".")
        add(f"- **Uncalibrated.** A questionnaire was given but calibration was not possible: {err}. "
            "The literature's directions were used instead, so nothing here has been checked against "
            "how people describe themselves.")
    else:
        add("- **Uncalibrated.** No questionnaire was supplied, so nothing here has been checked against "
            "how people describe themselves. The weights are the literature's directions only.")
    add("- **Transcripts are imperfect.** Speech recognition drops and mishears words, and a "
        "small local model labels each line's tone and intent; both add noise.")
    add("- **Thirds**: \"low\", \"middle\" and \"high\" below mean a score under -0.43, between -0.43 "
        "and +0.43, and over +0.43. If scores were spread like a normal bell curve, each third would "
        "hold about a third of the people.")
    add("")

    # ---- Data used
    add("## Data used")
    add("")
    first, last = s.get("first_date") or s.get("since") or "?", s.get("last_date") or s.get("until") or "?"
    add(f"- Dates: {first} to {last}")
    add(f"- Lines analysed: {s.get('lines', '-')}"
        + (" (only lines labelled as social chat were used for the scores)" if s.get("social_only") else ""))
    add(f"- Calls: {hab.get('calls', '-')}, hours of speech: {_fmt(hab.get('hours_spoken'), 1)}")
    spoke = s.get("people_spoke", s.get("people_total", "-"))
    if s.get("social_only"):
        add(f"- People who spoke: {spoke}; with social-chat lines: {s.get('people_total', '-')}; "
            f"included in the scores: {s.get('people_included', '-')}; "
            f"below the {s.get('min_words', '-')}-word minimum in social lines (left out of the scores): "
            f"{s.get('people_below_min', '-')}")
    else:
        add(f"- People who spoke: {spoke}; "
            f"included in the scores: {s.get('people_included', '-')}; "
            f"below the {s.get('min_words', '-')}-word minimum (left out of the scores): "
            f"{s.get('people_below_min', '-')}")
    if s.get("excluded_listed"):
        add(f"- Exclusion list: {s['excluded_listed']} id(s); {s.get('excluded_count', 0)} of them spoke in "
            "this server and date range, and all their lines were removed before any analysis in this run (they "
            "appear nowhere in these outputs and were not shown to the labelling model in this run).")
        if s.get("excluded_unmatched"):
            add(f"  {s['excluded_unmatched']} id(s) on the list matched no line here (they may not have spoken "
                "in this period, or the id may be wrong: check it is a user id).")
        add("  Limitation: the list removes what excluded people SAID. If others mention them by name, those "
            "other people's lines are still analysed (and can appear as text in sample.csv).")
    add("")

    # ---- Label quality
    add("## Label quality")
    add("")
    add(f"Each line was labelled by `{ls.get('model', '?')}` for sentiment, kind of remark, emotion, "
        "tone and topic.")
    add("")
    add(f"- Lines with labels: {ls.get('labelled', '-')} of {ls.get('lines', '-')}; "
        f"answers that could be read: {_pct(ls.get('parsed_share'))}")
    if _finite(ls.get("unlabelled")) and _num(ls.get("unlabelled")) > 0:
        add(f"- Lines not labelled yet (run stopped early or --limit): {int(_num(ls['unlabelled']))}. "
            "Run again to continue; finished lines are kept.")
    agree = ls.get("agreement") or rel.get("agreement")
    if _agreement_present(agree):
        add("- Agreement with a person's hand labels (Cohen's kappa: 1 is perfect, 0 is chance, "
            "above about 0.6 is usually considered good):")
        add("")
        add("| label | rows checked | kappa | accuracy |")
        add("|---|---|---|---|")
        for name, v in agree.items():
            v = v or {}
            add(f"| {name} | {v.get('n', '-')} | {_fmt(v.get('kappa'))} | {_pct(v.get('accuracy'))} |")
    else:
        add("- Not yet checked against a person's labels. Run `python -m study sample` and "
            "`python -m study agree` to measure this.")
    dists = ls.get("distribution") or {}
    if dists:
        add("")
        add("How often each label was given:")
        add("")
        for name, counts in dists.items():
            total = sum(counts.values()) or 1
            parts = ", ".join(f"{k} {100 * c / total:.0f}%"
                              for k, c in sorted(counts.items(), key=lambda kv: -kv[1]))
            add(f"- {name}: {parts}")
    add("")

    # ---- Traits
    add("## Trait spread across the server")
    add("")
    add("Share of included people in each third. The interval width is the median width of each "
        "person's 95% range from resampling the server's calls: the wider it is, the less a single score "
        f"means (a width of 2 covers most of the scale). People with fewer than {FEW_CALLS} calls get no "
        "interval (too few calls to show how much they vary).")
    add("")
    add("| trait | people | low | middle | high | median interval width |")
    add("|---|---|---|---|---|---|")
    for t in TRAITS:
        d = dist.get(t) or {}
        add(f"| {t} | {d.get('n', 0)} | {_pct(d.get('low'))} | {_pct(d.get('middle'))} | "
            f"{_pct(d.get('high'))} | {_fmt(d.get('median_ci_width'))} |")
    add("")

    # ---- Reliability
    add("## Reliability")
    add("")
    sh = rel.get("split_half") or {}
    n_sh = rel.get("split_half_users")
    add("**Split-half consistency.** The server's calls were split into two halves (alternate whole calls, "
        "so everyone in a call is in the same half) and scored separately. The number is how well the two "
        "halves agree across people, corrected to full length (1 = perfectly consistent, 0 = no "
        "consistency). Below about 0.5 the scores mostly reflect which calls happened to be recorded. "
        "A negative value is shown uncorrected and also means no consistency."
        + (f" Compared over {n_sh} people who spoke in both halves." if isinstance(n_sh, int) else ""))
    add("")
    if sh:
        add("| trait | split-half r |")
        add("|---|---|")
        for t in TRAITS:
            add(f"| {t} | {_fmt(sh.get(t))} |")
        if not any(_finite(v) for v in sh.values()):
            add("")
            add("(Not computable: it needs at least 5 included people.)")
    else:
        add("(Not computed.)")
    add("")
    cal = rel.get("calibration")
    if cal:
        add(f"**Calibration** against {cal.get('n', '?')} questionnaires. Leave-one-out correlation "
            "between predicted and self-reported scores (each person predicted by a model fitted "
            "without them):")
        add("")
        add("| trait | answers | leave-one-out r | chance level (95th pct) | p |")
        add("|---|---|---|---|---|")
        for t in TRAITS:
            add(f"| {t} | {(cal.get('n_by_trait') or {}).get(t, '-')} | {_fmt((cal.get('loo_r') or {}).get(t))} | "
                f"{_fmt((cal.get('loo_null95') or {}).get(t))} | {_fmt((cal.get('loo_p') or {}).get(t))} |")
        add("")
        add("The chance level is what the same procedure gives when the questionnaire answers are shuffled "
            "between people: with few questionnaires and many features, values of 0.3-0.6 happen by luck. "
            "A leave-one-out r below the chance level (or p above 0.05) means the calibrated scores for that "
            "trait are no better than chance. With only a handful of questionnaires this is common. "
            "Leave-one-out r can also come out clearly negative when there is no real signal (each person "
            "is predicted from the others' average), so judge by p, not by the sign of r."
            + (f" Ridge penalty: {cal.get('alpha')}." if cal.get("alpha") is not None else ""))
        add("")

    # ---- Habits
    add("## Server habits")
    add("")
    add("Spoken minutes by weekday and hour (local time of the machine that ran the study):")
    add("")
    L.extend(_heatmap_table(hab.get("heatmap")))
    add("")
    sizes = hab.get("call_sizes") or {}
    if sizes:
        parts = ", ".join(f"{k} people: {v}" for k, v in sorted(sizes.items(), key=lambda kv: int(_num(kv[0]))
                                                                 if _finite(kv[0]) else 0))
        add(f"- Calls by number of speakers: {parts}")
    lens = hab.get("call_lengths_min") or {}
    if lens:
        add(f"- Call length in minutes: median {_fmt(lens.get('median'), 0)}, mean {_fmt(lens.get('mean'), 0)}, "
            f"longest tenth over {_fmt(lens.get('p90'), 0)}")
    if _finite(hab.get("talk_gini")):
        add(f"- How evenly talk time is shared (Gini, 0 = everyone equal, 1 = one person talks): "
            f"{_fmt(hab.get('talk_gini'))}")
    roles = hab.get("role_counts") or {}
    if roles:
        add("- Roles (rule-based, from who opens calls, how many people they talk with, and how much "
            "they talk): " + ", ".join(f"{k} {v}" for k, v in sorted(roles.items())))
    net = hab.get("reply_network") or {}
    if net:
        add(f"- Reply network: {net.get('pairs', 0)} pairs of people who answer each other; a typical person "
            f"talks back and forth with {_fmt(net.get('median_partners'), 0)} others.")
    add("")

    # ---- Settings
    add("## Settings")
    add("")
    for k in ("engine", "model", "min_words", "limit", "social_only", "questionnaire", "since", "until",
              "boot", "created"):
        if k in s and s[k] not in (None, ""):
            add(f"- {k}: {s[k]}")
    add("")
    add("Per-person results are in people.csv, for the researcher only. Do not share that file.")
    add("")
    return "\n".join(L)


def write_report(out_dir, guild_name, people_features, scores, cis, reliability, habits,
                 label_stats, settings) -> None:
    """Write people.csv, server.json and report.md into out_dir.

    people_features  user -> feature dict (included people only)
    scores           user -> trait -> score (relative z-score, or calibrated prediction)
    cis              user -> trait -> (lo, hi) 95% bootstrap interval
    reliability      {"split_half": {trait: r}, "calibration": fit dict or None, "agreement": {...}?}
    habits           output of habits.server_habits
    label_stats      {"model", "lines", "labelled", "parsed_share", "unlabelled", "distribution", "agreement"?}
    settings         run settings. Two optional per-person keys are read and NOT written to server.json:
                     "names" (user -> display name, for people.csv) and
                     "labelled_share_by_user" (user -> share of their lines labelled, for flags).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(settings or {})
    names = settings.get("names") or settings.get("display_names") or {}
    labelled_share = settings.get("labelled_share_by_user") or {}
    public_settings = {k: v for k, v in settings.items() if k not in _PRIVATE_SETTINGS}

    people_features = people_features or {}
    scores = scores or {}
    cis = cis or {}

    dist = trait_distribution(scores, cis)
    hab = habits_summary(habits)

    # Render everything in memory first, then replace each file atomically.
    people_text = _people_csv(people_features, scores, cis, habits, names, labelled_share)

    server = {
        "guild_name": guild_name,
        "settings": public_settings,
        "label_stats": label_stats or {},
        "reliability": reliability or {},
        "trait_distribution": dist,
        "habits": hab,
    }
    # Calibration weights are aggregate (one weight per feature), keep them;
    # nothing in reliability is per person.
    server_text = json.dumps(jsonable(server), indent=2, allow_nan=False) + "\n"
    report_text = _report_md(guild_name, public_settings, label_stats, reliability, dist, hab)
    # report.md last: if an earlier file fails, the old report (describing the
    # old files) is not replaced by one that describes files that were not written.
    _write_atomic(out / "people.csv", people_text)
    _write_atomic(out / "server.json", server_text)
    _write_atomic(out / "report.md", report_text)
