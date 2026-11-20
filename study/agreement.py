"""Human-vs-model label agreement.

export_sample writes a CSV a person fills in by hand (one human_<label> column
per label); score_sample reads it back and reports Cohen's kappa and raw
accuracy per label, using only the rows where the human filled that column.

CSV columns (in order):
    key, text, previous,
    model_sentiment, model_act, model_emotion, model_tone, model_context,
    human_sentiment, human_act, human_emotion, human_tone, human_context

No user ids or display names are written: only the line key and the spoken text.
Files are written as UTF-8 with a BOM so Excel on Windows opens them correctly;
reading accepts files with or without the BOM.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

# Kept in sync with labels.LABELS by name; imported lazily so this module works
# even if labels.py (and its model backends) cannot be imported.
_LABEL_NAMES = ["sentiment", "act", "emotion", "tone", "context"]


def _label_names() -> list[str]:
    try:
        from study.labels import LABELS  # noqa: WPS433 (lazy import is intentional)
        return list(LABELS.keys())
    except Exception:
        return list(_LABEL_NAMES)


def _previous_texts(lines) -> dict[str, str]:
    """key -> text of the line just before it in the same call (or channel when
    calls were not grouped). '' for the first line of a call."""
    prev: dict[str, str] = {}
    last_in_group: dict[str, str] = {}
    for ln in sorted(lines, key=lambda l: (l.t, l.idx)):
        group = ln.call or ln.ch
        prev[ln.key] = last_in_group.get(group, "")
        last_in_group[group] = ln.text
    return prev


def export_sample(lines, labels: dict, out_csv, n: int = 200, seed: int = 0) -> int:
    """Write a hand-labelling sheet of up to n labelled lines to out_csv.

    Sampling is "stratified-ish": labelled lines are grouped by the model's
    `act` label, each group is shuffled, and lines are taken round-robin across
    groups so rare acts (greetings, disagreements) still show up. The final
    sheet is shuffled again so the strata are not visible in row order.
    Deterministic for a given seed. Lines with no labels (None) are skipped.
    Returns the number of rows written.
    """
    names = _label_names()
    rng = random.Random(seed)
    prev = _previous_texts(lines)

    strata: dict[str, list] = defaultdict(list)
    for ln in lines:
        lab = labels.get(ln.key)
        if not lab:
            continue
        strata[str(lab.get("act", ""))].append(ln)

    order = sorted(strata)              # stable stratum order before shuffling
    for k in order:
        rng.shuffle(strata[k])
    picked = []
    pos = 0
    while len(picked) < n and any(pos < len(strata[k]) for k in order):
        for k in order:
            if pos < len(strata[k]) and len(picked) < n:
                picked.append(strata[k][pos])
        pos += 1
    rng.shuffle(picked)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = (["key", "text", "previous"] + [f"model_{k}" for k in names]
              + [f"human_{k}" for k in names])
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for ln in picked:
            lab = labels[ln.key]
            w.writerow([ln.key, ln.text, prev.get(ln.key, "")]
                       + [lab.get(k, "") for k in names] + [""] * len(names))
    return len(picked)


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two equal-length label lists.

    1.0 if the lists are identical (including the degenerate one-category
    case), NaN if empty. Raises ValueError if the lengths differ.
    """
    if len(a) != len(b):
        raise ValueError("kappa: lists must have the same length")
    n = len(a)
    if n == 0:
        return float("nan")
    if list(a) == list(b):
        return 1.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb.get(c, 0) for c in ca) / (n * n)
    if expected >= 1.0:                 # cannot happen unless identical; guard anyway
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def score_sample(in_csv) -> dict[str, dict]:
    """Score a filled-in sheet: per label {"n", "kappa", "accuracy"} computed on
    the rows where human_<label> is non-empty. Values are compared lowercased
    and stripped. Labels with no filled rows get n=0 and NaN kappa/accuracy.
    Accepts model columns named either model_<label> or plain <label>.
    """
    with open(in_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = set(rows[0].keys()) if rows else set()

    # Labels present in the sheet: every human_<x> column.
    names = [c[len("human_"):] for c in (rows[0].keys() if rows else [])
             if c and c.startswith("human_")] or _label_names()

    out: dict[str, dict] = {}
    for name in names:
        model_col = f"model_{name}" if f"model_{name}" in fields else name
        human, model = [], []
        for r in rows:
            h = (r.get(f"human_{name}") or "").strip().lower()
            if not h:
                continue
            human.append(h)
            model.append((r.get(model_col) or "").strip().lower())
        n = len(human)
        acc = sum(1 for x, y in zip(human, model) if x == y) / n if n else float("nan")
        out[name] = {"n": n, "kappa": kappa(human, model) if n else math.nan, "accuracy": acc}
    return out
