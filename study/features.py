"""Per-person counted features for study mode.

Everything here is plain counting over the transcript lines plus the per-line
labels produced by study.labels. No model is called. See study/SPEC.md for the
feature list; every value returned is a float (NaN where it cannot be computed).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime

from study import lexicon
from study.data import Line, group_calls, overlaps as find_overlaps, replies as find_replies

NAN = float("nan")

# Window size for the moving-average type-token ratio (MATTR).
MATTR_WINDOW = 50

# Rate feature name -> lexicon word list. Rates are per 100 words.
# The word sets are used directly (rather than lexicon.counts) so that the
# feature names do not depend on how counts() names its keys.
RATE_LISTS = {
    "i_rate": "I_WORDS",
    "we_rate": "WE_WORDS",
    "you_rate": "YOU_WORDS",
    "swear_rate": "SWEAR",
    "pos_rate": "POSITIVE",
    "neg_rate": "NEGATIVE",
    "anger_rate": "ANGER",
    "anxiety_rate": "ANXIETY",
    "sadness_rate": "SADNESS",
    "tentative_rate": "TENTATIVE",
    "certainty_rate": "CERTAINTY",
    "social_rate": "SOCIAL",
    "achieve_rate": "ACHIEVE",
    "filler_rate": "FILLERS",
    # Assent / backchannel words (yeah, ok, right ...). Kept apart from
    # POSITIVE as LIWC does: in voice chat they are mostly "I'm listening".
    "assent_rate": "ASSENT",
}

# Share feature name -> (label name, label value). Shares are the fraction of
# a person's LABELLED lines carrying that value (0..1).
SHARES = {
    "question_share": ("act", "question"),
    "joke_share": ("act", "joke"),
    "agree_share": ("act", "agreement"),
    "disagree_share": ("act", "disagreement"),
    "request_share": ("act", "request"),
    "pos_sent_share": ("sentiment", "positive"),
    "neg_sent_share": ("sentiment", "negative"),
    "warm_share": ("tone", "warm"),
    "hostile_share": ("tone", "hostile"),
    "emo_joy_share": ("emotion", "joy"),
    "emo_anger_share": ("emotion", "anger"),
    "emo_sadness_share": ("emotion", "sadness"),
    "emo_fear_share": ("emotion", "fear"),
    "social_ctx_share": ("context", "social"),
    "gameplay_ctx_share": ("context", "gameplay"),
}


def _split_wordlist(words) -> tuple[set[str], list[tuple[str, ...]]]:
    """Split a lexicon list into single words and multi-word phrases.

    Phrases (e.g. "you know") are matched as consecutive tokens. Each entry is
    run through lexicon.tokenize so phrases use the same token rules as text.
    """
    singles: set[str] = set()
    phrases: list[tuple[str, ...]] = []
    for entry in words:
        toks = lexicon.tokenize(entry)
        if len(toks) == 1:
            singles.add(toks[0])
        elif len(toks) > 1:
            phrases.append(tuple(toks))
    return singles, phrases


# Built once at import time.
_MATCHERS = {
    feat: _split_wordlist(getattr(lexicon, name, ()))
    for feat, name in RATE_LISTS.items()
}


def _count_matches(tokens: list[str], singles: set[str],
                   phrases: list[tuple[str, ...]]) -> int:
    """Count whole-word hits of single words plus phrase hits in a token list."""
    n = sum(1 for tok in tokens if tok in singles)
    for phrase in phrases:
        k = len(phrase)
        for i in range(len(tokens) - k + 1):
            if tuple(tokens[i:i + k]) == phrase:
                n += 1
    return n


def _mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio.

    Mean of the type/token ratio over every window of `window` consecutive
    tokens. NaN with fewer tokens than one window: a plain TTR on a short
    sample is on a different, upward-biased scale (near 1.0) and would make
    people with little data look lexically rich. NaN becomes z = 0 (the server
    average) in the trait scores. Uses a sliding counter so it is O(n).
    """
    n = len(tokens)
    if n < window:
        return NAN
    counter = Counter(tokens[:window])
    total = len(counter)  # sum of distinct-type counts across windows
    for i in range(window, n):
        out_tok = tokens[i - window]
        counter[out_tok] -= 1
        if counter[out_tok] == 0:
            del counter[out_tok]
        counter[tokens[i]] += 1
        total += len(counter)
    n_windows = n - window + 1
    return total / (n_windows * window)


def _entropy_bits(counts: list[int]) -> float:
    """Shannon entropy (bits) of a histogram; NaN for an empty histogram."""
    total = sum(counts)
    if total == 0:
        return NAN
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _median(values: list[float]) -> float:
    if not values:
        return NAN
    v = sorted(values)
    m = len(v) // 2
    return float(v[m]) if len(v) % 2 else (v[m - 1] + v[m]) / 2.0


def _per100(count: float, base: float) -> float:
    return 100.0 * count / base if base > 0 else NAN


def _label_of(labels: dict[str, dict], line: Line) -> dict | None:
    """The usable label dict for a line, or None when missing / unparsed."""
    lab = labels.get(line.key) if labels else None
    return lab if isinstance(lab, dict) else None


def person_features(lines: list[Line], labels: dict[str, dict], replies,
                    overlaps, min_words: int = 0,
                    turn_lines: list[Line] | None = None) -> dict[str, dict[str, float]]:
    """Compute the per-person feature dict (see SPEC.md for the list).

    lines:     loaded lines (already filtered for exclusions by data.load_lines).
    labels:    line.key -> label dict (or None when the model answer failed).
    replies:   output of data.replies: (responder, previous_speaker, latency_ms).
    overlaps:  output of data.overlaps: (interrupter, interrupted).
    min_words: people with fewer total words than this are left out of the
               result entirely (the report counts them separately).
    turn_lines: the line set `replies` and `overlaps` were computed on, when it
               differs from `lines` (e.g. --social-only: turn-taking on all
               lines, text features on social lines). The *_per_100_lines
               rates then divide by the person's line count in turn_lines, so
               numerator and denominator always come from the same lines.
               Default: `lines`. See features_for, which does this for you.

    If any line has no call id yet, data.group_calls is run on `lines` first
    (it sets line.call in place, same as the main pipeline does).
    """
    if any(not ln.call for ln in lines):
        group_calls(lines)

    # Talk time and distinct speakers per call (all speakers) for talk_share.
    call_talk_s: dict[str, float] = defaultdict(float)
    call_speakers: dict[str, set] = defaultdict(set)
    for ln in lines:
        call_talk_s[ln.call] += max(0, ln.end - ln.t) / 1000.0
        call_speakers[ln.call].add(ln.user)

    # Line counts for the turn-taking denominators (see turn_lines above).
    turn_n_lines = Counter(ln.user for ln in (lines if turn_lines is None else turn_lines))

    # Group lines per user, in time order (lines are sorted by t after load,
    # but sort again so callers passing subsets still get a stable order).
    by_user: dict[str, list[Line]] = defaultdict(list)
    for ln in sorted(lines, key=lambda x: (x.t, x.idx)):
        by_user[ln.user].append(ln)

    # Interaction counts.
    interrupts = Counter(a for a, _ in overlaps)
    interrupted = Counter(b for _, b in overlaps)
    replies_given = Counter(r for r, _, _ in replies)
    pos_latencies: dict[str, list[float]] = defaultdict(list)
    for responder, _, latency in replies:
        if latency > 0:
            pos_latencies[responder].append(float(latency))

    people: dict[str, dict[str, float]] = {}
    for user, ulines in by_user.items():
        tokens: list[str] = []
        per_line_tokens: list[list[str]] = []
        for ln in ulines:
            toks = lexicon.tokenize(ln.text)
            per_line_tokens.append(toks)
            tokens.extend(toks)
        n_words = len(tokens)
        if n_words < min_words:
            continue
        n_lines = len(ulines)

        talk_s = sum(max(0, ln.end - ln.t) for ln in ulines) / 1000.0
        user_calls = {ln.call for ln in ulines}
        calls_total_s = sum(call_talk_s[c] for c in user_calls)
        # talk_share_rel: own share of each call relative to an equal share
        # (1.0 = fair share among the speakers present), averaged over calls
        # weighted by call talk time. Raw talk_share scales with 1/call size,
        # so someone who mostly hangs out in 2-person calls would look twice as
        # talkative as an identical person in 4-person calls. Weighted mean of
        # (own_c / total_c * n_c) with weights total_c = sum(own_c * n_c) / sum(total_c).
        own_by_call: dict[str, float] = defaultdict(float)
        for ln in ulines:
            own_by_call[ln.call] += max(0, ln.end - ln.t) / 1000.0
        rel_num = sum(own * len(call_speakers[c]) for c, own in own_by_call.items())

        f: dict[str, float] = {
            "words": float(n_words),
            "lines": float(n_lines),
            "talk_s": talk_s,
            "calls": float(len(user_calls)),
            # Raw share of talk in the person's calls: descriptive only (it
            # depends on call size); the trait weights use talk_share_rel.
            "talk_share": talk_s / calls_total_s if calls_total_s > 0 else NAN,
            "talk_share_rel": rel_num / calls_total_s if calls_total_s > 0 else NAN,
            "words_per_line": n_words / n_lines if n_lines else NAN,
            "wpm": n_words / talk_s * 60.0 if talk_s > 0 else NAN,
            "ttr_mattr": _mattr(tokens),
            "mean_word_len": (sum(len(t) for t in tokens) / n_words) if n_words else NAN,
        }

        # Lexicon rates per 100 words. Phrases are matched within a line only,
        # so a phrase never spans two separate utterances.
        for feat, (singles, phrases) in _MATCHERS.items():
            hits = sum(_count_matches(toks, singles, phrases) for toks in per_line_tokens)
            f[feat] = _per100(hits, n_words)

        # Label shares over labelled lines only; NaN when none are labelled.
        labelled = [lab for lab in (_label_of(labels, ln) for ln in ulines) if lab is not None]
        for feat, (name, value) in SHARES.items():
            if labelled:
                f[feat] = sum(1 for lab in labelled if lab.get(name) == value) / len(labelled)
            else:
                f[feat] = NAN

        # Turn-taking. Denominator = the person's lines in the same line set the
        # interactions were counted on (turn_lines), never a filtered subset.
        n_turn = turn_n_lines.get(user, 0)
        f["interrupts_per_100_lines"] = _per100(interrupts[user], n_turn)
        f["interrupted_per_100_lines"] = _per100(interrupted[user], n_turn)
        f["reply_latency_median_ms"] = _median(pos_latencies.get(user, []))
        f["replies_given_per_100_lines"] = _per100(replies_given[user], n_turn)

        # Hour-of-day spread, in the machine's local time zone (same as habits).
        hours = [0] * 24
        for ln in ulines:
            hours[datetime.fromtimestamp(ln.t / 1000.0).hour] += 1
        f["active_hours_entropy"] = _entropy_bits(hours)

        people[user] = f
    return people


def social_only(lines: list[Line], labels: dict[str, dict]) -> list[Line]:
    """Lines whose context label is "social". Unlabelled lines are excluded."""
    out = []
    for ln in lines:
        lab = _label_of(labels, ln)
        if lab is not None and lab.get("context") == "social":
            out.append(ln)
    return out


def features_for(lines: list[Line], labels: dict[str, dict], social: bool = False,
                 include: set[str] | None = None) -> dict[str, dict[str, float]]:
    """The one feature definition used for the point estimate, the bootstrap
    and split-half, so all three describe the same thing.

    lines   ALL lines of the calls in question (every speaker, every topic).
            Turn-taking (replies, overlaps and their per-100-lines rates) is
            always measured on these: who answers whom does not depend on topic,
            and dropping speakers would distort everyone else's rates.
    social  if True, the text and label features (and talk share) come from
            the lines labelled as social chat only (see social_only).
    include if given, only these users are returned (they are still computed
            in the context of everyone else in the calls).
    """
    if any(not ln.call for ln in lines):
        group_calls(lines)
    lines = sorted(lines, key=lambda ln: (ln.t, ln.idx))
    feat_lines = social_only(lines, labels) if social else lines
    feats = person_features(feat_lines, labels, find_replies(lines), find_overlaps(lines),
                            min_words=0, turn_lines=lines)
    if include is not None:
        feats = {u: f for u, f in feats.items() if u in include}
    return feats
