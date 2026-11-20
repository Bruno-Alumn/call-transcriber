"""Personality estimates from counted language features.

This module turns per-person feature dicts (from study/features.py) into Big
Five trait scores, and provides the statistics used to judge how much those
scores can be trusted:

  * PRIOR_WEIGHTS / prior_scores: an uncalibrated score built from the
    directions of published language-personality correlations.
  * bootstrap_ci: per-person confidence intervals by resampling calls.
  * split_half: internal reliability (odd vs even calls, Spearman-Brown).
  * fit_calibration / apply_calibration: ridge regression against
    questionnaire answers, with honest leave-one-out validity.
  * load_questionnaire: read those answers from a CSV.

numpy only. All scores are RELATIVE TO THE SERVER they were computed on.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Callable

import numpy as np

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# ---------------------------------------------------------------------------
# Prior weights
# ---------------------------------------------------------------------------
# Each weight is roughly the size of the correlation reported between that
# kind of language and the trait (typical |r| 0.05-0.30), with the sign of the
# reported direction. The weights are only ever used relative to each other
# (prior_scores divides by sum(|w|) and re-z-scores), so what matters is the
# direction and the rough ordering of magnitudes.
#
# Sources (short names used in the comments below):
#   PK99   Pennebaker & King 1999, J Pers Soc Psychol 77(6): LIWC categories in
#          student essays vs Big Five (the "LIWC style" findings).
#   M06    Mehl, Gosling & Pennebaker 2006, J Pers Soc Psychol 90(5): everyday
#          spoken language recorded with the EAR vs Big Five. The closest match
#          to spoken voice-chat transcripts.
#   M07    Mairesse, Walker, Mehl & Moore 2007, JAIR 30: conversational speech
#          and text features (LIWC, MRC, utterance type, prosody) -> Big Five.
#   Y10    Yarkoni 2010, J Res Pers 44(3): LIWC categories in ~700 bloggers.
#   S13    Schwartz et al. 2013, PLoS ONE 8(9): open-vocabulary analysis of
#          75,000 Facebook users.
#   T19    Tackman et al. 2019, J Pers Soc Psychol 116(5): I-talk meta-analysis.
#
# Features ending in _share come from the small LLM's per-line labels, not
# from word counts. The literature measured words, not labels, so for those we
# use the nearest word-category finding as an analogue and keep the weight
# small. Features with no published estimate at all (interruptions, fillers)
# get the smallest weights and say so.
#
# Deliberately NOT used: words, lines, talk_s, calls (they measure how much
# data we have on a person, not their style), active_hours_entropy and
# reply_latency_median_ms (no published personality link we trust).

PRIOR_WEIGHTS: dict[str, dict[str, float]] = {
    "openness": {
        # PK99, Y10: open people use longer words (PK99 "six-letter words")
        # and more articles / complex language.
        "mean_word_len": 0.15,
        # M07 used lexical richness features; higher type-token ratio went
        # with openness in their text models. Moderate.
        "ttr_mattr": 0.10,
        # PK99, Y10: fewer first-person singular pronouns ("immediacy" factor
        # loads negatively on openness).
        "i_rate": -0.10,
        # PK99: weak positive link with tentative / "making distinctions".
        "tentative_rate": 0.05,
        # Y10, S13: weak negative link with swearing.
        "swear_rate": -0.05,
        # Analogue (weak): curiosity expressed as questions. No direct
        # published estimate on labelled question share.
        "question_share": 0.05,
    },
    "conscientiousness": {
        # M06 (spoken), Y10, S13: fewer swear words; one of the most
        # replicated findings for conscientiousness.
        "swear_rate": -0.15,
        # PK99, Y10: fewer negative emotion words.
        "neg_rate": -0.10,
        # Y10: fewer anger words.
        "anger_rate": -0.10,
        # Y10, S13 ("work", "ready", planning words): more achievement words.
        "achieve_rate": 0.10,
        # Extrapolated, weak: disfluencies / fillers. M07 included filler-type
        # features; links were small. No direct estimate for voice chat.
        "filler_rate": -0.05,
        # Extrapolated, weak: interrupting others. No published estimate;
        # kept tiny so it cannot drive the score.
        "interrupts_per_100_lines": -0.05,
    },
    "extraversion": {
        # M06: amount of talking in daily life is the strongest spoken
        # correlate of extraversion (r around .3). We use the person's share of
        # talk relative to an equal share among the people in each call
        # (talk_share_rel), which removes data volume AND call size; the raw
        # share would make people who hang out in small calls look extraverted.
        "talk_share_rel": 0.20,
        # Y10, PK99: more social-process words.
        "social_rate": 0.15,
        # PK99, Y10: more positive emotion words.
        "pos_rate": 0.15,
        # M07 (speech): longer utterances among extravert markers. Moderate.
        "words_per_line": 0.10,
        # M07 used speech rate as a prosodic feature; faster speech is a
        # classic extraversion marker (Scherer 1979, cited by M07). Moderate.
        "wpm": 0.10,
        # Y10, S13 ("we", "everyone", party words): more inclusive pronouns.
        "we_rate": 0.10,
        # PK99, M07: fewer tentative words.
        "tentative_rate": -0.10,
        # Analogue: M06 found laughing in daily life went with extraversion;
        # we approximate it with the share of lines labelled as jokes.
        "joke_share": 0.10,
        # Analogue: time spent in social (vs task) talk, M06 "socializing".
        "social_ctx_share": 0.10,
    },
    "agreeableness": {
        # M06 (spoken), Y10, S13: fewer swear words; the strongest single
        # language marker of low agreeableness (|r| around .2).
        "swear_rate": -0.20,
        # Y10: fewer anger words.
        "anger_rate": -0.15,
        # Y10, PK99: more positive emotion words.
        "pos_rate": 0.15,
        # Y10: fewer negative emotion words.
        "neg_rate": -0.10,
        # Analogue of positive emotion / anger at the line level (LLM tone).
        "warm_share": 0.15,
        "hostile_share": -0.15,
        # Analogue of LIWC assent words (Y10) and of argumentativeness (M07
        # dialogue-act features), labelled per line.
        "agree_share": 0.10,
        "disagree_share": -0.10,
    },
    "neuroticism": {
        # PK99, Y10: more negative emotion words.
        "neg_rate": 0.15,
        # Y10 (around .15): more anxiety words.
        "anxiety_rate": 0.15,
        # Y10: more sadness words.
        "sadness_rate": 0.10,
        # Y10: more anger words.
        "anger_rate": 0.10,
        # T19 meta-analysis: I-talk and negative emotionality r around .10.
        "i_rate": 0.10,
        # PK99: weakly fewer positive emotion words.
        "pos_rate": -0.05,
        # Analogues of the word-level findings, labelled per line.
        "emo_fear_share": 0.10,
        "neg_sent_share": 0.10,
    },
}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _to_float(v) -> float:
    """Coerce a feature value to float; anything unusable becomes NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """Z-score each column of X across rows, ignoring NaN.

    Columns with fewer than 2 finite values, or zero variance, carry no
    information about differences between people: they become all zeros.
    NaN cells become 0 (the server average) after scoring.
    """
    X = np.asarray(X, dtype=float)
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        mu = col[ok].mean()
        sd = col[ok].std(ddof=1)
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        z = (col - mu) / sd
        z[~ok] = 0.0
        out[:, j] = z
    return out


def _pearson(a, b) -> float:
    """Pearson r over pairs where both values are finite; NaN if undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return math.nan
    a, b = a[ok], b[ok]
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if den <= 1e-12:
        return math.nan
    return float((a * b).sum() / den)


# ---------------------------------------------------------------------------
# Prior scores
# ---------------------------------------------------------------------------

def zscore_matrix(people: dict[str, dict[str, float]], feats: list[str]) -> tuple[list[str], np.ndarray]:
    """Rows = users (sorted), columns = feats, z-scored across users.

    A feature missing from a user's dict counts as NaN. NaN -> 0 after
    scoring; zero-variance columns are all 0 (see _zscore_columns).
    """
    users = sorted(people)
    X = np.full((len(users), len(feats)), np.nan)
    for i, u in enumerate(users):
        row = people[u] or {}
        for j, f in enumerate(feats):
            X[i, j] = _to_float(row.get(f, math.nan))
    if len(users) == 0:
        return users, X
    return users, _zscore_columns(X)


def prior_scores(people: dict[str, dict[str, float]],
                 weights: dict[str, dict[str, float]] = PRIOR_WEIGHTS) -> dict[str, dict[str, float]]:
    """Literature-weighted trait scores, relative to this server.

    For each trait: score = sum(w * z) / sum(|w|) over that trait's features,
    then the scores are z-scored again across users so that 0 = server
    average and +-1 = one standard deviation. A trait with no variation
    (e.g. a single user) scores 0 for everyone.

    A user with a missing (NaN) feature gets z = 0 for it, which pulls their
    score toward the server average: the conservative choice.
    """
    users = sorted(people)
    result: dict[str, dict[str, float]] = {u: {} for u in users}
    if not users:
        return result
    for trait, wmap in weights.items():
        feats = list(wmap)
        w = np.array([float(wmap[f]) for f in feats])
        denom = float(np.abs(w).sum())
        if not feats or denom <= 0:
            raw = np.zeros(len(users))
        else:
            _, Z = zscore_matrix(people, feats)
            raw = Z @ w / denom
        scores = _zscore_columns(raw.reshape(-1, 1))[:, 0]
        for i, u in enumerate(users):
            result[u][trait] = float(scores[i])
    return result


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

# Offset applied to the k-th extra copy of a resampled call, so duplicated
# calls do not overlap in time with the original (which would create fake
# replies/overlaps). A whole number of weeks keeps hour-of-day and weekday.
_COPY_OFFSET_MS = 1000 * 7 * 24 * 3600 * 1000


def _shifted_copy(line, k: int):
    """Return the line itself for k == 0, else a time-shifted copy.

    The copy keeps its `key`, so label lookups by key still work; its call id
    gets a '#bK' suffix so it counts as a separate call.
    """
    if k == 0 or not dataclasses.is_dataclass(line):
        return line
    off = k * _COPY_OFFSET_MS
    changes = {"t": line.t + off, "end": line.end + off}
    if hasattr(line, "call"):
        changes["call"] = f"{line.call}#b{k}"
    return dataclasses.replace(line, **changes)


# A bootstrap interval needs this many distinct calls for the person: with
# fewer, resampling calls cannot show how much their behaviour varies from
# call to call, and the interval comes out falsely narrow (one call can only
# ever be drawn as itself). Below it the interval is (nan, nan).
MIN_CI_CALLS = 5


def bootstrap_ci(lines_by_user_call: dict[str, dict[str, list]],
                 compute: Callable[[list], dict[str, dict[str, float]]],
                 n: int = 200, seed: int = 0,
                 min_calls: int = MIN_CI_CALLS) -> dict[str, dict[str, tuple[float, float]]]:
    """Per-user, per-trait 95% percentile intervals by resampling calls.

    lines_by_user_call: user -> call id -> that user's lines in that call.
    Pass EVERY speaker's lines (not only the people you want intervals for):
    the scores are relative and interaction features need the other side.

    Cluster bootstrap over calls: in each of `n` replicates the server's calls
    (the union over users) are drawn with replacement, as many draws as there
    are calls, and every user's lines from each drawn call are kept, so people
    keep their real conversation partners. A call drawn more than once is
    time-shifted for each extra copy (see _shifted_copy). The pooled lines,
    sorted by time, are passed to `compute`. A user who was in none of the
    drawn calls simply has no value for that replicate.

    (Resampling each user's calls independently, as an earlier version did,
    broke shared calls apart: most replicate calls lost some participants,
    which inflated talk share by ~80% and halved reply rates.)

    Because scores are relative to the server, the interval reflects both the
    user's own sampling noise and that of everyone else. Deterministic with
    `seed`. Users with fewer than `min_calls` distinct calls, and user/trait
    pairs with no finite replicate, get (nan, nan).
    """
    rng = np.random.default_rng(seed)
    users = sorted(u for u, calls in lines_by_user_call.items() if calls)
    call_ids = sorted({c for u in users for c in lines_by_user_call[u]})
    # call id -> every user's lines in that call
    by_call: dict[str, list] = {c: [] for c in call_ids}
    for u in users:
        for c, lns in lines_by_user_call[u].items():
            by_call[c].extend(lns)
    draws: dict[str, dict[str, list[float]]] = {u: {} for u in users}

    for _ in range(max(int(n), 0) if call_ids else 0):
        picks = rng.integers(0, len(call_ids), size=len(call_ids))
        seen: dict[str, int] = {}
        subset = []
        for p in picks:
            cid = call_ids[int(p)]
            k = seen.get(cid, 0)
            seen[cid] = k + 1
            subset.extend(_shifted_copy(ln, k) for ln in by_call[cid])
        subset.sort(key=lambda ln: (getattr(ln, "t", 0), getattr(ln, "idx", 0)))
        scores = compute(subset) or {}
        for u in users:
            for trait, v in (scores.get(u) or {}).items():
                draws[u].setdefault(trait, []).append(_to_float(v))

    out: dict[str, dict[str, tuple[float, float]]] = {}
    for u in users:
        out[u] = {}
        enough = len(lines_by_user_call[u]) >= min_calls
        for trait in sorted(set(TRAITS) | set(draws[u])):
            vals = np.array(draws[u].get(trait, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0 or not enough:
                out[u][trait] = (math.nan, math.nan)
            else:
                lo, hi = np.percentile(vals, [2.5, 97.5])
                out[u][trait] = (float(lo), float(hi))
    return out


# ---------------------------------------------------------------------------
# Split-half reliability
# ---------------------------------------------------------------------------

def split_half(score_fn: Callable[[list], dict[str, dict[str, float]]], lines: list,
               info: dict | None = None) -> dict[str, float]:
    """Odd/even-call split-half reliability per trait, Spearman-Brown corrected.

    All calls of the server are ordered by start time (the earliest line of
    the call) and WHOLE calls are assigned alternately to half A (1st, 3rd,
    ...) and half B (2nd, 4th, ...), so every speaker in a call lands in the
    same half and interactions stay intact. Pass every speaker's lines; users
    whose calls all fall in one half drop out of the comparison.

    (An earlier version gave each user their own odd/even call order. A shared
    call could then be in A for one person and in B for the others, which made
    occasional attenders look alone in the call in a way that repeats in both
    halves: spurious "reliability" that measured attendance patterns.)

    score_fn is applied to each half; the per-trait Pearson r across users
    present in both halves is corrected to full length with 2r / (1 + r).
    When r <= 0 the uncorrected r is returned instead: the correction is
    only meaningful for positive r (for r = -0.6 it would give -3.0), and a
    value <= 0 means "no consistency" either way. NaN when fewer than 5 users
    have finite scores in both halves.

    info: optional dict; filled with {"n_users": users compared,
    "n_calls": (calls in A, calls in B)}.

    Lines without a call id are grouped first with study.data.group_calls,
    which sets `call` on those line objects.
    """
    if any(not getattr(ln, "call", "") for ln in lines):
        from study.data import group_calls  # lazy: only needed for ungrouped input
        group_calls(lines)

    call_start: dict[str, int] = {}
    for ln in lines:
        cs = call_start.get(ln.call)
        if cs is None or ln.t < cs:
            call_start[ln.call] = ln.t
    order = {c: i for i, c in enumerate(sorted(call_start, key=lambda c: (call_start[c], c)))}

    half_a = [ln for ln in lines if order[ln.call] % 2 == 0]
    half_b = [ln for ln in lines if order[ln.call] % 2 == 1]
    sa = score_fn(half_a) or {}
    sb = score_fn(half_b) or {}

    traits = list(TRAITS)
    for s in (sa, sb):
        for d in s.values():
            for t in d or {}:
                if t not in traits:
                    traits.append(t)

    common = sorted(set(sa) & set(sb))
    out: dict[str, float] = {}
    n_used = 0
    for trait in traits:
        a = np.array([_to_float((sa[u] or {}).get(trait)) for u in common], dtype=float)
        b = np.array([_to_float((sb[u] or {}).get(trait)) for u in common], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        n_used = max(n_used, int(ok.sum()))
        if ok.sum() < 5:
            out[trait] = math.nan
            continue
        r = _pearson(a[ok], b[ok])
        if not math.isfinite(r):
            out[trait] = math.nan
        elif r <= 0:
            out[trait] = r          # uncorrected: Spearman-Brown is not meaningful here
        else:
            out[trait] = 2 * r / (1 + r)
    if info is not None:
        info["n_users"] = n_used
        info["n_calls"] = (len({c for c in order if order[c] % 2 == 0}),
                           len({c for c in order if order[c] % 2 == 1}))
    return out


# ---------------------------------------------------------------------------
# Calibration against questionnaire answers
# ---------------------------------------------------------------------------

MIN_CALIBRATION_USERS = 8


def _standardize(X: np.ndarray):
    """Column means/sds over finite values; returns (Z, mean, sd).

    NaN -> 0 after standardizing. Columns with < 2 finite values or zero
    variance get sd = inf, so they become all-zero and their weight is 0.
    """
    mu = np.zeros(X.shape[1])
    sd = np.full(X.shape[1], np.inf)
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = np.isfinite(col)
        if ok.sum() >= 2:
            mu[j] = col[ok].mean()
            s = col[ok].std(ddof=1)
            if np.isfinite(s) and s > 1e-12:
                sd[j] = s
    Z = (X - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z, mu, sd


def _ridge(Z: np.ndarray, y: np.ndarray, alpha: float):
    """Closed-form ridge on standardized Z and raw y; intercept = mean(y)."""
    b0 = float(y.mean())
    yc = y - b0
    p = Z.shape[1]
    A = Z.T @ Z + alpha * np.eye(p)
    try:
        w = np.linalg.solve(A, Z.T @ yc)
    except np.linalg.LinAlgError:
        # alpha = 0 with collinear/constant columns: minimum-norm solution.
        w = np.linalg.lstsq(A, Z.T @ yc, rcond=None)[0]
    return w, b0


# Alpha the CLI uses: fixed, never tuned on the LOO r that is then reported
# (that would inflate it). Larger values were tried and rejected: with ~10
# people and ~30 features, alpha 10 or 50 gave no more power to detect a real
# signal (7% at p < .05 for every alpha, simulated) and biased leave-one-out
# r DOWNWARD (mean r under no signal: 0.00 at alpha 1, -0.09 at 10, -0.31 at
# 50), because predictions shrink to the fold mean, which moves opposite to
# the held-out answer. At such sizes calibration is mostly noise whatever the
# alpha; the permutation chance level below is what says so honestly.
CLI_ALPHA = 1.0

# Permutations for the chance level of loo_r.
N_PERM = 200


def _loo_folds(Xt: np.ndarray):
    """Per held-out row: (training mask, standardized training Z, held-out z).

    Standardization does not depend on y, so it is done once and reused for
    the real answers and for every permutation.
    """
    folds = []
    for i in range(Xt.shape[0]):
        keep = np.arange(Xt.shape[0]) != i
        Ztr, mu_tr, sd_tr = _standardize(Xt[keep])
        zi = (Xt[i] - mu_tr) / sd_tr
        zi[~np.isfinite(zi)] = 0.0
        folds.append((keep, Ztr, zi))
    return folds


def _loo_r(folds, y: np.ndarray, alpha: float) -> float:
    """Pearson r between leave-one-out predictions and y."""
    preds = np.full(len(y), np.nan)
    for i, (keep, Ztr, zi) in enumerate(folds):
        if Ztr.shape[1]:
            wi, bi = _ridge(Ztr, y[keep], alpha)
            preds[i] = bi + float(zi @ wi)
        else:
            preds[i] = float(y[keep].mean())
    return _pearson(preds, y)


def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    """95% interval for a correlation via Fisher's z (needs n > 3)."""
    if not math.isfinite(r) or n <= 3:
        return (math.nan, math.nan)
    r = max(min(r, 0.999999), -0.999999)
    z, se = math.atanh(r), 1.0 / math.sqrt(n - 3)
    return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))


def fit_calibration(people: dict[str, dict[str, float]], truth: dict[str, dict[str, float]],
                    feats: list[str], alpha: float = 1.0, n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Ridge regression of questionnaire scores on z-scored features, per trait.

    Uses users that appear in both `people` and `truth`; raises ValueError
    with fewer than 8 such users. For each trait only users with a finite
    answer are used; a trait with fewer than 8 answers gets zero weights,
    intercept NaN and loo_r NaN.

    Leave-one-out: for each held-out user the standardization AND the ridge
    fit are redone on the others, so loo_r is an honest out-of-sample
    estimate (in-sample r would be inflated, badly so with ~40 features and
    ~10 people).

    Chance level: with ~10 people and ~30 features a LOO r of 0.3-0.6 is
    common even when the answers have nothing to do with the features. So
    the same LOO procedure is rerun `n_perm` times with the answers shuffled
    across people (seeded); "loo_null95" is the 95th percentile of those r
    values and "loo_p" the share of them >= the real r (with the +1
    correction). A real r below loo_null95 is indistinguishable from chance.
    "loo_ci" is a Fisher-z 95% interval for loo_r: rough only, since LOO
    predictions are not independent and LOO r is biased (downward under
    strong shrinkage); judge by loo_p / loo_null95. Not shown in report.md.

    Returns {"weights": {trait: {feat: w}}, "intercept": {trait: b0},
             "loo_r": {trait: r}, "loo_null95": {trait: r95}, "loo_p": {trait: p},
             "loo_ci": {trait: (lo, hi)}, "n": n_users,
             "n_by_trait": {trait: n}, "feature_mean": {feat: m},
             "feature_sd": {feat: sd or None for unused}, "alpha": alpha}.
    Weights apply to features z-scored with feature_mean / feature_sd
    (see apply_calibration).
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    users = sorted(u for u in people if u in truth)
    if len(users) < MIN_CALIBRATION_USERS:
        raise ValueError(
            f"calibration needs at least {MIN_CALIBRATION_USERS} people with both transcripts "
            f"and questionnaire answers; got {len(users)}")
    feats = list(feats)
    X = np.array([[_to_float((people[u] or {}).get(f)) for f in feats] for u in users],
                 dtype=float).reshape(len(users), len(feats))

    traits = list(TRAITS)
    for u in users:
        for t in truth[u] or {}:
            if t not in traits:
                traits.append(t)

    # Standardization stats on all calibration users (reported for applying).
    _, mu_all, sd_all = _standardize(X)

    weights: dict[str, dict[str, float]] = {}
    intercept: dict[str, float] = {}
    loo_r: dict[str, float] = {}
    loo_null95: dict[str, float] = {}
    loo_p: dict[str, float] = {}
    loo_ci: dict[str, tuple[float, float]] = {}
    n_by_trait: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for trait in traits:
        y_all = np.array([_to_float((truth[u] or {}).get(trait)) for u in users], dtype=float)
        ok = np.isfinite(y_all)
        n_by_trait[trait] = int(ok.sum())
        if ok.sum() < MIN_CALIBRATION_USERS:
            weights[trait] = {f: 0.0 for f in feats}
            intercept[trait] = math.nan
            loo_r[trait] = math.nan
            loo_null95[trait] = math.nan
            loo_p[trait] = math.nan
            loo_ci[trait] = (math.nan, math.nan)
            continue
        Xt, yt = X[ok], y_all[ok]

        # Final model: standardize with this trait's users, fit once.
        Zt, mu_t, sd_t = _standardize(Xt)
        w, b0 = _ridge(Zt, yt, alpha) if Xt.shape[1] else (np.zeros(0), float(yt.mean()))
        # Re-express weights against the all-user standardization so one
        # (feature_mean, feature_sd) pair applies to every trait.
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(np.isfinite(sd_t) & np.isfinite(sd_all), sd_all / sd_t, 0.0)
            w_all = w * scale
            shift = np.where(np.isfinite(sd_t), (mu_all - mu_t) / sd_t, 0.0)
        b0_all = b0 + float((w * shift).sum())
        weights[trait] = {f: float(w_all[j]) for j, f in enumerate(feats)}
        intercept[trait] = b0_all

        # Leave-one-out predictions, refitting everything without the held-out user.
        folds = _loo_folds(Xt)
        r_real = _loo_r(folds, yt, alpha)
        loo_r[trait] = r_real
        loo_ci[trait] = _fisher_ci(r_real, len(yt))
        # Permutation null: identical procedure, answers shuffled across people.
        null = np.array([_loo_r(folds, rng.permutation(yt), alpha) for _ in range(max(int(n_perm), 0))])
        null = null[np.isfinite(null)]
        if null.size and math.isfinite(r_real):
            loo_null95[trait] = float(np.percentile(null, 95))
            loo_p[trait] = float((1 + (null >= r_real).sum()) / (1 + null.size))
        else:
            loo_null95[trait] = math.nan
            loo_p[trait] = math.nan

    return {
        "weights": weights,
        "intercept": intercept,
        "loo_r": loo_r,
        "loo_null95": loo_null95,
        "loo_p": loo_p,
        "loo_ci": loo_ci,
        "n": len(users),
        "n_by_trait": n_by_trait,
        "feature_mean": {f: float(mu_all[j]) for j, f in enumerate(feats)},
        "feature_sd": {f: (float(sd_all[j]) if np.isfinite(sd_all[j]) else None)
                       for j, f in enumerate(feats)},
        "alpha": float(alpha),
    }


def apply_calibration(people: dict[str, dict[str, float]], calib: dict,
                      relative: bool = True) -> dict[str, dict[str, float]]:
    """Score users with a fit_calibration result.

    Features are z-scored with the calibration sample's mean/sd (missing or
    unused features contribute 0). With relative=True (default) the
    predictions are re-z-scored across these users, matching prior_scores'
    "relative to this server" scale; with relative=False they are on the
    questionnaire scale (0-100, see load_questionnaire). Traits whose
    intercept is NaN (too few answers) score NaN.
    """
    users = sorted(people)
    out: dict[str, dict[str, float]] = {u: {} for u in users}
    mean, sd = calib["feature_mean"], calib["feature_sd"]
    for trait, wmap in calib["weights"].items():
        b0 = _to_float(calib["intercept"].get(trait))
        pred = np.full(len(users), np.nan)
        if math.isfinite(b0):
            for i, u in enumerate(users):
                s = b0
                for f, w in wmap.items():
                    x, m, d = _to_float((people[u] or {}).get(f)), mean.get(f), sd.get(f)
                    if w and d and math.isfinite(x):
                        s += w * (x - m) / d
                pred[i] = s
            if relative:
                pred = _zscore_columns(pred.reshape(-1, 1))[:, 0]
        for i, u in enumerate(users):
            out[u][trait] = float(pred[i])
    return out


# ---------------------------------------------------------------------------
# Questionnaire CSV
# ---------------------------------------------------------------------------

def load_questionnaire(path) -> dict[str, dict[str, float]]:
    """Read questionnaire answers: user -> trait -> score on a 0-100 scale.

    CSV header: user plus any of the five trait names, in any order (case and
    surrounding spaces ignored; other columns are ignored). Blank or
    non-numeric cells become NaN; rows without a user id are skipped.

    Scale: if every non-missing trait value in the file lies within 1-5, the
    file is treated as a 1-5 Likert mean and mapped to 0-100 with
    (x - 1) / 4 * 100; otherwise it is taken as already 0-100. The check is
    file-wide so one file never mixes scales. (Values within 1-5 on a 0-100
    file would be implausible.) Values outside 0-100 after this raise
    ValueError, as they point to a wrong scale or a typo.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty questionnaire file")
        cols = {(name or "").strip().lower(): name for name in reader.fieldnames}
        if "user" not in cols:
            raise ValueError(f"{path}: questionnaire needs a 'user' column")
        present = [t for t in TRAITS if t in cols]
        if not present:
            raise ValueError(f"{path}: no trait columns found (expected some of {', '.join(TRAITS)})")
        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            user = (row.get(cols["user"]) or "").strip()
            if not user:
                continue
            rows[user] = {t: _to_float((row.get(cols[t]) or "").strip() or math.nan) for t in present}

    values = [v for r in rows.values() for v in r.values() if math.isfinite(v)]
    likert = bool(values) and all(1.0 <= v <= 5.0 for v in values)
    for r in rows.values():
        for t, v in r.items():
            if not math.isfinite(v):
                continue
            if likert:
                v = (v - 1.0) / 4.0 * 100.0
            if v < 0 or v > 100:
                raise ValueError(f"{path}: value {r[t]} for {t} is outside the 1-5 or 0-100 scale")
            r[t] = v
    return rows
