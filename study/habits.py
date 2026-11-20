"""Server-wide habits: when people talk, how big and long calls are, how evenly
talk time is shared, who replies to whom, and a simple rule-based role per user.

Everything here is aggregate or per-user-id; display names are never used.
Excluded users are already removed by data.load_lines, so nothing here needs to
filter them again.

Only the Line attributes from the spec are used (t, end, user, ch, call), so
this module does not import study.data.

Roles (assigned in this priority order; the first rule that matches wins):

  host       Opened (spoke first in) at least HOST_MIN_OPENED calls, and those
             are at least HOST_MIN_OPEN_SHARE (30%) of the calls they were in.
             These are the people who get calls going.
  connector  Reply-network degree (distinct reply partners, either direction)
             of at least CONNECTOR_MIN_DEGREE (3), in the top quarter of all
             users' degrees (>= 75th percentile) AND strictly above the median
             degree. The median test matters on small servers where everyone
             has replied to everyone: then all degrees tie and nobody is a
             connector. They talk with many different people.
  lurker     Their average share of talk time in the calls they attend is less
             than half of an "equal share" (the mean of 1/people over those
             calls). Present, but mostly listening.
  regular    Everyone else.

The thresholds are module constants so they can be tuned and are reported as-is.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np

HOST_MIN_OPENED = 3
HOST_MIN_OPEN_SHARE = 0.30
CONNECTOR_MIN_DEGREE = 3
CONNECTOR_PERCENTILE = 75
LURKER_SHARE_FACTOR = 0.5
TOP_EDGES = 200
TOP_CONNECTORS = 10


def _duration_ms(line) -> int:
    """Spoken duration of a line in ms (never negative)."""
    return max(0, int(line.end) - int(line.t))


def _add_to_heatmap(heat: list[list[float]], t_ms: int, end_ms: int) -> None:
    """Add the minutes of [t_ms, end_ms) to the weekday x hour grid, splitting the
    span at local hour boundaries so a line crossing an hour counts in both."""
    if end_ms <= t_ms:
        return
    cur = datetime.fromtimestamp(t_ms / 1000)        # local time
    stop = datetime.fromtimestamp(end_ms / 1000)
    while cur < stop:
        next_hour = cur.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        seg_end = min(next_hour, stop)
        heat[cur.weekday()][cur.hour] += (seg_end - cur).total_seconds() / 60.0
        cur = seg_end


def gini(values) -> float:
    """Gini coefficient of non-negative values: 0 = perfectly equal, ->1 = one
    person has everything. NaN for no values, 0 when all values are zero."""
    x = np.sort(np.asarray([float(v) for v in values], dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    total = x.sum()
    if total <= 0:
        return 0.0
    # Standard formula on sorted values: G = (2 * sum(i * x_i) / (n * sum x)) - (n + 1) / n
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x)) / (n * total) - (n + 1.0) / n)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    return float(np.percentile(np.asarray(sorted_vals, dtype=float), q))


def server_habits(lines, calls: dict, replies) -> dict:
    """Aggregate habits for one server.

    lines   : list[Line] (already filtered to one guild, excluded users removed)
    calls   : {call_id: [Line, ...]} from data.group_calls
    replies : [(responder_user, previous_speaker_user, latency_ms), ...] from data.replies

    Returns the dict described in SPEC.md (heatmap, call_sizes, call_lengths_min,
    calls, people, hours_spoken, talk_gini, reply_network, roles).
    """
    # --- Heatmap: spoken minutes by weekday (Mon=0) x local hour -------------
    heat = [[0.0] * 24 for _ in range(7)]
    talk_ms: dict[str, int] = defaultdict(int)
    for ln in lines:
        _add_to_heatmap(heat, int(ln.t), int(ln.end))
        talk_ms[ln.user] += _duration_ms(ln)
    heat = [[round(v, 2) for v in row] for row in heat]

    # --- Calls: sizes, lengths, who opened each, per-user talk share ---------
    call_sizes: Counter = Counter()
    lengths_min: list[float] = []
    opened: Counter = Counter()                 # user -> calls they spoke first in
    attended: Counter = Counter()               # user -> calls they spoke in
    shares: dict[str, list[float]] = defaultdict(list)    # user -> talk share per call
    equal: dict[str, list[float]] = defaultdict(list)     # user -> 1/people per call
    for call_lines in calls.values():
        if not call_lines:
            continue
        cl = sorted(call_lines, key=lambda l: l.t)
        people = {l.user for l in cl}
        call_sizes[len(people)] += 1
        start = min(l.t for l in cl)
        end = max(max(l.end, l.t) for l in cl)
        lengths_min.append((end - start) / 60000.0)
        opened[cl[0].user] += 1
        per_user: dict[str, int] = defaultdict(int)
        for l in cl:
            per_user[l.user] += _duration_ms(l)
        total = sum(per_user.values())
        for u in people:
            attended[u] += 1
            # A call with zero total duration gives everyone an equal share.
            shares[u].append(per_user[u] / total if total > 0 else 1.0 / len(people))
            equal[u].append(1.0 / len(people))

    lengths_sorted = sorted(lengths_min)
    call_lengths = {
        "median": _percentile(lengths_sorted, 50),
        "p90": _percentile(lengths_sorted, 90),
        "mean": float(np.mean(lengths_sorted)) if lengths_sorted else float("nan"),
    }

    # --- Reply network --------------------------------------------------------
    # Edges are directed: [responder, previous_speaker, count]. Degree counts
    # distinct partners in either direction. Self-replies are ignored.
    edge_counts: Counter = Counter()
    partners: dict[str, set] = defaultdict(set)
    for item in replies:
        a, b = item[0], item[1]
        if not a or not b or a == b:
            continue
        edge_counts[(a, b)] += 1
        partners[a].add(b)
        partners[b].add(a)
    edges = [[a, b, c] for (a, b), c in
             sorted(edge_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_EDGES]]

    users = sorted({ln.user for ln in lines} | set(partners))
    degree = {u: len(partners.get(u, ())) for u in users}
    top_connectors = [u for u, d in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
                      if d > 0][:TOP_CONNECTORS]

    # --- Roles (rules in the module docstring) --------------------------------
    deg_sorted = sorted(degree.values())
    deg_cut = _percentile(deg_sorted, CONNECTOR_PERCENTILE) if degree else float("inf")
    deg_median = _percentile(deg_sorted, 50) if degree else float("inf")
    roles: dict[str, str] = {}
    for u in users:
        n_att = attended.get(u, 0)
        n_open = opened.get(u, 0)
        if n_open >= HOST_MIN_OPENED and n_att > 0 and n_open / n_att >= HOST_MIN_OPEN_SHARE:
            roles[u] = "host"
        elif (degree[u] >= CONNECTOR_MIN_DEGREE and degree[u] >= deg_cut
              and degree[u] > deg_median):
            roles[u] = "connector"
        elif shares.get(u) and np.mean(shares[u]) < LURKER_SHARE_FACTOR * np.mean(equal[u]):
            roles[u] = "lurker"
        else:
            roles[u] = "regular"

    return {
        "heatmap": heat,
        "call_sizes": {int(k): int(v) for k, v in sorted(call_sizes.items())},
        "call_lengths_min": call_lengths,
        "calls": sum(1 for c in calls.values() if c),
        "people": len({ln.user for ln in lines}),
        "hours_spoken": round(sum(talk_ms.values()) / 3_600_000.0, 3),
        "talk_gini": gini([talk_ms[u] / 1000.0 for u in talk_ms]),
        "reply_network": {
            "edges": edges,
            "degree": degree,
            "top_connectors": top_connectors,
        },
        "roles": roles,
    }
