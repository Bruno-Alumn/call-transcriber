"""Loading transcript lines and deriving conversation structure.

Transcripts live in `transcripts/YYYY-MM-DD.jsonl`, one JSON object per line
(see SPEC.md, "Data"). This module turns them into `Line` objects for a single
server (guild), and derives calls, replies and overlaps from timing alone.

Privacy rule: users on the exclusion list are dropped inside `load_lines`
before any other filtering or processing, so nothing downstream (labels,
cache, features, reports) ever sees their lines.
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

# Whisper's well-known phantom outputs on silence. A local copy on purpose:
# study must not import from the server package.
HALLUCINATIONS = {"thank you.", "thanks for watching.", "you", "bye.", ""}

# Transcript files are named by date; anything else (e.g. the .md exports) is ignored.
_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")

# Minimum lead an interrupter must have over the end of the line it cuts into.
OVERLAP_MIN_MS = 300


@dataclass
class Line:
    t: int
    end: int
    user: str
    speaker: str
    text: str
    ch: str
    ch_name: str
    guild: str
    guild_name: str
    call: str = ""   # call id, set by group_calls
    idx: int = 0     # position in the loaded list (stable id within a run)
    key: str = ""    # sha1(f"{t}|{user}|{text}")[:16], stable across runs


def line_key(t: int, user: str, text: str) -> str:
    """Stable id for a line across runs (used by the label cache)."""
    return hashlib.sha1(f"{t}|{user}|{text}".encode("utf-8")).hexdigest()[:16]


def _transcript_files(transcripts_dir: Path, since: str | None = None,
                      until: str | None = None) -> list[tuple[str, Path]]:
    """(date, path) for every YYYY-MM-DD.jsonl in the folder, within [since, until], by date."""
    out = []
    d = Path(transcripts_dir)
    if not d.is_dir():
        return out
    for p in d.iterdir():
        m = _FILE_RE.match(p.name)
        if not m or not p.is_file():
            continue
        date = m.group(1)
        # ISO dates compare correctly as strings.
        if since and date < since:
            continue
        if until and date > until:
            continue
        out.append((date, p))
    out.sort()
    return out


def _iter_records(path: Path):
    """Yield the JSON objects in one .jsonl file, skipping blank or corrupt lines."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue  # a half-written last line, for example
            if isinstance(rec, dict):
                yield rec


def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_lines(transcripts_dir: Path, guild: str, since: str | None = None, until: str | None = None,
               exclude: set[str] = frozenset(), stats: dict | None = None) -> list[Line]:
    """Load every usable line of one server, sorted by start time.

    Order of filters matters: excluded users are removed first, before any
    other check, so their text is never even inspected further. Then lines
    from other servers, lines without a user or server, too-short lines
    (< 2 words) and known Whisper hallucinations are dropped.

    stats: optional dict; if given, stats["excluded_lines"] is filled with
    excluded id -> number of raw lines of THIS server (in the date range) that
    were dropped for it. Ids that matched nothing are present with 0, so the
    caller can warn about an id that may be wrong.
    """
    exclude = {str(u) for u in exclude}
    dropped = {u: 0 for u in exclude}
    guild = str(guild)
    lines: list[Line] = []
    for _date, path in _transcript_files(transcripts_dir, since, until):
        for rec in _iter_records(path):
            user = str(rec.get("user") or "")
            # 1. Exclusion list, before anything else touches the line.
            if user in exclude:
                if str(rec.get("guild") or "") == guild:
                    dropped[user] += 1
                continue
            # 2. Attribution: old lines without a server or user are unusable.
            if not user or not rec.get("ch") or not rec.get("guild"):
                continue
            if str(rec.get("guild")) != guild:
                continue
            # 3. Content filters.
            text = str(rec.get("text") or "").strip()
            if text.lower() in HALLUCINATIONS or len(text.split()) < 2:
                continue
            t = _to_int(rec.get("t"))
            if t is None:
                continue
            end = _to_int(rec.get("end"))
            if end is None or end < t:
                end = t  # zero-length rather than negative duration
            lines.append(Line(
                t=t, end=end, user=user,
                speaker=str(rec.get("speaker") or ""),
                text=text,
                ch=str(rec.get("ch")), ch_name=str(rec.get("chName") or ""),
                guild=guild, guild_name=str(rec.get("guildName") or ""),
            ))
    # Files are in append order, not strictly time order; sort (stable) by start.
    lines.sort(key=lambda ln: ln.t)
    for i, ln in enumerate(lines):
        ln.idx = i
        ln.key = line_key(ln.t, ln.user, ln.text)
    if stats is not None:
        stats["excluded_lines"] = dropped
    return lines


# A Discord user id (a "snowflake"): 15-21 digits in practice.
_ID_RE = re.compile(r"^\d{15,21}$")
# A pasted mention, <@123...> or <@!123...>, is accepted and reduced to the id.
_MENTION_RE = re.compile(r"^<@!?(\d+)>$")


def _decode_text_file(raw: bytes) -> str:
    """Decode a small text file written by any common editor or shell.

    UTF-16 (with a BOM, as PowerShell 5 `>` writes it, or without one) and
    UTF-8 with or without a BOM (Notepad's "UTF-8 with BOM") are handled. A
    BOM left in place would glue itself to the first id and silently stop
    that person from being excluded, which is why this is not a plain
    read_text().
    """
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    # UTF-16 without a BOM: ASCII digits interleaved with NUL bytes.
    if len(raw) >= 4 and raw.count(b"\x00") >= len(raw) // 3:
        le = raw[1::2].count(b"\x00") > raw[0::2].count(b"\x00")
        return raw.decode("utf-16-le" if le else "utf-16-be")
    return raw.decode("utf-8-sig")


def load_exclude(path: Path | None) -> set[str]:
    """Read an exclusion list: one user id per line, '#' starts a comment.

    A missing path or file gives an empty set (the CLI refuses a path that was
    given but does not exist, before calling this).

    The list fails CLOSED: an entry that is not a plain Discord user id after
    decoding and trimming raises ValueError naming it, rather than being kept
    as a string that matches nobody (which would leave that person in the
    study while the report says they were removed). A pasted mention such as
    <@123...> or <@!123...> is accepted and reduced to the id.
    """
    if path is None:
        return set()
    p = Path(path)
    if not p.is_file():
        return set()
    try:
        text = _decode_text_file(p.read_bytes())
    except UnicodeDecodeError as e:
        raise ValueError(f"{p}: cannot read the exclusion list as UTF-8 or UTF-16 ({e})") from e
    out, bad = set(), []
    for n, raw in enumerate(text.splitlines(), 1):
        entry = raw.split("#", 1)[0].strip().strip("\ufeff").strip()
        if not entry:
            continue
        m = _MENTION_RE.match(entry)
        if m:
            entry = m.group(1)
        if _ID_RE.match(entry):
            out.add(entry)
        else:
            bad.append(f"line {n}: {entry[:40]!r}")
    if bad:
        raise ValueError(f"{p}: not a user id (expected 15-21 digits, one per line): " + "; ".join(bad[:5])
                         + (f" (and {len(bad) - 5} more)" if len(bad) > 5 else ""))
    return out


def list_guilds(transcripts_dir: Path) -> list[dict]:
    """Summarise the servers present in the transcripts.

    Returns [{"guild", "guildName", "lines", "first", "last"}] sorted by line
    count, descending. "first"/"last" are the dates (file names) of the first
    and last transcript file containing that server. Counts every attributed
    line (no content filters), as a quick overview.
    """
    info: dict[str, dict] = {}
    for date, path in _transcript_files(transcripts_dir):
        for rec in _iter_records(path):
            g = rec.get("guild")
            if not g or not rec.get("ch"):
                continue
            g = str(g)
            row = info.setdefault(g, {"guild": g, "guildName": "", "lines": 0,
                                      "first": date, "last": date})
            row["lines"] += 1
            row["first"] = min(row["first"], date)
            row["last"] = max(row["last"], date)
            # Keep the most recent non-empty name (servers get renamed).
            if rec.get("guildName"):
                row["guildName"] = str(rec["guildName"])
    return sorted(info.values(), key=lambda r: (-r["lines"], r["guild"]))


def group_calls(lines: list[Line], gap_ms: int = 15 * 60 * 1000) -> dict[str, list[Line]]:
    """Split lines into calls and set `line.call` on each.

    A call is a run of consecutive lines in one channel where each line starts
    less than `gap_ms` after the previous one ended (or started, if that is
    later; overlapping speech keeps the call going). Call id: f"{ch}@{first_t}".
    Returns call id -> lines in time order.
    """
    by_ch: dict[str, list[Line]] = {}
    for ln in sorted(lines, key=lambda x: x.t):
        by_ch.setdefault(ln.ch, []).append(ln)
    calls: dict[str, list[Line]] = {}
    for ch, chl in by_ch.items():
        current: list[Line] = []
        last_end = None  # latest end time seen in the current call
        for ln in chl:
            if current and ln.t - last_end >= gap_ms:
                calls[f"{ch}@{current[0].t}"] = current
                current, last_end = [], None
            current.append(ln)
            last_end = ln.end if last_end is None else max(last_end, ln.end)
        if current:
            calls[f"{ch}@{current[0].t}"] = current
    for cid, cl in calls.items():
        for ln in cl:
            ln.call = cid
    return calls


def _by_call(lines: list[Line]) -> dict[str, list[Line]]:
    """Lines grouped by call id, each group sorted by start time.

    If group_calls has not been run (empty call ids), it is run here so that
    replies/overlaps never pair lines across separate calls.
    """
    if any(not ln.call for ln in lines):
        group_calls(lines)
    groups: dict[str, list[Line]] = {}
    for ln in lines:
        groups.setdefault(ln.call, []).append(ln)
    for g in groups.values():
        g.sort(key=lambda x: x.t)
    return groups


def replies(lines: list[Line], max_gap_ms: int = 3000) -> list[tuple[str, str, int]]:
    """Detect reply turns from timing.

    For each line by user B, take the most recent earlier-starting line in the
    same call by a different user A. If B starts no later than `max_gap_ms`
    after A's END, it counts as B replying to A: (B, A, latency_ms) with
    latency = B.t - A.end. The latency is negative when B started before A
    finished (an overlap). Each line yields at most one reply.

    Only a line that TAKES the turn counts: B's own previous line in the call
    must have started before A's line. Whisper often cuts one turn into
    several segments; without this rule the 2nd, 3rd ... segment of B's turn
    would each count as another reply to A (with ever larger latencies), so
    people whose speech gets segmented more would look more responsive.
    """
    out: list[tuple[str, str, int]] = []
    for group in _by_call(lines).values():
        # The latest line of the two most recent distinct speakers is enough
        # to find "the most recent line by someone other than B".
        recent: list[Line] = []  # up to 2 lines, latest first, distinct users
        for ln in group:
            prev = next((r for r in recent if r.user != ln.user), None)
            own = next((r for r in recent if r.user == ln.user), None)
            taking_turn = own is None or (prev is not None and own.t < prev.t)
            if prev is not None and prev.t < ln.t and taking_turn:
                latency = ln.t - prev.end
                if latency <= max_gap_ms:
                    out.append((ln.user, prev.user, int(latency)))
            # Update the "latest line per distinct user" window.
            recent = [ln] + [r for r in recent if r.user != ln.user][:1]
    return out


def overlaps(lines: list[Line]) -> list[tuple[str, str]]:
    """Detect interruptions from timing.

    (B, A) is emitted when B's line starts while A's line (same call,
    different user) is still running, at least OVERLAP_MIN_MS before A's end,
    and A started first. One pair per (B line, A line) combination.
    """
    out: list[tuple[str, str]] = []
    for group in _by_call(lines).values():
        starts = [ln.t for ln in group]
        # No line can still be running if it started more than the longest
        # line duration in this call ago; that bounds the backwards scan.
        longest = max(ln.end - ln.t for ln in group)
        for j, b in enumerate(group):
            # Candidates are lines that started strictly before B.
            i = bisect_right(starts, b.t - 1, 0, j) - 1
            while i >= 0 and b.t - group[i].t <= longest:
                a = group[i]
                if a.user != b.user and b.t <= a.end - OVERLAP_MIN_MS:
                    out.append((b.user, a.user))
                i -= 1
    return out
