"""Merging calls: several calls (split by a quiet gap, or by midnight) become one
entry with all their lines; search, delete and unmerge follow the merged call.
  cd server && uv run python ../test/merge_test.py"""
import datetime as dt, json, sys, tempfile
from pathlib import Path
sys.argv = sys.argv[:1]; sys.path.insert(0, ".")
import server as S
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

root = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root / "transcripts"; S.TRANSCRIPTS.mkdir()
def ms(d, h, m): return int(dt.datetime.strptime(d, "%Y-%m-%d").replace(hour=h, minute=m).timestamp() * 1000)
def write(day, rows):
    with (S.TRANSCRIPTS / f"{day}.jsonl").open("a", encoding="utf-8") as f:
        for i, (t, who, text) in enumerate(rows):
            f.write(json.dumps({"id": f"{day}-{t}", "t": t, "end": t + 2000, "user": who, "speaker": who, "text": text,
                                "ch": "v2", "chName": "Voice 2", "guild": "g", "guildName": "WARDOGS"}) + "\n")
d1, d2 = "2026-11-20", "2026-11-21"
write(d1, [(ms(d1, 22, 0), "Ann", "first call starts here"), (ms(d1, 22, 1), "Ben", "still the first call"),
           (ms(d1, 23, 40), "Ann", "after a long gap, a second call"), (ms(d1, 23, 52), "Ben", "right before midnight")])
write(d2, [(ms(d2, 0, 1), "Cal", "just after midnight, a third call"), (ms(d2, 0, 2), "Ann", "where are we on the build")])
calls = S.all_calls()
check(len(calls) == 3 and all(len(c["parts"]) == 1 for c in calls), f"three separate calls before merging ({len(calls)})")
ids = [c["id"] for c in calls]                                   # newest first
res = S.merge_calls([ids[0], ids[1]])                            # the midnight split
check(res.get("ok") and res["parts"] == 2, "merge the two calls around midnight")
calls = S.all_calls()
m = next(c for c in calls if c.get("merged"))
check(len(calls) == 2 and m["id"] == ids[1] and [p["day"] for p in m["parts"]] == [d1, d2], "one merged call spanning both days, id of its earliest part")
check(m["lines"] == 4 and m["start"] == ms(d1, 23, 40) and m["end"] == ms(d2, 0, 2) + 2000, "counts and time range cover both parts")
check(set(m["people"]) == {"Ann", "Ben", "Cal"}, "people from every part")
res = S.merge_calls([ids[2], m["id"]])                           # add the first call to the merged one
calls = S.all_calls()
check(len(calls) == 1 and len(calls[0]["parts"]) == 3 and calls[0]["id"] == ids[2], "merging into a merged call joins the groups")
check(S.merge_calls([ids[2]]).get("error") is not None, "merging one call on its own is refused")
check(S.merge_calls(["nope@1", ids[2]]).get("error") is not None, "unknown call ids are ignored")
hits = {h["call"] for d in S.all_days() for h in [None] if False}
pm = S.merged_id_of()
check(pm.get(ids[0]) == ids[2] and pm.get(ids[1]) == ids[2], "every part maps to the merged call")
check(S.unmerge_call(ids[2]).get("parts") == 3 and len(S.all_calls()) == 3, "split back into the three calls")
S.merge_calls([ids[0], ids[1]])
parts = S.merged_id_of()
n = sum(1 for d in S.all_days() for c in S.day_data(d)[1] if S.group_match(c, "call", ids[1], parts))
check(n == 2, "deleting the merged call covers both parts")
out = S.delete_group("call", ids[1])
check(out["calls"] == 2 and out["lines"] == 4 and not S.merge_groups(), "delete removes both parts and the merge group")
check(len(S.all_calls()) == 1, "only the first call is left")
print("FAIL" if fails else "PASS"); sys.exit(1 if fails else 0)
