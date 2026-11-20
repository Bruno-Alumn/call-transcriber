# Study mode: specification

Batch analysis of one server's saved transcripts: per-line labels from a small
local model, counted features, personality estimates grounded in published
research, reliability checks, and a server-wide report. Runs offline on the
files in `transcripts/`; the live app is not needed.

Scope of this version: the analysis mechanisms only. Announcements and consent
notices are out of scope (handled separately with the server owner). The
exclusion list IS in scope: excluded user ids must never reach any output,
model prompt or cache. Limitation (documented in report.md): the list removes what
excluded people SAID; other people's lines that mention them by name are still analysed.

Python 3.12, standard library + `numpy` only (scipy, pandas, networkx and
sklearn are not guaranteed on Windows). The LLM backends import their own
packages lazily (`mlx_vlm` on Apple Silicon, `torch` + `transformers` with an
NVIDIA GPU). Run from the repo root with the server's venv:

```
server/.venv/bin/python -m study <command> [options]        # Mac
server\.venv\Scripts\python.exe -m study <command> [options] # Windows
```

## Data

`transcripts/YYYY-MM-DD.jsonl`, one JSON object per line:
`{"t": ms, "end": ms, "user": "<user id or ''>", "speaker": "<display name>",
"text": "...", "ch": "<channel id>", "chName": "...", "guild": "<server id>",
"guildName": "..."}`. `ch`/`guild` may be missing on old lines (skip those
lines: they can't be attributed to a server). `user` may be empty on very old
lines (skip). Lines are sorted by `t` after loading (files are append-order).

## Package layout (each module owned by one author; keep to these interfaces)

### `study/data.py`
```python
@dataclass
class Line:
    t: int; end: int; user: str; speaker: str; text: str
    ch: str; ch_name: str; guild: str; guild_name: str
    call: str = ""          # call id, set by group_calls
    idx: int = 0            # position in the loaded list (stable id within a run)
    key: str = ""           # sha1(f"{t}|{user}|{text}")[:16], stable across runs

def load_lines(transcripts_dir: Path, guild: str, since: str | None = None, until: str | None = None,
               exclude: set[str] = frozenset(), stats: dict | None = None) -> list[Line]
    # only lines with line.guild == guild; dates are 'YYYY-MM-DD' inclusive (by file name);
    # drops excluded users FIRST; drops lines whose text has < 2 words or is in HALLUCINATIONS
    # (import HALLUCINATIONS from nowhere: define a local copy {"thank you.", "thanks for watching.", "you", "bye.", ""});
    # sorts by t; sets idx and key. stats (optional) gets {"excluded_lines": {id: lines dropped}}.
def load_exclude(path: Path | None) -> set[str]   # one user id per line, '#' comments allowed; missing file -> empty set
    # UTF-8 (with/without BOM) or UTF-16; <@id>/<@!id> mentions reduced to the id; any other entry that is
    # not 15-21 digits raises ValueError (fail closed: a typo must not silently leave someone in).
def list_guilds(transcripts_dir: Path) -> list[dict]  # [{"guild","guildName","lines","first","last"}] sorted by lines desc
def group_calls(lines: list[Line], gap_ms: int = 15*60*1000) -> dict[str, list[Line]]
    # a call = consecutive lines in one channel with gaps < gap_ms; call id f"{ch}@{first_t}"; sets line.call
def replies(lines: list[Line], max_gap_ms: int = 3000) -> list[tuple[str, str, int]]
    # (responder_user, previous_speaker_user, latency_ms) when a line from user B starts within
    # max_gap_ms after the END of the most recent line by a different user A in the same call
    # (latency may be negative when B starts before A finished: that is an overlap).
    # Only a line that takes the turn counts: B's own previous line in the call must have started
    # before A's line (so a turn Whisper split into several segments is one reply, not several).
def overlaps(lines: list[Line]) -> list[tuple[str, str]]
    # (interrupter, interrupted) when B's line starts while A's line (same call, A != B) is still running
    # and at least 300 ms before A's end
```

### `study/lexicon.py`
Small built-in word lists (lowercase, whole-word match) used for counted features:
`I_WORDS` (i, me, my, mine, myself, i'm, i've, i'll, i'd), `WE_WORDS`, `YOU_WORDS`,
`SWEAR`, `POSITIVE`, `NEGATIVE`, `ANGER`, `ANXIETY`, `SADNESS`, `TENTATIVE`
(maybe, perhaps, guess, probably, kinda, sorta, idk, think, unsure, ...),
`CERTAINTY` (always, never, definitely, clearly, sure, obviously, ...),
`SOCIAL` (friend, guys, bro, dude, everyone, together, talk, ...),
`ACHIEVE` (win, won, goal, plan, done, finish, try, ...), `FILLERS` (um, uh, you know; bare "like" is
not counted, it is mostly a verb), `ASSENT` (yeah, yes, ok, okay, right, sure, ...: backchannels, kept out
of POSITIVE as LIWC keeps assent apart from posemo). Gameplay verbs (kill, dead, died, attack, ...) are kept
out of NEGATIVE/ANGER/SADNESS.
50–200 entries each where the category is large. A `tokenize(text) -> list[str]` that lowercases,
keeps apostrophes inside words, and splits on everything else. `counts(tokens) -> dict[str,int]`
returning per-category counts.

### `study/labels.py`
Per-line labelling with a local LLM. One narrow question set per line, answers from fixed lists.
```python
LABELS = {
  "sentiment": ["positive", "neutral", "negative"],
  "act": ["question", "statement", "request", "joke", "agreement", "disagreement", "greeting"],
  "emotion": ["joy", "anger", "sadness", "fear", "surprise", "none"],
  "tone": ["warm", "neutral", "hostile"],
  "context": ["gameplay", "social", "logistics", "other"],
}
class Labeller(Protocol):
    name: str
    def label(self, items: list[tuple[str, str, str]]) -> list[dict | None]
        # items: (key, previous_line_text_or_'', text). Returns one dict per item with exactly the
        # LABELS keys and allowed values, or None when the model's answer couldn't be parsed after 1 retry.
def make_labeller(engine: str = "auto", model: str | None = None) -> Labeller
    # "auto": mlx on Apple Silicon if mlx_vlm imports, torch if torch.cuda.is_available(), else raise RuntimeError
    # with a clear message. mlx repo default "LiquidAI/LFM3-1.7B-MLX-8bit" (mlx_vlm load/generate/apply_chat_template,
    # see PersonalityReader in server/server.py for the exact calls); torch default "LiquidAI/LFM3-1.7B"
    # (AutoModelForCausalLM bf16 .to("cuda"), tokenizer.apply_chat_template(..., enable_thinking=False)).
    # temperature 0 / greedy, max ~60 new tokens. On mlx, decoding is constrained to a JSON schema of the
    # LABELS enums (mlx_vlm.structured, llguidance) when available, else free decoding. Labeller.name is
    # f"{backend}:{repo}:{labeller_version}" where the version hashes the prompt, token cap, LABELS, synonym
    # map and constrained flag, so a prompt change relabels instead of reusing stale cache entries.
    # engine "mlx"/"torch" check the hardware before loading a model and raise RuntimeError if it is missing.
class FakeLabeller:   # deterministic, no model: derives labels from lexicon counts (for tests and dry runs)
class LabelCache:     # JSON Lines at <out>/labels.jsonl: {"key","model","labels"}; get(key, model) / put(...)
def label_lines(lines, labeller, cache, progress=None, limit=None) -> dict[str, dict]
    # labels every line not already cached for labeller.name; returns key -> labels (or None).
    # A generation that RAISED (not an unparseable answer) is not cached and is left out, so it is retried.
```
Prompt design (important for a 1.7B model): a fixed system prompt that defines each label in one
sentence with the allowed values, and asks for ONE line of JSON with exactly those keys. The user
message gives the previous line (for context) and the line to label. Parse the first `{...}` in the
answer; lowercase values; near-synonyms are mapped (SYNONYMS, e.g. act "command" -> request); a value
not in the allowed list counts as invalid; one retry with a short
"answer with only the JSON" reminder; then None. Never put any other user's name or id in the prompt.

### `study/features.py`
```python
def person_features(lines, labels: dict[str, dict], replies, overlaps, min_words=0, turn_lines=None) -> dict[str, dict[str, float]]
    # user -> feature dict. Features (all floats; rates are per 100 words unless noted):
    # words, lines, talk_s (sum of end-t), calls (distinct calls), talk_share (talk_s / total talk_s of the
    # calls they were in; descriptive, it scales with 1/call size), talk_share_rel (own share x speakers in the
    # call, i.e. 1.0 = an equal share, averaged over calls weighted by call talk time; used for extraversion),
    # words_per_line, wpm (words / talk_s * 60), ttr_mattr (moving-average TTR, window 50; NaN below 50 tokens),
    # mean_word_len, i_rate, we_rate, you_rate, swear_rate, pos_rate, neg_rate, anger_rate, anxiety_rate,
    # sadness_rate, tentative_rate, certainty_rate, social_rate, achieve_rate, filler_rate, assent_rate,
    # question_share, joke_share, agree_share, disagree_share, request_share (from labels.act, share of labelled lines),
    # pos_sent_share, neg_sent_share, warm_share, hostile_share, emo_joy_share, emo_anger_share,
    # emo_sadness_share, emo_fear_share, social_ctx_share, gameplay_ctx_share,
    # interrupts_per_100_lines, interrupted_per_100_lines, reply_latency_median_ms (median of positive latencies),
    # replies_given_per_100_lines, active_hours_entropy (Shannon entropy of the hour-of-day histogram, bits).
    # Labels for a line may be missing (None): share features use only labelled lines; if a user has no
    # labelled lines those features are NaN. The *_per_100_lines rates divide by the person's lines in
    # turn_lines (default: lines), the line set replies/overlaps were computed on.
def social_only(lines, labels) -> list[Line]   # lines whose context label is "social" (unlabelled lines excluded)
def features_for(lines, labels, social=False, include=None) -> dict[str, dict[str, float]]
    # THE feature definition used by the CLI for the point estimate, the bootstrap and split-half: turn-taking
    # on all `lines`, text/label features and talk share on social_only(lines) when social=True.
```

### `study/traits.py`
```python
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
PRIOR_WEIGHTS: dict[str, dict[str, float]]   # trait -> {feature: signed weight}
    # Directions from the literature, each with a short citation comment:
    # Mairesse et al. 2007 (JAIR, conversational speech & text features -> Big Five),
    # Yarkoni 2010 (J Res Pers, word use of bloggers), Schwartz et al. 2013 (PLoS ONE, open vocabulary),
    # Pennebaker & King 1999 (LIWC style). Use only features from features.py. Magnitudes in [-1, 1]
    # roughly proportional to reported correlations (typically |r| 0.05-0.3). Examples of expected directions:
    # extraversion + talk_share_rel, + words_per_line, + we_rate, + social_rate, + pos_rate, + joke_share, - tentative_rate;
    # agreeableness + pos_rate, + warm_share, + agree_share, - swear_rate, - anger_rate, - hostile_share, - disagree_share;
    # conscientiousness - swear_rate, - neg_rate, + achieve_rate, - filler_rate, - interrupts_per_100_lines;
    # neuroticism + i_rate, + neg_rate, + anxiety_rate, + sadness_rate, + anger_rate, + emo_fear_share;
    # openness + ttr_mattr, + mean_word_len, + question_share, - swear_rate (weak), + tentative_rate (weak).
def zscore_matrix(people: dict[str, dict[str, float]], feats: list[str]) -> tuple[list[str], np.ndarray]
    # rows = users (sorted), cols = feats; z-scored across users (mean 0, sd 1); NaN -> 0 after scoring.
def prior_scores(people, weights=PRIOR_WEIGHTS) -> dict[str, dict[str, float]]
    # user -> trait -> score = sum(w * z) / sum(|w|), then re-z-scored across users per trait.
    # Scores are RELATIVE TO THIS SERVER (0 = server average, ±1 = one SD).
def bootstrap_ci(lines_by_user_call: dict[str, dict[str, list]], compute, n=200, seed=0, min_calls=5) -> dict[str, dict[str, tuple[float,float]]]
    # cluster bootstrap: resample the server's CALLS with replacement, keeping every user's lines of each drawn
    # call (pass every speaker's lines), recompute scores, return 2.5/97.5 percentiles per user per trait.
    # Users with fewer than min_calls distinct calls get (nan, nan): one call can only be drawn as itself.
    # `compute` is a callable(lines_subset) -> user->trait->score. Deterministic with seed.
def split_half(score_fn, lines, info=None) -> dict[str, float]
    # split the server's WHOLE calls into odd/even (by call start order; everyone in a call in the same half),
    # compute scores on each half, return per-trait Pearson r across users (Spearman-Brown corrected:
    # 2r/(1+r); uncorrected when r <= 0); NaN if < 5 users. info (optional) gets {"n_users", "n_calls"}.
def fit_calibration(people, truth: dict[str, dict[str, float]], feats, alpha=1.0, n_perm=200, seed=0) -> dict
    # ridge regression per trait (numpy closed form on z-scored features), leave-one-out predictions,
    # returns {"weights": {trait: {feat: w}}, "intercept": {...}, "loo_r": {trait: r}, "n": int,
    # "loo_null95"/"loo_p": permutation chance level of loo_r, "loo_ci": Fisher 95% interval, ...}.
    # The CLI passes alpha=CLI_ALPHA (50, fixed, not tuned on the reported LOO r).
    # Refuse (raise ValueError) with fewer than 8 users that have both features and truth.
def load_questionnaire(path) -> dict[str, dict[str, float]]
    # CSV with header user,openness,conscientiousness,extraversion,agreeableness,neuroticism (any order, 1-5 or 0-100 scale).
```

### `study/habits.py`
```python
def server_habits(lines, calls: dict[str, list[Line]], replies) -> dict
    # {"heatmap": 7x24 list of spoken minutes by weekday (Mon=0) x local hour,
    #  "call_sizes": {n_people: count}, "call_lengths_min": {"median","p90","mean"},
    #  "calls": n, "people": n, "hours_spoken": float,
    #  "talk_gini": Gini coefficient of talk_s across users,
    #  "reply_network": {"edges": [[a,b,count]...] top 200 by count, "degree": {user: distinct partners},
    #                    "top_connectors": [users sorted by degree desc][:10]},
    #  "roles": {user: "host" | "regular" | "connector" | "lurker"}   # rule-based, documented in the docstring
    # }
```

### `study/agreement.py`
```python
def export_sample(lines, labels, out_csv, n=200, seed=0)   # stratified-ish random sample: key, text, previous,
    # then the model's labels, then EMPTY columns human_sentiment, human_act, human_emotion, human_tone, human_context
def kappa(a: list[str], b: list[str]) -> float              # Cohen's kappa; 1.0 if identical; nan if empty
def score_sample(in_csv) -> dict[str, dict]                 # per label: {"n", "kappa", "accuracy"} on rows the human filled in
```

### `study/report.py`
```python
def write_report(out_dir, guild_name, people_features, scores, cis, reliability, habits, label_stats, settings) -> None
    # writes people.csv (user, display name, words, calls, each trait score and CI, flags), server.json (everything
    # aggregate), and report.md: a plain-language server report: data used (dates, calls, hours, people included and
    # how many were below the min-words threshold), label quality (share parsed, agreement if scored), trait
    # distribution (mean ± sd is 0/1 by construction for prior scores, so report the share of people in each third
    # and the reliability numbers instead), habits (heatmap as a text table, call sizes, roles), and a "How to read
    # this" section stating clearly: scores are relative to this server, based on published language-personality
    # correlations, typical validity r≈0.2-0.4, uncalibrated unless a questionnaire was supplied.
    # report.md must not list per-person scores (people.csv holds them, for the researcher only).
```

### `study/__main__.py` (CLI)
```
python -m study guilds                                   # list servers in transcripts/
python -m study run --guild ID [--since D] [--until D] [--exclude FILE] [--out DIR]
                    [--engine auto|mlx|torch|fake] [--min-words 1000] [--limit N] [--social-only] [--questionnaire CSV]
    # load -> label (cached, resumable, prints progress every 100 lines) -> features -> scores (prior, or
    # calibrated if --questionnaire) -> CIs -> split-half -> habits -> report. Default out: study/out/<guild>/
python -m study sample --guild ID [--n 200] [--out DIR]   # export the hand-labelling sheet (labels must exist)
python -m study agree --sheet FILE                        # score a filled-in sheet (refuses non-sheets;
                                                          # saves nothing when no human_* cell is filled)
```
`--transcripts DIR` (default `transcripts/` next to the repo root) on every command. Inputs (folder,
dates, exclusion list, questionnaire, engine hardware) are checked before any labelling. With an exclusion
list, `run` also scrubs earlier outputs in --out: label-cache rows and sample.csv rows whose line is no
longer a valid line of the server (excluded or deleted), sample.csv 'previous' cells, and agreement.json.
Output files are written atomically (temp file + replace). `study/out/` is
git-ignored (it holds personal data).

## Tests: `test/study_test.py`
Runs with no model (FakeLabeller) on synthetic data generated in the test (a fictional server, 8–12
fictional people with deliberately different styles: e.g. one swears a lot, one asks many questions, one
interrupts, one is warm), checks: excluded user absent from every output file and the label cache; replies/
overlaps detected on a hand-built example; kappa on known pairs (identical -> 1, chance -> ~0); ridge LOO on
synthetic data where truth is a linear function of features gives r > 0.9; split-half returns finite values;
prior directions: the heavy swearer scores lower on agreeableness than the warm person; report.md contains no
per-person names; the whole `run` command completes end to end with `--engine fake`. Prints PASS/FAIL per check
and exits non-zero on failure.
