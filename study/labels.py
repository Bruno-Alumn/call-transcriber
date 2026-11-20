"""Per-line labelling with a small local LLM.

Each transcript line gets five labels, each from a fixed list (LABELS). The
model sees only the text of the previous line (for context) and the line to
label: never a name, user id, channel or server. Answers are parsed from one
line of JSON; anything outside the allowed values counts as a failed answer,
which is retried once and then recorded as None.

Backends:
  mlx    Apple Silicon, via mlx_vlm (same calls as PersonalityReader in server/server.py)
  torch  NVIDIA GPU, via transformers
  fake   FakeLabeller: deterministic labels from word lists, no model (tests, dry runs)

The model packages are imported lazily inside functions so that importing this
module needs only the standard library.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Protocol

LABELS: dict[str, list[str]] = {
    "sentiment": ["positive", "neutral", "negative"],
    "act": ["question", "statement", "request", "joke", "agreement", "disagreement", "greeting"],
    "emotion": ["joy", "anger", "sadness", "fear", "surprise", "none"],
    "tone": ["warm", "neutral", "hostile"],
    "context": ["gameplay", "social", "logistics", "other"],
}

MLX_REPO = "LiquidAI/LFM3-1.7B-MLX-8bit"
TORCH_REPO = "LiquidAI/LFM3-1.7B"
MAX_NEW_TOKENS = 60          # one line of JSON with five short values fits in ~45 tokens

# ---------------------------------------------------------------- prompt
# Written for a 1.7B model: one short sentence per label, the allowed values
# spelled out, one worked example, and a strict output format. The example
# uses the exact key order we want back, which small models tend to copy.
SYSTEM_PROMPT = """You label one line of spoken chat from a voice call among friends (often while gaming). The text is a speech-to-text transcript, so expect missing punctuation and small errors.

Give exactly these five labels:
sentiment: the overall feeling of the line. One of: positive, neutral, negative.
act: what the line does. One of: question, statement, request, joke, agreement, disagreement, greeting.
emotion: the main emotion shown, or none. One of: joy, anger, sadness, fear, surprise, none.
tone: how the speaker treats the other people. Being annoyed at the game (not at a person) is neutral. One of: warm, neutral, hostile.
context: what the line is about. One of: gameplay (the game being played: moves, enemies, tactics, scores), social (feelings, life, jokes, chat between people), logistics (when to meet, setup, audio, mics, tech), other.

Label only the LINE. The PREVIOUS line is context only.
Never invent a value. If no value fits well, pick the closest listed one (act: statement, tone: neutral, emotion: none).
Answer with ONE line of JSON and nothing else, using only the values listed above, for example:
{"sentiment": "neutral", "act": "question", "emotion": "none", "tone": "neutral", "context": "logistics"}"""

RETRY_REMINDER = ("Answer with only the JSON line, with keys sentiment, act, emotion, tone, context "
                  "and only the allowed values.")


def retry_reminder(answer: str) -> str:
    """The retry message: the general reminder plus the allowed values of each key that was wrong."""
    bad = invalid_keys(answer)
    if not bad:
        return RETRY_REMINDER
    fixes = " ".join(f"{k} must be one of: {', '.join(LABELS[k])}." for k in bad)
    return f"{RETRY_REMINDER} {fixes}"


def build_messages(previous: str, text: str) -> list[dict]:
    """Chat messages for one line. Only line texts go in: no names or ids."""
    prev = previous.strip() if previous else ""
    user = f"PREVIOUS: {prev if prev else '(none)'}\nLINE: {text.strip()}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- parsing
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.S)


def _first_obj(answer: str) -> dict | None:
    """The first {...} in the answer as a dict with lowercased keys, or None."""
    m = _JSON_OBJ.search(answer or "")
    if not m:
        return None
    raw = m.group(0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Small models sometimes use single quotes or leave a trailing comma.
        try:
            obj = json.loads(re.sub(r",\s*}", "}", raw.replace("'", '"')))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    return {str(k).strip().lower(): v for k, v in obj.items()}


def _clean_value(val, allowed: list[str]) -> str | None:
    """Lowercased allowed value, or None. "exclamation/statement" -> "statement" (first allowed part)."""
    if not isinstance(val, str):
        return None
    val = val.strip().strip(".").lower()
    if val in allowed:
        return val
    for part in re.split(r"[/|,;]| or ", val):
        part = part.strip().strip(".")
        if part in allowed:
            return part
    return None


def invalid_keys(answer: str) -> list[str]:
    """Label keys that are missing or invalid in the answer (all of them if there is no JSON)."""
    obj = _first_obj(answer)
    if obj is None:
        return list(LABELS)
    return [k for k, allowed in LABELS.items() if _clean_value(obj.get(k), allowed) is None]


def parse_answer(answer: str) -> dict | None:
    """Parse the first {...} in the model's answer into a valid label dict, or None.

    Lenient about format (code fences, single quotes, case, extra keys, stray
    whitespace), strict about content: all five keys must be present and every
    value must be one of the allowed values.
    """
    obj = _first_obj(answer)
    if obj is None:
        return None
    out = {}
    for key, allowed in LABELS.items():
        val = _clean_value(obj.get(key), allowed)
        if val is None:
            return None
        out[key] = val
    return out


# ---------------------------------------------------------------- protocol
class Labeller(Protocol):
    name: str

    def label(self, items: list[tuple[str, str, str]]) -> list[dict | None]:
        """items: (key, previous_line_text_or_'', text) -> one label dict (or None) per item."""
        ...


class _LLMLabeller:
    """Shared logic for the model backends: prompt, greedy generate, parse, one retry."""

    name: str = ""

    def _generate(self, messages: list[dict]) -> str:      # backend-specific
        raise NotImplementedError

    def label_one(self, previous: str, text: str) -> dict | None:
        messages = build_messages(previous, text)
        answer = self._generate(messages)
        parsed = parse_answer(answer)
        if parsed is not None:
            return parsed
        # One retry: a fresh conversation with a reminder (naming the keys that were
        # wrong) added to the user turn. Showing the model its own bad answer was
        # tried and made it repeat the same invalid value.
        retry = [messages[0], {"role": "user",
                               "content": messages[1]["content"] + "\n\n" + retry_reminder(answer)}]
        return parse_answer(self._generate(retry))

    def label(self, items: list[tuple[str, str, str]]) -> list[dict | None]:
        out = []
        for _key, previous, text in items:
            try:
                out.append(self.label_one(previous, text))
            except Exception as e:        # one bad line must not stop a long run
                print(f"labels: generation failed ({type(e).__name__}: {e})"[:200], file=sys.stderr, flush=True)
                out.append(None)
        return out


class MlxLabeller(_LLMLabeller):
    """Apple Silicon backend (mlx_vlm), same load/generate calls as PersonalityReader."""

    def __init__(self, repo: str = MLX_REPO):
        from mlx_vlm import load
        self.repo = repo
        self.name = f"mlx:{repo}"
        self.model, self.proc = load(repo)

    def _generate(self, messages: list[dict]) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = apply_chat_template(self.proc, self.model.config, messages, num_images=0)
        # temperature 0.0 makes mlx_vlm use argmax (greedy), so answers are repeatable.
        out = generate(self.model, self.proc, prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.0, verbose=False)
        return (out.text if hasattr(out, "text") else str(out)).strip()


class TorchLabeller(_LLMLabeller):
    """NVIDIA backend (transformers), same calls as PersonalityReader, greedy decoding."""

    def __init__(self, repo: str = TORCH_REPO):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.repo = repo
        self.name = f"torch:{repo}"
        self.proc = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()

    def _generate(self, messages: list[dict]) -> str:
        import torch
        ids = self.proc.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, enable_thinking=False).to("cuda")
        with torch.inference_mode():
            gen = self.model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return self.proc.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------- fake
class FakeLabeller:
    """Deterministic labels from the lexicon word lists. No model; for tests and dry runs.

    Rules are simple and documented so tests can predict them:
      act       greeting words -> greeting; ends with '?' or starts with a question word -> question;
                'lol'/'haha'/'lmao' -> joke; agreement/disagreement openers; please/can you -> request;
                else statement.
      sentiment positive vs negative word counts (swear words count as negative).
      emotion   anger / sadness / anxiety word counts, else joy if positive, else none.
      tone      hostile if any swear or anger word, warm if positive or social words dominate, else neutral.
      context   gameplay words -> gameplay; logistics words -> logistics; social words or a joke -> social; else other.
    """

    name = "fake"

    _GREET = {"hi", "hey", "hello", "yo", "sup", "morning", "evening", "hiya", "howdy"}
    _QWORDS = {"what", "why", "how", "when", "where", "who", "which", "is", "are", "do", "does",
               "did", "can", "could", "would", "should", "will"}
    _JOKE = {"lol", "lmao", "haha", "hahaha", "rofl", "lmfao", "jk"}
    _AGREE = {"yes", "yeah", "yep", "agreed", "exactly", "true", "right", "totally", "same"}
    _DISAGREE = {"no", "nah", "nope", "wrong"}
    _REQUEST = {"please", "pls", "plz"}
    _GAME = {"game", "level", "boss", "loot", "spawn", "respawn", "kill", "killed", "map", "match", "round",
             "enemy", "enemies", "heal", "push", "rush", "gun", "ammo", "quest", "raid", "rank", "ranked",
             "lobby", "team", "build", "base", "craft", "server", "mob", "mobs", "dps", "tank", "nerf", "buff"}
    _LOGISTICS = {"mic", "audio", "hear", "muted", "mute", "lag", "ping", "internet", "join", "invite",
                  "tonight", "tomorrow", "later", "time", "minutes", "brb", "afk", "restart", "update",
                  "download", "install", "schedule", "o'clock"}

    def _labels(self, text: str) -> dict:
        from study import lexicon    # imported here so this module stays importable on its own
        toks = lexicon.tokenize(text)
        s = set(toks)

        def n(words) -> int:
            return sum(1 for t in toks if t in words)

        pos, neg = n(lexicon.POSITIVE), n(lexicon.NEGATIVE) + n(lexicon.SWEAR)
        anger, sad, anx = n(lexicon.ANGER), n(lexicon.SADNESS), n(lexicon.ANXIETY)
        swear, social = n(lexicon.SWEAR), n(lexicon.SOCIAL)
        first = toks[0] if toks else ""

        if first in self._GREET and len(toks) <= 4:
            act = "greeting"
        elif text.rstrip().endswith("?") or first in self._QWORDS:
            act = "question"
        elif s & self._JOKE:
            act = "joke"
        elif first in self._AGREE:
            act = "agreement"
        elif first in self._DISAGREE:
            act = "disagreement"
        elif s & self._REQUEST or text.lower().startswith(("can you", "could you", "let's", "lets")):
            act = "request"
        else:
            act = "statement"

        sentiment = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        if anger:
            emotion = "anger"
        elif sad:
            emotion = "sadness"
        elif anx:
            emotion = "fear"
        elif pos > neg:
            emotion = "joy"
        else:
            emotion = "none"
        if swear or anger:
            tone = "hostile"
        elif pos > neg or (social and neg == 0):
            tone = "warm"
        else:
            tone = "neutral"
        if s & self._GAME:
            context = "gameplay"
        elif s & self._LOGISTICS:
            context = "logistics"
        elif social or act in ("joke", "greeting"):
            context = "social"
        else:
            context = "other"
        return {"sentiment": sentiment, "act": act, "emotion": emotion, "tone": tone, "context": context}

    def label(self, items: list[tuple[str, str, str]]) -> list[dict | None]:
        return [self._labels(text) for _key, _prev, text in items]


# ---------------------------------------------------------------- factory
def _mlx_ok() -> bool:
    import platform
    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        return False
    try:
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def make_labeller(engine: str = "auto", model: str | None = None) -> Labeller:
    """Build a labeller. engine: auto | mlx | torch | fake. model overrides the default repo."""
    if engine == "fake":
        return FakeLabeller()
    if engine == "auto":
        if _mlx_ok():
            engine = "mlx"
        elif _cuda_ok():
            engine = "torch"
        else:
            raise RuntimeError("No local model backend available: labelling needs an Apple Silicon Mac with "
                               "mlx_vlm installed (cd server && uv sync) or an NVIDIA GPU with torch + "
                               "transformers. Use --engine fake for a dry run without a model.")
    if engine == "mlx":
        return MlxLabeller(model or MLX_REPO)
    if engine == "torch":
        return TorchLabeller(model or TORCH_REPO)
    raise ValueError(f"unknown engine {engine!r} (expected auto, mlx, torch or fake)")


# ---------------------------------------------------------------- cache
class LabelCache:
    """Labels stored as JSON Lines: {"key", "model", "labels"} per line (labels may be null).

    Only the line key (a hash) is stored, never text, names or user ids. A later
    entry for the same (key, model) replaces an earlier one. Failed lines (None)
    are cached too, so a resumed run does not retry them forever.

    `path` may be the output directory (the file is then <path>/labels.jsonl) or
    the .jsonl file itself.
    """

    def __init__(self, path: Path | str):
        path = Path(path)
        self.path = path if path.suffix == ".jsonl" else path / "labels.jsonl"
        self._data: dict[tuple[str, str], dict | None] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for row in f:
                    row = row.strip()
                    if not row:
                        continue
                    try:
                        rec = json.loads(row)
                        self._data[(rec["key"], rec["model"])] = rec.get("labels")
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue      # a half-written last line after a crash: ignore it

    def has(self, key: str, model: str) -> bool:
        return (key, model) in self._data

    def get(self, key: str, model: str) -> dict | None:
        """Cached labels, or None when missing or when the model's answer failed (see has())."""
        return self._data.get((key, model))

    def put(self, key: str, model: str, labels: dict | None) -> None:
        self._data[(key, model)] = labels
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": model, "labels": labels}) + "\n")

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------- driver
def _previous_texts(lines: Iterable) -> dict[str, str]:
    """key -> text of the line just before it in the same call (or channel when calls aren't set)."""
    prev_by_group: dict[str, str] = {}
    out: dict[str, str] = {}
    for ln in sorted(lines, key=lambda x: x.t):
        group = ln.call or ln.ch
        out[ln.key] = prev_by_group.get(group, "")
        prev_by_group[group] = ln.text
    return out


def label_lines(lines, labeller: Labeller, cache: LabelCache,
                progress: Callable[[int, int], None] | None = None,
                limit: int | None = None, batch: int = 16) -> dict[str, dict | None]:
    """Label every line not yet cached for labeller.name; return key -> labels (or None).

    lines    already-filtered Line objects (excluded users must be removed before this call).
    progress called as progress(done, todo) after each batch of new lines.
    limit    label at most this many NEW lines this run (cached lines are always returned).
    Lines past the limit are left out of the result rather than given None.
    """
    lines = list(lines)
    prev = _previous_texts(lines)
    result: dict[str, dict | None] = {}
    todo = []
    for ln in lines:
        if cache.has(ln.key, labeller.name):
            result[ln.key] = cache.get(ln.key, labeller.name)
        elif ln.key not in result:
            result[ln.key] = None     # placeholder; replaced below or removed if past the limit
            todo.append(ln)
    if limit is not None:
        for ln in todo[limit:]:
            result.pop(ln.key, None)
        todo = todo[:limit]

    done = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        answers = labeller.label([(ln.key, prev.get(ln.key, ""), ln.text) for ln in chunk])
        for ln, lab in zip(chunk, answers):
            cache.put(ln.key, labeller.name, lab)
            result[ln.key] = lab
        done += len(chunk)
        if progress:
            progress(done, len(todo))
    return result
