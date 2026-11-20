"""Own-voice detection: is this utterance the person, or something near their mic?

Speech that is not the person (a video or TV playing near their microphone,
someone else in their room) would otherwise be transcribed as if they said it.
Per person we keep a voiceprint, a speaker embedding averaged over their own
speech, and compare every utterance with it. Lines clearly unlike the person
get "bg": true; they are never dropped (the dashboard decides how to show them).

Model: WeSpeaker CAM++ (VoxCeleb, 29 MB ONNX, Apache-2.0), run with onnxruntime
on CPU on every platform; downloaded from Hugging Face on first use. About 12 ms
per second of speech on an M-series Mac with 2 threads (ResNet34 was 3x slower
for the same accuracy).

The first voice we hear might itself be the video, so nobody is enrolled from a
single utterance. Clean utterances that become lines are grouped into voices;
the person is the voice with the most speech this session, and only once it has
enough (several utterances, 30 s). Embeddings alone can't tell "a video, then the
person" from "the person, then a video", so whenever it is unclear we flag
nothing rather than risk hiding the person's own words:
- the voice holding the voiceprint keeps it until another has 1.5x its speech,
  and does not fade while others talk (a long video doesn't wear it away);
  while another voice has more than 2/3 of its speech, nothing is flagged;
- a voice that ever held the voiceprint is never flagged again: if a video
  took over, the person's own lines stay unflagged when they come back. (If a
  video heard first held it, it isn't flagged after the person takes over:
  missing a video is the safer mistake.) Enrolment waits for 30 s of one voice
  so that a short video intro rarely holds it at all.
"""

import threading
import time
from collections import deque

import numpy as np

SR = 16000
REPO, FILE = "Wespeaker/wespeaker-voxceleb-campplus", "voxceleb_CAM++.onnx"

# Cosine similarity to a voiceprint, measured on AMI headset audio (16 speakers):
# the same person scores a median 0.81 (5th percentile 0.60) on 3 s of speech,
# another person a median 0.07 (95th percentile 0.27, 99th 0.39).
SAME = 0.45          # "the same voice" when grouping utterances into voices
BACKGROUND = 0.3     # below this an utterance is flagged as background
MIN_S = 1.0          # shorter utterances are never judged (too little voice)
CLEAN_S = 1.2        # only utterances this long shape the voices (our segmenter cuts at 0.7 s pauses, so most are short)
MAX_S = 12.0         # longer audio is cropped to its middle (speed)
ENROL_N = 4          # utterances, and
ENROL_S = 30.0       # seconds of speech the person's voice needs before anything is flagged
LEAD = 1.5           # another voice needs this many times the holder's speech to take the voiceprint
FADE = 0.995         # per clean utterance: old speech counts a little less (half after ~140)
MAX_VOICES = 8


# ---------------------------------------------------------------- features
def _mel_banks(n_fft: int, n_mels: int = 80, low: float = 20.0) -> np.ndarray:
    """Kaldi-style triangular mel filters (on the mel scale, as Kaldi does it)."""
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    lo, hi = mel(low), mel(SR / 2)
    centers = lo + (hi - lo) * np.arange(n_mels + 2) / (n_mels + 1)
    fm = mel(np.arange(n_fft // 2) * SR / n_fft)            # Kaldi leaves out the Nyquist bin
    banks = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = centers[m], centers[m + 1], centers[m + 2]
        up, down = (fm - l) / (c - l), (r - fm) / (r - c)
        banks[m, : n_fft // 2] = np.maximum(0.0, np.minimum(up, down))
    return banks


_BANKS = _mel_banks(512)
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85     # Kaldi's "povey" window


def fbank(audio: np.ndarray) -> np.ndarray:
    """80-dim log mel filterbank, matching Kaldi/torchaudio defaults (25 ms
    frames every 10 ms, no dither) as WeSpeaker was trained, then mean-normalised."""
    x = audio.astype(np.float64) * 32768.0                  # WeSpeaker feeds int16-scale samples
    n = 1 + (len(x) - 400) // 160
    if n < 1:
        return np.zeros((0, 80), dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160][:n].copy()
    frames -= frames.mean(axis=1, keepdims=True)             # remove DC per frame
    frames = np.concatenate([frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)   # pre-emphasis
    frames *= _WINDOW
    power = np.abs(np.fft.rfft(frames, 512)) ** 2
    feats = np.log(np.maximum(power @ _BANKS.T.astype(np.float64), np.finfo(np.float32).eps)).astype(np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------- model
class Embedder:
    """Loads the ONNX model on first use (in the background: the first lines of a
    call never wait for a download) and turns audio into a unit vector."""

    def __init__(self):
        self.sess = None
        self.error = None
        self.lock = threading.Lock()
        self.loading = False

    def load(self, wait: bool = False):
        if self.sess is not None or self.error:
            return
        with self.lock:
            if self.loading:
                return
            self.loading = True
        def go():
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2               # leave the cores to speech recognition
                opts.inter_op_num_threads = 1
                self.sess = ort.InferenceSession(hf_hub_download(REPO, FILE), opts, providers=["CPUExecutionProvider"])
                print("Own-voice detection ready.", flush=True)
            except Exception as e:
                self.error = str(e)[:200]
                print("Own-voice detection unavailable:", self.error, flush=True)
        if wait:
            go()
        else:
            threading.Thread(target=go, daemon=True).start()

    def __call__(self, audio: np.ndarray) -> np.ndarray | None:
        if self.sess is None:
            self.load()
            return None
        if len(audio) > MAX_S * SR:
            mid = len(audio) // 2
            audio = audio[mid - int(MAX_S * SR / 2): mid + int(MAX_S * SR / 2)]
        f = fbank(audio)
        if len(f) < 50:
            return None
        e = self.sess.run(None, {"feats": f[None]})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


# ---------------------------------------------------------------- per person
class Person:
    """The voices heard on one person's mic this session, as clusters of clean
    utterances weighted by seconds of speech (slowly fading, so the voiceprint
    follows a changed mic or room). The person is the voice heard most; see the
    module notes for how a takeover is handled."""

    def __init__(self):
        # [centroid (unit vector), seconds, utterances, has held the voiceprint]
        self.voices: list[list] = []
        self.holder = None                 # the voice whose centroid is the voiceprint
        self.flagged = self.judged = 0

    @property
    def print(self):
        """The voiceprint, or None while it is not yet clear whose mic this is."""
        h = self.holder
        if h is None:
            return None
        if any(LEAD * v[1] > h[1] for v in self.voices if v is not h):
            return None                    # another voice is catching up: unclear, flag nothing
        return h[0]

    def _pick_holder(self):
        top = max(self.voices, key=lambda v: v[1], default=None)
        if top is None or top is self.holder or top[1] < ENROL_S or top[2] < ENROL_N:
            return
        if self.holder is None:
            # First enrolment: it must clearly lead every other voice.
            if all(top[1] >= LEAD * v[1] for v in self.voices if v is not top):
                self.holder = top
        elif top[1] >= LEAD * self.holder[1]:
            self.holder = top
        if self.holder is top:
            top[3] = True

    def _learn(self, e: np.ndarray, secs: float):
        best = max(self.voices, key=lambda v: float(v[0] @ e), default=None)
        joins = best if best is not None and float(best[0] @ e) > SAME else None
        for v in self.voices:
            # The holder only fades on its own speech: a long video must not wear it away.
            if v is not self.holder or joins is self.holder:
                v[1] *= FADE
        if joins is not None:
            best = joins
            c = best[0] * best[1] + e * secs
            best[0], best[1], best[2] = c / np.linalg.norm(c), best[1] + secs, best[2] + 1
            # A voice can start as two groups (a laugh, a quieter stretch) that
            # grow together: merge them once their centres agree.
            twins = [v for v in self.voices if v is not best and float(v[0] @ best[0]) > SAME]
            for v in twins:
                c = best[0] * best[1] + v[0] * v[1]
                best[0], best[1], best[2], best[3] = c / np.linalg.norm(c), best[1] + v[1], best[2] + v[2], best[3] or v[3]
                if v is self.holder:
                    self.holder = best
            self.voices = [v for v in self.voices if not any(v is t for t in twins)]
        else:
            new = [e, secs, 1, False]
            if len(self.voices) >= MAX_VOICES:
                # Forget the least heard older voice: never the one just heard
                # (it could never grow), the holder, or a former holder.
                old = [v for v in self.voices if v is not self.holder and not v[3]] or \
                      [v for v in self.voices if v is not self.holder]
                least = min(old, key=lambda v: v[1])
                self.voices = [v for v in self.voices if v is not least]
            self.voices.append(new)
        self._pick_holder()

    def judge(self, e: np.ndarray, secs: float) -> tuple[bool, float | None]:
        """Background or not, and the similarity (None while not enrolled)."""
        if secs >= CLEAN_S:
            self._learn(e, secs)
        p = self.print
        if p is None:
            return False, None
        s = float(p @ e)
        # Like a voice that held the voiceprint before? Then it may be the person.
        like_former = any(v[3] and float(v[0] @ e) >= BACKGROUND for v in self.voices)
        bg = secs >= MIN_S and s < BACKGROUND and not like_former
        self.judged += 1; self.flagged += bg
        return bg, s

    check = judge


class Voiceprints:
    """One voiceprint per person (by user id), in memory only: voiceprints are
    biometric data, so nothing is written to disk."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.people: dict[str, Person] = {}
        self.ms: deque = deque(maxlen=200)     # embedding time per utterance (ms)

    def embed(self, audio: np.ndarray):
        """The speaker embedding (None if too short, or the model isn't loaded).
        Runs next to transcription; judge() once we know a line will be written."""
        if len(audio) / SR < MIN_S:
            return None
        t = time.perf_counter()
        try:
            e = self.embedder(audio)
        except Exception as ex:                 # never lose a line over this
            print("voiceprint error:", ex, flush=True)
            return None
        if e is not None:
            self.ms.append((time.perf_counter() - t) * 1000)
        return e

    def judge(self, key: str, e, secs: float) -> bool:
        """True when the utterance clearly is not this person's voice. Only call
        it for utterances that became lines: music, noise and hallucinations
        must not shape the voices."""
        if e is None:
            return False
        return self.people.setdefault(key or "?", Person()).judge(e, secs)[0]

    def check(self, key: str, audio: np.ndarray) -> bool:
        """embed() and judge() in one (tests)."""
        return self.judge(key, self.embed(audio), len(audio) / SR)
