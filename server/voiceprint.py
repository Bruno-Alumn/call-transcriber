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
single utterance. Clean utterances (2 s or more) are grouped into voices; the
person is the voice with the most speech this session, and only once it has
enough (several utterances, 20 s) and clearly leads every other voice. Until
then nothing is flagged. A video heard first is overtaken as soon as the person
has talked more than it, and the voiceprint then moves to the person.
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
ENROL_S = 20.0       # seconds of speech the person's voice needs before anything is flagged
LEAD = 1.5           # ... and it must have this many times the speech of the next voice
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
    follows a changed mic or room). The person is the voice heard most; a video
    heard first loses that place once the person has talked more than it."""

    def __init__(self):
        self.voices: list[list] = []       # [centroid (unit vector), seconds, utterances]
        self.flagged = self.judged = 0

    @property
    def print(self):
        """The voiceprint, or None while it is not yet clear whose mic this is."""
        if not self.voices:
            return None
        v = sorted(self.voices, key=lambda x: -x[1])
        top, second = v[0], (v[1][1] if len(v) > 1 else 0.0)
        if top[1] < ENROL_S or top[2] < ENROL_N or top[1] < LEAD * second:
            return None
        return top[0]

    def _learn(self, e: np.ndarray, secs: float):
        for v in self.voices:
            v[1] *= FADE
        best = max(self.voices, key=lambda v: float(v[0] @ e), default=None)
        if best is not None and float(best[0] @ e) > SAME:
            c = best[0] * best[1] + e * secs
            best[0], best[1], best[2] = c / np.linalg.norm(c), best[1] + secs, best[2] + 1
            # A voice can start as two groups (a laugh, a quieter stretch) that
            # grow together: merge them once their centres agree.
            twins = [v for v in self.voices if v is not best and float(v[0] @ best[0]) > SAME]
            for v in twins:
                c = best[0] * best[1] + v[0] * v[1]
                best[0], best[1], best[2] = c / np.linalg.norm(c), best[1] + v[1], best[2] + v[2]
            self.voices = [v for v in self.voices if not any(v is t for t in twins)]
        else:
            self.voices.append([e, secs, 1])
            if len(self.voices) > MAX_VOICES:                 # forget the least heard
                least = min(self.voices, key=lambda v: v[1])
                self.voices = [v for v in self.voices if v is not least]

    def check(self, e: np.ndarray, secs: float) -> tuple[bool, float | None]:
        """Background or not, and the similarity (None while not enrolled)."""
        if secs >= CLEAN_S:
            self._learn(e, secs)
        p = self.print
        if p is None:
            return False, None
        s = float(p @ e)
        bg = secs >= MIN_S and s < BACKGROUND
        self.judged += 1; self.flagged += bg
        return bg, s


class Voiceprints:
    """One voiceprint per person (by user id), in memory only: voiceprints are
    biometric data, so nothing is written to disk."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.people: dict[str, Person] = {}
        self.ms: deque = deque(maxlen=200)     # embedding time per utterance (ms)

    def check(self, key: str, audio: np.ndarray) -> bool:
        """True when the utterance clearly is not this person's voice."""
        secs = len(audio) / SR
        if secs < MIN_S:
            return False
        t = time.perf_counter()
        try:
            e = self.embedder(audio)
        except Exception as ex:                 # never lose a line over this
            print("voiceprint error:", ex, flush=True)
            return False
        if e is None:
            return False
        self.ms.append((time.perf_counter() - t) * 1000)
        return self.people.setdefault(key or "?", Person()).check(e, secs)[0]
