# Call Transcriber

Live, local transcripts of Discord voice calls: every person is transcribed from
their own audio stream, grouped by server and voice channel, with a dashboard
(history, search, analytics). Audio never leaves your computer. macOS and
Windows share one codebase:

| Part | Where | Platforms |
|---|---|---|
| Chrome extension | `extension/` | both |
| Transcription server + dashboard | `server/` | both (engine picked per machine) |
| Mac app (Swift, WKWebView) | `app/mac/` | macOS |
| Windows app (pywebview + tray) + installer | `app/windows/`, `install.cmd` | Windows |
| Tests (fake Discord call, server e2e) | `test/` | both, run by CI |

CI (`.github/workflows/ci.yml`) runs the same end-to-end test on macOS and
Windows and builds both apps on every push, so the two stay at parity.

## Updating
- **Mac:** `git pull`, then `app/mac/build.sh` (rebuilds the app), reload the
  extension in `chrome://extensions`.
- **Windows:** `git pull`, then double-click `install.cmd`, reload the extension.
- Transcripts, names and archive choices are personal data and are never
  committed (see `.gitignore`); they stay on each machine.


Live, local transcript of a Discord voice call. Each person is transcribed
from their own audio stream, so overlapping speech stays separate and every
line is labeled with who said it. Audio never leaves your Mac.

## How it works
Discord's web app receives each participant as a separate audio stream tagged
with their user id. The extension reads those streams inside the Discord tab
(read-only; your playback is untouched) and sends them to a local Whisper
server, which writes the transcript.

## One-time setup
1. `chrome://extensions` → Developer mode on → **Load unpacked** → `~/discord-transcriber/extension`.

## The app
**Call Transcriber** (in ~/Applications, also in Launchpad/Spotlight) runs the
transcription server and shows a live dashboard: transcript with search, who's
in the call and talking, talk time per person, activity over time, a 10-minute
timeline, crosstalk and most-said words, per call session or whole day.
Closing the window keeps transcribing; Quit (Cmd+Q) stops it.
Rebuild after changes: `app/mac/build.sh`. Dashboard also at http://127.0.0.1:8765/.

## Each time (without the app)
1. Run `~/discord-transcriber/start.sh` and leave it running.
2. Open discord.com in Chrome, join the voice channel. That's it — no button.
   The toolbar badge shows how many people are being transcribed
   (OFF = server not running, II = paused; clicking the icon pauses/resumes).
3. Transcript: `~/discord-transcriber/transcripts/YYYY-MM-DD.md`, also printed live.

While the server runs, **any** Discord call you're in is transcribed.

Notes
- Muting Chrome at the system level is fine. Don't use Discord's *Deafen* — it stops audio from arriving.
- Your own microphone isn't transcribed (only other people).
- People whose name can't be read from Discord's page show as `user-1234`
  (last digits of their Discord id) — never a guessed name.
- After reloading the extension, reload the Discord tab (or rejoin the call).

## Windows
`app/windows/make-zip.sh` builds `dist/CallTranscriber-Windows.zip` (installer,
tray app, server, extension; see README-WINDOWS.md). The server picks its speech
engine per machine: MLX on Apple Silicon, Granite on CUDA with an NVIDIA GPU,
otherwise Parakeet on ONNX Runtime (CPU: 7.4% word error, ~50x real time).
Override with `--engine mlx|torch|onnx`.

## Model
Default: IBM Granite Speech 5.0 470M (Apache-2.0), run on Apple Silicon via
mlx-audio. Benchmarked here on 120 real conversational clips (AMI meetings,
same data as the Open ASR Leaderboard):

| model | word error | speed | peak memory |
|---|---|---|---|
| **granite-speech-5.0-470m-turboctc** | **5.5%** | **214x realtime** | **1.6 GB** |
| parakeet-tdt-0.6b-v2 | 7.2% | 29x | 3.5 GB |
| fluid-speech-pro | 8.2% | 38x | 3.4 GB |
| qwen3-asr-1.7b (8-bit) | 8.6% | 16x | 3.9 GB |
| whisper-large-v3-turbo | 13.0% | 14x | 2.5 GB |

Swap with `./start.sh --model mlx-community/parakeet-tdt-0.6b-v2` (any mlx-audio model).
Fillers (um/uh) are removed and "i" capitalised for readability.

## Health check
`~/discord-transcriber/status.log` gets a line every 5 s from the Discord tab:
streams found, audio chunks, audio-engine state, names and errors.

## Tests
`cd test && node run.js /tmp/somewhere` runs the real extension in Chromium
against a fake call with three overlapping speakers and checks the transcript.
