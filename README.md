# Call Transcriber

Live, local transcripts of Discord voice calls: every person is transcribed from
their own audio stream, grouped by server and voice channel, with a dashboard
(history, search, analytics). Audio never leaves your computer. macOS and
Windows share one codebase and one app.

| Part | Where | Platforms |
|---|---|---|
| Desktop app (Electron): dashboard + Discord built in | `app/desktop/` | both |
| Transcription server + dashboard | `server/` | both (engine picked per machine) |
| Mac bundle build | `app/mac/build.sh` | macOS |
| Windows installer (+ launcher kept for old shortcuts) | `install.cmd`, `app/windows/` | Windows |
| Chrome extension (optional: use Discord in Chrome instead) | `extension/` | both |
| Tests (fake Discord call, app e2e, server e2e) | `test/` | both, run by CI |

CI (`.github/workflows/ci.yml`) runs the same end-to-end tests on macOS and
Windows on every push, so the two stay at parity.

## How it works
Discord's web client receives each participant as a separate audio stream
tagged with their user id. The app opens discord.com in its own built-in
Chromium window and adds a small read-only hook (`extension/page-hook.js`)
before Discord's code runs; it reads those streams (your playback is untouched)
and the app passes them to the local server, which transcribes and writes the
transcript. No Chrome or extension needed.

## Setup (Mac)
1. Server: `cd server && uv sync`.
2. App: `app/mac/build.sh` (needs Node.js and Xcode's command-line tools). It
   installs **Call Transcriber** in ~/Applications.
3. Open it, sign in to Discord in its Discord window once (⌘D shows it), and
   join a voice channel. macOS asks for the microphone the first time.

## The app
- **Dashboard window:** transcript with search and export, who's in the call
  and talking, talk time, activity, timeline, crosstalk, most-said words, per
  call or whole day; history grouped by server and channel.
- **Discord window:** the real Discord web app (your account). Closing it
  hides it and keeps you in the call; the **Discord** button or ⌘D / Ctrl+D
  brings it back. Discord ▸ Pause Transcribing pauses.
- **Compact mode** (the layout button left of the call name, or ⌘⇧K / Ctrl+Shift+K):
  live transcript plus every analytics panel on one screen, no scrolling.
- Closing the dashboard keeps transcribing; Quit (⌘Q, or the Windows tray
  icon) stops it. The app starts and stops the server itself.
- Dashboard also at http://127.0.0.1:8765/.

## Updating
The dashboard shows when a new release is out; **Update now** installs it and
**Restart now** finishes (rebuilding the Mac app only when Electron changed).
By hand: `git pull`, then `app/mac/build.sh` (Mac) or `install.cmd` (Windows).
Transcripts, names and archive choices are personal data and are never
committed (see `.gitignore`); they stay on each machine.

## Using Chrome instead (optional)
Run `start.sh` (or the app), load `extension/` in `chrome://extensions`
(Developer mode → Load unpacked) and join the call in Chrome. Don't be in the
same call in both Chrome and the app.

Notes
- Don't use Discord's *Deafen*: it stops audio from arriving.
- Your own microphone isn't transcribed (only other people).
- People whose name can't be read show as `user-1234` (last digits of their id).

## Windows
`app/windows/make-zip.sh` builds `dist/CallTranscriber-Windows.zip` (installer,
app, server; see README-WINDOWS.md). The server picks its speech
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
`node test/app-test.js /tmp/somewhere` runs the real desktop app (its own server,
built-in Discord window pointed at the fake call) and checks injection timing,
capture rate with the window hidden, speakers, channels, and that quitting
stops the server.
