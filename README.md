# Earshot

Live transcripts of your voice calls, one person at a time, all on your own computer.

Earshot is a desktop app for macOS and Windows. Join a voice channel in its built-in call window and it transcribes each participant from their own audio stream. People who talk over each other stay on separate lines, and every line has the right name on it. The transcript, your call history and the stats sit in a dashboard next to the call. No audio leaves your machine.

![Dashboard during a call](docs/dashboard.png)
<sub>Demo data. The names and conversation are made up.</sub>

## What it does

- **One stream per person.** The voice platform sends each participant as a separate audio stream, and Earshot reads those streams directly. It never has to guess who's speaking.
- **Call window built in.** The app opens the voice platform's web client in its own window, signed in with your normal account. You don't need a browser or an extension. Closing the window keeps you in the call.
- **Dashboard.**
  - **Call view:** live transcript with search and export (Markdown, text, SRT, CSV, JSON), who's in the call, talk time, activity over time, a 10-minute timeline, crosstalk and most-said words.
  - **History:** past calls grouped by server and channel, with search across all of them. Right-click a call to archive, export or delete it.
- **Compact mode.** The transcript and every panel on one screen, no scrolling. ⌘⇧K / Ctrl+Shift+K.
- **Share it into the call.** Screen sharing from the call window opens a picker with a compact view made for sharing: live transcript and stats, none of your controls.
- **Personality reads.** Click **Read** next to anyone in Talk time, or right-click their name, for a light-hearted sketch of them: a one-liner, a rough Big Five guess and quotes as evidence. In compact mode the top three speakers still in the call get a one-line read that refreshes every 10 minutes. A small local model writes them (Liquid AI LFM3 1.7B, about 2 GB, downloaded on first use). Needs Apple Silicon or an NVIDIA GPU.
- **Mute the call for yourself.** Turn the sound off on your end and Earshot keeps transcribing (⌘⇧M / Ctrl+Shift+M).
- **Pick your speech model.** All of them run locally and you can switch mid-call:

  | Model | Words wrong* | Mac (Apple GPU) | Windows (NVIDIA GPU) | Notes |
  |---|---|---|---|---|
  | **Granite Speech 5.0 · 470M** (default) | 5.5% | 249× real time | 82× | best overall |
  | Granite Speech 5.0 · 2B | 5.4% | 13× | 11× | most accurate, 9 GB |
  | Parakeet TDT 0.6B v3 | 7.0% | 116× | 30× (CPU) | 25 European languages |
  | Qwen3-ASR · 1.7B | 8.5% | 18× | 10× | 52 languages |
  | Whisper large-v3-turbo | 13% | 15× | 30× | 99 languages |

  <sub>*Word error rate on 120 real meeting clips (AMI), measured on our own machines.</sub>
- **Updates itself.** The dashboard tells you when a release is out, and **Update now** installs it.

| Compact mode | Shared view (what the call sees) |
|---|---|
| ![Compact mode](docs/compact.png) | ![Shared view](docs/shared-view.png) |

## Install

### macOS (Apple Silicon)
You need [uv](https://docs.astral.sh/uv/), Node.js and Xcode's command-line tools.
```bash
git clone https://github.com/Bruno-Alumn/earshot.git ~/earshot
cd ~/earshot/server && uv sync
~/earshot/app/mac/build.sh
```
`build.sh` puts the app in `~/Applications`. Open it, sign in once in the call window (⌘D), and join a voice channel. macOS asks for the microphone the first time.

### Windows 10/11
```bash
git clone https://github.com/Bruno-Alumn/earshot.git %USERPROFILE%\Earshot
```
Then double-click **install.cmd** in that folder. It installs:
- Python and the transcription server
- the GPU engine if you have an NVIDIA card, otherwise a fast CPU engine
- the speech model and the app (plus Node.js if it's missing)
- Start-menu and desktop shortcuts

[README-WINDOWS.md](README-WINDOWS.md) has the details. Install with `git clone` rather than the release zip, or in-app updates won't work.

## Using it

1. Start the app. It runs the transcription server and opens the dashboard.
2. Open the call window (the button in the top right, or ⌘D / Ctrl+D) and join a voice channel. Transcription starts on its own.
3. Transcripts are saved in `transcripts/`, one Markdown and one JSON Lines file per day.

Closing the windows doesn't stop anything. Quit (⌘Q, or the Windows tray icon) stops it all. The dashboard also runs at http://127.0.0.1:8765/.

Notes:
- Your own microphone isn't transcribed, only other people.
- Don't deafen yourself in the call: that stops the audio from arriving. Use **Mute call audio** instead.
- If someone's name can't be read, they show up as `user-1234` (the last digits of their ID).
- Recording laws differ by place. Following the ones that apply to you and the people in the call is up to you.

## How it works

```
call window (built-in Chromium)
  └─ page-hook.js, injected before the web client's own code
       reads each participant's WebRTC audio track, in 16 kHz chunks
  └─ app main process ──WebSocket──▶ local server (Python)
                                      ├─ voice activity detection per person
                                      ├─ speech model (MLX / CUDA / ONNX)
                                      └─ transcripts/*.md + *.jsonl
dashboard ◀──HTTP + live updates── local server
```

| Part | Where |
|---|---|
| Desktop app (Electron): dashboard, call window, share picker, tray | `app/desktop/` |
| Transcription server and dashboard | `server/` |
| Hook that reads each person's audio | `app/desktop/page-hook.js` |
| Mac app build | `app/mac/build.sh` |
| Windows installer | `install.cmd`, `install.ps1`, `app/windows/` |
| Tests | `test/` |

The server picks the speech engine for your machine: MLX on Apple Silicon, CUDA with an NVIDIA GPU, otherwise ONNX Runtime on the CPU. `--engine mlx|torch|onnx` overrides it.

## Development

- `node test/app-test.js <dir>` runs the real app against a fake call with four speakers. It checks that the hook loads before the web client's code, the capture rate with the window hidden, names, channels, mute and screen sharing, and that quitting stops the server.
- `cd server && uv run python ../test/server_e2e.py` runs the server test with three people talking over each other.
- CI runs both on macOS and Windows for every push. Pushing a `vX.Y.Z` tag that matches `VERSION` publishes a release, which the app then offers as an update.

Transcripts, learned names and archive choices are personal data and never get committed (see `.gitignore`).
