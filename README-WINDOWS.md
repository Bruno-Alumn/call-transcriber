# Call Transcriber for Windows

Live, local transcripts of Discord voice calls - each person transcribed from
their own audio stream, grouped by server and voice channel, with a dashboard.
Audio never leaves your PC.

## Install (once)
1. Get the files: `git clone <repo url> %USERPROFILE%\CallTranscriber`
   (or unzip the Windows package from a CI build somewhere permanent).
2. Double-click **install.cmd**. It installs Python, the speech model and the
   app (a few minutes the first time; Node.js too if it's missing) and creates
   Start-menu and desktop shortcuts.
   - With an NVIDIA graphics card it also installs the GPU engine (best accuracy).
   - Without one it uses a fast CPU engine (Parakeet, ~50x faster than real time).
3. Start **Call Transcriber** and sign in to Discord in its Discord window (once).

## Use
- Start **Call Transcriber** (Start menu / desktop). It runs the transcription
  server, shows the dashboard and opens Discord in its own window (Ctrl+D or the
  Discord button shows it). Join a voice channel there; that's it.
- Closing the window keeps transcribing in the system tray (bottom-right).
  Closing the Discord window hides it and keeps you in the call.
  Right-click the tray icon for Pause, Restart/Stop server and Quit.
- Transcripts: the `transcripts` folder (one .md file per day).

## Notes
- Requires Windows 10 or 11 (64-bit). Chrome isn't needed. Don't use Discord's Deafen.
- To update: use **Update now** in the dashboard, or `git pull` and run install.cmd again
  (your `transcripts` folder is kept).
- Problems: see `server.log` in this folder.
