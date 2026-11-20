# Call Transcriber for Windows

Live, local transcripts of Discord voice calls - each person transcribed from
their own audio stream, grouped by server and voice channel, with a dashboard.
Audio never leaves your PC.

## Install (once)
1. Get the files: `git clone <repo url> %USERPROFILE%\CallTranscriber`
   (or unzip the Windows package from a CI build somewhere permanent).
2. Double-click **install.cmd**. It installs Python and the speech model
   (a few minutes the first time) and creates Start-menu and desktop shortcuts.
   - With an NVIDIA graphics card it also installs the GPU engine (best accuracy).
   - Without one it uses a fast CPU engine (Parakeet, ~50x faster than real time).
3. Load the Chrome extension: `chrome://extensions` -> Developer mode ->
   **Load unpacked** -> pick the `extension` folder (the installer opens it).

## Use
- Start **Call Transcriber** (Start menu / desktop). It runs the transcription
  server and shows the dashboard. Join a voice channel in Discord in Chrome.
- Closing the window keeps transcribing in the system tray (bottom-right).
  Right-click the tray icon for Restart/Stop server and Quit.
- Transcripts: the `transcripts` folder (one .md file per day).

## Notes
- Requires Windows 10 or 11 (64-bit) and Chrome. Don't use Discord's Deafen.
- To update: `git pull` in the folder, then run install.cmd again
  (your `transcripts` folder is kept).
- Problems: see `server.log` in this folder.
