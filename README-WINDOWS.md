# Earshot for Windows

Live transcripts of your voice calls, with each person transcribed from their own audio stream, grouped by server and channel, in a dashboard. No audio leaves your PC.

## Install (once)
1. Get the files: `git clone https://github.com/Bruno-Alumn/earshot.git %USERPROFILE%\Earshot`. You can also unzip the Windows package from a release, but then in-app updates won't work.
2. Double-click **install.cmd**. The first run takes a few minutes. It installs Python, the speech model and the app (plus Node.js if it's missing) and adds Start-menu and desktop shortcuts.
   - With an NVIDIA graphics card it also installs the GPU engine, which is the most accurate.
   - Without one it uses a CPU engine (Parakeet, about 50× faster than real time).
3. Start the app and sign in once in its call window.

## Use
- Start the app from the Start menu or desktop. It runs the transcription server, opens the dashboard and opens the call window (Ctrl+D brings it back). Join a voice channel there and transcription starts.
- Closing the dashboard keeps it running in the system tray, bottom right. Closing the call window hides it and keeps you in the call. Right-click the tray icon to pause, restart or stop the server, or quit.
- Transcripts go in the `transcripts` folder, one .md file per day.

## Notes
- Needs Windows 10 or 11 (64-bit). Don't deafen yourself in the call: use **Mute call audio** instead.
- To update, click **Update now** in the dashboard, or run `git pull` and then install.cmd again. Your `transcripts` folder is kept.
- If something goes wrong, check `server.log` in this folder.
