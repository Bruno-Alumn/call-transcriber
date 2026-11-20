#!/bin/zsh
# Starts the local transcription server. Leave this window open while recording.
cd "$(dirname "$0")/server" && exec uv run server.py "$@"
