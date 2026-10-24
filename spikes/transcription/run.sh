#!/bin/zsh
# uv run --locked --project spikes/transcription python spikes/transcription/<script>.py ARGS
cd "$(dirname "$0")/../.."
s=$1; shift
exec uv run --locked --project spikes/transcription python spikes/transcription/$s.py "$@"
