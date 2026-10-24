#!/bin/zsh
# CER passes over the whole corpus (resumable per leg: rerun the same command after an interruption).
#   zsh spikes/transcription/cer_passes.sh TAG PROMPT [TAG PROMPT ...]     e.g. terms100 terms none100 none
# CALLS=call01,call02,... restricts the pass. Writes spikes/results/transcription_cer_<TAG>.json (+ .partial.jsonl while running)
cd "$(dirname "$0")/../.."
mkdir -p spikes/transcription/data/logs
while [ $# -ge 2 ]; do
  tag=$1; prompt=$2; shift 2
  spikes/transcription/run.sh harness cer --out spikes/results/transcription_cer_$tag.json --prompt $prompt --speed 4 ${CALLS:+--calls} ${CALLS:+$CALLS} \
    >> spikes/transcription/data/logs/cer_$tag.log 2>&1
done
