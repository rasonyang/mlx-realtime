#!/bin/zsh
# Commit-to-completed latency with 1, 2 and 4 sessions open, in real time, and the bots' first audio
# alone (2 bots) and beside 2 busy transcription sessions (4). Needs the stack and AICC_REPO (aiccflow's fixture).
#   zsh spikes/transcription/latency_runs.sh TAG [CALLS] [SECONDS]     files: spikes/results/transcription_latency_${tag}_<TAG>_{1,2,4,bots2}.json
#   clean = call04,call07 (no hold music, no DTMF); holdmusic = call06,call14 (music and DTMF)
cd "$(dirname "$0")/../.."
tag=${1:-clean}; calls=${2:-call04,call07}; sec=${3:-240}
H="spikes/transcription/run.sh harness latency --calls $calls --seconds $sec"
spikes/transcription/run.sh harness latency --sessions bots2 --out spikes/results/transcription_latency_${tag}_bots2.json --bot-calls 8 --bot-lead 0 --seconds $sec
$=H --sessions 1 --out spikes/results/transcription_latency_${tag}_1.json
$=H --sessions 2 --out spikes/results/transcription_latency_${tag}_2.json
$=H --sessions 4 --out spikes/results/transcription_latency_${tag}_4.json --bot-calls 8
