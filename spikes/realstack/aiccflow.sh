#!/bin/zsh
# AICC's flow with spoken caller turns (aiccflow.py): N calls alone, then N calls while two bench
# sessions talk (wide, 24 kHz PCM), against the running gateway. Rows go to
# spikes/results/realstack_aiccflow_{alone,load}.jsonl.
#   zsh spikes/realstack/aiccflow.sh [N]           (AICC_REPO: an AICC checkout)
set -u
cd "$(dirname "$0")/../.."
n=${1:-10}
LOGS=spikes/data/realstack/logs
mkdir -p $LOGS
[ -f spikes/data/realstack/clips/transfer.wav ] || uv run --locked --project spikes/realstack python spikes/realstack/make_clips.py
run() { uv run --locked --project spikes/realstack python spikes/realstack/aiccflow.py "$@"; }
rm -f spikes/results/realstack_aiccflow_alone.jsonl spikes/results/realstack_aiccflow_load.jsonl
echo "$(date +%H:%M:%S) alone"
run spikes/results/realstack_aiccflow_alone.jsonl $n > $LOGS/aiccflow_alone.log 2>&1
echo "$(date +%H:%M:%S) with two bench sessions"
uv run --locked python -m bench.bench --url ws://127.0.0.1:8000/v1/realtime \
    --input spikes/data/manifest.jsonl --variant wide --concurrency 2 > $LOGS/aiccflow_bench.log 2>&1 &
BENCH=$!
sleep 15
run spikes/results/realstack_aiccflow_load.jsonl $n > $LOGS/aiccflow_load.log 2>&1
kill -INT $BENCH 2>/dev/null; sleep 3; kill $BENCH 2>/dev/null
echo "$(date +%H:%M:%S) done"
