#!/bin/zsh
# S6 soak: 3 concurrent sessions looping over the manifest through the running gateway for at
# least SOAK_S seconds (bench passes back to back: wide pcm, tel8k pcm, tel8k pcmu, ...), with
# per-process memory every 5 minutes (spikes/results/realstack_soak_memory.jsonl).
#   zsh spikes/realstack/soak.sh [SOAK_S]          (from the repository root, make dev running)
set -u
cd "$(dirname "$0")/../.."
SOAK_S=${1:-1800}
LOGS=spikes/data/realstack/logs
mkdir -p $LOGS
python3 -I spikes/realstack/memsample.py spikes/results/realstack_soak_memory.jsonl 300 $((SOAK_S + 900)) > /dev/null &
SAMPLER=$!
start=$(date +%s)
echo "$(date +%H:%M:%S) soak start ($SOAK_S s)"
pass=0
specs=(wide:pcm tel8k:pcm tel8k:pcmu)
while [ $(( $(date +%s) - start )) -lt $SOAK_S ]; do
  spec=${specs[$(( pass % 3 + 1 ))]}
  IFS=: read variant codec <<< "$spec"
  pass=$((pass + 1))
  name=soak-$pass-$variant-$codec
  echo "$(date +%H:%M:%S) pass $pass: $variant $codec, 3 sessions"
  uv run --locked python -m bench.bench --url ws://127.0.0.1:8000/v1/realtime \
      --input spikes/data/manifest.jsonl --variant $variant --codec $codec \
      --concurrency 3 --out bench/results/real-$name > $LOGS/bench-$name.log 2>&1
  echo "$(date +%H:%M:%S) pass $pass exit $?"
  grep -E "^- utterances|first audio delta|longest gap|last audio delta|no audio \|" $LOGS/bench-$name.log
done
echo "$(date +%H:%M:%S) soak end after $(( $(date +%s) - start )) s, $pass passes"
python3 -I spikes/realstack/memsample.py spikes/results/realstack_soak_memory.jsonl 1 1 > /dev/null
kill $SAMPLER 2>/dev/null
