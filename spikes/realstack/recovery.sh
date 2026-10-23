#!/bin/zsh
# Kill each backend in turn (kill -9), probe a spoken turn while it is down, start it again with
# the command `mlxrt dev` uses, time /readyz's return to 200, probe a turn after it.
# Needs the stack started process by process (not `mlxrt dev`, which stops everything when a
# backend exits): run this script's `start` first.
#   zsh spikes/realstack/recovery.sh start            # llm, asr, tts and the gateway, in the background
#   zsh spikes/realstack/recovery.sh kill asr tts llm  # the experiment
#   zsh spikes/realstack/recovery.sh stop
set -u
cd "$(dirname "$0")/../.."
LOGS=spikes/data/realstack/logs
mkdir -p $LOGS
SNAPSHOT=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
export HF_HUB_OFFLINE=1
unset VIRTUAL_ENV

start_one() {
  local log=$LOGS/proc-$1-$(date +%H%M%S).log
  case $1 in
    llm) nohup uv run --locked --project backends/llm python -m mlxrt_llm.serve --host 127.0.0.1 --port 8101 --model $SNAPSHOT --chat-template-args '{"enable_thinking":false}' --max-tokens 2048 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --prompt-cache-size 64 --prompt-cache-bytes 8GB --tools-checkpoint --shared-prefix-checkpoint > $log 2>&1 &;;
    asr) nohup uv run --locked --project backends/audio mlx_audio.server --host 127.0.0.1 --port 8102 --log-dir /tmp/mlxrt-asr > $log 2>&1 &;;
    tts) nohup uv run --locked --project backends/audio mlx_audio.server --host 127.0.0.1 --port 8103 --log-dir /tmp/mlxrt-tts > $log 2>&1 &;;
    gateway) MLXRT__OBSERVABILITY__DEBUG_ENDPOINTS=true nohup uv run --locked mlxrt serve > $log 2>&1 &;;
  esac
}
port_of() { case $1 in llm) echo 8101;; asr) echo 8102;; tts) echo 8103;; gateway) echo 8000;; esac }
probe() { uv run --locked --project spikes/realstack python spikes/realstack/turnprobe.py "$@" 2>&1 | grep -E "response.created|completed|failed|first audio|error|response.done|outcome" | cut -c1-220; }

case $1 in
  start)
    for name in llm asr tts gateway; do start_one $name; done
    until curl -sf -o /dev/null http://127.0.0.1:8000/readyz; do sleep 1; done
    echo "ready";;
  stop)
    for name in gateway llm asr tts; do pid=$(lsof -t -iTCP:$(port_of $name) -sTCP:LISTEN); [ -n "$pid" ] && kill -TERM $pid; done;;
  kill)
    shift
    for name in "$@"; do
      pid=$(lsof -t -iTCP:$(port_of $name) -sTCP:LISTEN)
      echo "=== $name (pid $pid): kill -9 at $(date +%H:%M:%S)"
      kill -9 $pid; sleep 1
      echo "readyz 1 s after: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/readyz)"
      probe $name-down --wait 15
      start_one $name
      t0=$(date +%s.%N)
      until curl -sf -o /dev/null http://127.0.0.1:8000/readyz; do sleep 0.25; done
      printf "readyz 200 again %.1f s after the restart\n" $(( $(date +%s.%N) - t0 ))
      probe $name-restarted --wait 20
    done;;
esac
