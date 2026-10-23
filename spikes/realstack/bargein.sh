#!/bin/zsh
# AICC's TestMLXBargeIn N times in each profile (PCM, then G.711) against the running gateway:
# speech_started after the caller's onset (2.5 s into the line + the clip's 0.24 s lead).
#   zsh spikes/realstack/bargein.sh [N]            (AICC_REPO: an AICC checkout)
set -u
cd "$(dirname "$0")/../.."
AICC=${AICC_REPO:-../ai-native-callcenter}
n=${1:-10}
for profile in pcm g711; do
  flag=""; [ $profile = g711 ] && flag="--g711"
  for i in $(seq 1 $n); do
    out=$(uv run --locked python tests/aicc/probe.py --aicc-repo $AICC --url ws://127.0.0.1:8000/v1/realtime --run '^TestMLXBargeIn$' $flag 2>&1)
    into=$(echo "$out" | grep -o "speech_started [0-9.]*s into its playback" | head -1 | grep -o "[0-9.]*s" | tr -d s)
    result=$(echo "$out" | grep -E "^--- (PASS|FAIL)" | head -1 | cut -c1-30)
    heard=$(echo "$out" | grep -o 'heard "[^"]*"' | head -1)
    late=$( [ -n "$into" ] && python3 -c "print(round($into - 2.74, 2))" )
    echo "$profile $i | speech_started ${into}s into the line, ${late}s after the onset | $result | $heard"
  done
done
