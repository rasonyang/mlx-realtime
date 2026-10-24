#!/bin/zsh
# AICC's own transcription client (openairt.go) and audio pump (pump.go) against a running
# mlx-realtime, configuration only: AICC_TRANSCRIBE_PROVIDER / _ENDPOINT, any non-empty key. The two
# legs of one corpus call are fed as the human call's two speakers. Nothing in the AICC checkout is
# modified (go test -overlay adds tests/aicc/zz_mlx_transcribe_test.go to internal/streamin).
#   zsh spikes/transcription/aicc_e2e.sh [CALL] [SECONDS] [SPEED]    (AICC_REPO, ENDPOINT override)
set -eu
cd "$(dirname "$0")/../.."
call=${1:-call04}; seconds=${2:-240}; speed=${3:-1}
AICC=${AICC_REPO:-$HOME/workspaces/github/ai-native-callcenter}
DATA=spikes/transcription/data/aicc
mkdir -p $DATA
uv run --locked --project spikes/transcription python - "$call" "$seconds" <<'PY'
import sys
sys.path.insert(0, "spikes/transcription")
import common as c
call, seconds = sys.argv[1], float(sys.argv[2])
for leg in ("agent", "customer"):
    pcm = c.read_wav(c.DATA / "legs" / f"{call}_{leg}.wav")[: int(seconds * c.RATE)]
    (c.DATA / "aicc" / f"{call}_{leg}.pcm").write_bytes(pcm.astype("<i2").tobytes())
PY
OVERLAY=$DATA/overlay.json
print -r -- "{\"Replace\": {\"$AICC/internal/streamin/zz_mlx_transcribe_test.go\": \"$PWD/tests/aicc/zz_mlx_transcribe_test.go\"}}" > $OVERLAY
cd $AICC
AICC_TRANSCRIBE_PROVIDER=openai \
AICC_TRANSCRIBE_ENDPOINT=${ENDPOINT:-ws://127.0.0.1:8000/v1/realtime?intent=transcription} \
OPENAI_API_KEY=local \
MLX_TRANSCRIBE_SPEED=$speed \
MLX_TRANSCRIBE_HINTS="NovaNet, Orbix, Lumio, Veltra, Zentra, SkyWave, Wi-Fi 7, eSIM, 5G 畅享 199, 5G 尊享 299" \
MLX_TRANSCRIBE_LEGS="HUMAN_AGENT=$OLDPWD/$DATA/${call}_agent.pcm,CUSTOMER=$OLDPWD/$DATA/${call}_customer.pcm" \
go test -count=1 -overlay=$OLDPWD/$OVERLAY -run '^TestMLXTranscribeLegs$' -v ./internal/streamin/
