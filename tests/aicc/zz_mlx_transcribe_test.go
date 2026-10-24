// mlx-realtime's acceptance wrapper around AICC's own transcription client and audio pump.
//
// It is not part of AICC. It is added to AICC's internal/streamin package with `go test -overlay`
// (spikes/transcription/aicc_e2e.sh), so no AICC checkout is modified. It drives AICC's real
// transcribe.New (openairt.go) and newPump (pump.go) against a mlx-realtime transcription session,
// the way a human call's two legs do: one session and one pump per leg, 20 ms frames of 24 kHz
// PCM16 written to the pump, the pump's silence detector committing each utterance.
//
//	AICC_TRANSCRIBE_PROVIDER=openai \
//	AICC_TRANSCRIBE_ENDPOINT='ws://127.0.0.1:8000/v1/realtime?intent=transcription' \
//	OPENAI_API_KEY=local \
//	MLX_TRANSCRIBE_LEGS='HUMAN_AGENT=agent.pcm,CUSTOMER=customer.pcm' \
//	go test -count=1 -overlay=overlay.json -run '^TestMLXTranscribeLegs$' -v ./internal/streamin/
//
// MLX_TRANSCRIBE_SPEED (default 1) streams faster than real time; MLX_TRANSCRIBE_HINTS is the
// comma-separated hint list AICC would send as the prompt.

package streamin

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rasonyang/ai-native-callcenter/internal/transcribe"
)

func TestMLXTranscribeLegs(t *testing.T) {
	legs := os.Getenv("MLX_TRANSCRIBE_LEGS")
	if legs == "" {
		t.Skip("MLX_TRANSCRIBE_LEGS is not set")
	}
	provider := os.Getenv("AICC_TRANSCRIBE_PROVIDER")
	endpoint := os.Getenv("AICC_TRANSCRIBE_ENDPOINT")
	profile, err := transcribe.ProfileFor(provider, transcribe.Override{Endpoint: endpoint, Model: os.Getenv("AICC_TRANSCRIBE_MODEL")})
	if err != nil {
		t.Fatalf("profile: %v", err)
	}
	key := os.Getenv(profile.APIKeyEnv)
	if key == "" {
		key = "local"
	}
	speed := 1.0
	if s := os.Getenv("MLX_TRANSCRIBE_SPEED"); s != "" {
		speed, _ = strconv.ParseFloat(s, 64)
	}
	var hints []string
	for _, h := range strings.Split(os.Getenv("MLX_TRANSCRIBE_HINTS"), ",") {
		if h = strings.TrimSpace(h); h != "" {
			hints = append(hints, h)
		}
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))

	var out sync.Mutex
	var wg sync.WaitGroup
	counts := map[string]map[transcribe.EventType]int{}
	for _, spec := range strings.Split(legs, ",") {
		speaker, path, _ := strings.Cut(spec, "=")
		pcm, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		session, err := transcribe.New(profile, key, log)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		if err := session.Start(context.Background(), transcribe.Config{Language: "zh", Hints: hints}); err != nil {
			t.Fatalf("start %s: %v", speaker, err)
		}
		counts[speaker] = map[transcribe.EventType]int{}
		p := newPump(speaker, profile.Name, session, profile.OwnsEndpointing, log)
		wg.Add(2)
		go func() { // the leg's events, as the ingest reads them
			defer wg.Done()
			for ev := range session.Events() {
				out.Lock()
				counts[speaker][ev.Type]++
				fmt.Printf("EVENT\t%s\t%s\t%s\t%s\t%v\n", speaker, ev.Type, ev.UtteranceID, ev.Text, ev.Err)
				out.Unlock()
			}
		}()
		go func() { // the leg's audio, 20 ms frames
			defer wg.Done()
			const frame = 960 // 20 ms of 24 kHz s16le
			start := time.Now()
			for i := 0; (i+1)*frame <= len(pcm); i++ {
				p.write(pcm[i*frame : (i+1)*frame])
				if d := time.Until(start.Add(time.Duration(float64((i+1)*20)/speed) * time.Millisecond)); d > 0 {
					time.Sleep(d)
				}
			}
			time.Sleep(3 * time.Second) // the last utterance's commit and final
			p.close(context.Background())
		}()
	}
	wg.Wait()
	for speaker, c := range counts {
		fmt.Printf("SUMMARY\t%s\tpartial=%d final=%d error=%d closed=%d\n", speaker,
			c[transcribe.EventPartial], c[transcribe.EventFinal], c[transcribe.EventError], c[transcribe.EventClosed])
		if c[transcribe.EventFinal] == 0 {
			t.Errorf("%s: no final transcript", speaker)
		}
		if c[transcribe.EventError] != 0 {
			t.Errorf("%s: %d error events", speaker, c[transcribe.EventError])
		}
	}
}
