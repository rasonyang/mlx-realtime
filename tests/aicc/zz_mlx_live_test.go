// mlx-realtime's acceptance wrapper around AICC's own Realtime client harness (gateway profile).
//
// It is not part of AICC. tests/aicc/probe.py adds it to AICC's internal/provider package with
// `go test -overlay`, so no AICC checkout is modified. It drives AICC's real client, exactly as a
// call would, against MLX_REALTIME_URL:
//
//	MLX_REALTIME_URL=ws://127.0.0.1:8000/v1/realtime go test -count=1 -run '^TestMLX' -v ./internal/provider/
//
// AICC's own deadlines apply throughout: the handshake within 3 s; response.created within 3 s of
// response.create (measured here, enforced by probe.py); the first audio within 3 s of
// response.created and no 2 s gap between deltas (AICC's watchdog logs "response abandoned").

package provider

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/rasonyang/ai-native-callcenter/internal/media"
)

func mlxProfile(t *testing.T) Profile {
	t.Helper()
	url := os.Getenv("MLX_REALTIME_URL")
	if url == "" {
		t.Skip("MLX_REALTIME_URL is not set")
	}
	p := GatewayProfile()
	p.Endpoint = url
	if m := os.Getenv("MLX_REALTIME_MODEL"); m != "" {
		p.Model = m
	}
	if os.Getenv(p.APIKeyEnv) == "" {
		t.Setenv(p.APIKeyEnv, "local")
	}
	return p
}

// Handshake, opening turn on an empty conversation, 24 kHz audio back.
func TestMLXGreeting(t *testing.T) {
	p := mlxProfile(t)
	_, output := p.FormatsFor(media.LawMu)
	result := probe(t, p, "You answer the phone for NovaNet. Greet the caller in one short "+
		"sentence and ask how you can help.", 12*time.Second)
	checkAudioArrived(t, result, output)
}

// response.create -> response.created latency against the 3 s bound.
func TestMLXResponseCreatedLatency(t *testing.T) {
	p := mlxProfile(t)
	measureResponseCreated(t, p, "You answer the phone for NovaNet. Reply in one short sentence.", "en", 8)
}

// A flow-chosen voice is accepted (D19), and the opening line and a mid-call line are spoken
// word for word (D16): AICC's SayExactly directions, sent as the client sends them.
func TestMLXLines(t *testing.T) {
	p := mlxProfile(t)
	var logs lockedBuffer
	session, err := New(p, slog.New(slog.NewTextHandler(&logs,
		&slog.HandlerOptions{Level: slog.LevelWarn})))
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	defer func() { _ = session.Close(context.Background()) }()

	input, output := p.FormatsFor(media.LawMu)
	opening := "Good morning, NovaNet support."
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if err := session.Start(ctx, SessionConfig{
		Instructions: "You answer the phone for NovaNet.",
		Voice:        "flow-voice",
		Language:     "en",
		Turn:         DefaultTurnDetection(),
		OpeningText:  opening,
		InputFormat:  input,
		OutputFormat: output,
	}); err != nil {
		t.Fatalf("start: %v\nlogs: %s", err, logs.String())
	}
	stop := make(chan struct{})
	defer close(stop)
	go feedRoomTone(session, media.LawMu, input, stop)

	said, audio := awaitLine(t, session)
	t.Logf("opening line: %q, %d audio bytes", said, audio)
	if said != opening || audio == 0 {
		t.Errorf("opening: want %q with audio, got %q and %d bytes", opening, said, audio)
	}
	line := "One moment, let me check that."
	if err := session.SpeakText(line, false); err != nil {
		t.Fatalf("speak: %v", err)
	}
	said, audio = awaitLine(t, session)
	t.Logf("wait line: %q, %d audio bytes", said, audio)
	if said != line || audio == 0 {
		t.Errorf("line: want %q with audio, got %q and %d bytes", line, said, audio)
	}
	if text := logs.String(); strings.Contains(text, "response abandoned") ||
		strings.Contains(text, "unanswered") {
		t.Errorf("the client gave up on a turn:\n%s", text)
	}
}

// awaitLine collects one turn's final output transcript and audio, up to its end.
func awaitLine(t *testing.T, session *Realtime) (string, int) {
	t.Helper()
	said, audio := "", 0
	deadline := time.After(15 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatalf("the turn never finished (said %q so far)", said)
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("the session ended")
			}
			switch event.Type {
			case EventTypeAudioDelta:
				audio += len(event.Audio)
			case EventTypeOutputTranscript:
				if event.IsFinal {
					said += event.Text
				}
			case EventTypeError:
				t.Errorf("provider error (fatal=%v): %v", event.IsFatal, event.Err)
			case EventTypeResponseDone:
				return said, audio
			}
		}
	}
}
