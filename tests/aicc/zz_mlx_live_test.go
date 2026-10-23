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
//
// The barge-in scenarios (M4, AICC acceptance part 2: TestMLXBargeIn, TestMLXKeypressCancel) stand
// in for AICC's call session where it drives the client, and TestMLXBargeIn plays the caller's
// speech from MLX_REALTIME_CALLER_CLIP: 24 kHz mono PCM16, which probe.py writes from
// tests/fixtures/speech/en_dinosaurs.wav.
//
// With MLX_REALTIME_G711 set (probe.py --g711; M7) the profile is GatewayProfile with
// AcceptsG711: true, as a dedicated AICC profile for the gateway would be: G.711 μ-law both ways,
// passed through untouched ({"type": "audio/pcmu"}, 160-byte frames every 20 ms). The caller's
// clip is converted to the session's format by AICC's own media.Converter.

package provider

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
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
	if os.Getenv("MLX_REALTIME_G711") != "" {
		p.AcceptsG711 = true // a dedicated profile: telephone audio passed through (M7)
	}
	if m := os.Getenv("MLX_REALTIME_MODEL"); m != "" {
		p.Model = m
	}
	if os.Getenv(p.APIKeyEnv) == "" {
		t.Setenv(p.APIKeyEnv, "local")
	}
	return p
}

// Handshake, opening turn on an empty conversation, audio back (24 kHz PCM, or G.711).
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

//
// Barge-in (M4, AICC acceptance part 2).
//
// On a call it is AICC's session (internal/aicall) that drives the client, and these tests do what
// it does: on SPEECH_STARTED it flushes playback, counts what reached the caller and calls
// Interrupt(SPEECH, ms) (Session.bargeIn); on a keypress it calls Interrupt(DTMF, ms) and then
// SendUserText with the keypress note (Session.pumpDigits). On the gateway profile
// (CancelsResponseItself) the client sends no response.cancel for speech, only the truncate.
//

// billingLine is a mid-call line long enough that its response is still open when the caller
// talks over it. The gateway sends a turn's audio up to output.max_lead_ms (10 s) ahead of real
// time and ends the response once the last of it is sent, so with a line under ~13 s the response
// is already over when the caller's speech is confirmed (bargeInAfter, the clip's 0.24 s lead-in
// and 400 ms of speech). The mock TTS speaks 60 ms per character: this is ~19 s.
const billingLine = "Thank you for calling NovaNet billing. Your October invoice is now ready. " +
	"The total due this month is forty two dollars. Payment is taken on the fifth of each month " +
	"from the card on file. To change that card, press one or stay on the line. To set up a " +
	"payment plan, press two. To hear these options again, press nine. An agent will be with " +
	"you shortly, and we thank you for your patience."

// bargeInAfter is when the caller starts talking over the line, from its first audio: past the
// gateway's echo guard (barge_in.echo_guard_ms, 2 s), inside which no speech onset interrupts.
const bargeInAfter = 2500 * time.Millisecond

// frameMs is one frame of call audio. AICC hands audio to the wire, and counts what was played,
// in whole frames.
const frameMs = int(FrameInterval / time.Millisecond)

// What the client logs (realtime.go) for the events these tests look for.
const (
	// logUnmapped is a provider event it has no mapping for, such as conversation.item.truncated.
	logUnmapped = "unmapped provider event"
	// logUnheardDeleted is its conversation.item.delete of a turn nobody heard (deleteUnheardItem).
	logUnheardDeleted = "took a turn the caller heard none of out of the conversation"
	// logItemRemoved is the provider's conversation.item.deleted.
	logItemRemoved = "the provider removed an item from the conversation"
)

// The caller talks over a long line, as on a call: playback stops on SPEECH_STARTED and AICC
// reports how much of the line was heard. The gateway must confirm the speech only once it is past
// the echo guard, end the line as cancelled right behind speech_started, accept the truncate, and
// answer the caller once their speech has stopped.
func TestMLXBargeIn(t *testing.T) {
	p := mlxProfile(t)
	input, output := p.FormatsFor(media.LawMu)
	clip := callerClip(t, input)
	calls := &callLog{}
	session := startCall(t, p, calls, input, output)
	caller := &callerLine{}
	stop := make(chan struct{})
	defer close(stop)
	go caller.feed(session, media.LawMu, input, stop)
	hearOpening(t, session, output)

	mark := calls.mark()
	if err := session.SpeakText(billingLine, false); err != nil {
		t.Fatalf("speak: %v", err)
	}
	var (
		lineStarted                              bool
		lineAudio, replyAudio, playedMs          int
		firstAudio, spoke, started, cut, stopped time.Time
		reply, replyAudioAt                      time.Time
		heard, replied                           string
		bargeIn                                  <-chan time.Time
	)
	// Offsets into the line's playback, for the messages.
	into := func(at time.Time) string {
		if at.IsZero() || firstAudio.IsZero() {
			return "never"
		}
		return at.Sub(firstAudio).Round(time.Millisecond).String()
	}
	deadline := time.After(30 * time.Second)
	for done := false; !done; {
		select {
		case <-deadline:
			t.Fatalf("no reply to the caller within 30s; into the line: caller spoke %s, "+
				"speech_started %s, line cut %s, speech_stopped %s, reply %s",
				into(spoke), into(started), into(cut), into(stopped), into(reply))
		case <-bargeIn:
			bargeIn = nil
			spoke = time.Now()
			caller.say(clip, input)
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("the session ended")
			}
			now := time.Now()
			switch event.Type {
			case EventTypeResponseStarted:
				switch {
				case !lineStarted:
					if !event.IsSpokenLine {
						t.Fatalf("the turn that started is not the line")
					}
					lineStarted = true
				case stopped.IsZero():
					t.Fatalf("a turn started %s into the line, before the caller's speech stopped",
						into(now))
				case !reply.IsZero():
					t.Fatalf("a second turn started after the reply")
				default:
					reply = now
				}
			case EventTypeAudioDelta:
				switch {
				case !reply.IsZero():
					if replyAudioAt.IsZero() {
						replyAudioAt = now
					}
					replyAudio += len(event.Audio)
				case !started.IsZero():
					t.Fatalf("audio of the line arrived after speech_started (%s into the line)",
						into(now))
				default:
					if firstAudio.IsZero() {
						firstAudio = now
						bargeIn = time.After(bargeInAfter)
					}
					lineAudio += len(event.Audio)
				}
			case EventTypeSpeechStarted:
				switch {
				case spoke.IsZero():
					t.Fatalf("speech_started %s into the line, before the caller spoke: the line's "+
						"audio or room tone was taken for speech", into(now))
				case !started.IsZero():
					t.Fatalf("a second speech_started, %s after the first: the caller's speech was split",
						now.Sub(started).Round(time.Millisecond))
				}
				started = now
				// As aicall.Session.bargeIn: playback stops here, and what reached the wire is
				// what the caller heard.
				playedMs = playedLikeAICC(lineAudio, now.Sub(firstAudio), output)
				if err := session.Interrupt(InterruptReasonSpeech, playedMs); err != nil {
					t.Fatalf("interrupt: %v", err)
				}
			case EventTypeInterrupted:
				if started.IsZero() {
					t.Fatalf("the line was cut short (by %s) %s into it, before speech_started",
						event.InterruptedBy, into(now))
				}
				if event.InterruptedBy != InterruptReasonSpeech {
					t.Errorf("the line was cut short by %s, want %s",
						event.InterruptedBy, InterruptReasonSpeech)
				}
				cut = now
			case EventTypeSpeechStopped:
				if started.IsZero() {
					t.Fatalf("speech_stopped %s into the line without speech_started", into(now))
				}
				stopped = now
			case EventTypeInputTranscript:
				if event.IsFinal {
					heard += event.Text
				}
			case EventTypeOutputTranscript:
				if event.IsFinal && !reply.IsZero() {
					replied += event.Text
				}
			case EventTypeError:
				t.Fatalf("provider error %s into the line (fatal=%v): %v",
					into(now), event.IsFatal, event.Err)
			case EventTypeResponseDone:
				if reply.IsZero() {
					t.Fatalf("the line ran to its end (%s into it) instead of being cut short; "+
						"the caller spoke %s into it, speech_started %s",
						into(now), into(spoke), into(started))
				}
				done = true
			}
		}
	}

	if cut.IsZero() {
		t.Fatalf("the line's turn never ended as cancelled (no response.done cancelled)")
	}
	if replyAudio == 0 {
		t.Fatalf("the reply had no audio")
	}
	if !calls.await(mark, logUnmapped, "type", "conversation.item.truncated", 2*time.Second) {
		t.Errorf("the truncate to %d ms was never acknowledged (no conversation.item.truncated); "+
			"unmapped events: %v", playedMs, calls.values(mark, logUnmapped, "type"))
	}
	if deleted := calls.values(mark, logUnheardDeleted, "itemId"); len(deleted) > 0 {
		t.Errorf("the client deleted %v, but the caller heard %d ms of the line", deleted, playedMs)
	}
	checkNoWarnings(t, calls)
	t.Logf("barge-in: caller audio from %s into the line, speech_started %s into its playback",
		into(spoke), into(started))
	t.Logf("barge-in: reported %d ms played (%d ms had arrived), line cancelled %s after "+
		"speech_started, truncate acknowledged",
		playedMs, int(audioDuration(lineAudio, output)/time.Millisecond),
		cut.Sub(started).Round(time.Millisecond))
	t.Logf("barge-in: speech_stopped %s after speech_started; reply turn %s and its audio %s "+
		"after speech_stopped (%d ms of audio); heard %q, replied %q",
		stopped.Sub(started).Round(time.Millisecond), reply.Sub(stopped).Round(time.Millisecond),
		replyAudioAt.Sub(stopped).Round(time.Millisecond),
		int(audioDuration(replyAudio, output)/time.Millisecond), heard, replied)
}

// Keypresses over a line, as AICC's session handles them: the turn is cut, and the keypress note
// gets a reply of its own once it has ended (SendUserText waits for the cut turn's done).
//
//   - Key 1 as the line's first audio arrives: the audio was queued and none of it played, so
//     AICC reports 0 ms. The client sends response.cancel and a truncate to 0, and deletes the
//     item once the response is done (removeUnheardTurn).
//   - Key 2 the moment the next line's turn starts, before any of its audio: nothing was queued,
//     so AICC reports -1. The client sends response.cancel only, and at response.done deletes the
//     turn's spoken items, none of whose audio arrived (removeUnheardTurn).
func TestMLXKeypressCancel(t *testing.T) {
	p := mlxProfile(t)
	input, output := p.FormatsFor(media.LawMu)
	calls := &callLog{}
	session := startCall(t, p, calls, input, output)
	stop := make(chan struct{})
	defer close(stop)
	go feedRoomTone(session, media.LawMu, input, stop)
	hearOpening(t, session, output)

	mark := calls.mark()
	key := pressKeyOverLine(t, session, "1", false)
	if key.itemID == "" || key.lineAudio == 0 {
		t.Fatalf("key 1: pressed with no audio of the line (item %q, %d bytes)",
			key.itemID, key.lineAudio)
	}
	checkKeypress(t, key)
	if !calls.await(mark, logUnmapped, "type", "conversation.item.truncated", 2*time.Second) {
		t.Errorf("key 1: the truncate to 0 ms was never acknowledged (no "+
			"conversation.item.truncated); unmapped events: %v", calls.values(mark, logUnmapped, "type"))
	}
	if deleted := calls.values(mark, logUnheardDeleted, "itemId"); !slices.Equal(deleted, []string{key.itemID}) {
		t.Errorf("key 1: the client deleted %v, want the unheard line %s", deleted, key.itemID)
	}
	if !calls.await(mark, logItemRemoved, "itemId", key.itemID, 2*time.Second) {
		t.Errorf("key 1: the deletion of %s was never confirmed (no conversation.item.deleted)",
			key.itemID)
	}
	key.log(t, output)

	mark = calls.mark()
	key = pressKeyOverLine(t, session, "2", true)
	if key.lineAudio != 0 {
		t.Fatalf("key 2: %d ms of the line's audio arrived before its cancel took effect; the key "+
			"was to land before any", int(audioDuration(key.lineAudio, output)/time.Millisecond))
	}
	checkKeypress(t, key)
	deleted := calls.values(mark, logUnheardDeleted, "itemId")
	switch {
	case len(deleted) == 0:
		t.Errorf("key 2: the cut line's spoken item was never deleted (no conversation.item.delete)")
	case key.itemID != "" && !slices.Equal(deleted, []string{key.itemID}):
		t.Errorf("key 2: the client deleted %v, want the cut line %s", deleted, key.itemID)
	}
	for _, id := range deleted {
		if !calls.await(mark, logItemRemoved, "itemId", id, 2*time.Second) {
			t.Errorf("key 2: the deletion of %s was never confirmed (no conversation.item.deleted)", id)
		}
	}
	key.log(t, output)
	checkNoWarnings(t, calls)
}

// keypress is one key pressed over a line, and what followed it.
type keypress struct {
	digit string
	// playedMs is what AICC reported heard of the line: 0 (its audio was queued, none of it
	// played) or -1 (none of it had arrived).
	playedMs int
	// itemID is the line's spoken item as the client knew it when the key was pressed.
	itemID string
	// lineAudio and replyAudio are the bytes of the line's audio that arrived and of the reply's.
	lineAudio, replyAudio int
	by                    InterruptReason
	replied               string
	// cutAfter and replyAfter run from the key to the line's turn ending and to the reply's audio.
	cutAfter, replyAfter time.Duration
}

func (k keypress) log(t *testing.T, output media.AudioFormat) {
	t.Helper()
	t.Logf("key %s: reported %d ms played (%d ms of the line arrived), line cancelled %s after the "+
		"key, reply audio %s after it (%d ms); replied %q",
		k.digit, k.playedMs, int(audioDuration(k.lineAudio, output)/time.Millisecond),
		k.cutAfter.Round(time.Millisecond), k.replyAfter.Round(time.Millisecond),
		int(audioDuration(k.replyAudio, output)/time.Millisecond), k.replied)
}

// pressKeyOverLine has the flow say billingLine and the caller press digit over it: on the line's
// first audio, or with beforeAudio the moment its turn starts. It returns once the reply to the
// keypress is done.
func pressKeyOverLine(t *testing.T, session *Realtime, digit string, beforeAudio bool) keypress {
	t.Helper()
	key := keypress{digit: digit}
	if err := session.SpeakText(billingLine, false); err != nil {
		t.Fatalf("key %s: speak: %v", digit, err)
	}
	lineStarted := false
	var pressed, cut, reply, replyAudioAt time.Time
	press := func(playedMs int) {
		pressed = time.Now()
		key.playedMs = playedMs
		session.mu.Lock()
		key.itemID = session.responseItemID
		session.mu.Unlock()
		// As aicall.Session.pumpDigits: bargeIn(DTMF), then the note for the model.
		if err := session.Interrupt(InterruptReasonDTMF, playedMs); err != nil {
			t.Fatalf("key %s: interrupt: %v", digit, err)
		}
		if err := session.SendUserText(keypressText(digit)); err != nil {
			t.Fatalf("key %s: keypress note: %v", digit, err)
		}
	}
	deadline := time.After(15 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatalf("key %s: no reply within 15s (line started %v, key pressed %v, line cut %v, "+
				"reply started %v)", digit, lineStarted, !pressed.IsZero(), !cut.IsZero(), !reply.IsZero())
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("key %s: the session ended", digit)
			}
			now := time.Now()
			switch event.Type {
			case EventTypeResponseStarted:
				switch {
				case !lineStarted:
					if !event.IsSpokenLine {
						t.Fatalf("key %s: the turn that started is not the line", digit)
					}
					lineStarted = true
					if beforeAudio {
						press(-1) // nothing of the turn queued yet (aicall.Session.bargeIn)
					}
				case cut.IsZero() || !reply.IsZero():
					t.Fatalf("key %s: a turn started out of order (line cut %v, reply started %v)",
						digit, !cut.IsZero(), !reply.IsZero())
				default:
					reply = now
				}
			case EventTypeAudioDelta:
				switch {
				case !reply.IsZero():
					if replyAudioAt.IsZero() {
						replyAudioAt = now
					}
					key.replyAudio += len(event.Audio)
				case !cut.IsZero():
					t.Fatalf("key %s: audio of the line arrived after its turn ended", digit)
				default:
					key.lineAudio += len(event.Audio)
					if pressed.IsZero() {
						press(0) // the turn's audio is queued, none of it on the wire yet
					}
				}
			case EventTypeInterrupted:
				if pressed.IsZero() {
					t.Fatalf("key %s: the line was cut short (by %s) before the key",
						digit, event.InterruptedBy)
				}
				key.by = event.InterruptedBy
				cut = now
			case EventTypeOutputTranscript:
				if event.IsFinal && !reply.IsZero() {
					key.replied += event.Text
				}
			case EventTypeSpeechStarted:
				t.Fatalf("key %s: speech detected with only room tone on the line", digit)
			case EventTypeError:
				t.Fatalf("key %s: provider error (fatal=%v): %v", digit, event.IsFatal, event.Err)
			case EventTypeResponseDone:
				if reply.IsZero() {
					t.Fatalf("key %s: the line ran to its end: the keypress never cancelled it", digit)
				}
				key.cutAfter = cut.Sub(pressed)
				if !replyAudioAt.IsZero() {
					key.replyAfter = replyAudioAt.Sub(pressed)
				}
				return key
			}
		}
	}
}

// checkKeypress: the line ended cancelled for the key, and the keypress note got a spoken reply.
func checkKeypress(t *testing.T, key keypress) {
	t.Helper()
	if key.by != InterruptReasonDTMF {
		t.Errorf("key %s: the line was cut short by %s, want %s", key.digit, key.by, InterruptReasonDTMF)
	}
	if key.replyAudio == 0 {
		t.Errorf("key %s: the reply to the keypress had no audio", key.digit)
	}
}

// keypressText is how AICC's session tells the model about a key (aicall.keypressText).
func keypressText(digit string) string {
	return fmt.Sprintf("(The caller pressed %s on their keypad.)", digit)
}

// startCall opens a session as AICC opens a call on the gateway profile: its formats, the default
// turn detection, and an opening line. Every record the client logs is kept in calls.
func startCall(t *testing.T, p Profile, calls *callLog, input, output media.AudioFormat) *Realtime {
	t.Helper()
	session, err := New(p, slog.New(calls))
	if err != nil {
		t.Fatalf("new session: %v", err)
	}
	t.Cleanup(func() { _ = session.Close(context.Background()) })
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if err := session.Start(ctx, SessionConfig{
		Instructions: "You answer the phone for NovaNet billing. Reply in one short sentence.",
		Language:     "en",
		Turn:         DefaultTurnDetection(),
		OpeningText:  "Hello, NovaNet billing.",
		InputFormat:  input,
		OutputFormat: output,
	}); err != nil {
		t.Fatalf("start: %v\nwarnings:\n%s", err, calls.warnings())
	}
	return session
}

// hearOpening waits for the opening turn, then for the caller to hear it out. The gateway sends a
// short turn's audio at once and ends the turn while the caller hears it in real time; waiting it
// out has the next turn play from its first frame, on AICC's side and in the gateway's model of
// it, as when a flow speaks after the greeting.
func hearOpening(t *testing.T, session *Realtime, output media.AudioFormat) {
	t.Helper()
	var first time.Time
	audio := 0
	deadline := time.After(15 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatalf("the opening never finished (%d audio bytes)", audio)
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("the session ended during the opening")
			}
			switch event.Type {
			case EventTypeAudioDelta:
				if first.IsZero() {
					first = time.Now()
				}
				audio += len(event.Audio)
			case EventTypeInterrupted:
				t.Fatalf("the opening was cut short (by %s)", event.InterruptedBy)
			case EventTypeSpeechStarted:
				t.Fatalf("speech detected during the opening, with only room tone on the line")
			case EventTypeError:
				t.Fatalf("provider error during the opening (fatal=%v): %v", event.IsFatal, event.Err)
			case EventTypeResponseDone:
				if audio == 0 {
					t.Fatalf("the opening had no audio")
				}
				heard := first.Add(audioDuration(audio, output) + 300*time.Millisecond)
				listen(t, session, time.Until(heard))
				return
			}
		}
	}
}

// listen reads events for d, while nothing is to happen on the call.
func listen(t *testing.T, session *Realtime, d time.Duration) {
	t.Helper()
	timer := time.NewTimer(max(d, 0))
	defer timer.Stop()
	for {
		select {
		case <-timer.C:
			return
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("the session ended")
			}
			switch event.Type {
			case EventTypeError:
				t.Fatalf("provider error (fatal=%v): %v", event.IsFatal, event.Err)
			case EventTypeSpeechStarted, EventTypeResponseStarted:
				t.Fatalf("%s while the caller was listening", event.Type)
			}
		}
	}
}

// audioDuration is how long this much audio plays: PCM16 has 2 bytes a sample, G.711 one.
func audioDuration(bytes int, format media.AudioFormat) time.Duration {
	return time.Duration(bytes/bytesPerSample(format)) * time.Second / time.Duration(format.RateHz)
}

// bytesPerSample is 2 for PCM16 and 1 for G.711.
func bytesPerSample(format media.AudioFormat) int {
	if format.Encoding == media.EncodingPCM16 {
		return 2
	}
	return 1
}

// playedLikeAICC is what AICC's session reports heard of a turn it cuts short
// (aicall.Session.stopPlayback): the frames already handed to the wire, one per 20 ms tick since
// the turn's first audio, and never more than had arrived.
func playedLikeAICC(audioBytes int, sinceFirstAudio time.Duration, output media.AudioFormat) int {
	arrived := int(audioDuration(audioBytes, output) / time.Millisecond)
	elapsed := int(sinceFirstAudio / time.Millisecond)
	return min(arrived, elapsed) / frameMs * frameMs
}

// checkNoWarnings fails on any warning the client logged: AICC's watchdog giving up on a turn
// ("response abandoned"), a request the provider never created, an error it could not place. A
// healthy call logs none.
func checkNoWarnings(t *testing.T, calls *callLog) {
	t.Helper()
	if warnings := calls.warnings(); warnings != "" {
		t.Errorf("the client logged warnings:\n%s", warnings)
	}
}

// callerClip is the caller's speech (en_dinosaurs.wav, speech from 0.24 to 3.92 s) in the session's
// input format: 24 kHz mono PCM16 as probe.py writes it, or converted to G.711 the way AICC's media
// path converts (media.Converter) when the session passes G.711 through.
func callerClip(t *testing.T, input media.AudioFormat) []byte {
	t.Helper()
	path := os.Getenv("MLX_REALTIME_CALLER_CLIP")
	if path == "" {
		t.Skip("MLX_REALTIME_CALLER_CLIP is not set (tests/aicc/probe.py writes the caller's speech)")
	}
	clip, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("the caller's speech: %v", err)
	}
	wide := media.PCM16Format(media.RateProviderOut)
	if len(clip)%2 != 0 || audioDuration(len(clip), wide) < time.Second {
		t.Fatalf("%s holds %d bytes: not a second or more of PCM16", path, len(clip))
	}
	if input == wide {
		return clip
	}
	converter, err := media.NewConverter(wide, input)
	if err != nil {
		t.Fatalf("the caller's speech for %s: %v", input, err)
	}
	return converter.Convert(nil, clip)
}

// callerLine is a call's input as AICC's media loop sends it: one 20 ms frame per tick, room tone
// (as feedRoomTone makes it) while the caller listens, and the caller's speech once say queues it.
type callerLine struct {
	mu     sync.Mutex
	frames [][]byte
}

// say queues speech in the session's input format (PCM16 or G.711), cut into 20 ms frames, the
// last one padded with silence. It goes out in place of room tone from the next tick.
func (c *callerLine) say(speech []byte, input media.AudioFormat) {
	size := input.RateHz * frameMs / 1000 * bytesPerSample(input)
	var pad byte
	if law, ok := g711Law(input); ok {
		pad = law.Silence()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for start := 0; start < len(speech); start += size {
		frame := make([]byte, size)
		for i := range frame {
			frame[i] = pad
		}
		copy(frame, speech[start:])
		c.frames = append(c.frames, frame)
	}
}

func (c *callerLine) next() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.frames) == 0 {
		return nil
	}
	frame := c.frames[0]
	c.frames = c.frames[1:]
	return frame
}

// feed sends the call's input until stop is closed.
func (c *callerLine) feed(session *Realtime, law media.Law, to media.AudioFormat, stop <-chan struct{}) {
	converter, err := media.NewConverter(media.G711Format(law), to)
	if err != nil {
		return
	}
	ticker := time.NewTicker(FrameInterval)
	defer ticker.Stop()

	samples := make([]int16, media.FrameSamples)
	frame := make([]byte, 0, media.FrameSamples)
	converted := make([]byte, 0, media.FrameSamples*8)
	phase := 0
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
		}
		out := c.next()
		if out == nil {
			for i := range samples {
				samples[i] = int16(30 * math.Sin(float64(phase+i)/7.0))
			}
			frame = law.Encode(frame, samples)
			converted = converter.Convert(converted, frame)
			out = converted
		}
		phase += media.FrameSamples
		if err := session.SendAudio(out); err != nil {
			return
		}
	}
}

// callLog keeps every record the client logs, at every level. The acks it has no mapping for show
// up there (a truncate's as an "unmapped provider event" debug line, as debugTap reads them; a
// delete's as "the provider removed an item from the conversation"), and so do AICC's own verdicts
// on a turn, as warnings.
type callLog struct {
	mu      sync.Mutex
	records []callRecord
}

type callRecord struct {
	level   slog.Level
	message string
	attrs   map[string]string
}

func (c *callLog) Enabled(context.Context, slog.Level) bool { return true }
func (c *callLog) WithAttrs([]slog.Attr) slog.Handler       { return c }
func (c *callLog) WithGroup(string) slog.Handler            { return c }
func (c *callLog) Handle(_ context.Context, rec slog.Record) error {
	attrs := make(map[string]string, rec.NumAttrs())
	rec.Attrs(func(a slog.Attr) bool {
		attrs[a.Key] = a.Value.String()
		return true
	})
	c.mu.Lock()
	c.records = append(c.records, callRecord{level: rec.Level, message: rec.Message, attrs: attrs})
	c.mu.Unlock()
	return nil
}

// mark is the log's position now. The queries below read what was logged after a mark.
func (c *callLog) mark() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.records)
}

// values is the attribute key of every record with this message, logged after mark.
func (c *callLog) values(mark int, message, key string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, rec := range c.records[mark:] {
		if rec.message == message {
			out = append(out, rec.attrs[key])
		}
	}
	return out
}

// await waits up to timeout for a record with this message and attribute value after mark: an
// ack arrives one round trip after the request it answers.
func (c *callLog) await(mark int, message, key, value string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for !slices.Contains(c.values(mark, message, key), value) {
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(10 * time.Millisecond)
	}
	return true
}

// warnings is every record at WARN or above, one per line.
func (c *callLog) warnings() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var text strings.Builder
	for _, rec := range c.records {
		if rec.level >= slog.LevelWarn {
			fmt.Fprintf(&text, "%s %s %v\n", rec.level, rec.message, rec.attrs)
		}
	}
	return text.String()
}

// g711Law is the law of a G.711 format.
func g711Law(format media.AudioFormat) (media.Law, bool) {
	switch format.Encoding {
	case media.EncodingMuLaw:
		return media.LawMu, true
	case media.EncodingALaw:
		return media.LawAlaw, true
	}
	return 0, false
}
