"""G.711 (M7): the codec exhaustively against ``audioop``, the wire formats' arithmetic, the
codecs to and from canonical PCM, and the soxr quality chosen for the 8 kHz steps (M0 S7)."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from mlx_realtime.audio import g711
from mlx_realtime.audio.codec import InputDecoder, OutputEncoder, from_linear, to_linear
from mlx_realtime.audio.formats import PCM, PCMA, PCMU, AudioFormat
from mlx_realtime.audio.resample import Quality, Resampler
from mlx_realtime.config import Config, ConfigError, validate_config

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # 3.12 still has it (removed in 3.13): the reference, in tests only

LAWS: list[tuple[g711.Law, Callable[[bytes, int], bytes], Callable[[bytes, int], bytes]]] = [
    ("ulaw", audioop.ulaw2lin, audioop.lin2ulaw),
    ("alaw", audioop.alaw2lin, audioop.lin2alaw),
]
EVERY_SAMPLE = np.arange(-32768, 32768, dtype=np.int16)
EVERY_CODE = bytes(range(256))


@pytest.mark.parametrize(("law", "decode", "encode"), LAWS)
def test_decoding_matches_audioop_on_every_code(
    law: g711.Law, decode: Callable[[bytes, int], bytes], encode: Callable[[bytes, int], bytes]
) -> None:
    assert len(g711.DECODE[law]) == 256
    assert g711.decode(EVERY_CODE, law) == decode(EVERY_CODE, 2)
    assert g711.decode_samples(EVERY_CODE, law).tobytes() == decode(EVERY_CODE, 2)


@pytest.mark.parametrize(("law", "decode", "encode"), LAWS)
def test_encoding_matches_audioop_on_every_sample(
    law: g711.Law, decode: Callable[[bytes, int], bytes], encode: Callable[[bytes, int], bytes]
) -> None:
    pcm = EVERY_SAMPLE.astype("<i2").tobytes()
    assert len(g711.ENCODE[law]) == 65536
    expected = encode(pcm, 2)
    assert g711.encode(pcm, law) == expected
    assert g711.encode_samples(EVERY_SAMPLE, law) == expected
    # Sample by sample too: the table is indexed by each sample's own bit pattern.
    mismatches = [
        int(sample)
        for sample, code in zip(EVERY_SAMPLE, g711.encode_samples(EVERY_SAMPLE, law), strict=True)
        if code != expected[int(sample) + 32768]
    ]
    assert mismatches == []


@pytest.mark.parametrize("law", ["ulaw", "alaw"])
def test_every_code_survives_a_round_trip(law: g711.Law) -> None:
    """Decoding then encoding is the identity, except μ-law's negative zero (0x7F → 0xFF)."""
    again = g711.encode(g711.decode(EVERY_CODE, law), law)
    changed = [code for code in range(256) if again[code] != code]
    assert changed == ([0x7F] if law == "ulaw" else [])
    assert g711.encode(bytes(2), law) == bytes([g711.SILENCE[law]])


def test_the_tables_do_not_follow_suns_later_g711_c() -> None:
    """S7: Sun's g711.c disagrees with audioop on negative inputs; spot-check one."""
    sample = np.array([-1], dtype=np.int16)
    assert g711.encode_samples(sample, "ulaw") == audioop.lin2ulaw(sample.tobytes(), 2)
    assert g711.encode_samples(sample, "alaw") == audioop.lin2alaw(sample.tobytes(), 2)


@pytest.mark.parametrize(
    ("audio_format", "rate", "width", "law"),
    [(PCM, 24000, 2, None), (PCMU, 8000, 1, "ulaw"), (PCMA, 8000, 1, "alaw")],
)
def test_wire_format_arithmetic(
    audio_format: AudioFormat, rate: int, width: int, law: g711.Law | None
) -> None:
    assert (audio_format.rate, audio_format.sample_width, audio_format.law) == (rate, width, law)
    assert audio_format.ms_to_bytes(100) == rate // 10 * width  # one 100 ms delta
    assert audio_format.ms_to_bytes(20) == rate // 50 * width  # one 20 ms telephone frame
    assert audio_format.bytes_to_ms(rate // 10 * width) == 100
    assert audio_format.bytes_to_ms(rate // 10 * width - 1) == 99  # rounded down
    assert audio_format.duration_ms(rate * width) == 1000.0
    assert audio_format.bytes_per_second == rate * width


def test_g711_frames_and_deltas_are_whole_milliseconds() -> None:
    assert PCMU.ms_to_bytes(20) == 160  # AICC's appends
    assert PCMU.bytes_to_ms(800) == 100  # max_delta_ms
    assert PCMU.bytes_to_ms(7) == 0
    assert PCMA.bytes_to_ms(8) == 1


def sine(rate: int, seconds: float, frequency: float = 440.0, amplitude: float = 8000) -> bytes:
    t = np.arange(round(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype("<i2").tobytes()


@pytest.mark.parametrize("audio_format", [PCMU, PCMA])
@pytest.mark.parametrize("quality", ["LQ", "HQ"])
def test_g711_input_becomes_canonical_audio(audio_format: AudioFormat, quality: Quality) -> None:
    law = audio_format.law
    assert law is not None
    tone = sine(8000, 1.0)
    data = g711.encode(tone, law)
    decoder = InputDecoder(audio_format, quality)
    # 20 ms appends of 160 bytes, as AICC sends them
    out = b"".join(decoder.decode(data[start : start + 160]) for start in range(0, len(data), 160))
    out += decoder.flush()
    assert len(out) == 32_000  # 1 s at 16 kHz: the timeline is exact at 8 kHz
    reference = Resampler(8000, 16000, quality)
    expected = reference.process(g711.decode(data, law)) + reference.flush()
    assert out == expected


def test_any_g711_byte_count_is_whole_samples() -> None:
    decoder = InputDecoder(PCMU, "LQ")
    out = b"".join(decoder.decode(bytes([0xFF]) * size) for size in (1, 3, 7, 149))
    out += decoder.flush()
    assert len(out) == 2 * 2 * (1 + 3 + 7 + 149)


@pytest.mark.parametrize("audio_format", [PCMU, PCMA])
def test_output_becomes_g711_deltas_of_at_most_max_delta_ms(audio_format: AudioFormat) -> None:
    law = audio_format.law
    assert law is not None
    encoder = OutputEncoder(100, audio_format, "HQ")
    tts = sine(24000, 1.0)
    deltas: list[bytes] = []
    for start in range(0, len(tts), 7680):  # the TTS's 160 ms chunks
        deltas += encoder.encode(tts[start : start + 7680], 24000)
    deltas += encoder.flush()
    assert all(0 < len(delta) <= 800 for delta in deltas)  # 100 ms at 8 kHz, 1 byte a sample
    data = b"".join(deltas)
    assert len(data) == 8000  # 1 s: the segment's tail is flushed
    reference = Resampler(24000, 8000, "HQ")
    expected = g711.encode(reference.process(tts) + reference.flush(), law)
    assert data == expected
    # The tone survives the trip: correlate the decoded deltas with the original, at 8 kHz.
    decoded = np.frombuffer(g711.decode(data, law), dtype="<i2").astype(float)
    original = np.frombuffer(sine(8000, 1.0), dtype="<i2").astype(float)
    middle = slice(400, 7600)
    correlation = np.corrcoef(decoded[middle], original[middle])[0, 1]
    assert correlation > 0.99


def test_linear_conversions() -> None:
    pcm = sine(8000, 0.1)
    assert to_linear(pcm, PCM) == pcm
    assert from_linear(pcm, PCM) == pcm
    assert to_linear(from_linear(pcm, PCMU), PCMU) == g711.decode(g711.encode(pcm, "ulaw"), "ulaw")


def stream_lag_ms(quality: Quality, chunk_ms: int = 20, chunks: int = 250) -> float:
    """The mean lag of a streaming 8 → 16 kHz soxr resampler fed 20 ms appends: how much of
    the audio sent has not come out yet."""
    resampler = Resampler(8000, 16000, quality)
    chunk = bytes(2 * 8000 * chunk_ms // 1000)
    sent = produced = 0
    lags: list[float] = []
    for _ in range(chunks):
        produced += len(resampler.process(chunk)) // 2
        sent += len(chunk) // 2
        lags.append(sent * 2 - produced)
    return float(np.mean(lags[10:])) * 1000 / 16000


def band_gain_db(quality: Quality, frequency: float) -> float:
    resampler = Resampler(8000, 16000, quality)
    tone = sine(8000, 2.0, frequency, amplitude=16000)
    out = np.frombuffer(resampler.process(tone) + resampler.flush(), dtype="<i2").astype(float)
    middle = out[4000:-4000]
    return 20 * float(np.log10(np.sqrt(np.mean(middle**2)) / (16000 / np.sqrt(2))))


def test_lq_halves_the_8k_to_16k_lag_and_keeps_the_voice_band() -> None:
    """S7: HQ lags ~64 ms on 8 → 16 kHz, which delays every speech event of a G.711 call.
    LQ halves it; its passband is flat to 3 kHz and -8 dB at 3.4 kHz (the telephone band's
    edge), which the default accepts for the VAD's sake (ASR CER on LQ vs HQ is pending)."""
    lq, hq = stream_lag_ms("LQ"), stream_lag_ms("HQ")
    assert lq <= 35, lq
    assert lq <= 0.55 * hq, (lq, hq)
    for frequency in (300.0, 1000.0, 2000.0, 3000.0):
        assert band_gain_db("LQ", frequency) > -0.5, frequency
        assert band_gain_db("HQ", frequency) > -0.1, frequency
    assert band_gain_db("LQ", 3400.0) > -9.0
    assert Config().g711.input_quality == "LQ"
    assert Config().g711.output_quality == "HQ"


def silence_ratio(samples: npt.NDArray[np.int16]) -> float:
    return float(np.mean(samples == 0))


def test_g711_silence_codes_decode_to_near_zero() -> None:
    for law in ("ulaw", "alaw"):
        frame = bytes([g711.SILENCE[law]]) * 160
        decoded = g711.decode_samples(frame, law)
        assert np.max(np.abs(decoded.astype(int))) <= 8
    assert silence_ratio(g711.decode_samples(bytes([0xFF]) * 10, "ulaw")) == 1.0


def test_the_g711_qualities_are_configurable_and_checked() -> None:
    config = validate_config({"g711": {"input_quality": "HQ", "output_quality": "VHQ"}})
    assert (config.g711.input_quality, config.g711.output_quality) == ("HQ", "VHQ")
    with pytest.raises(ConfigError, match=r"g711\.input_quality"):
        validate_config({"g711": {"input_quality": "best"}})
    with pytest.raises(ConfigError, match=r"g711\.rate"):
        validate_config({"g711": {"rate": 16000}})
