from __future__ import annotations

import numpy as np
import pytest

from mlx_realtime.audio.codec import InputDecoder, OutputEncoder
from mlx_realtime.audio.pcm import bytes_to_ms, ms_to_bytes
from mlx_realtime.audio.resample import Resampler


def noise(samples: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(samples) * 4000).astype("<i2").tobytes()


def stream(resampler: Resampler, pcm: bytes, chunk_bytes: int) -> bytes:
    out = b"".join(
        resampler.process(pcm[start : start + chunk_bytes])
        for start in range(0, len(pcm), chunk_bytes)
    )
    return out + resampler.flush()


@pytest.mark.parametrize(
    ("in_rate", "out_rate"), [(24_000, 16_000), (16_000, 24_000), (8_000, 16_000)]
)
@pytest.mark.parametrize("chunk_bytes", [2, 14, 960, 3842])
def test_chunked_output_equals_one_shot(in_rate: int, out_rate: int, chunk_bytes: int) -> None:
    pcm = noise(in_rate + 7)  # a second and a bit, not a multiple of any chunk
    one_shot = stream(Resampler(in_rate, out_rate), pcm, len(pcm))
    assert stream(Resampler(in_rate, out_rate), pcm, chunk_bytes) == one_shot
    # The stream keeps the timeline: n input samples become n * out / in samples, rounded half up.
    assert len(one_shot) // 2 == (len(pcm) // 2 * out_rate * 2 + in_rate) // (2 * in_rate)


def test_flush_starts_a_fresh_stream() -> None:
    resampler = Resampler(24_000, 16_000)
    pcm = noise(4800)
    first = stream(resampler, pcm, 960)
    assert stream(resampler, pcm, 960) == first


def test_equal_rates_pass_through() -> None:
    resampler = Resampler(24_000, 24_000)
    assert resampler.process(b"\x01\x00") == b"\x01\x00"
    assert resampler.flush() == b""


def test_input_decoder_produces_canonical_audio() -> None:
    decoder = InputDecoder()
    pcm = noise(24_000)
    out = b"".join(decoder.decode(pcm[start : start + 960]) for start in range(0, len(pcm), 960))
    out += decoder.flush()
    assert len(out) == 32_000  # 1 s at 16 kHz, 2 bytes per sample


def test_output_encoder_splits_into_bounded_deltas() -> None:
    encoder = OutputEncoder(max_delta_ms=100)
    deltas = encoder.encode(noise(24_000 * 3 // 10), 24_000)  # 300 ms
    assert [bytes_to_ms(len(delta), 24_000) for delta in deltas] == [100, 100, 100]
    assert encoder.flush() == []


def test_output_encoder_resamples_other_rates_to_the_wire_rate() -> None:
    encoder = OutputEncoder(max_delta_ms=100)
    pcm = noise(16_000)  # 1 s at 16 kHz
    out = b"".join(encoder.encode(pcm, 16_000)) + b"".join(encoder.flush())
    assert len(out) == ms_to_bytes(1000, 24_000)
    with pytest.raises(ValueError, match="sample rate changed"):
        encoder.encode(pcm, 22_050)
