"""The AICC harness runner (tests/aicc/probe.py): how it reads AICC's output and judges the
deadlines, the caller's speech it hands the barge-in tests, and a run of AICC's own client against
a gateway with mock providers (marker e2e; needs ``AICC_REPO`` pointing at an AICC checkout, and
Go)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.aicc.probe import Report, go_duration, write_caller_clip
from tests.speech import EN, clip, to_wire

REPO = Path(__file__).parents[2]
PROBE = Path(__file__).parent / "probe.py"
AICC_REPO = os.environ.get("AICC_REPO", "")
# Every test in tests/aicc/zz_mlx_live_test.go: the M3 connectivity checks, then barge-in (M4).
TESTS = (
    "TestMLXGreeting",
    "TestMLXResponseCreatedLatency",
    "TestMLXLines",
    "TestMLXBargeIn",
    "TestMLXKeypressCancel",
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("0s", 0.0),
        ("2ms", 0.002),
        ("350.5ms", 0.3505),
        ("1.25s", 1.25),
        ("1m2s", 62.0),
        ("12µs", 12e-6),
        ("3us", 3e-6),
        ("40ns", 4e-8),
        ("1h", 3600.0),
    ],
)
def test_go_durations(text: str, seconds: float) -> None:
    assert go_duration(text) == pytest.approx(seconds)


def test_a_malformed_duration_is_refused() -> None:
    with pytest.raises(ValueError, match="not a Go duration"):
        go_duration("3 seconds")


PASSING = """\
=== RUN   TestMLXGreeting
    zz_mlx_live_test.go:47: session ready in 4ms, first audio after 320ms, 9 deltas, 31680 bytes
--- PASS: TestMLXGreeting (12.01s)
=== RUN   TestMLXResponseCreatedLatency
    zz_mlx_live_test.go:53: gateway: sample 1: request -> response.created 2ms
    zz_mlx_live_test.go:53: gateway: sample 2: request -> response.created 1.2s
    zz_mlx_live_test.go:53: gateway (cascade): n=2/2 median=2ms p95=1.2s max=1.2s (nearest rank)
--- PASS: TestMLXResponseCreatedLatency (6.90s)
PASS
"""


def read(text: str) -> Report:
    report = Report()
    for line in text.splitlines(keepends=True):
        report.read(line)
    return report


def test_a_clean_run_passes() -> None:
    report = read(PASSING)
    assert report.results == {
        "TestMLXGreeting": "PASS",
        "TestMLXResponseCreatedLatency": "PASS",
    }
    assert report.created_s == [0.002, 1.2]
    assert report.created_max_s == 1.2
    assert report.first_audio_s == [0.32]
    assert report.failures(0) == []


@pytest.mark.parametrize(
    ("extra", "exit_code", "problem"),
    [
        ("    x: gateway: sample 3: request -> response.created 3.1s\n", 0, "later than 3 s"),
        ("    x: gateway (cascade): n=7/8 median=2ms p95=2ms max=2ms\n", 0, "unanswered"),
        ('time=… level=WARN msg="response abandoned" reason="the provider…"\n', 0, "gave up"),
        ("--- FAIL: TestMLXLines (3.00s)\n", 1, "TestMLXLines"),
        ("", 2, "exited with 2"),
    ],
)
def test_deadline_misses_and_failures_fail_the_run(
    extra: str, exit_code: int, problem: str
) -> None:
    problems = read(PASSING + extra).failures(exit_code)
    assert any(problem in item for item in problems), problems


def test_a_run_where_everything_was_skipped_fails() -> None:
    skipped = "--- SKIP: TestMLXGreeting (0.00s)\n"
    assert "no test ran" in read(skipped).failures(0)


def test_the_caller_clip_is_the_english_fixture_as_wire_audio(tmp_path: Path) -> None:
    """MLX_REALTIME_CALLER_CLIP: raw 24 kHz mono s16le, the fixture through the gateway's own
    resampler, as AICC sends a caller on the gateway profile."""
    written = write_caller_clip(tmp_path / "caller.pcm").read_bytes()
    assert written == to_wire(clip(EN))
    assert len(written) == len(clip(EN)) * 3 // 2  # 16 → 24 kHz, 4.228 s


@pytest.mark.e2e
@pytest.mark.skipif(
    not AICC_REPO or shutil.which("go") is None,
    reason="set AICC_REPO to an AICC checkout (and install Go) to run AICC's own client",
)
@pytest.mark.parametrize("g711", [False, True], ids=["pcm", "g711"])
def test_aiccs_client_against_the_mock_gateway(g711: bool) -> None:
    """``make aicc-probe-mock``: AICC's client, unchanged, against the gateway with mocks,
    including the barge-in scenarios; and (M7) on the profile's G.711 variant
    (``AcceptsG711: true``), μ-law passed through both ways."""
    extra = ["--g711"] if g711 else []
    finished = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--aicc-repo",
            AICC_REPO,
            "--serve",
            "config.mock.yaml",
            *extra,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert finished.returncode == 0, finished.stdout[-4000:] + finished.stderr[-2000:]
    for name in TESTS:
        assert f"--- PASS: {name} " in finished.stdout, name
