"""SpeexDSP's echo canceller for mlx-realtime (vendored SpeexDSP 1.2.1, BSD-3-Clause).

``EchoCanceller`` is one stream's MDF canceller (a linear adaptive filter in the frequency
domain), optionally followed by SpeexDSP's preprocessor. It is built from the C sources in
``speexdsp/`` (see ``speexdsp/SOURCE.json`` for their provenance) by this package's
``setup.py``, so nothing outside the uv lockfile is loaded at run time.
"""

from speexdsp_aec._speexdsp import SPEEXDSP_VERSION, EchoCanceller

__all__ = ["SPEEXDSP_VERSION", "EchoCanceller"]
