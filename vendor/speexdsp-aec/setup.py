"""Builds SpeexDSP's echo canceller and preprocessor, with their CPython binding, as one extension.

The C sources under ``speexdsp/`` are copied unchanged from the SpeexDSP 1.2.1 release tarball
(BSD-3-Clause; version, URL, sha256 and per-file hashes in ``speexdsp/SOURCE.json``). The macros
are what SpeexDSP's own ``configure`` defines on macOS (floating point, the smallft FFT, C99
variable-size arrays), plus ``OS_SUPPORT_CUSTOM`` for ``include/os_support_custom.h`` (warnings
counted, not printed). Every symbol but the module's init function stays hidden, so this copy
never collides with another SpeexDSP in the process. Nothing is linked but libc and libm.
"""

from setuptools import Extension, setup

SPEEX = "speexdsp/libspeexdsp"

setup(
    ext_modules=[
        Extension(
            "speexdsp_aec._speexdsp",
            sources=[
                "src/speexdsp_aec/_speexdsp.c",
                f"{SPEEX}/mdf.c",
                f"{SPEEX}/preprocess.c",
                f"{SPEEX}/filterbank.c",
                f"{SPEEX}/fftwrap.c",
                f"{SPEEX}/smallft.c",
            ],
            include_dirs=["include", "speexdsp/include", SPEEX],
            define_macros=[
                ("FLOATING_POINT", None),
                ("USE_SMALLFT", None),
                ("VAR_ARRAYS", None),
                ("EXPORT", ""),
                ("OS_SUPPORT_CUSTOM", None),
            ],
            extra_compile_args=["-std=c11", "-fvisibility=hidden"],
        )
    ]
)
