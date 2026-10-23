/* The CPython binding of SpeexDSP's echo canceller (MDF) and, optionally, its preprocessor.
 *
 * Built by setup.py from the vendored SpeexDSP 1.2.1 sources (speexdsp/, BSD-3-Clause; the
 * version, URL and sha256 are in speexdsp/SOURCE.json), as floating point with the smallft FFT,
 * as SpeexDSP's own configure builds it on macOS.
 *
 * The canceller must never crash the process that hosts it:
 *
 * - every buffer is checked before the native call: C-contiguous, raw bytes or 2-byte items, and
 *   exactly one frame long; it is then copied into the instance's own aligned scratch buffers,
 *   so the native code never reads caller memory;
 * - one call at a time per instance: a busy flag, set and cleared while holding the GIL, makes a
 *   second thread get an exception instead of racing on the state;
 * - the native work runs with the GIL released;
 * - sizes are bounded at construction, so SpeexDSP's int arithmetic cannot overflow;
 * - SpeexDSP's warnings (it prints one when it resets a diverged filter) are counted per
 *   instance instead of printed (os_support_custom.h).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "speex/speex_echo.h"
#include "speex/speex_preprocess.h"

#define SPEEXDSP_AEC_VERSION "1.2.1"
#define MIN_FRAME 16
#define MAX_FRAME 4096
#define MAX_FILTER 65536
#define MIN_RATE 8000
#define MAX_RATE 48000

/* Incremented by SpeexDSP's speex_warning() on the calling thread (os_support_custom.h). */
_Thread_local long speexdsp_aec_warnings = 0;

typedef struct {
    PyObject_HEAD
    SpeexEchoState *echo;
    SpeexPreprocessState *preprocess; /* NULL: the linear canceller alone */
    spx_int16_t *capture;             /* scratch: one frame each */
    spx_int16_t *reference;
    spx_int16_t *output;
    int frame_size;
    int filter_length;
    int sample_rate;
    int with_preprocess;
    int busy;
    long long frames;
    long long warnings;
} EchoCanceller;

static void
free_states(EchoCanceller *self)
{
    if (self->preprocess != NULL) {
        speex_preprocess_state_destroy(self->preprocess);
        self->preprocess = NULL;
    }
    if (self->echo != NULL) {
        speex_echo_state_destroy(self->echo);
        self->echo = NULL;
    }
    free(self->capture);
    free(self->reference);
    free(self->output);
    self->capture = self->reference = self->output = NULL;
}

/* The preprocessor linked to the echo state, with its noise suppressor on: SpeexDSP applies its
 * residual-echo suppression only through the denoiser's spectral gain (M0 S11). */
static int
make_preprocess(EchoCanceller *self)
{
    int on = 1;
    self->preprocess = speex_preprocess_state_init(self->frame_size, self->sample_rate);
    if (self->preprocess == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    speex_preprocess_ctl(self->preprocess, SPEEX_PREPROCESS_SET_ECHO_STATE, self->echo);
    speex_preprocess_ctl(self->preprocess, SPEEX_PREPROCESS_SET_DENOISE, &on);
    return 0;
}

static int
EchoCanceller_init(EchoCanceller *self, PyObject *args, PyObject *kwargs)
{
    static char *keywords[] = {"frame_size", "filter_length", "sample_rate", "preprocess", NULL};
    int frame_size, filter_length, sample_rate, with_preprocess = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii|$p", keywords, &frame_size,
                                     &filter_length, &sample_rate, &with_preprocess)) {
        return -1;
    }
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError, "the echo canceller is processing a frame");
        return -1;
    }
    if (frame_size < MIN_FRAME || frame_size > MAX_FRAME) {
        PyErr_Format(PyExc_ValueError, "frame_size must be %d to %d samples, not %d", MIN_FRAME,
                     MAX_FRAME, frame_size);
        return -1;
    }
    if (filter_length < frame_size || filter_length > MAX_FILTER) {
        PyErr_Format(PyExc_ValueError,
                     "filter_length must be frame_size (%d) to %d samples, not %d", frame_size,
                     MAX_FILTER, filter_length);
        return -1;
    }
    if (sample_rate < MIN_RATE || sample_rate > MAX_RATE) {
        PyErr_Format(PyExc_ValueError, "sample_rate must be %d to %d Hz, not %d", MIN_RATE,
                     MAX_RATE, sample_rate);
        return -1;
    }
    free_states(self); /* __init__ called again */
    self->frame_size = frame_size;
    self->filter_length = filter_length;
    self->sample_rate = sample_rate;
    self->with_preprocess = with_preprocess;
    self->frames = 0;
    self->warnings = 0;
    self->capture = calloc((size_t)frame_size, sizeof(spx_int16_t));
    self->reference = calloc((size_t)frame_size, sizeof(spx_int16_t));
    self->output = calloc((size_t)frame_size, sizeof(spx_int16_t));
    if (self->capture == NULL || self->reference == NULL || self->output == NULL) {
        free_states(self);
        PyErr_NoMemory();
        return -1;
    }
    self->echo = speex_echo_state_init(frame_size, filter_length);
    if (self->echo == NULL) {
        free_states(self);
        PyErr_NoMemory();
        return -1;
    }
    speex_echo_ctl(self->echo, SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate);
    if (with_preprocess && make_preprocess(self) < 0) {
        free_states(self);
        return -1;
    }
    return 0;
}

static void
EchoCanceller_dealloc(EchoCanceller *self)
{
    free_states(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
ready(EchoCanceller *self)
{
    if (self->echo == NULL) {
        PyErr_SetString(PyExc_ValueError, "the echo canceller is closed");
        return 0;
    }
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError,
                        "the echo canceller is processing a frame in another thread");
        return 0;
    }
    return 1;
}

/* Copy one frame from a buffer the caller passed into `scratch`, after checking it. */
static int
take_frame(PyObject *object, const char *name, int frame_size, spx_int16_t *scratch)
{
    Py_buffer view;
    if (PyObject_GetBuffer(object, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        return -1;
    }
    const char *format = view.format == NULL ? "B" : view.format;
    if (*format == '<' || *format == '=' || *format == '@') {
        format++;
    }
    int raw = view.itemsize == 1 && (strcmp(format, "B") == 0 || strcmp(format, "b") == 0 ||
                                     strcmp(format, "c") == 0);
    int samples = view.itemsize == 2 && strcmp(format, "h") == 0;
    Py_ssize_t expected = (Py_ssize_t)frame_size * (Py_ssize_t)sizeof(spx_int16_t);
    if (!raw && !samples) {
        PyErr_Format(PyExc_ValueError, "the %s frame must be int16 samples or bytes, not '%s'",
                     name, view.format == NULL ? "B" : view.format);
        PyBuffer_Release(&view);
        return -1;
    }
    if (view.len != expected) {
        PyErr_Format(PyExc_ValueError, "the %s frame must be %zd bytes (%d samples), not %zd",
                     name, expected, frame_size, view.len);
        PyBuffer_Release(&view);
        return -1;
    }
    memcpy(scratch, view.buf, (size_t)expected);
    PyBuffer_Release(&view);
    return 0;
}

PyDoc_STRVAR(process_doc,
             "process(capture, reference, /) -> bytes\n\n"
             "Cancel the echo of one reference frame from one capture frame (both exactly\n"
             "frame_size int16 samples, as bytes or an int16 buffer); returns the output frame\n"
             "as little-endian int16 bytes.");

static PyObject *
EchoCanceller_process(EchoCanceller *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "process() takes 2 arguments (capture, reference), not %zd",
                     nargs);
        return NULL;
    }
    if (!ready(self)) {
        return NULL;
    }
    /* Busy from here: taking a buffer can run Python code (a __buffer__ method), and so let
     * another thread in. */
    self->busy = 1;
    int n = self->frame_size;
    if (take_frame(args[0], "capture", n, self->capture) < 0 ||
        take_frame(args[1], "reference", n, self->reference) < 0) {
        self->busy = 0;
        return NULL;
    }
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)n * (Py_ssize_t)sizeof(spx_int16_t));
    if (out == NULL) {
        self->busy = 0;
        return NULL;
    }
    long warnings = 0;
    Py_BEGIN_ALLOW_THREADS
    long before = speexdsp_aec_warnings;
    speex_echo_cancellation(self->echo, self->capture, self->reference, self->output);
    if (self->preprocess != NULL) {
        speex_preprocess_run(self->preprocess, self->output);
    }
    warnings = speexdsp_aec_warnings - before;
    Py_END_ALLOW_THREADS
    self->busy = 0;
    self->frames += 1;
    self->warnings += warnings;
    memcpy(PyBytes_AS_STRING(out), self->output, (size_t)n * sizeof(spx_int16_t));
    return out;
}

PyDoc_STRVAR(reset_doc, "reset() -> None\n\nForget the adapted filter (and the preprocessor's state).");

static PyObject *
EchoCanceller_reset(EchoCanceller *self, PyObject *Py_UNUSED(ignored))
{
    if (!ready(self)) {
        return NULL;
    }
    speex_echo_state_reset(self->echo);
    if (self->preprocess != NULL) {
        speex_preprocess_state_destroy(self->preprocess);
        self->preprocess = NULL;
        if (make_preprocess(self) < 0) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(close_doc, "close() -> None\n\nFree the native state; later calls raise ValueError.");

static PyObject *
EchoCanceller_close(EchoCanceller *self, PyObject *Py_UNUSED(ignored))
{
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError, "the echo canceller is processing a frame");
        return NULL;
    }
    free_states(self);
    Py_RETURN_NONE;
}

static PyObject *
get_frame_size(EchoCanceller *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->frame_size);
}

static PyObject *
get_filter_length(EchoCanceller *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->filter_length);
}

static PyObject *
get_sample_rate(EchoCanceller *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->sample_rate);
}

static PyObject *
get_preprocess(EchoCanceller *self, void *Py_UNUSED(closure))
{
    return PyBool_FromLong(self->with_preprocess);
}

static PyObject *
get_frames(EchoCanceller *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLongLong(self->frames);
}

static PyObject *
get_warnings(EchoCanceller *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLongLong(self->warnings);
}

static PyObject *
get_closed(EchoCanceller *self, void *Py_UNUSED(closure))
{
    return PyBool_FromLong(self->echo == NULL);
}

static PyMethodDef EchoCanceller_methods[] = {
    {"process", (PyCFunction)(void (*)(void))EchoCanceller_process, METH_FASTCALL, process_doc},
    {"reset", (PyCFunction)EchoCanceller_reset, METH_NOARGS, reset_doc},
    {"close", (PyCFunction)EchoCanceller_close, METH_NOARGS, close_doc},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef EchoCanceller_getset[] = {
    {"frame_size", (getter)get_frame_size, NULL, "Samples per frame.", NULL},
    {"filter_length", (getter)get_filter_length, NULL, "Echo tail the filter covers, in samples.", NULL},
    {"sample_rate", (getter)get_sample_rate, NULL, "Sample rate in Hz.", NULL},
    {"preprocess", (getter)get_preprocess, NULL, "Whether the preprocessor (denoise and residual-echo suppression) runs.", NULL},
    {"frames", (getter)get_frames, NULL, "Frames processed.", NULL},
    {"warnings", (getter)get_warnings, NULL, "SpeexDSP warnings raised by this instance (a reset of a diverged filter).", NULL},
    {"closed", (getter)get_closed, NULL, "Whether close() freed the native state.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

PyDoc_STRVAR(EchoCanceller_doc,
             "EchoCanceller(frame_size, filter_length, sample_rate, *, preprocess=False)\n\n"
             "One stream's SpeexDSP MDF echo canceller: a linear adaptive filter of\n"
             "filter_length taps, fed one frame of capture and one of reference per call.\n"
             "preprocess=True also runs SpeexDSP's preprocessor on the output, linked to the\n"
             "echo state, with its noise suppressor (and so its residual-echo suppression) on.\n"
             "Not reentrant: one call at a time, frames in order.");

static PyTypeObject EchoCancellerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "speexdsp_aec._speexdsp.EchoCanceller",
    .tp_doc = EchoCanceller_doc,
    .tp_basicsize = sizeof(EchoCanceller),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)EchoCanceller_init,
    .tp_dealloc = (destructor)EchoCanceller_dealloc,
    .tp_methods = EchoCanceller_methods,
    .tp_getset = EchoCanceller_getset,
};

static int
module_exec(PyObject *module)
{
    if (PyType_Ready(&EchoCancellerType) < 0) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "EchoCanceller", (PyObject *)&EchoCancellerType) < 0) {
        return -1;
    }
    return PyModule_AddStringConstant(module, "SPEEXDSP_VERSION", SPEEXDSP_AEC_VERSION);
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, module_exec},
    /* The type is static: one interpreter only. */
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
    {0, NULL},
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "speexdsp_aec._speexdsp",
    .m_doc = "SpeexDSP 1.2.1's echo canceller (vendored), for mlx-realtime.",
    .m_size = 0,
    .m_slots = module_slots,
};

PyMODINIT_FUNC
PyInit__speexdsp(void)
{
    return PyModuleDef_Init(&module_def);
}
