/* Included by SpeexDSP's os_support.h when OS_SUPPORT_CUSTOM is defined (setup.py does).
 *
 * SpeexDSP prints its warnings to stderr; here they are counted on the calling thread instead,
 * and the binding (_speexdsp.c) adds them to the instance that raised them. In this build the
 * only one the echo canceller can raise is the reset of a filter that diverged. */

#ifndef SPEEXDSP_AEC_OS_SUPPORT_CUSTOM_H
#define SPEEXDSP_AEC_OS_SUPPORT_CUSTOM_H

extern _Thread_local long speexdsp_aec_warnings;

#define OVERRIDE_SPEEX_WARNING
static inline void speex_warning(const char *str)
{
   (void)str;
   speexdsp_aec_warnings++;
}

#define OVERRIDE_SPEEX_WARNING_INT
static inline void speex_warning_int(const char *str, int val)
{
   (void)str;
   (void)val;
   speexdsp_aec_warnings++;
}

#define OVERRIDE_SPEEX_NOTIFY
static inline void speex_notify(const char *str)
{
   (void)str;
}

#endif
