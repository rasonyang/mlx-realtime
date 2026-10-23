/* What SpeexDSP's configure writes from speexdsp_config_types.h.in. speexdsp_types.h includes it
 * only on platforms it does not know (macOS uses <sys/types.h> instead). */

#ifndef __SPEEX_TYPES_H__
#define __SPEEX_TYPES_H__

#include <stdint.h>

typedef int16_t spx_int16_t;
typedef uint16_t spx_uint16_t;
typedef int32_t spx_int32_t;
typedef uint32_t spx_uint32_t;

#endif
