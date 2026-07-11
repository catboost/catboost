#include "jemalloc-linux.h"

/* format(gnu_printf, ...) is not supported on macOS; redefine to plain printf variant. */
#undef JEMALLOC_HAVE_ATTR_FORMAT_GNU_PRINTF

/* macOS does not use JEMALLOC_SYS_NOTHROW (the Apple/FreeBSD branch makes it empty,
 * same as the non-Apple expansion of JEMALLOC_NOTHROW on GCC/Clang). */
#undef JEMALLOC_SYS_NOTHROW
#define JEMALLOC_SYS_NOTHROW
