#include "jemalloc-linux.h"

/* format(gnu_printf, ...) is not supported on macOS; redefine to plain printf variant. */
#undef JEMALLOC_HAVE_ATTR_FORMAT_GNU_PRINTF

/* macOS does not use JEMALLOC_SYS_NOTHROW (the Apple/FreeBSD branch makes it empty,
 * same as the non-Apple expansion of JEMALLOC_NOTHROW on GCC/Clang). */
#undef JEMALLOC_SYS_NOTHROW
#define JEMALLOC_SYS_NOTHROW

/* memalign is not available on macOS. */
#undef JEMALLOC_OVERRIDE_MEMALIGN
#undef je_memalign

/* Different GID / smallocx symbol on macOS build. */
#undef je_smallocx_54eaed1d8b56b1aa528be3bdd1877e59c56fa90c
#define je_smallocx_ea6b3e973b477b8061e0076bb257dbd7f3faa756 smallocx_ea6b3e973b477b8061e0076bb257dbd7f3faa756

/* malloc_conf_2_conf_harder is absent on the macOS build. */
#undef je_malloc_conf_2_conf_harder
