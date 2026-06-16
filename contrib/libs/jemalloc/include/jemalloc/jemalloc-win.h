#include "jemalloc-linux.h"

/* LG_SIZEOF_PTR is determined at compile time via _WIN64 on Windows. */
#undef LG_SIZEOF_PTR
#define LG_SIZEOF_PTR LG_SIZEOF_PTR_WIN

/* JEMALLOC_SYS_NOTHROW is not used on Windows; make it a no-op. */
#undef JEMALLOC_SYS_NOTHROW
#define JEMALLOC_SYS_NOTHROW

/* JEMALLOC_EXPORT is unconditionally empty on Windows (no dllexport/import). */
#undef JEMALLOC_EXPORT
#define JEMALLOC_EXPORT

/* JEMALLOC_NOTHROW with attrs on Windows should use __attribute__((nothrow)). */
#undef JEMALLOC_NOTHROW
#define JEMALLOC_NOTHROW JEMALLOC_ATTR(nothrow)

/* memalign and valloc are not available on Windows. */
#undef JEMALLOC_OVERRIDE_MEMALIGN
#undef JEMALLOC_OVERRIDE_VALLOC
#undef je_memalign
#undef je_valloc

/* Different GID / smallocx symbol on the Windows build. */
#undef je_smallocx_54eaed1d8b56b1aa528be3bdd1877e59c56fa90c
#define je_smallocx_ea6b3e973b477b8061e0076bb257dbd7f3faa756 smallocx_ea6b3e973b477b8061e0076bb257dbd7f3faa756

/* malloc_conf_2_conf_harder is absent on the Windows build. */
#undef je_malloc_conf_2_conf_harder
