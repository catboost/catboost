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
