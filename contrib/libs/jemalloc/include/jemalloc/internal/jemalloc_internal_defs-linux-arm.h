#ifndef JEMALLOC_INTERNAL_DEFS_H_
#include "jemalloc_internal_defs-linux.h"

#define CPU_SPINWAIT
#define HAVE_CPU_SPINWAIT 1

#define LG_VADDR 32

#undef JEMALLOC_C11_ATOMICS

#undef JEMALLOC_RETAIN

#undef JEMALLOC_HAVE_MADVISE_HUGE

#define LG_SIZEOF_LONG 2

#endif /* JEMALLOC_INTERNAL_DEFS_H_ */
