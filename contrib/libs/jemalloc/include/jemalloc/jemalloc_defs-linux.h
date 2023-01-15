#pragma once

/* include/jemalloc/jemalloc_defs.h.  Generated from jemalloc_defs.h.in by configure.  */
/* Defined if __attribute__((...)) syntax is supported. */
#define JEMALLOC_HAVE_ATTR 

/* Support the experimental API. */
#define JEMALLOC_EXPERIMENTAL 

/*
 * Define overrides for non-standard allocator-related functions if they are
 * present on the system.
 */
#define JEMALLOC_OVERRIDE_MEMALIGN 
#define JEMALLOC_OVERRIDE_VALLOC 

/*
 * At least Linux omits the "const" in:
 *
 *   size_t malloc_usable_size(const void *ptr);
 *
 * Match the operating system's prototype.
 */
#define JEMALLOC_USABLE_SIZE_CONST 

/* sizeof(void *) == 2^LG_SIZEOF_PTR. */
#define LG_SIZEOF_PTR 3
