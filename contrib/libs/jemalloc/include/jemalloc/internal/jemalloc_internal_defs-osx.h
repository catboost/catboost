#pragma once

#include "jemalloc_internal_defs-linux.h"

/*
 * Define overrides for non-standard allocator-related functions if they are
 * present on the system.
 */
#undef JEMALLOC_OVERRIDE___LIBC_CALLOC
#undef JEMALLOC_OVERRIDE___LIBC_FREE
#undef JEMALLOC_OVERRIDE___LIBC_MALLOC
#undef JEMALLOC_OVERRIDE___LIBC_MEMALIGN
#undef JEMALLOC_OVERRIDE___LIBC_REALLOC
#undef JEMALLOC_OVERRIDE___LIBC_VALLOC

/*
 * Defined if os_unfair_lock_*() functions are available, as provided by Darwin.
 */
#define JEMALLOC_OS_UNFAIR_LOCK 

/* Defined if syscall(2) is usable. */
#undef JEMALLOC_USE_SYSCALL

/*
 * Defined if secure_getenv(3) is available.
 */
#undef JEMALLOC_HAVE_SECURE_GETENV

/*
 * Defined if issetugid(2) is available.
 */
#define JEMALLOC_HAVE_ISSETUGID 

/* Defined if pthread_setname_np(3) is available. */
#undef JEMALLOC_HAVE_PTHREAD_SETNAME_NP

/*
 * Defined if clock_gettime(CLOCK_MONOTONIC_COARSE, ...) is available.
 */
#undef JEMALLOC_HAVE_CLOCK_MONOTONIC_COARSE

/*
 * Defined if clock_gettime(CLOCK_MONOTONIC, ...) is available.
 */
#undef JEMALLOC_HAVE_CLOCK_MONOTONIC

/*
 * Defined if mach_absolute_time() is available.
 */
#define JEMALLOC_HAVE_MACH_ABSOLUTE_TIME 1

/*
 * Defined if threaded initialization is known to be safe on this platform.
 * Among other things, it must be possible to initialize a mutex without
 * triggering allocation in order for threaded allocation to be safe.
 */
#undef JEMALLOC_THREADED_INIT

/* JEMALLOC_PROF enables allocation profiling. */
#undef JEMALLOC_PROF

/* Use libunwind for profile backtracing if defined. */
#undef JEMALLOC_PROF_LIBUNWIND

/*
 * JEMALLOC_DSS enables use of sbrk(2) to allocate extents from the data storage
 * segment (DSS).
 */
#undef JEMALLOC_DSS

/*
 * If defined, retain memory for later reuse by default rather than using e.g.
 * munmap() to unmap freed extents.  This is enabled on 64-bit Linux because
 * common sequences of mmap()/munmap() calls will cause virtual memory map
 * holes.
 */
#undef JEMALLOC_RETAIN

/* TLS is used to map arenas and magazine caches to threads. */
#undef JEMALLOC_TLS

/*
 * Darwin (OS X) uses zones to work around Mach-O symbol override shortcomings.
 */
#define JEMALLOC_ZONE 

/*
 * Methods for determining whether the OS overcommits.
 * JEMALLOC_PROC_SYS_VM_OVERCOMMIT_MEMORY: Linux's
 *                                         /proc/sys/vm.overcommit_memory file.
 * JEMALLOC_SYSCTL_VM_OVERCOMMIT: FreeBSD's vm.overcommit sysctl.
 */
#undef JEMALLOC_PROC_SYS_VM_OVERCOMMIT_MEMORY

/*
 * Defined if transparent huge pages are supported via the MADV_[NO]HUGEPAGE
 * arguments to madvise(2).
 */
#undef JEMALLOC_HAVE_MADVISE_HUGE

/*
 * Methods for purging unused pages differ between operating systems.
 *
 *   madvise(..., MADV_FREE) : This marks pages as being unused, such that they
 *                             will be discarded rather than swapped out.
 *   madvise(..., MADV_DONTNEED) : If JEMALLOC_PURGE_MADVISE_DONTNEED_ZEROS is
 *                                 defined, this immediately discards pages,
 *                                 such that new pages will be demand-zeroed if
 *                                 the address region is later touched;
 *                                 otherwise this behaves similarly to
 *                                 MADV_FREE, though typically with higher
 *                                 system overhead.
 */
#undef JEMALLOC_PURGE_MADVISE_DONTNEED_ZEROS

/* Defined if madvise(2) is available but MADV_FREE is not (x86 Linux only). */
#undef JEMALLOC_DEFINE_MADVISE_FREE

/*
 * Defined if MADV_DO[NT]DUMP is supported as an argument to madvise.
 */
#undef JEMALLOC_MADVISE_DONTDUMP

/* Define if operating system has alloca.h header. */
#undef JEMALLOC_HAS_ALLOCA_H

/* glibc malloc hooks (__malloc_hook, __realloc_hook, __free_hook). */
#undef JEMALLOC_GLIBC_MALLOC_HOOK

/* glibc memalign hook. */
#undef JEMALLOC_GLIBC_MEMALIGN_HOOK

/* Adaptive mutex support in pthreads. */
#undef JEMALLOC_HAVE_PTHREAD_MUTEX_ADAPTIVE_NP

/* GNU specific sched_getcpu support */
#undef JEMALLOC_HAVE_SCHED_GETCPU

/* GNU specific sched_setaffinity support */
#undef JEMALLOC_HAVE_SCHED_SETAFFINITY

/*
 * If defined, all the features necessary for background threads are present.
 */
#undef JEMALLOC_BACKGROUND_THREAD

/*
 * Defined if strerror_r returns char * if _GNU_SOURCE is defined.
 */
#undef JEMALLOC_STRERROR_R_RETURNS_CHAR_WITH_GNU_SOURCE
