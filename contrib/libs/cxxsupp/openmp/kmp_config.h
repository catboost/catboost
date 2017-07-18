/*
 * kmp_config.h -- Feature macros
 */
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
#ifndef KMP_CONFIG_H
#define KMP_CONFIG_H

#include "kmp_platform.h"

// cmakedefine01 MACRO will define MACRO as either 0 or 1
// cmakedefine MACRO 1 will define MACRO as 1 or leave undefined
#define DEBUG_BUILD 0
#define RELWITHDEBINFO_BUILD 0
#define LIBOMP_USE_ITT_NOTIFY 1
#define USE_ITT_NOTIFY LIBOMP_USE_ITT_NOTIFY
#if ! LIBOMP_USE_ITT_NOTIFY
# define INTEL_NO_ITTNOTIFY_API
#endif
#define LIBOMP_USE_VERSION_SYMBOLS 0
#if LIBOMP_USE_VERSION_SYMBOLS
# define KMP_USE_VERSION_SYMBOLS
#endif
#define LIBOMP_HAVE_WEAK_ATTRIBUTE 1
#define KMP_HAVE_WEAK_ATTRIBUTE LIBOMP_HAVE_WEAK_ATTRIBUTE
#define LIBOMP_HAVE_PSAPI 0
#define KMP_HAVE_PSAPI LIBOMP_HAVE_PSAPI
#define LIBOMP_STATS 0
#define KMP_STATS_ENABLED LIBOMP_STATS
#define LIBOMP_USE_DEBUGGER 0
#define USE_DEBUGGER LIBOMP_USE_DEBUGGER
#define LIBOMP_OMPT_DEBUG 0
#define OMPT_DEBUG LIBOMP_OMPT_DEBUG
#define LIBOMP_OMPT_SUPPORT 0
#define OMPT_SUPPORT LIBOMP_OMPT_SUPPORT
#define LIBOMP_OMPT_BLAME 1
#define OMPT_BLAME LIBOMP_OMPT_BLAME
#define LIBOMP_OMPT_TRACE 1
#define OMPT_TRACE LIBOMP_OMPT_TRACE
#define LIBOMP_USE_ADAPTIVE_LOCKS 1
#define KMP_USE_ADAPTIVE_LOCKS LIBOMP_USE_ADAPTIVE_LOCKS
#define KMP_DEBUG_ADAPTIVE_LOCKS 0
#define LIBOMP_USE_INTERNODE_ALIGNMENT 0
#define KMP_USE_INTERNODE_ALIGNMENT LIBOMP_USE_INTERNODE_ALIGNMENT
#define LIBOMP_ENABLE_ASSERTIONS 1
#define KMP_USE_ASSERT LIBOMP_ENABLE_ASSERTIONS
#define STUBS_LIBRARY 0
#define LIBOMP_USE_HWLOC 0
#define KMP_USE_HWLOC LIBOMP_USE_HWLOC
#define KMP_ARCH_STR "Intel(R) 64"
#define KMP_LIBRARY_FILE "libomp.so"
#define KMP_VERSION_MAJOR 5
#define KMP_VERSION_MINOR 0
#define LIBOMP_OMP_VERSION 41
#define OMP_50_ENABLED (LIBOMP_OMP_VERSION >= 50)
#define OMP_41_ENABLED (LIBOMP_OMP_VERSION >= 41)
#define OMP_40_ENABLED (LIBOMP_OMP_VERSION >= 40)
#define OMP_30_ENABLED (LIBOMP_OMP_VERSION >= 30)

// Configured cache line based on architecture
#if KMP_ARCH_PPC64
# define CACHE_LINE 128
#else
# define CACHE_LINE 64
#endif

#define KMP_DYNAMIC_LIB 1
#define KMP_NESTED_HOT_TEAMS 1
#define KMP_ADJUST_BLOCKTIME 1
#define BUILD_PARALLEL_ORDERED 1
#define KMP_ASM_INTRINS 1
#define USE_ITT_BUILD 1
#define INTEL_ITTNOTIFY_PREFIX __kmp_itt_
#if ! KMP_MIC
# define USE_LOAD_BALANCE 1
#endif
#if ! (KMP_OS_WINDOWS || KMP_OS_DARWIN)
# define KMP_TDATA_GTID 1
#endif
#if STUBS_LIBRARY
# define KMP_STUB 1
#endif
#if DEBUG_BUILD || RELWITHDEBINFO_BUILD
# define KMP_DEBUG 1
#endif

#if KMP_OS_WINDOWS
# define KMP_WIN_CDECL
#else
# define BUILD_TV
# define KMP_GOMP_COMPAT
#endif

#endif // KMP_CONFIG_H
