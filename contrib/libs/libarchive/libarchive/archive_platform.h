/*-
 * Copyright (c) 2003-2007 Tim Kientzle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* !!ONLY FOR USE INTERNALLY TO LIBARCHIVE!! */

/*
 * This header is the first thing included in any of the libarchive
 * source files.  As far as possible, platform-specific issues should
 * be dealt with here and not within individual source files.  I'm
 * actively trying to minimize #if blocks within the main source,
 * since they obfuscate the code.
 */

#ifndef ARCHIVE_PLATFORM_H_INCLUDED
#define	ARCHIVE_PLATFORM_H_INCLUDED

/* archive.h and archive_entry.h require this. */
#define	__LIBARCHIVE_BUILD 1

#if defined(PLATFORM_CONFIG_H)
/* Use hand-built config.h in environments that need it. */
#error #include PLATFORM_CONFIG_H
#elif defined(HAVE_CONFIG_H)
/* Most POSIX platforms use the 'configure' script to build config.h */
#include "config.h"
#else
/* Warn if the library hasn't been (automatically or manually) configured. */
#error Oops: No config.h and no pre-built configuration in archive_platform.h.
#endif

/* On macOS check for some symbols based on the deployment target version.  */
#if defined(__APPLE__)
# undef HAVE_FUTIMENS
# undef HAVE_UTIMENSAT
# include <AvailabilityMacros.h>
# if MAC_OS_X_VERSION_MIN_REQUIRED >= 101300
#  define HAVE_FUTIMENS 1
#  define HAVE_UTIMENSAT 1
# endif
#endif

/* For cygwin, to avoid missing LONG, ULONG, PUCHAR, ... definitions */
#ifdef __CYGWIN__
#include <windef.h>
#endif

/* It should be possible to get rid of this by extending the feature-test
 * macros to cover Windows API functions, probably along with non-trivial
 * refactoring of code to find structures that sit more cleanly on top of
 * either Windows or Posix APIs. */
#if (defined(__WIN32__) || defined(_WIN32) || defined(__WIN32)) && !defined(__CYGWIN__)
#include "archive_windows.h"
/* The C library on Windows specifies a calling convention for callback
 * functions and exports; when we interact with them (capture pointers,
 * call and pass function pointers) we need to match their calling
 * convention.
 * This only matters when libarchive is built with /Gr, /Gz or /Gv
 * (which change the default calling convention.) */
#define __LA_LIBC_CC __cdecl
#else
#define la_stat(path,stref)		stat(path,stref)
#define __LA_LIBC_CC
#endif

/*
 * The config files define a lot of feature macros.  The following
 * uses those macros to select/define replacements and include key
 * headers as required.
 */

/* Try to get standard C99-style integer type definitions. */
#if HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#if HAVE_STDINT_H
#include <stdint.h>
#endif

/* Borland warns about its own constants!  */
#if defined(__BORLANDC__)
# if HAVE_DECL_UINT64_MAX
#  undef	UINT64_MAX
#  undef	HAVE_DECL_UINT64_MAX
# endif
# if HAVE_DECL_UINT64_MIN
#  undef	UINT64_MIN
#  undef	HAVE_DECL_UINT64_MIN
# endif
# if HAVE_DECL_INT64_MAX
#  undef	INT64_MAX
#  undef	HAVE_DECL_INT64_MAX
# endif
# if HAVE_DECL_INT64_MIN
#  undef	INT64_MIN
#  undef	HAVE_DECL_INT64_MIN
# endif
#endif

/* Some platforms lack the standard *_MAX definitions. */
#if !HAVE_DECL_SIZE_MAX
#define	SIZE_MAX (~(size_t)0)
#endif
#if !HAVE_DECL_SSIZE_MAX
#define	SSIZE_MAX ((ssize_t)(SIZE_MAX >> 1))
#endif
#if !HAVE_DECL_UINT32_MAX
#define	UINT32_MAX (~(uint32_t)0)
#endif
#if !HAVE_DECL_INT32_MAX
#define	INT32_MAX ((int32_t)(UINT32_MAX >> 1))
#endif
#if !HAVE_DECL_INT32_MIN
#define	INT32_MIN ((int32_t)(~INT32_MAX))
#endif
#if !HAVE_DECL_UINT64_MAX
#define	UINT64_MAX (~(uint64_t)0)
#endif
#if !HAVE_DECL_INT64_MAX
#define	INT64_MAX ((int64_t)(UINT64_MAX >> 1))
#endif
#if !HAVE_DECL_INT64_MIN
#define	INT64_MIN ((int64_t)(~INT64_MAX))
#endif
#if !HAVE_DECL_UINTMAX_MAX
#define	UINTMAX_MAX (~(uintmax_t)0)
#endif
#if !HAVE_DECL_INTMAX_MAX
#define	INTMAX_MAX ((intmax_t)(UINTMAX_MAX >> 1))
#endif
#if !HAVE_DECL_INTMAX_MIN
#define	INTMAX_MIN ((intmax_t)(~INTMAX_MAX))
#endif

/* Some platforms lack the standard PRIxN/PRIdN definitions. */
#if !HAVE_INTTYPES_H || !defined(PRIx32) || !defined(PRId32)
#ifndef PRIx32
#if SIZEOF_INT == 4
#define PRIx32 "x"
#elif SIZEOF_LONG == 4
#define PRIx32 "lx"
#else
#error No suitable 32-bit unsigned integer type found for this platform
#endif
#endif // PRIx32
#ifndef PRId32
#if SIZEOF_INT == 4
#define PRId32 "d"
#elif SIZEOF_LONG == 4
#define PRId32 "ld"
#else
#error No suitable 32-bit signed integer type found for this platform
#endif
#endif // PRId32
#endif // !HAVE_INTTYPES_H || !defined(PRIx32) || !defined(PRId32)

/*
 * If we can't restore metadata using a file descriptor, then
 * for compatibility's sake, close files before trying to restore metadata.
 */
#if defined(HAVE_FCHMOD) || defined(HAVE_FUTIMES) || defined(HAVE_ACL_SET_FD) || defined(HAVE_ACL_SET_FD_NP) || defined(HAVE_FCHOWN)
#define	CAN_RESTORE_METADATA_FD
#endif

/*
 * glibc 2.24 deprecates readdir_r
 * bionic c deprecates readdir_r too
 */
#if defined(HAVE_READDIR_R) && (!defined(__GLIBC__) || !defined(__GLIBC_MINOR__) || __GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 24)) && (!defined(__ANDROID__))
#define	USE_READDIR_R	1
#else
#undef	USE_READDIR_R
#endif

/* Set up defaults for internal error codes. */
#ifndef ARCHIVE_ERRNO_FILE_FORMAT
#if HAVE_EFTYPE
#define	ARCHIVE_ERRNO_FILE_FORMAT EFTYPE
#else
#if HAVE_EILSEQ
#define	ARCHIVE_ERRNO_FILE_FORMAT EILSEQ
#else
#define	ARCHIVE_ERRNO_FILE_FORMAT EINVAL
#endif
#endif
#endif

#ifndef ARCHIVE_ERRNO_PROGRAMMER
#define	ARCHIVE_ERRNO_PROGRAMMER EINVAL
#endif

#ifndef ARCHIVE_ERRNO_MISC
#define	ARCHIVE_ERRNO_MISC (-1)
#endif

#if defined(__GNUC__) && (__GNUC__ >= 7)
#define	__LA_FALLTHROUGH	__attribute__((fallthrough))
#else
#define	__LA_FALLTHROUGH
#endif

#endif /* !ARCHIVE_PLATFORM_H_INCLUDED */
