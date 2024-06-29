/* stat-related time functions.

   Copyright (C) 2005, 2007, 2009-2018 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Paul Eggert.  */

#ifndef STAT_TIME_H
#define STAT_TIME_H 1

#include "intprops.h"

#include <errno.h>
#include <stddef.h>
#include <sys/stat.h>
#include <time.h>

#ifndef _GL_INLINE_HEADER_BEGIN
 #error "Please include config.h first."
#endif
_GL_INLINE_HEADER_BEGIN
#ifndef _GL_STAT_TIME_INLINE
# define _GL_STAT_TIME_INLINE _GL_INLINE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* STAT_TIMESPEC (ST, ST_XTIM) is the ST_XTIM member for *ST of type
   struct timespec, if available.  If not, then STAT_TIMESPEC_NS (ST,
   ST_XTIM) is the nanosecond component of the ST_XTIM member for *ST,
   if available.  ST_XTIM can be st_atim, st_ctim, st_mtim, or st_birthtim
   for access, status change, data modification, or birth (creation)
   time respectively.

   These macros are private to stat-time.h.  */
#if _GL_WINDOWS_STAT_TIMESPEC || defined HAVE_STRUCT_STAT_ST_ATIM_TV_NSEC
# if _GL_WINDOWS_STAT_TIMESPEC || defined TYPEOF_STRUCT_STAT_ST_ATIM_IS_STRUCT_TIMESPEC
#  define STAT_TIMESPEC(st, st_xtim) ((st)->st_xtim)
# else
#  define STAT_TIMESPEC_NS(st, st_xtim) ((st)->st_xtim.tv_nsec)
# endif
#elif defined HAVE_STRUCT_STAT_ST_ATIMESPEC_TV_NSEC
# define STAT_TIMESPEC(st, st_xtim) ((st)->st_xtim##espec)
#elif defined HAVE_STRUCT_STAT_ST_ATIMENSEC
# define STAT_TIMESPEC_NS(st, st_xtim) ((st)->st_xtim##ensec)
#elif defined HAVE_STRUCT_STAT_ST_ATIM_ST__TIM_TV_NSEC
# define STAT_TIMESPEC_NS(st, st_xtim) ((st)->st_xtim.st__tim.tv_nsec)
#endif

/* Return the nanosecond component of *ST's access time.  */
_GL_STAT_TIME_INLINE long int _GL_ATTRIBUTE_PURE
get_stat_atime_ns (struct stat const *st)
{
# if defined STAT_TIMESPEC
  return STAT_TIMESPEC (st, st_atim).tv_nsec;
# elif defined STAT_TIMESPEC_NS
  return STAT_TIMESPEC_NS (st, st_atim);
# else
  return 0;
# endif
}

/* Return the nanosecond component of *ST's status change time.  */
_GL_STAT_TIME_INLINE long int _GL_ATTRIBUTE_PURE
get_stat_ctime_ns (struct stat const *st)
{
# if defined STAT_TIMESPEC
  return STAT_TIMESPEC (st, st_ctim).tv_nsec;
# elif defined STAT_TIMESPEC_NS
  return STAT_TIMESPEC_NS (st, st_ctim);
# else
  return 0;
# endif
}

/* Return the nanosecond component of *ST's data modification time.  */
_GL_STAT_TIME_INLINE long int _GL_ATTRIBUTE_PURE
get_stat_mtime_ns (struct stat const *st)
{
# if defined STAT_TIMESPEC
  return STAT_TIMESPEC (st, st_mtim).tv_nsec;
# elif defined STAT_TIMESPEC_NS
  return STAT_TIMESPEC_NS (st, st_mtim);
# else
  return 0;
# endif
}

/* Return the nanosecond component of *ST's birth time.  */
_GL_STAT_TIME_INLINE long int _GL_ATTRIBUTE_PURE
get_stat_birthtime_ns (struct stat const *st _GL_UNUSED)
{
# if defined HAVE_STRUCT_STAT_ST_BIRTHTIMESPEC_TV_NSEC
  return STAT_TIMESPEC (st, st_birthtim).tv_nsec;
# elif defined HAVE_STRUCT_STAT_ST_BIRTHTIMENSEC
  return STAT_TIMESPEC_NS (st, st_birthtim);
# else
  return 0;
# endif
}

/* Return *ST's access time.  */
_GL_STAT_TIME_INLINE struct timespec _GL_ATTRIBUTE_PURE
get_stat_atime (struct stat const *st)
{
#ifdef STAT_TIMESPEC
  return STAT_TIMESPEC (st, st_atim);
#else
  struct timespec t;
  t.tv_sec = st->st_atime;
  t.tv_nsec = get_stat_atime_ns (st);
  return t;
#endif
}

/* Return *ST's status change time.  */
_GL_STAT_TIME_INLINE struct timespec _GL_ATTRIBUTE_PURE
get_stat_ctime (struct stat const *st)
{
#ifdef STAT_TIMESPEC
  return STAT_TIMESPEC (st, st_ctim);
#else
  struct timespec t;
  t.tv_sec = st->st_ctime;
  t.tv_nsec = get_stat_ctime_ns (st);
  return t;
#endif
}

/* Return *ST's data modification time.  */
_GL_STAT_TIME_INLINE struct timespec _GL_ATTRIBUTE_PURE
get_stat_mtime (struct stat const *st)
{
#ifdef STAT_TIMESPEC
  return STAT_TIMESPEC (st, st_mtim);
#else
  struct timespec t;
  t.tv_sec = st->st_mtime;
  t.tv_nsec = get_stat_mtime_ns (st);
  return t;
#endif
}

/* Return *ST's birth time, if available; otherwise return a value
   with tv_sec and tv_nsec both equal to -1.  */
_GL_STAT_TIME_INLINE struct timespec _GL_ATTRIBUTE_PURE
get_stat_birthtime (struct stat const *st _GL_UNUSED)
{
  struct timespec t;

#if (defined HAVE_STRUCT_STAT_ST_BIRTHTIMESPEC_TV_NSEC \
     || defined HAVE_STRUCT_STAT_ST_BIRTHTIM_TV_NSEC)
  t = STAT_TIMESPEC (st, st_birthtim);
#elif defined HAVE_STRUCT_STAT_ST_BIRTHTIMENSEC
  t.tv_sec = st->st_birthtime;
  t.tv_nsec = st->st_birthtimensec;
#elif defined _WIN32 && ! defined __CYGWIN__
  /* Native Windows platforms (but not Cygwin) put the "file creation
     time" in st_ctime (!).  See
     <https://msdn.microsoft.com/en-us/library/14h5k7ff(VS.80).aspx>.  */
# if _GL_WINDOWS_STAT_TIMESPEC
  t = st->st_ctim;
# else
  t.tv_sec = st->st_ctime;
  t.tv_nsec = 0;
# endif
#else
  /* Birth time is not supported.  */
  t.tv_sec = -1;
  t.tv_nsec = -1;
#endif

#if (defined HAVE_STRUCT_STAT_ST_BIRTHTIMESPEC_TV_NSEC \
     || defined HAVE_STRUCT_STAT_ST_BIRTHTIM_TV_NSEC \
     || defined HAVE_STRUCT_STAT_ST_BIRTHTIMENSEC)
  /* FreeBSD and NetBSD sometimes signal the absence of knowledge by
     using zero.  Attempt to work around this problem.  Alas, this can
     report failure even for valid timestamps.  Also, NetBSD
     sometimes returns junk in the birth time fields; work around this
     bug if it is detected.  */
  if (! (t.tv_sec && 0 <= t.tv_nsec && t.tv_nsec < 1000000000))
    {
      t.tv_sec = -1;
      t.tv_nsec = -1;
    }
#endif

  return t;
}

/* If a stat-like function returned RESULT, normalize the timestamps
   in *ST, in case this platform suffers from the Solaris 11 bug where
   tv_nsec might be negative.  Return the adjusted RESULT, setting
   errno to EOVERFLOW if normalization overflowed.  This function
   is intended to be private to this .h file.  */
_GL_STAT_TIME_INLINE int
stat_time_normalize (int result, struct stat *st _GL_UNUSED)
{
#if defined __sun && defined STAT_TIMESPEC
  if (result == 0)
    {
      long int timespec_hz = 1000000000;
      short int const ts_off[] = { offsetof (struct stat, st_atim),
                                   offsetof (struct stat, st_mtim),
                                   offsetof (struct stat, st_ctim) };
      int i;
      for (i = 0; i < sizeof ts_off / sizeof *ts_off; i++)
        {
          struct timespec *ts = (struct timespec *) ((char *) st + ts_off[i]);
          long int q = ts->tv_nsec / timespec_hz;
          long int r = ts->tv_nsec % timespec_hz;
          if (r < 0)
            {
              r += timespec_hz;
              q--;
            }
          ts->tv_nsec = r;
          /* Overflow is possible, as Solaris 11 stat can yield
             tv_sec == TYPE_MINIMUM (time_t) && tv_nsec == -1000000000.
             INT_ADD_WRAPV is OK, since time_t is signed on Solaris.  */
          if (INT_ADD_WRAPV (q, ts->tv_sec, &ts->tv_sec))
            {
              errno = EOVERFLOW;
              return -1;
            }
        }
    }
#endif
  return result;
}

#ifdef __cplusplus
}
#endif

_GL_INLINE_HEADER_END

#endif
