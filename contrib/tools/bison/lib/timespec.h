/* timespec -- System time interface

   Copyright (C) 2000, 2002, 2004-2005, 2007, 2009-2020 Free Software
   Foundation, Inc.

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

#if ! defined TIMESPEC_H
#define TIMESPEC_H

#include <time.h>

#ifndef _GL_INLINE_HEADER_BEGIN
 #error "Please include config.h first."
#endif
_GL_INLINE_HEADER_BEGIN
#ifndef _GL_TIMESPEC_INLINE
# define _GL_TIMESPEC_INLINE _GL_INLINE
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "arg-nonnull.h"

/* Inverse resolution of timespec timestamps (in units per second),
   and log base 10 of the inverse resolution.  */

enum { TIMESPEC_HZ = 1000000000 };
enum { LOG10_TIMESPEC_HZ = 9 };

/* Obsolescent names for backward compatibility.
   They are misnomers, because TIMESPEC_RESOLUTION is not a resolution.  */

enum { TIMESPEC_RESOLUTION = TIMESPEC_HZ };
enum { LOG10_TIMESPEC_RESOLUTION = LOG10_TIMESPEC_HZ };

/* Return a timespec with seconds S and nanoseconds NS.  */

_GL_TIMESPEC_INLINE struct timespec
make_timespec (time_t s, long int ns)
{
  struct timespec r;
  r.tv_sec = s;
  r.tv_nsec = ns;
  return r;
}

/* Return negative, zero, positive if A < B, A == B, A > B, respectively.  */

_GL_TIMESPEC_INLINE int _GL_ATTRIBUTE_PURE
timespec_cmp (struct timespec a, struct timespec b)
{
  return 2 * _GL_CMP (a.tv_sec, b.tv_sec) + _GL_CMP (a.tv_nsec, b.tv_nsec);
}

/* Return -1, 0, 1, depending on the sign of A.  A.tv_nsec must be
   nonnegative.  */
_GL_TIMESPEC_INLINE int _GL_ATTRIBUTE_PURE
timespec_sign (struct timespec a)
{
  return _GL_CMP (a.tv_sec, 0) + (!a.tv_sec & !!a.tv_nsec);
}

struct timespec timespec_add (struct timespec, struct timespec)
  _GL_ATTRIBUTE_CONST;
struct timespec timespec_sub (struct timespec, struct timespec)
  _GL_ATTRIBUTE_CONST;
struct timespec dtotimespec (double)
  _GL_ATTRIBUTE_CONST;

/* Return an approximation to A, of type 'double'.  */
_GL_TIMESPEC_INLINE double
timespectod (struct timespec a)
{
  return a.tv_sec + a.tv_nsec / 1e9;
}

struct timespec current_timespec (void);
void gettime (struct timespec *) _GL_ARG_NONNULL ((1));
int settime (struct timespec const *) _GL_ARG_NONNULL ((1));

#ifdef __cplusplus
}
#endif

_GL_INLINE_HEADER_END

#endif
