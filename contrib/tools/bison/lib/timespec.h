/* timespec -- System time interface

   Copyright (C) 2000, 2002, 2004-2005, 2007, 2009-2018 Free Software
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
#include "verify.h"

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

/* Return negative, zero, positive if A < B, A == B, A > B, respectively.

   For each timestamp T, this code assumes that either:

     * T.tv_nsec is in the range 0..999999999; or
     * T.tv_sec corresponds to a valid leap second on a host that supports
       leap seconds, and T.tv_nsec is in the range 1000000000..1999999999; or
     * T.tv_sec is the minimum time_t value and T.tv_nsec is -1; or
       T.tv_sec is the maximum time_t value and T.tv_nsec is 2000000000.
       This allows for special struct timespec values that are less or
       greater than all possible valid timestamps.

   In all these cases, it is safe to subtract two tv_nsec values and
   convert the result to integer without worrying about overflow on
   any platform of interest to the GNU project, since all such
   platforms have 32-bit int or wider.

   Replacing "a.tv_nsec - b.tv_nsec" with something like
   "a.tv_nsec < b.tv_nsec ? -1 : a.tv_nsec > b.tv_nsec" would cause
   this function to work in some cases where the above assumption is
   violated, but not in all cases (e.g., a.tv_sec==1, a.tv_nsec==-2,
   b.tv_sec==0, b.tv_nsec==999999999) and is arguably not worth the
   extra instructions.  Using a subtraction has the advantage of
   detecting some invalid cases on platforms that detect integer
   overflow.  */

_GL_TIMESPEC_INLINE int _GL_ATTRIBUTE_PURE
timespec_cmp (struct timespec a, struct timespec b)
{
  if (a.tv_sec < b.tv_sec)
    return -1;
  if (a.tv_sec > b.tv_sec)
    return 1;

  /* Pacify gcc -Wstrict-overflow (bleeding-edge circa 2017-10-02).  See:
     https://lists.gnu.org/r/bug-gnulib/2017-10/msg00006.html  */
  assume (-1 <= a.tv_nsec && a.tv_nsec <= 2 * TIMESPEC_HZ);
  assume (-1 <= b.tv_nsec && b.tv_nsec <= 2 * TIMESPEC_HZ);

  return a.tv_nsec - b.tv_nsec;
}

/* Return -1, 0, 1, depending on the sign of A.  A.tv_nsec must be
   nonnegative.  */
_GL_TIMESPEC_INLINE int _GL_ATTRIBUTE_PURE
timespec_sign (struct timespec a)
{
  return a.tv_sec < 0 ? -1 : a.tv_sec || a.tv_nsec;
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
