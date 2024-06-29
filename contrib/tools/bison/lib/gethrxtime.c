/* gethrxtime -- get high resolution real time

   Copyright (C) 2005-2007, 2009-2019 Free Software Foundation, Inc.

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

#include <config.h>

#define GETHRXTIME_INLINE _GL_EXTERN_INLINE
#include "gethrxtime.h"

#if ! (HAVE_ARITHMETIC_HRTIME_T && HAVE_DECL_GETHRTIME)

#include "timespec.h"

/* Get the current time, as a count of the number of nanoseconds since
   an arbitrary epoch (e.g., the system boot time).  Prefer a
   high-resolution clock that is not subject to resetting or
   drifting.  */

xtime_t
gethrxtime (void)
{
# if HAVE_NANOUPTIME
  {
    struct timespec ts;
    nanouptime (&ts);
    return xtime_make (ts.tv_sec, ts.tv_nsec);
  }
# else

#  if defined CLOCK_MONOTONIC && HAVE_CLOCK_GETTIME
  {
    struct timespec ts;
    if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
      return xtime_make (ts.tv_sec, ts.tv_nsec);
  }
#  endif

#  if HAVE_MICROUPTIME
  {
    struct timeval tv;
    microuptime (&tv);
    return xtime_make (tv.tv_sec, 1000 * tv.tv_usec);
  }

#  else
  /* No monotonically increasing clocks are available; fall back on a
     clock that might jump backwards, since it's the best we can do.  */
  {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return xtime_make (ts.tv_sec, ts.tv_nsec);
  }
#  endif
# endif
}

#endif
