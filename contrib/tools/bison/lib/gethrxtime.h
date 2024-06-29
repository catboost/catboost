/* gethrxtime -- get high resolution real time

   Copyright (C) 2005, 2009-2019 Free Software Foundation, Inc.

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

#ifndef GETHRXTIME_H_
#define GETHRXTIME_H_ 1

#include "xtime.h"

#ifndef _GL_INLINE_HEADER_BEGIN
 #error "Please include config.h first."
#endif
_GL_INLINE_HEADER_BEGIN
#ifndef GETHRXTIME_INLINE
# define GETHRXTIME_INLINE _GL_INLINE
#endif

#ifdef  __cplusplus
extern "C" {
#endif

/* Get the current time, as a count of the number of nanoseconds since
   an arbitrary epoch (e.g., the system boot time).  Prefer a
   high-resolution clock that is not subject to resetting or
   drifting.  */

#if HAVE_ARITHMETIC_HRTIME_T && HAVE_DECL_GETHRTIME
# include <time.h>
GETHRXTIME_INLINE xtime_t gethrxtime (void) { return gethrtime (); }
# else
xtime_t gethrxtime (void);
#endif

_GL_INLINE_HEADER_END

#ifdef  __cplusplus
}
#endif

#endif
