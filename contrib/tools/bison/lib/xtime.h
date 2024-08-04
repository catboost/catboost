/* xtime -- extended-resolution integer timestamps

   Copyright (C) 2005-2006, 2009-2020 Free Software Foundation, Inc.

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

#ifndef XTIME_H_
#define XTIME_H_ 1

#ifndef _GL_INLINE_HEADER_BEGIN
 #error "Please include config.h first."
#endif
_GL_INLINE_HEADER_BEGIN
#ifndef XTIME_INLINE
# define XTIME_INLINE _GL_INLINE
#endif

/* xtime_t is a signed type used for timestamps.  It is an integer
   type that is a count of nanoseconds.  */
typedef long long int xtime_t;
#define XTIME_PRECISION 1000000000

#ifdef  __cplusplus
extern "C" {
#endif

/* Return an extended time value that contains S seconds and NS
   nanoseconds.  S and NS should be nonnegative; otherwise, integer
   overflow can occur even if the result is in range.  */
XTIME_INLINE xtime_t
xtime_make (xtime_t s, long int ns)
{
  return XTIME_PRECISION * s + ns;
}

/* The following functions split an extended time value:
   T = XTIME_PRECISION * xtime_sec (T) + xtime_nsec (T)
   with  0 <= xtime_nsec (T) < XTIME_PRECISION.  */

/* Return the number of seconds in T, which must be nonnegative.  */
XTIME_INLINE xtime_t
xtime_nonnegative_sec (xtime_t t)
{
  return t / XTIME_PRECISION;
}

/* Return the number of seconds in T.  */
XTIME_INLINE xtime_t
xtime_sec (xtime_t t)
{
  return (t + (t < 0)) / XTIME_PRECISION - (t < 0);
}

/* Return the number of nanoseconds in T, which must be nonnegative.  */
XTIME_INLINE long int
xtime_nonnegative_nsec (xtime_t t)
{
  return t % XTIME_PRECISION;
}

/* Return the number of nanoseconds in T.  */
XTIME_INLINE long int
xtime_nsec (xtime_t t)
{
  long int ns = t % XTIME_PRECISION;
  if (ns < 0)
    ns += XTIME_PRECISION;
  return ns;
}

#ifdef  __cplusplus
}
#endif

#endif
