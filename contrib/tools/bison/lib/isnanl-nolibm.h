/* Test for NaN that does not need libm.
   Copyright (C) 2007-2020 Free Software Foundation, Inc.

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

#if HAVE_ISNANL_IN_LIBC
/* Get declaration of isnan macro or (older) isnanl function.  */
# include <math.h>
# if (__GNUC__ >= 4) || (__clang_major__ >= 4)
   /* GCC >= 4.0 and clang provide a type-generic built-in for isnan.
      GCC >= 4.0 also provides __builtin_isnanl, but clang doesn't.  */
#  undef isnanl
#  define isnanl(x) __builtin_isnan ((long double)(x))
# elif defined isnan
#  undef isnanl
#  define isnanl(x) isnan ((long double)(x))
# endif
#else
/* Test whether X is a NaN.  */
# undef isnanl
# define isnanl rpl_isnanl
extern int isnanl (long double x);
#endif
