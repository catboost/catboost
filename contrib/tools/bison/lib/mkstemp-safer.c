/* Invoke mkstemp, but avoid some glitches.

   Copyright (C) 2005-2007, 2009-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* Written by Paul Eggert.  */

#include <config.h>

#include "stdlib-safer.h"

#include "stdlib--.h"
#include "unistd-safer.h"


/* Like mkstemp, but do not return STDIN_FILENO, STDOUT_FILENO, or
   STDERR_FILENO.  */

int
mkstemp_safer (char *templ)
{
  return fd_safer (mkstemp (templ));
}

#if GNULIB_MKOSTEMP
/* Like mkostemp, but do not return STDIN_FILENO, STDOUT_FILENO, or
   STDERR_FILENO.  */
int
mkostemp_safer (char *templ, int flags)
{
  return fd_safer_flag (mkostemp (templ, flags), flags);
}
#endif

#if GNULIB_MKOSTEMPS
/* Like mkostemps, but do not return STDIN_FILENO, STDOUT_FILENO, or
   STDERR_FILENO.  */
int
mkostemps_safer (char *templ, int suffixlen, int flags)
{
  return fd_safer_flag (mkostemps (templ, suffixlen, flags), flags);
}
#endif

#if GNULIB_MKSTEMPS
/* Like mkstemps, but do not return STDIN_FILENO, STDOUT_FILENO, or
   STDERR_FILENO.  */
int mkstemps_safer (char *templ, int suffixlen)
{
  return fd_safer (mkstemps (templ, suffixlen));
}
#endif
