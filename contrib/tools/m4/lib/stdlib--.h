/* Like stdlib.h, but redefine some names to avoid glitches.

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

#include <stdlib.h>
#include "stdlib-safer.h"

#undef mkstemp
#define mkstemp mkstemp_safer

#if GNULIB_MKOSTEMP
# define mkostemp mkostemp_safer
#endif

#if GNULIB_MKOSTEMPS
# define mkostemps mkostemps_safer
#endif

#if GNULIB_MKSTEMPS
# define mkstemps mkstemps_safer
#endif
