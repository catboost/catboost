/* Provide access to the last buffer returned by localtime() or gmtime().

   Copyright (C) 2001-2003, 2005-2007, 2009-2018 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

/* written by Jim Meyering */

#include <time.h>

#if GETTIMEOFDAY_CLOBBERS_LOCALTIME || TZSET_CLOBBERS_LOCALTIME

/* The address of the last buffer returned by localtime() or gmtime().  */
extern struct tm *localtime_buffer_addr;

#endif
