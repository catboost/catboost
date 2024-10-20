/* vasprintf and asprintf with out-of-memory checking.
   Copyright (C) 1999, 2002-2004, 2006, 2009-2016 Free Software Foundation,
   Inc.

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

#include <config.h>

/* Specification.  */
#include "xvasprintf.h"

char *
xasprintf (const char *format, ...)
{
  va_list args;
  char *result;

  va_start (args, format);
  result = xvasprintf (format, args);
  va_end (args);

  return result;
}
