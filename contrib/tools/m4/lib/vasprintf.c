/* Formatted output to strings.
   Copyright (C) 1999, 2002, 2006-2013 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, see <http://www.gnu.org/licenses/>.  */

#include <config.h>

/* Specification.  */
#ifdef IN_LIBASPRINTF
# include "vasprintf.h"
#else
# include <stdio.h>
#endif

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#include "vasnprintf.h"

int
vasprintf (char **resultp, const char *format, va_list args)
{
  size_t length;
  char *result = vasnprintf (NULL, &length, format, args);
  if (result == NULL)
    return -1;

  if (length > INT_MAX)
    {
      free (result);
      errno = EOVERFLOW;
      return -1;
    }

  *resultp = result;
  /* Return the number of resulting bytes, excluding the trailing NUL.  */
  return length;
}
