/* vasprintf and asprintf with out-of-memory checking.
   Copyright (C) 1999, 2002-2004, 2006-2016 Free Software Foundation, Inc.

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

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>

#include "xalloc.h"

/* Checked size_t computations.  */
#include "xsize.h"

static char *
xstrcat (size_t argcount, va_list args)
{
  char *result;
  va_list ap;
  size_t totalsize;
  size_t i;
  char *p;

  /* Determine the total size.  */
  totalsize = 0;
  va_copy (ap, args);
  for (i = argcount; i > 0; i--)
    {
      const char *next = va_arg (ap, const char *);
      totalsize = xsum (totalsize, strlen (next));
    }
  va_end (ap);

  /* Test for overflow in the summing pass above or in (totalsize + 1) below.
     Also, don't return a string longer than INT_MAX, for consistency with
     vasprintf().  */
  if (totalsize == SIZE_MAX || totalsize > INT_MAX)
    {
      errno = EOVERFLOW;
      return NULL;
    }

  /* Allocate and fill the result string.  */
  result = XNMALLOC (totalsize + 1, char);
  p = result;
  for (i = argcount; i > 0; i--)
    {
      const char *next = va_arg (args, const char *);
      size_t len = strlen (next);
      memcpy (p, next, len);
      p += len;
    }
  *p = '\0';

  return result;
}

char *
xvasprintf (const char *format, va_list args)
{
  char *result;

  /* Recognize the special case format = "%s...%s".  It is a frequently used
     idiom for string concatenation and needs to be fast.  We don't want to
     have a separate function xstrcat() for this purpose.  */
  {
    size_t argcount = 0;
    const char *f;

    for (f = format;;)
      {
        if (*f == '\0')
          /* Recognized the special case of string concatenation.  */
          return xstrcat (argcount, args);
        if (*f != '%')
          break;
        f++;
        if (*f != 's')
          break;
        f++;
        argcount++;
      }
  }

  if (vasprintf (&result, format, args) < 0)
    {
      if (errno == ENOMEM)
        xalloc_die ();
      return NULL;
    }

  return result;
}
