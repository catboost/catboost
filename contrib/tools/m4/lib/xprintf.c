/* printf wrappers that fail immediately for non-file-related errors
   Copyright (C) 2007, 2009-2016 Free Software Foundation, Inc.

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

#include "xprintf.h"

#include <errno.h>

#include "error.h"
#include "exitfail.h"
#include "gettext.h"

/* written by Jim Meyering */

/* Just like printf, but call error if it fails without setting the
   stream's error indicator.  */
int
xprintf (char const *restrict format, ...)
{
  va_list args;
  int retval;
  va_start (args, format);
  retval = xvprintf (format, args);
  va_end (args);

  return retval;
}

/* Just like vprintf, but call error if it fails without setting the
   stream's error indicator.  */
int
xvprintf (char const *restrict format, va_list args)
{
  int retval = vprintf (format, args);
  if (retval < 0 && ! ferror (stdout))
    error (exit_failure, errno, gettext ("cannot perform formatted output"));

  return retval;
}

/* Just like fprintf, but call error if it fails without setting the
   stream's error indicator.  */
int
xfprintf (FILE *restrict stream, char const *restrict format, ...)
{
  va_list args;
  int retval;
  va_start (args, format);
  retval = xvfprintf (stream, format, args);
  va_end (args);

  return retval;
}

/* Just like vfprintf, but call error if it fails without setting the
   stream's error indicator.  */
int
xvfprintf (FILE *restrict stream, char const *restrict format, va_list args)
{
  int retval = vfprintf (stream, format, args);
  if (retval < 0 && ! ferror (stream))
    error (exit_failure, errno, gettext ("cannot perform formatted output"));

  return retval;
}
