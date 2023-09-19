/* va_list error handler for noninteractive utilities
   Copyright (C) 2006-2007, 2009-2013 Free Software Foundation, Inc.

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

/* Written by Eric Blake.  */

#include <config.h>

#include "verror.h"
#include "xvasprintf.h"

#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>

#if ENABLE_NLS
# include "gettext.h"
# define _(msgid) gettext (msgid)
#endif

#ifndef _
# define _(String) String
#endif

/* Print a message with 'vfprintf (stderr, FORMAT, ARGS)';
   if ERRNUM is nonzero, follow it with ": " and strerror (ERRNUM).
   If STATUS is nonzero, terminate the program with 'exit (STATUS)'.
   Use the globals error_print_progname and error_message_count similarly
   to error().  */
void
verror (int status, int errnum, const char *format, va_list args)
{
  verror_at_line (status, errnum, NULL, 0, format, args);
}

/* Print a message with 'vfprintf (stderr, FORMAT, ARGS)';
   if ERRNUM is nonzero, follow it with ": " and strerror (ERRNUM).
   If STATUS is nonzero, terminate the program with 'exit (STATUS)'.
   If FNAME is not NULL, prepend the message with "FNAME:LINENO:".
   Use the globals error_print_progname, error_message_count, and
   error_one_per_line similarly to error_at_line().  */
void
verror_at_line (int status, int errnum, const char *file,
                unsigned int line_number, const char *format, va_list args)
{
  char *message = xvasprintf (format, args);
  if (message)
    {
      /* Until http://sourceware.org/bugzilla/show_bug.cgi?id=2997 is fixed,
         glibc violates GNU Coding Standards when the file argument to
         error_at_line is NULL.  */
      if (file)
        error_at_line (status, errnum, file, line_number, "%s", message);
      else
        error (status, errnum, "%s", message);
    }
  else
    {
      /* EOVERFLOW, EINVAL, and EILSEQ from xvasprintf are signs of
         serious programmer errors.  */
      error (0, errno, _("unable to display error message"));
      abort ();
    }
  free (message);
}
