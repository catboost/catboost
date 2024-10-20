/* Close standard input, rewinding seekable stdin if necessary.

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

#include "closein.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include "gettext.h"
#define _(msgid) gettext (msgid)

#include "close-stream.h"
#include "closeout.h"
#include "error.h"
#include "exitfail.h"
#include "freadahead.h"
#include "quotearg.h"

static const char *file_name;

/* Set the file name to be reported in the event an error is detected
   on stdin by close_stdin.  See also close_stdout_set_file_name, if
   an error is detected when closing stdout.  */
void
close_stdin_set_file_name (const char *file)
{
  file_name = file;
}

/* Close standard input, rewinding any unused input if stdin is
   seekable.  On error, issue a diagnostic and _exit with status
   'exit_failure'.  Then call close_stdout.

   Most programs can get by with close_stdout.  close_stdin is only
   needed when a program wants to guarantee that partially read input
   from seekable stdin is not consumed, for any subsequent clients.
   For example, POSIX requires that these two commands behave alike:

     (sed -ne 1q; cat) < file
     tail -n +2 file

   Since close_stdin is commonly registered via 'atexit', POSIX
   and the C standard both say that it should not call 'exit',
   because the behavior is undefined if 'exit' is called more than
   once.  So it calls '_exit' instead of 'exit'.  If close_stdin
   is registered via atexit before other functions are registered,
   the other functions can act before this _exit is invoked.

   Applications that use close_stdout should flush any streams other
   than stdin, stdout, and stderr before exiting, since the call to
   _exit will bypass other buffer flushing.  Applications should be
   flushing and closing other streams anyway, to check for I/O errors.
   Also, applications should not use tmpfile, since _exit can bypass
   the removal of these files.

   It's important to detect such failures and exit nonzero because many
   tools (most notably 'make' and other build-management systems) depend
   on being able to detect failure in other tools via their exit status.  */

void
close_stdin (void)
{
  bool fail = false;

  /* There is no need to flush stdin if we can determine quickly that stdin's
     input buffer is empty; in this case we know that if stdin is seekable,
     (fseeko (stdin, 0, SEEK_CUR), ftello (stdin))
     == lseek (0, 0, SEEK_CUR).  */
  if (freadahead (stdin) > 0)
    {
      /* Only attempt flush if stdin is seekable, as fflush is entitled to
         fail on non-seekable streams.  */
      if (fseeko (stdin, 0, SEEK_CUR) == 0 && fflush (stdin) != 0)
        fail = true;
    }
  if (close_stream (stdin) != 0)
    fail = true;
  if (fail)
    {
      /* Report failure, but defer exit until after closing stdout,
         since the failure report should still be flushed.  */
      char const *close_error = _("error closing file");
      if (file_name)
        error (0, errno, "%s: %s", quotearg_colon (file_name),
               close_error);
      else
        error (0, errno, "%s", close_error);
    }

  close_stdout ();

  if (fail)
    _exit (exit_failure);
}
