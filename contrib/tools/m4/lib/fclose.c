/* fclose replacement.
   Copyright (C) 2008-2016 Free Software Foundation, Inc.

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
#include <stdio.h>

#include <errno.h>
#include <unistd.h>

#include "freading.h"
#include "msvc-inval.h"

#undef fclose

#if HAVE_MSVC_INVALID_PARAMETER_HANDLER
static int
fclose_nothrow (FILE *fp)
{
  int result;

  TRY_MSVC_INVAL
    {
      result = fclose (fp);
    }
  CATCH_MSVC_INVAL
    {
      result = EOF;
      errno = EBADF;
    }
  DONE_MSVC_INVAL;

  return result;
}
#else
# define fclose_nothrow fclose
#endif

/* Override fclose() to call the overridden fflush() or close().  */

int
rpl_fclose (FILE *fp)
{
  int saved_errno = 0;
  int fd;
  int result = 0;

  /* Don't change behavior on memstreams.  */
  fd = fileno (fp);
  if (fd < 0)
    return fclose_nothrow (fp);

  /* We only need to flush the file if it is not reading or if it is
     seekable.  This only guarantees the file position of input files
     if the fflush module is also in use.  */
  if ((!freading (fp) || lseek (fileno (fp), 0, SEEK_CUR) != -1)
      && fflush (fp))
    saved_errno = errno;

  /* fclose() calls close(), but we need to also invoke all hooks that our
     overridden close() function invokes.  See lib/close.c.  */
#if WINDOWS_SOCKETS
  /* Call the overridden close(), then the original fclose().
     Note about multithread-safety: There is a race condition where some
     other thread could open fd between our close and fclose.  */
  if (close (fd) < 0 && saved_errno == 0)
    saved_errno = errno;

  fclose_nothrow (fp); /* will fail with errno = EBADF,
                          if we did not lose a race */

#else /* !WINDOWS_SOCKETS */
  /* Call fclose() and invoke all hooks of the overridden close().  */

# if REPLACE_FCHDIR
  /* Note about multithread-safety: There is a race condition here as well.
     Some other thread could open fd between our calls to fclose and
     _gl_unregister_fd.  */
  result = fclose_nothrow (fp);
  if (result == 0)
    _gl_unregister_fd (fd);
# else
  /* No race condition here.  */
  result = fclose_nothrow (fp);
# endif

#endif /* !WINDOWS_SOCKETS */

  if (saved_errno != 0)
    {
      errno = saved_errno;
      result = EOF;
    }

  return result;
}
