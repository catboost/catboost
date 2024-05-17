/* fflush.c -- allow flushing input streams
   Copyright (C) 2007-2013 Free Software Foundation, Inc.

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

/* Written by Eric Blake. */

#include <config.h>

/* Specification.  */
#include "stdio--.h"

#include <errno.h>
#include <unistd.h>

#include "freading.h"

#include "stdio-impl.h"

#include "unused-parameter.h"

#undef fflush


#if defined _IO_EOF_SEEN || defined _IO_ftrylockfile || __GNU_LIBRARY__ == 1 /* GNU libc, BeOS, Haiku, Linux libc5 */

/* Clear the stream's ungetc buffer, preserving the value of ftello (fp).  */
static void
clear_ungetc_buffer_preserving_position (FILE *fp)
{
  if (fp->_flags & _IO_IN_BACKUP)
    /* _IO_free_backup_area is a bit complicated.  Simply call fseek.  */
    fseeko (fp, 0, SEEK_CUR);
}

#else

/* Clear the stream's ungetc buffer.  May modify the value of ftello (fp).  */
static void
clear_ungetc_buffer (FILE *fp)
{
# if defined __sferror || defined __DragonFly__ /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */
  if (HASUB (fp))
    {
      fp_->_p += fp_->_r;
      fp_->_r = 0;
    }
# elif defined __EMX__              /* emx+gcc */
  if (fp->_ungetc_count > 0)
    {
      fp->_ungetc_count = 0;
      fp->_rcount = - fp->_rcount;
    }
# elif defined _IOERR               /* Minix, AIX, HP-UX, IRIX, OSF/1, Solaris, OpenServer, mingw, NonStop Kernel */
  /* Nothing to do.  */
# else                              /* other implementations */
  fseeko (fp, 0, SEEK_CUR);
# endif
}

#endif

#if ! (defined _IO_EOF_SEEN || defined _IO_ftrylockfile || __GNU_LIBRARY__ == 1 /* GNU libc, BeOS, Haiku, Linux libc5 */)

# if (defined __sferror || defined __DragonFly__) && defined __SNPT /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */

static int
disable_seek_optimization (FILE *fp)
{
  int saved_flags = fp_->_flags & (__SOPT | __SNPT);
  fp_->_flags = (fp_->_flags & ~__SOPT) | __SNPT;
  return saved_flags;
}

static void
restore_seek_optimization (FILE *fp, int saved_flags)
{
  fp_->_flags = (fp_->_flags & ~(__SOPT | __SNPT)) | saved_flags;
}

# else

static void
update_fpos_cache (FILE *fp _GL_UNUSED_PARAMETER,
                   off_t pos _GL_UNUSED_PARAMETER)
{
#  if defined __sferror || defined __DragonFly__ /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */
#   if defined __CYGWIN__
  /* fp_->_offset is typed as an integer.  */
  fp_->_offset = pos;
#   else
  /* fp_->_offset is an fpos_t.  */
  /* Use a union, since on NetBSD, the compilation flags determine
     whether fpos_t is typedef'd to off_t or a struct containing a
     single off_t member.  */
  union
    {
      fpos_t f;
      off_t o;
    } u;
  u.o = pos;
  fp_->_offset = u.f;
#   endif
  fp_->_flags |= __SOFF;
#  endif
}
# endif
#endif

/* Flush all pending data on STREAM according to POSIX rules.  Both
   output and seekable input streams are supported.  */
int
rpl_fflush (FILE *stream)
{
  /* When stream is NULL, POSIX and C99 only require flushing of "output
     streams and update streams in which the most recent operation was not
     input", and all implementations do this.

     When stream is "an output stream or an update stream in which the most
     recent operation was not input", POSIX and C99 requires that fflush
     writes out any buffered data, and all implementations do this.

     When stream is, however, an input stream or an update stream in
     which the most recent operation was input, C99 specifies nothing,
     and POSIX only specifies behavior if the stream is seekable.
     mingw, in particular, drops the input buffer, leaving the file
     descriptor positioned at the end of the input buffer. I.e. ftell
     (stream) is lost.  We don't want to call the implementation's
     fflush in this case.

     We test ! freading (stream) here, rather than fwriting (stream), because
     what we need to know is whether the stream holds a "read buffer", and on
     mingw this is indicated by _IOREAD, regardless of _IOWRT.  */
  if (stream == NULL || ! freading (stream))
    return fflush (stream);

#if defined _IO_EOF_SEEN || defined _IO_ftrylockfile || __GNU_LIBRARY__ == 1 /* GNU libc, BeOS, Haiku, Linux libc5 */

  clear_ungetc_buffer_preserving_position (stream);

  return fflush (stream);

#else
  {
    /* Notes about the file-position indicator:
       1) The file position indicator is incremented by fgetc() and decremented
          by ungetc():
          <http://www.opengroup.org/susv3/functions/fgetc.html>
            "... the fgetc() function shall ... advance the associated file
             position indicator for the stream ..."
          <http://www.opengroup.org/susv3/functions/ungetc.html>
            "The file-position indicator is decremented by each successful
             call to ungetc()..."
       2) <http://www.opengroup.org/susv3/functions/ungetc.html> says:
            "The value of the file-position indicator for the stream after
             reading or discarding all pushed-back bytes shall be the same
             as it was before the bytes were pushed back."
          Here we are discarding all pushed-back bytes.  But more specifically,
       3) <http://www.opengroup.org/austin/aardvark/latest/xshbug3.txt> says:
            "[After fflush(),] the file offset of the underlying open file
             description shall be set to the file position of the stream, and
             any characters pushed back onto the stream by ungetc() ... shall
             be discarded."  */

    /* POSIX does not specify fflush behavior for non-seekable input
       streams.  Some implementations purge unread data, some return
       EBADF, some do nothing.  */
    off_t pos = ftello (stream);
    if (pos == -1)
      {
        errno = EBADF;
        return EOF;
      }

    /* Clear the ungetc buffer.  */
    clear_ungetc_buffer (stream);

    /* To get here, we must be flushing a seekable input stream, so the
       semantics of fpurge are now appropriate to clear the buffer.  To
       avoid losing data, the lseek is also necessary.  */
    {
      int result = fpurge (stream);
      if (result != 0)
        return result;
    }

# if (defined __sferror || defined __DragonFly__) && defined __SNPT /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */

    {
      /* Disable seek optimization for the next fseeko call.  This tells the
         following fseeko call to seek to the desired position directly, rather
         than to seek to a block-aligned boundary.  */
      int saved_flags = disable_seek_optimization (stream);
      int result = fseeko (stream, pos, SEEK_SET);

      restore_seek_optimization (stream, saved_flags);
      return result;
    }

# else

    pos = lseek (fileno (stream), pos, SEEK_SET);
    if (pos == -1)
      return EOF;
    /* After a successful lseek, update the file descriptor's position cache
       in the stream.  */
    update_fpos_cache (stream, pos);

    return 0;

# endif
  }
#endif
}
