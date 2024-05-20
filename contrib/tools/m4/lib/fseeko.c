/* An fseeko() function that, together with fflush(), is POSIX compliant.
   Copyright (C) 2007-2013 Free Software Foundation, Inc.

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
#include <stdio.h>

/* Get off_t, lseek, _POSIX_VERSION.  */
#include <unistd.h>

#include "stdio-impl.h"

int
fseeko (FILE *fp, off_t offset, int whence)
#undef fseeko
#if !HAVE_FSEEKO
# undef fseek
# define fseeko fseek
#endif
#if _GL_WINDOWS_64_BIT_OFF_T
# undef fseeko
# if HAVE__FSEEKI64 /* msvc, mingw64 */
#  define fseeko _fseeki64
# else /* mingw */
#  define fseeko fseeko64
# endif
#endif
{
#if LSEEK_PIPE_BROKEN
  /* mingw gives bogus answers rather than failure on non-seekable files.  */
  if (lseek (fileno (fp), 0, SEEK_CUR) == -1)
    return EOF;
#endif

  /* These tests are based on fpurge.c.  */
#if defined _IO_EOF_SEEN || defined _IO_ftrylockfile || __GNU_LIBRARY__ == 1 /* GNU libc, BeOS, Haiku, Linux libc5 */
  if (fp->_IO_read_end == fp->_IO_read_ptr
      && fp->_IO_write_ptr == fp->_IO_write_base
      && fp->_IO_save_base == NULL)
#elif defined __sferror || defined __DragonFly__ /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */
# if defined __SL64 && defined __SCLE /* Cygwin */
  if ((fp->_flags & __SL64) == 0)
    {
      /* Cygwin 1.5.0 through 1.5.24 failed to open stdin in 64-bit
         mode; but has an fseeko that requires 64-bit mode.  */
      FILE *tmp = fopen ("/dev/null", "r");
      if (!tmp)
        return -1;
      fp->_flags |= __SL64;
      fp->_seek64 = tmp->_seek64;
      fclose (tmp);
    }
# endif
  if (fp_->_p == fp_->_bf._base
      && fp_->_r == 0
      && fp_->_w == ((fp_->_flags & (__SLBF | __SNBF | __SRD)) == 0 /* fully buffered and not currently reading? */
                     ? fp_->_bf._size
                     : 0)
      && fp_ub._base == NULL)
#elif defined __EMX__               /* emx+gcc */
  if (fp->_ptr == fp->_buffer
      && fp->_rcount == 0
      && fp->_wcount == 0
      && fp->_ungetc_count == 0)
#elif defined __minix               /* Minix */
  if (fp_->_ptr == fp_->_buf
      && (fp_->_ptr == NULL || fp_->_count == 0))
#elif defined _IOERR                /* AIX, HP-UX, IRIX, OSF/1, Solaris, OpenServer, mingw, NonStop Kernel */
  if (fp_->_ptr == fp_->_base
      && (fp_->_ptr == NULL || fp_->_cnt == 0))
#elif WIN_SDK10
  if (((TWinSdk10File*)fp)->_ptr == ((TWinSdk10File*)fp)->_base
      && (((TWinSdk10File*)fp)->_ptr == NULL || ((TWinSdk10File*)fp)->_cnt == 0))
#elif defined __UCLIBC__            /* uClibc */
  if (((fp->__modeflags & __FLAG_WRITING) == 0
       || fp->__bufpos == fp->__bufstart)
      && ((fp->__modeflags & (__FLAG_READONLY | __FLAG_READING)) == 0
          || fp->__bufpos == fp->__bufread))
#elif defined __QNX__               /* QNX */
  if ((fp->_Mode & 0x2000 /* _MWRITE */ ? fp->_Next == fp->_Buf : fp->_Next == fp->_Rend)
      && fp->_Rback == fp->_Back + sizeof (fp->_Back)
      && fp->_Rsave == NULL)
#elif defined __MINT__              /* Atari FreeMiNT */
  if (fp->__bufp == fp->__buffer
      && fp->__get_limit == fp->__bufp
      && fp->__put_limit == fp->__bufp
      && !fp->__pushed_back)
#elif defined EPLAN9                /* Plan9 */
  if (fp->rp == fp->buf
      && fp->wp == fp->buf)
#elif FUNC_FFLUSH_STDIN < 0 && 200809 <= _POSIX_VERSION
  /* Cross-compiling to some other system advertising conformance to
     POSIX.1-2008 or later.  Assume fseeko and fflush work as advertised.
     If this assumption is incorrect, please report the bug to
     bug-gnulib.  */
  if (0)
#else
  #error "Please port gnulib fseeko.c to your platform! Look at the code in fseeko.c, then report this to bug-gnulib."
#endif
    {
      /* We get here when an fflush() call immediately preceded this one (or
         if ftell() has created buffers but no I/O has occurred on a
         newly-opened stream).  We know there are no buffers.  */
      off_t pos = lseek (fileno (fp), offset, whence);
      if (pos == -1)
        {
#if defined __sferror || defined __DragonFly__ /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */
          fp_->_flags &= ~__SOFF;
#endif
          return -1;
        }

#if defined _IO_EOF_SEEN || defined _IO_ftrylockfile || __GNU_LIBRARY__ == 1 /* GNU libc, BeOS, Haiku, Linux libc5 */
      fp->_flags &= ~_IO_EOF_SEEN;
      fp->_offset = pos;
#elif defined __sferror || defined __DragonFly__ /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */
# if defined __CYGWIN__
      /* fp_->_offset is typed as an integer.  */
      fp_->_offset = pos;
# else
      /* fp_->_offset is an fpos_t.  */
      {
        /* Use a union, since on NetBSD, the compilation flags
           determine whether fpos_t is typedef'd to off_t or a struct
           containing a single off_t member.  */
        union
          {
            fpos_t f;
            off_t o;
          } u;
        u.o = pos;
        fp_->_offset = u.f;
      }
# endif
      fp_->_flags |= __SOFF;
      fp_->_flags &= ~__SEOF;
#elif defined __EMX__               /* emx+gcc */
      fp->_flags &= ~_IOEOF;
#elif defined _IOERR                /* AIX, HP-UX, IRIX, OSF/1, Solaris, OpenServer, mingw, NonStop Kernel */
      fp->_flag &= ~_IOEOF;
#elif defined __MINT__              /* Atari FreeMiNT */
      fp->__offset = pos;
      fp->__eof = 0;
#endif
      return 0;
    }
  return fseeko (fp, offset, whence);
}
