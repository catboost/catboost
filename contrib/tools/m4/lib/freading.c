/* Retrieve information about a FILE stream.
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

#include <config.h>

/* Specification.  */
#include "freading.h"

#include "stdio-impl.h"

/* Don't use glibc's __freading function in glibc < 2.7, see
   <http://sourceware.org/bugzilla/show_bug.cgi?id=4359>  */
#if !(HAVE___FREADING && (!defined __GLIBC__ || defined __UCLIBC__ || __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 7)))

bool
freading (FILE *fp)
{
  /* Most systems provide FILE as a struct and the necessary bitmask in
     <stdio.h>, because they need it for implementing getc() and putc() as
     fast macros.  */
# if defined _IO_EOF_SEEN || defined _IO_ftrylockfile || __GNU_LIBRARY__ == 1 /* GNU libc, BeOS, Haiku, Linux libc5 */
  return ((fp->_flags & _IO_NO_WRITES) != 0
          || ((fp->_flags & (_IO_NO_READS | _IO_CURRENTLY_PUTTING)) == 0
              && fp->_IO_read_base != NULL));
# elif defined __sferror || defined __DragonFly__ /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin */
  return (fp_->_flags & __SRD) != 0;
# elif defined __EMX__               /* emx+gcc */
  return (fp->_flags & _IOREAD) != 0;
# elif defined __minix               /* Minix */
  return (fp->_flags & _IOREADING) != 0;
# elif defined _IOERR                /* AIX, HP-UX, IRIX, OSF/1, Solaris, OpenServer, mingw, NonStop Kernel */
#  if defined __sun                  /* Solaris */
  return (fp->_flag & _IOREAD) != 0 && (fp->_flag & _IOWRT) == 0;
#  else
  return (fp->_flag & _IOREAD) != 0;
#  endif
# elif WIN_SDK10
  return (((TWinSdk10File*)fp)->_flags & WIN_SDK10_IOREAD) != 0;
# elif defined __UCLIBC__            /* uClibc */
  return (fp->__modeflags & (__FLAG_READONLY | __FLAG_READING)) != 0;
# elif defined __QNX__               /* QNX */
  return ((fp->_Mode & 0x2 /* _MOPENW */) == 0
          || (fp->_Mode & 0x1000 /* _MREAD */) != 0);
# elif defined __MINT__              /* Atari FreeMiNT */
  if (!fp->__mode.__write)
    return 1;
  if (!fp->__mode.__read)
    return 0;
#  ifdef _IO_CURRENTLY_GETTING /* Flag added on 2009-02-28 */
  return (fp->__flags & _IO_CURRENTLY_GETTING) != 0;
#  else
  return (fp->__buffer < fp->__get_limit /*|| fp->__bufp == fp->__put_limit ??*/);
#  endif
# elif defined EPLAN9                /* Plan9 */
  if (fp->state == 0 /* CLOSED */ || fp->state == 4 /* WR */)
    return 0;
  return (fp->state == 3 /* RD */ && (fp->bufl == 0 || fp->rp < fp->wp));
# else
#  error "Please port gnulib freading.c to your platform!"
# endif
}

#endif
