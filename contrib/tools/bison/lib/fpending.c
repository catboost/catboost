/* fpending.c -- return the number of pending output bytes on a stream
   Copyright (C) 2000, 2004, 2006-2007, 2009-2020 Free Software Foundation,
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
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Jim Meyering. */

#include <config.h>

/* Specification.  */
#include "fpending.h"

#include "stdio-impl.h"

/* This file is not used on systems that already have the __fpending function,
   namely glibc >= 2.2, Solaris >= 7, Android API >= 23.  */

/* Return the number of pending (aka buffered, unflushed)
   bytes on the stream, FP, that is open for writing.  */
size_t
__fpending (FILE *fp)
{
  /* Most systems provide FILE as a struct and the necessary bitmask in
     <stdio.h>, because they need it for implementing getc() and putc() as
     fast macros.  */
#if defined _IO_EOF_SEEN || defined _IO_ftrylockfile || __GNU_LIBRARY__ == 1
  /* GNU libc, BeOS, Haiku, Linux libc5 */
  return fp->_IO_write_ptr - fp->_IO_write_base;
#elif defined __sferror || defined __DragonFly__ || defined __ANDROID__
  /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin, Minix 3, Android */
  return fp->_p - fp->_bf._base;
#elif defined __EMX__                /* emx+gcc */
  return fp->_ptr - fp->_buffer;
#elif defined __minix                /* Minix */
  return fp_->_ptr - fp_->_buf;
#elif defined _IOERR                 /* AIX, HP-UX, IRIX, OSF/1, Solaris, OpenServer, mingw, MSVC, NonStop Kernel, OpenVMS */
  return (fp_->_ptr ? fp_->_ptr - fp_->_base : 0);
#elif defined __UCLIBC__             /* uClibc */
  return (fp->__modeflags & __FLAG_WRITING ? fp->__bufpos - fp->__bufstart : 0);
#elif defined __QNX__                /* QNX */
  return (fp->_Mode & 0x2000 /*_MWRITE*/ ? fp->_Next - fp->_Buf : 0);
#elif defined __MINT__               /* Atari FreeMiNT */
  return fp->__bufp - fp->__buffer;
#elif defined EPLAN9                 /* Plan9 */
  return fp->wp - fp->buf;
#else
# error "Please port gnulib fpending.c to your platform!"
  return 1;
#endif
}
