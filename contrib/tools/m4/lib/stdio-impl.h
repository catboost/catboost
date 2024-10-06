/* Implementation details of FILE streams.
   Copyright (C) 2007-2008, 2010-2021 Free Software Foundation, Inc.

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

/* Many stdio implementations have the same logic and therefore can share
   the same implementation of stdio extension API, except that some fields
   have different naming conventions, or their access requires some casts.  */

/* Glibc 2.28 made _IO_UNBUFFERED and _IO_IN_BACKUP private.  For now, work
   around this problem by defining them ourselves.  FIXME: Do not rely on glibc
   internals.  */
#if defined _IO_EOF_SEEN
# if !defined _IO_UNBUFFERED
#  define _IO_UNBUFFERED 0x2
# endif
# if !defined _IO_IN_BACKUP
#  define _IO_IN_BACKUP 0x100
# endif
#endif

/* BSD stdio derived implementations.  */

#if defined __NetBSD__                         /* NetBSD */
/* Get __NetBSD_Version__.  */
# include <sys/param.h>
#endif

#include <errno.h>                             /* For detecting Plan9.  */

#if defined __sferror || defined __DragonFly__ || defined __ANDROID__
  /* FreeBSD, NetBSD, OpenBSD, DragonFly, Mac OS X, Cygwin, Minix 3, Android */

# if defined __DragonFly__          /* DragonFly */
  /* See <https://gitweb.dragonflybsd.org/dragonfly.git/blob_plain/HEAD:/lib/libc/stdio/priv_stdio.h>.  */
#  define fp_ ((struct { struct __FILE_public pub; \
                         struct { unsigned char *_base; int _size; } _bf; \
                         void *cookie; \
                         void *_close; \
                         void *_read; \
                         void *_seek; \
                         void *_write; \
                         struct { unsigned char *_base; int _size; } _ub; \
                         int _ur; \
                         unsigned char _ubuf[3]; \
                         unsigned char _nbuf[1]; \
                         struct { unsigned char *_base; int _size; } _lb; \
                         int _blksize; \
                         fpos_t _offset; \
                         /* More fields, not relevant here.  */ \
                       } *) fp)
  /* See <https://gitweb.dragonflybsd.org/dragonfly.git/blob_plain/HEAD:/include/stdio.h>.  */
#  define _p pub._p
#  define _flags pub._flags
#  define _r pub._r
#  define _w pub._w
# elif defined __ANDROID__ /* Android */
#  ifdef __LP64__
#   define _gl_flags_file_t int
#  else
#   define _gl_flags_file_t short
#  endif
  /* Up to this commit from 2015-10-12
     <https://android.googlesource.com/platform/bionic.git/+/f0141dfab10a4b332769d52fa76631a64741297a>
     the innards of FILE were public, and fp_ub could be defined like for OpenBSD,
     see <https://android.googlesource.com/platform/bionic.git/+/e78392637d5086384a5631ddfdfa8d7ec8326ee3/libc/stdio/fileext.h>
     and <https://android.googlesource.com/platform/bionic.git/+/e78392637d5086384a5631ddfdfa8d7ec8326ee3/libc/stdio/local.h>.
     After this commit, the innards of FILE are hidden.  */
#  define fp_ ((struct { unsigned char *_p; \
                         int _r; \
                         int _w; \
                         _gl_flags_file_t _flags; \
                         _gl_flags_file_t _file; \
                         struct { unsigned char *_base; size_t _size; } _bf; \
                         int _lbfsize; \
                         void *_cookie; \
                         void *_close; \
                         void *_read; \
                         void *_seek; \
                         void *_write; \
                         struct { unsigned char *_base; size_t _size; } _ext; \
                         unsigned char *_up; \
                         int _ur; \
                         unsigned char _ubuf[3]; \
                         unsigned char _nbuf[1]; \
                         struct { unsigned char *_base; size_t _size; } _lb; \
                         int _blksize; \
                         fpos_t _offset; \
                         /* More fields, not relevant here.  */ \
                       } *) fp)
# else
#  define fp_ fp
# endif

# if (defined __NetBSD__ && __NetBSD_Version__ >= 105270000) || defined __OpenBSD__ || defined __minix /* NetBSD >= 1.5ZA, OpenBSD, Minix 3 */
  /* See <http://cvsweb.netbsd.org/bsdweb.cgi/src/lib/libc/stdio/fileext.h?rev=HEAD&content-type=text/x-cvsweb-markup>
     and <https://cvsweb.openbsd.org/cgi-bin/cvsweb/src/lib/libc/stdio/fileext.h?rev=HEAD&content-type=text/x-cvsweb-markup>
     and <https://github.com/Stichting-MINIX-Research-Foundation/minix/blob/master/lib/libc/stdio/fileext.h> */
  struct __sfileext
    {
      struct  __sbuf _ub; /* ungetc buffer */
      /* More fields, not relevant here.  */
    };
#  define fp_ub ((struct __sfileext *) fp->_ext._base)->_ub
# elif defined __ANDROID__                     /* Android */
  struct __sfileext
    {
      struct { unsigned char *_base; size_t _size; } _ub; /* ungetc buffer */
      /* More fields, not relevant here.  */
    };
#  define fp_ub ((struct __sfileext *) fp_->_ext._base)->_ub
# else                                         /* FreeBSD, NetBSD <= 1.5Z, DragonFly, Mac OS X, Cygwin */
#  define fp_ub fp_->_ub
# endif

# define HASUB(fp) (fp_ub._base != NULL)

# if defined __ANDROID__ /* Android */
  /* Needed after this commit from 2016-01-25
     <https://android.googlesource.com/platform/bionic.git/+/e70e0e9267d069bf56a5078c99307e08a7280de7> */
#  ifndef __SEOF
#   define __SLBF 1
#   define __SNBF 2
#   define __SRD 4
#   define __SWR 8
#   define __SRW 0x10
#   define __SEOF 0x20
#   define __SERR 0x40
#  endif
#  ifndef __SOFF
#   define __SOFF 0x1000
#  endif
# endif

#endif


/* SystemV derived implementations.  */

#ifdef __TANDEM                     /* NonStop Kernel */
# ifndef _IOERR
/* These values were determined by the program 'stdioext-flags' at
   <https://lists.gnu.org/r/bug-gnulib/2010-12/msg00165.html>.  */
#  define _IOERR   0x40
#  define _IOREAD  0x80
#  define _IOWRT    0x4
#  define _IORW   0x100
# endif
#endif

#if defined _IOERR

# if defined __sun && defined _LP64 /* Solaris/{SPARC,AMD64} 64-bit */
#  define fp_ ((struct { unsigned char *_ptr; \
                         unsigned char *_base; \
                         unsigned char *_end; \
                         long _cnt; \
                         int _file; \
                         unsigned int _flag; \
                       } *) fp)
# elif defined __VMS                /* OpenVMS */
#  define fp_ ((struct _iobuf *) fp)
# else
#  define fp_ fp
# endif

# if defined _SCO_DS || (defined __SCO_VERSION__ || defined __sysv5__)  /* OpenServer 5, OpenServer 6, UnixWare 7 */
#  define _cnt __cnt
#  define _ptr __ptr
#  define _base __base
#  define _flag __flag
# endif

#elif defined _WIN32 && ! defined __CYGWIN__  /* newer Windows with MSVC */

/* <stdio.h> does not define the innards of FILE any more.  */
# define WINDOWS_OPAQUE_FILE

struct _gl_real_FILE
{
  /* Note: Compared to older Windows and to mingw, it has the fields
     _base and _cnt swapped. */
  unsigned char *_ptr;
  unsigned char *_base;
  int _cnt;
  int _flag;
  int _file;
  int _charbuf;
  int _bufsiz;
};
# define fp_ ((struct _gl_real_FILE *) fp)

/* These values were determined by a program similar to the one at
   <https://lists.gnu.org/r/bug-gnulib/2010-12/msg00165.html>.  */
# define _IOREAD   0x1
# define _IOWRT    0x2
# define _IORW     0x4
# define _IOEOF    0x8
# define _IOERR   0x10

#endif
