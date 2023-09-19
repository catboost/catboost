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

#include <stdbool.h>
#include <stdio.h>

/* Return true if the stream STREAM is opened read-only, or if the
   last operation on the stream was a read operation.  Return false if
   the stream is opened write-only or append-only, or if it supports
   writing and there is no current read operation (such as fgetc).

   freading and fwriting will never both be true.  If STREAM supports
   both reads and writes, then:
     - both freading and fwriting might be false when the stream is first
       opened, after read encounters EOF, or after fflush,
     - freading might be false or true and fwriting might be false
       after repositioning (such as fseek, fsetpos, or rewind),
   depending on the underlying implementation.

   STREAM must not be wide-character oriented.  */

#if HAVE___FREADING && (!defined __GLIBC__ || defined __UCLIBC__ || __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 7))
/* Solaris >= 7, not glibc >= 2.2, but glibc >= 2.7, or musl libc  */

# include <stdio_ext.h>
# define freading(stream) (__freading (stream) != 0)

#else

# ifdef __cplusplus
extern "C" {
# endif

extern bool freading (FILE *stream);

# ifdef __cplusplus
}
# endif

#endif
