/* An ftello() function that works around platform bugs.
   Copyright (C) 2007, 2009-2013 Free Software Foundation, Inc.

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

/* Get lseek.  */
#include <unistd.h>

#include "stdio-impl.h"

off_t
ftello (FILE *fp)
#undef ftello
#if !HAVE_FTELLO
# undef ftell
# define ftello ftell
#endif
#if _GL_WINDOWS_64_BIT_OFF_T
# undef ftello
# if HAVE__FTELLI64 /* msvc, mingw64 */
#  define ftello _ftelli64
# else /* mingw */
#  define ftello ftello64
# endif
#endif
{
#if LSEEK_PIPE_BROKEN
  /* mingw gives bogus answers rather than failure on non-seekable files.  */
  if (lseek (fileno (fp), 0, SEEK_CUR) == -1)
    return -1;
#endif

#if FTELLO_BROKEN_AFTER_SWITCHING_FROM_READ_TO_WRITE /* Solaris */
  /* The Solaris stdio leaves the _IOREAD flag set after reading from a file
     reaches EOF and the program then starts writing to the file.  ftello
     gets confused by this.  */
  if (fp_->_flag & _IOWRT)
    {
      off_t pos;

      /* Call ftello nevertheless, for the side effects that it does on fp.  */
      ftello (fp);

      /* Compute the file position ourselves.  */
      pos = lseek (fileno (fp), (off_t) 0, SEEK_CUR);
      if (pos >= 0)
        {
          if ((fp_->_flag & _IONBF) == 0 && fp_->_base != NULL)
            pos += fp_->_ptr - fp_->_base;
        }
      return pos;
    }
#endif

#if defined __SL64 && defined __SCLE /* Cygwin */
  if ((fp->_flags & __SL64) == 0)
    {
      /* Cygwin 1.5.0 through 1.5.24 failed to open stdin in 64-bit
         mode; but has an ftello that requires 64-bit mode.  */
      FILE *tmp = fopen ("/dev/null", "r");
      if (!tmp)
        return -1;
      fp->_flags |= __SL64;
      fp->_seek64 = tmp->_seek64;
      fclose (tmp);
    }
#endif
  return ftello (fp);
}
