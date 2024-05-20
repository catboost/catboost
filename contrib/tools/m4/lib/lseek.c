/* An lseek() function that detects pipes.
   Copyright (C) 2007, 2009-2013 Free Software Foundation, Inc.

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
#include <unistd.h>

#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
/* Windows platforms.  */
/* Get GetFileType.  */
# include <windows.h>
/* Get _get_osfhandle.  */
# include "msvc-nothrow.h"
#else
# include <sys/stat.h>
#endif
#include <errno.h>

#undef lseek

off_t
rpl_lseek (int fd, off_t offset, int whence)
{
#if (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__
  /* mingw lseek mistakenly succeeds on pipes, sockets, and terminals.  */
  HANDLE h = (HANDLE) _get_osfhandle (fd);
  if (h == INVALID_HANDLE_VALUE)
    {
      errno = EBADF;
      return -1;
    }
  if (GetFileType (h) != FILE_TYPE_DISK)
    {
      errno = ESPIPE;
      return -1;
    }
#else
  /* BeOS lseek mistakenly succeeds on pipes...  */
  struct stat statbuf;
  if (fstat (fd, &statbuf) < 0)
    return -1;
  if (!S_ISREG (statbuf.st_mode))
    {
      errno = ESPIPE;
      return -1;
    }
#endif
#if _GL_WINDOWS_64_BIT_OFF_T
  return _lseeki64 (fd, offset, whence);
#else
  return lseek (fd, offset, whence);
#endif
}
