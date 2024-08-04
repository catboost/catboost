/* Binary mode I/O.
   Copyright 2017-2020 Free Software Foundation, Inc.

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

#include <config.h>

#define BINARY_IO_INLINE _GL_EXTERN_INLINE
#include "binary-io.h"

#if defined __DJGPP__ || defined __EMX__
# include <unistd.h>

int
set_binary_mode (int fd, int mode)
{
  if (isatty (fd))
    /* If FD refers to a console (not a pipe, not a regular file),
       O_TEXT is the only reasonable mode, both on input and on output.
       Silently ignore the request.  If we were to return -1 here,
       all programs that use xset_binary_mode would fail when run
       with console input or console output.  */
    return O_TEXT;
  else
    return __gl_setmode (fd, mode);
}

#endif
