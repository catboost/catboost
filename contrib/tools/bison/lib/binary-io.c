/* Binary mode I/O.
   Copyright 2017-2019 Free Software Foundation, Inc.

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
# include <errno.h>
# include <unistd.h>

int
__gl_setmode_check (int fd)
{
  if (isatty (fd))
    {
      errno = EINVAL;
      return -1;
    }
  else
    return 0;
}
#endif
