/* Safe automatic memory allocation with out of memory checking.
   Copyright (C) 2003, 2006-2007, 2009-2013 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2003.

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
#include "xmalloca.h"

#include "xalloc.h"

#if HAVE_ALLOCA

void *
xmmalloca (size_t n)
{
  void *p;

  p = mmalloca (n);
  if (p == NULL)
    xalloc_die ();
  return p;
}

#endif
