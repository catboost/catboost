/* Safe automatic memory allocation with out of memory checking.
   Copyright (C) 2003, 2005, 2007, 2009-2016 Free Software Foundation, Inc.
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

#ifndef _XMALLOCA_H
#define _XMALLOCA_H

#include "malloca.h"
#include "xalloc.h"


#ifdef __cplusplus
extern "C" {
#endif


/* xmalloca(N) is a checking safe variant of alloca(N).  It allocates N bytes
   of memory allocated on the stack, that must be freed using freea() before
   the function returns.  Upon failure, it exits with an error message.  */
#if HAVE_ALLOCA
# define xmalloca(N) \
  ((N) < 4032 - sa_increment                                        \
   ? (void *) ((char *) alloca ((N) + sa_increment) + sa_increment) \
   : xmmalloca (N))
extern void * xmmalloca (size_t n);
#else
# define xmalloca(N) \
  xmalloc (N)
#endif

/* xnmalloca(N,S) is an overflow-safe variant of xmalloca (N * S).
   It allocates an array of N objects, each with S bytes of memory,
   on the stack.  S must be positive and N must be nonnegative.
   The array must be freed using freea() before the function returns.
   Upon failure, it exits with an error message.  */
#if HAVE_ALLOCA
/* Rely on xmalloca (SIZE_MAX) calling xalloc_die ().  */
# define xnmalloca(n, s) \
    xmalloca (xalloc_oversized ((n), (s)) ? (size_t) (-1) : (n) * (s))
#else
# define xnmalloca(n, s) \
    xnmalloc ((n), (s))
#endif


#ifdef __cplusplus
}
#endif


#endif /* _XMALLOCA_H */
