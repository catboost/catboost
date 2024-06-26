/* Safe automatic memory allocation.
   Copyright (C) 2003, 2006-2007, 2009-2018 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2003, 2018.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#define _GL_USE_STDLIB_ALLOC 1
#include <config.h>

/* Specification.  */
#include "malloca.h"

#include "verify.h"

/* The speed critical point in this file is freea() applied to an alloca()
   result: it must be fast, to match the speed of alloca().  The speed of
   mmalloca() and freea() in the other case are not critical, because they
   are only invoked for big memory sizes.
   Here we use a bit in the address as an indicator, an idea by Ondřej Bílka.
   malloca() can return three types of pointers:
     - Pointers ≡ 0 mod 2*sa_alignment_max come from stack allocation.
     - Pointers ≡ sa_alignment_max mod 2*sa_alignment_max come from heap
       allocation.
     - NULL comes from a failed heap allocation.  */

/* Type for holding very small pointer differences.  */
typedef unsigned char small_t;
/* Verify that it is wide enough.  */
verify (2 * sa_alignment_max - 1 <= (small_t) -1);

void *
mmalloca (size_t n)
{
#if HAVE_ALLOCA
  /* Allocate one more word, used to determine the address to pass to freea(),
     and room for the alignment ≡ sa_alignment_max mod 2*sa_alignment_max.  */
  size_t nplus = n + sizeof (small_t) + 2 * sa_alignment_max - 1;

  if (nplus >= n)
    {
      char *mem = (char *) malloc (nplus);

      if (mem != NULL)
        {
          char *p =
            (char *)((((uintptr_t)mem + sizeof (small_t) + sa_alignment_max - 1)
                      & ~(uintptr_t)(2 * sa_alignment_max - 1))
                     + sa_alignment_max);
          /* Here p >= mem + sizeof (small_t),
             and p <= mem + sizeof (small_t) + 2 * sa_alignment_max - 1
             hence p + n <= mem + nplus.
             So, the memory range [p, p+n) lies in the allocated memory range
             [mem, mem + nplus).  */
          ((small_t *) p)[-1] = p - mem;
          /* p ≡ sa_alignment_max mod 2*sa_alignment_max.  */
          return p;
        }
    }
  /* Out of memory.  */
  return NULL;
#else
# if !MALLOC_0_IS_NONNULL
  if (n == 0)
    n = 1;
# endif
  return malloc (n);
#endif
}

#if HAVE_ALLOCA
void
freea (void *p)
{
  /* Check argument.  */
  if ((uintptr_t) p & (sa_alignment_max - 1))
    {
      /* p was not the result of a malloca() call.  Invalid argument.  */
      abort ();
    }
  /* Determine whether p was a non-NULL pointer returned by mmalloca().  */
  if ((uintptr_t) p & sa_alignment_max)
    {
      void *mem = (char *) p - ((small_t *) p)[-1];
      free (mem);
    }
}
#endif

/*
 * Hey Emacs!
 * Local Variables:
 * coding: utf-8
 * End:
 */
