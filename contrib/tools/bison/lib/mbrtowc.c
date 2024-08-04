/* Convert multibyte character to wide character.
   Copyright (C) 1999-2002, 2005-2020 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2008.

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

/* Specification.  */
#include <wchar.h>

#if GNULIB_defined_mbstate_t
/* Implement mbrtowc() on top of mbtowc() for the non-UTF-8 locales
   and directly for the UTF-8 locales.  */

# include <errno.h>
# include <stdint.h>
# include <stdlib.h>

# if defined _WIN32 && !defined __CYGWIN__

#  define WIN32_LEAN_AND_MEAN  /* avoid including junk */
#  include <windows.h>

# elif HAVE_PTHREAD_API

#  include <pthread.h>
#  if HAVE_THREADS_H && HAVE_WEAK_SYMBOLS
#   include <threads.h>
#   pragma weak thrd_exit
#   define c11_threads_in_use() (thrd_exit != NULL)
#  else
#   define c11_threads_in_use() 0
#  endif

# elif HAVE_THREADS_H

#  include <threads.h>

# endif

# include "verify.h"
# error #include "lc-charset-dispatch.h"
# error #include "mbtowc-lock.h"

# ifndef FALLTHROUGH
#  if __GNUC__ < 7
#   define FALLTHROUGH ((void) 0)
#  else
#   define FALLTHROUGH __attribute__ ((__fallthrough__))
#  endif
# endif

verify (sizeof (mbstate_t) >= 4);
static char internal_state[4];

size_t
mbrtowc (wchar_t *pwc, const char *s, size_t n, mbstate_t *ps)
{
# define FITS_IN_CHAR_TYPE(wc)  ((wc) <= WCHAR_MAX)
# error #include "mbrtowc-impl.h"
}

#else
/* Override the system's mbrtowc() function.  */

# if MBRTOWC_IN_C_LOCALE_MAYBE_EILSEQ
#  include "hard-locale.h"
#  include <locale.h>
# endif

# undef mbrtowc

size_t
rpl_mbrtowc (wchar_t *pwc, const char *s, size_t n, mbstate_t *ps)
{
  size_t ret;
  wchar_t wc;

# if MBRTOWC_NULL_ARG2_BUG || MBRTOWC_RETVAL_BUG || MBRTOWC_EMPTY_INPUT_BUG
  if (s == NULL)
    {
      pwc = NULL;
      s = "";
      n = 1;
    }
# endif

# if MBRTOWC_EMPTY_INPUT_BUG
  if (n == 0)
    return (size_t) -2;
# endif

  if (! pwc)
    pwc = &wc;

# if MBRTOWC_RETVAL_BUG
  {
    static mbstate_t internal_state;

    /* Override mbrtowc's internal state.  We cannot call mbsinit() on the
       hidden internal state, but we can call it on our variable.  */
    if (ps == NULL)
      ps = &internal_state;

    if (!mbsinit (ps))
      {
        /* Parse the rest of the multibyte character byte for byte.  */
        size_t count = 0;
        for (; n > 0; s++, n--)
          {
            ret = mbrtowc (&wc, s, 1, ps);

            if (ret == (size_t)(-1))
              return (size_t)(-1);
            count++;
            if (ret != (size_t)(-2))
              {
                /* The multibyte character has been completed.  */
                *pwc = wc;
                return (wc == 0 ? 0 : count);
              }
          }
        return (size_t)(-2);
      }
  }
# endif

# if MBRTOWC_STORES_INCOMPLETE_BUG
  ret = mbrtowc (&wc, s, n, ps);
  if (ret < (size_t) -2 && pwc != NULL)
    *pwc = wc;
# else
  ret = mbrtowc (pwc, s, n, ps);
# endif

# if MBRTOWC_NUL_RETVAL_BUG
  if (ret < (size_t) -2 && !*pwc)
    return 0;
# endif

# if MBRTOWC_IN_C_LOCALE_MAYBE_EILSEQ
  if ((size_t) -2 <= ret && n != 0 && ! hard_locale (LC_CTYPE))
    {
      unsigned char uc = *s;
      *pwc = uc;
      return 1;
    }
# endif

  return ret;
}

#endif
