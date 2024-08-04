/* Determine the number of screen columns needed for a character.
   Copyright (C) 2006-2007, 2010-2020 Free Software Foundation, Inc.

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

/* Get iswprint.  */
#include <wctype.h>

#include "localcharset.h"
#include "streq.h"
#include "uniwidth.h"

/* Returns 1 if the current locale is an UTF-8 locale, 0 otherwise.  */
static inline int
is_locale_utf8 (void)
{
  const char *encoding = locale_charset ();
  return STREQ_OPT (encoding, "UTF-8", 'U', 'T', 'F', '-', '8', 0, 0, 0, 0);
}

#if GNULIB_WCHAR_SINGLE
/* When we know that the locale does not change, provide a speedup by
   caching the value of is_locale_utf8.  */
static int cached_is_locale_utf8 = -1;
static inline int
is_locale_utf8_cached (void)
{
  if (cached_is_locale_utf8 < 0)
    cached_is_locale_utf8 = is_locale_utf8 ();
  return cached_is_locale_utf8;
}
#else
/* By default, don't make assumptions, hence no caching.  */
# define is_locale_utf8_cached is_locale_utf8
#endif

int
wcwidth (wchar_t wc)
#undef wcwidth
{
  /* In UTF-8 locales, use a Unicode aware width function.  */
  if (is_locale_utf8_cached ())
    {
      /* We assume that in a UTF-8 locale, a wide character is the same as a
         Unicode character.  */
      return uc_width (wc, "UTF-8");
    }
  else
    {
      /* Otherwise, fall back to the system's wcwidth function.  */
#if HAVE_WCWIDTH
      return wcwidth (wc);
#else
      return wc == 0 ? 0 : iswprint (wc) ? 1 : -1;
#endif
    }
}
