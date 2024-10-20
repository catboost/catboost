/* hard-locale.c -- Determine whether a locale is hard.

   Copyright (C) 1997-1999, 2002-2004, 2006-2007, 2009-2016 Free Software
   Foundation, Inc.

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

#include "hard-locale.h"

#include <locale.h>
#include <stdlib.h>
#include <string.h>

#ifdef __GLIBC__
# define GLIBC_VERSION __GLIBC__
#elif defined __UCLIBC__
# define GLIBC_VERSION 2
#else
# define GLIBC_VERSION 0
#endif

/* Return true if the current CATEGORY locale is hard, i.e. if you
   can't get away with assuming traditional C or POSIX behavior.  */
bool
hard_locale (int category)
{
  bool hard = true;
  char const *p = setlocale (category, NULL);

  if (p)
    {
      if (2 <= GLIBC_VERSION)
        {
          if (strcmp (p, "C") == 0 || strcmp (p, "POSIX") == 0)
            hard = false;
        }
      else
        {
          char *locale = strdup (p);
          if (locale)
            {
              /* Temporarily set the locale to the "C" and "POSIX" locales
                 to find their names, so that we can determine whether one
                 or the other is the caller's locale.  */
              if (((p = setlocale (category, "C"))
                   && strcmp (p, locale) == 0)
                  || ((p = setlocale (category, "POSIX"))
                      && strcmp (p, locale) == 0))
                hard = false;

              /* Restore the caller's locale.  */
              setlocale (category, locale);
              free (locale);
            }
        }
    }

  return hard;
}
