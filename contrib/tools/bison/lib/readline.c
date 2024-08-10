/* readline.c --- Simple implementation of readline.
   Copyright (C) 2005-2007, 2009-2020 Free Software Foundation, Inc.
   Written by Simon Josefsson

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

/* This module is intended to be used when the application only needs
   the readline interface.  If you need more functions from the
   readline library, it is recommended to require the readline library
   (or improve this module) rather than #if-protect part of your
   application (doing so would add assumptions of this module into
   your application).  The application should use #include
   "readline.h", that header file will include <readline/readline.h>
   if the real library is present on the system. */

/* Get specification. */
#include "readline.h"

#include <stdio.h>
#include <string.h>

char *
readline (const char *prompt)
{
  char *out = NULL;
  size_t size = 0;

  if (prompt)
    {
      fputs (prompt, stdout);
      fflush (stdout);
    }

  if (getline (&out, &size, stdin) < 0)
    return NULL;

  while (*out && (out[strlen (out) - 1] == '\r'
                  || out[strlen (out) - 1] == '\n'))
    out[strlen (out) - 1] = '\0';

  return out;
}
