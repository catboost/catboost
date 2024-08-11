/* basename.c -- return the last element in a file name

   Copyright (C) 1990, 1998-2001, 2003-2006, 2009-2020 Free Software
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
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#include <config.h>

/* Specification.  */
#include "basename-lgpl.h"

#include <stdbool.h>
#include <string.h>

#include "filename.h"

char *
last_component (char const *name)
{
  char const *base = name + FILE_SYSTEM_PREFIX_LEN (name);
  char const *p;
  bool last_was_slash = false;

  while (ISSLASH (*base))
    base++;

  for (p = base; *p; p++)
    {
      if (ISSLASH (*p))
        last_was_slash = true;
      else if (last_was_slash)
        {
          base = p;
          last_was_slash = false;
        }
    }

  return (char *) base;
}

size_t
base_len (char const *name)
{
  size_t len;
  size_t prefix_len = FILE_SYSTEM_PREFIX_LEN (name);

  for (len = strlen (name);  1 < len && ISSLASH (name[len - 1]);  len--)
    continue;

  if (DOUBLE_SLASH_IS_DISTINCT_ROOT && len == 1
      && ISSLASH (name[0]) && ISSLASH (name[1]) && ! name[2])
    return 2;

  if (FILE_SYSTEM_DRIVE_PREFIX_CAN_BE_RELATIVE && prefix_len
      && len == prefix_len && ISSLASH (name[prefix_len]))
    return prefix_len + 1;

  return len;
}
