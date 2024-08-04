/* Concatenate path components.
   Copyright (C) 2018-2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 3 of the License, or any
   later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Akim Demaille <akim@lrde.epita.fr>.  */

#include <config.h>

/* Specification.  */
#include "path-join.h"

#include <concat-filename.h> /* xconcatenated_filename */
#include <filename.h> /* IS_ABSOLUTE_PATH */
#include <xalloc.h> /* xstrdup */

char *
xpath_join (const char *path1, const char *path2)
{
  if (!path2 || !*path2)
    return xstrdup (path1);
  else if (IS_ABSOLUTE_PATH (path2))
    return xstrdup (path2);
  else
    return xconcatenated_filename (path1, path2, NULL);
}
