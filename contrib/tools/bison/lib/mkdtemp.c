/* Copyright (C) 1999, 2001-2003, 2006-2007, 2009-2013 Free Software
   Foundation, Inc.
   This file is part of the GNU C Library.

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

/* Extracted from misc/mkdtemp.c.  */

#include <config.h>

/* Specification.  */
#include <stdlib.h>

#include "tempname.h"

/* Generate a unique temporary directory from XTEMPLATE.
   The last six characters of XTEMPLATE must be "XXXXXX";
   they are replaced with a string that makes the filename unique.
   The directory is created, mode 700, and its name is returned.
   (This function comes from OpenBSD.) */
char *
mkdtemp (char *xtemplate)
{
  if (gen_tempname (xtemplate, 0, 0, GT_DIR))
    return NULL;
  else
    return xtemplate;
}
