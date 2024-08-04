/* Run-time assert-like macros.

   Copyright (C) 2014-2020 Free Software Foundation, Inc.

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

/* Written by Paul Eggert.  */

#ifndef _GL_ASSURE_H
#define _GL_ASSURE_H

#include <assert.h>

/* Check E's value at runtime, and report an error and abort if not.
   However, do nothing if NDEBUG is defined.

   Unlike standard 'assert', this macro always compiles E even when NDEBUG
   is defined, so as to catch typos and avoid some GCC warnings.  */

#ifdef NDEBUG
# define assure(E) ((void) (0 && (E)))
#else
# define assure(E) assert (E)
#endif

#endif
