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
#include "verify.h"

/* Evaluate an assertion E that is guaranteed to be true.
   If NDEBUG is not defined, abort the program if E is false.
   If NDEBUG is defined, the compiler can assume E and behavior is
   undefined if E is false, fails to evaluate, or has side effects.

   Unlike standard 'assert', this macro evaluates E even when NDEBUG
   is defined, so as to catch typos, avoid some GCC warnings, and
   improve performance when E is simple enough.

   Also see the documentation for 'assume' in verify.h.  */

#ifdef NDEBUG
# define affirm(E) assume (E)
#else
# define affirm(E) assert (E)
#endif

/* Check E's value at runtime, and report an error and abort if not.
   However, do nothing if NDEBUG is defined.

   Unlike standard 'assert', this macro compiles E even when NDEBUG
   is defined, so as to catch typos and avoid some GCC warnings.
   Unlike 'affirm', it is OK for E to use hard-to-optimize features,
   since E is not executed if NDEBUG is defined.  */

#ifdef NDEBUG
# define assure(E) ((void) (0 && (E)))
#else
# define assure(E) assert (E)
#endif

#endif
