/* Scan memory for the first of two bytes.
   Copyright (C) 2008-2013 Free Software Foundation, Inc.

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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return the first address of either C1 or C2 (treated as unsigned
   char) that occurs within N bytes of the memory region S.  If
   neither byte appears, return NULL.  */

extern void *memchr2 (void const *s, int c1, int c2, size_t n)
  _GL_ATTRIBUTE_PURE;

#ifdef __cplusplus
}
#endif
