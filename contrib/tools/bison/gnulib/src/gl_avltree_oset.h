/* Ordered set data type implemented by a binary tree.
   Copyright (C) 2006, 2009-2013 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2006.

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

#ifndef _GL_AVLTREE_OSET_H
#define _GL_AVLTREE_OSET_H

#include "gl_oset.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const struct gl_oset_implementation gl_avltree_oset_implementation;
#define GL_AVLTREE_OSET &gl_avltree_oset_implementation

#ifdef __cplusplus
}
#endif

#endif /* _GL_AVLTREE_OSET_H */
