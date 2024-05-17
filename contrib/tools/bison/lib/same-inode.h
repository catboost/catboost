/* Determine whether two stat buffers refer to the same file.

   Copyright (C) 2006, 2009-2013 Free Software Foundation, Inc.

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

#ifndef SAME_INODE_H
# define SAME_INODE_H 1

# ifdef __VMS
#  define SAME_INODE(a, b)             \
    ((a).st_ino[0] == (b).st_ino[0]    \
     && (a).st_ino[1] == (b).st_ino[1] \
     && (a).st_ino[2] == (b).st_ino[2] \
     && (a).st_dev == (b).st_dev)
# else
#  define SAME_INODE(a, b)    \
    ((a).st_ino == (b).st_ino \
     && (a).st_dev == (b).st_dev)
# endif

#endif
